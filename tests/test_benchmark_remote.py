"""Bounded remote diagnostics use fake transports and never activate hardware."""

import asyncio
from types import SimpleNamespace

import pytest

from scripts.benchmark_remote import (
    compact_report,
    compare_denoising,
    compare_encodings,
    compare_quality_sweep,
    modal_sdk_measurements,
    profile_modal,
)
from yamkit.inference.performance import percentile_summary, summarize_measurements
from yamkit.inference.profiles import get_profile


def test_report_separates_first_request_and_marks_short_samples_incomplete():
    samples = [{"round_trip_s": 20.0, "instance_id": "same"},
               {"round_trip_s": 0.2, "instance_id": "same", "server_timing": {
                   "inference_s": 0.1, "preprocess_steps_s": {"0:Tokenizer": 0.01}}},
               {"round_trip_s": 0.4, "instance_id": "same", "server_timing": {
                   "inference_s": 0.2, "preprocess_steps_s": {"0:Tokenizer": 0.02}}}]
    report = summarize_measurements(samples)
    assert report["first_request"]["round_trip_s"] == 20
    assert report["warm_round_trip_s"]["p50"] == pytest.approx(0.3)
    assert report["warm_stages"]["server_timing.inference_s"]["p95"] == pytest.approx(0.195)
    assert report["warm_stages"]["server_timing.preprocess_steps_s.0:Tokenizer"]["p50"] == pytest.approx(0.015)
    assert not report["warm_sample_requirement_met"]
    assert report["container_instance_count"] == 1
    assert percentile_summary([]) == {"sample_count": 0}
    with pytest.raises(ValueError, match="finite"):
        percentile_summary([float("nan")])


def fake_profile_transport(monkeypatch, *, changed_instance_at=None):
    profile = get_profile("molmoact2")
    clock = [10.0]
    monkeypatch.setattr("scripts.benchmark_remote.time", SimpleNamespace(monotonic=lambda: clock[0]))

    class Transport:
        def __init__(self):
            self.last_timing = {}
            self.cancelled = False
            self.requests = []

        def ready(self, timeout_s):
            clock[0] += 5
            return {**profile.metadata(), "ready": True, "fresh_chunk": True,
                    "saved_processors": True, "instance_id": "same", "prediction_count": 0}

        def predict_chunk(self, request, timeout_s):
            self.requests.append(request)
            clock[0] += 1.5 if request["sequence_id"] == 0 else 0.3
            return {**{key: request[key] for key in ("protocol_version", "profile", "model_revision",
                                                    "session_id", "sequence_id", "observation_time")},
                    "instance_id": "other" if request["sequence_id"] == changed_instance_at else "same",
                    "action_units": "checkpoint_native", "action_names": list(profile.action_names),
                    "diagnostic_seed": request.get("diagnostic_seed"),
                    "diagnostic_num_inference_steps": request.get("diagnostic_num_inference_steps"),
                    "diagnostic_cuda_graph": request.get("diagnostic_cuda_graph"),
                    "model_execution": {"default_num_inference_steps": 10,
                                        "effective_num_inference_steps": request.get("diagnostic_num_inference_steps", 10),
                                        "cuda_graph_enabled": request.get("diagnostic_cuda_graph", False)},
                    "chunk": [[0.0] * len(profile.action_names) for _ in range(profile.chunk_size)],
                    "timing": {"preprocess_s": 0.02, "inference_s": 0.2, "postprocess_s": 0.02, "total_s": 0.24},
                    "lifecycle": {"prediction_count": request["sequence_id"] + 1,
                                  "first_prediction": request["sequence_id"] == 0}}

        def cancel(self):
            self.cancelled = True

    return Transport()


def test_nonexecutable_modal_profile_collects_100_warm_without_weakening_action_age(monkeypatch):
    transport = fake_profile_transport(monkeypatch)
    report = profile_modal(transport, warm_samples=100, image_hw=(8, 8), image_encoding="rgb8")
    assert report["warm_sample_count"] == 100 and report["warm_sample_requirement_met"]
    assert report["first_request"]["round_trip_s"] == 1.5
    assert report["warm_round_trip_s"]["p99"] == pytest.approx(0.3)
    assert report["readiness_s"] == 5
    assert report["queue_depth"] is None and report["underruns"] is None
    assert not report["physical_modal_rollout_allowed"] and not report["readiness_cold_start_verified"]
    assert transport.cancelled
    assert len(transport.requests) == 101
    assert all(request["mode"] == "native_fixture" for request in transport.requests)
    assert report["warm_stages"]["payload_bytes"]["p50"] == 3 * 8 * 8 * 3
    assert set(transport.requests[0]["images"]) == {"top", "left", "right"}


def test_paired_encoding_comparison_uses_identical_fixtures_and_seed(monkeypatch):
    transport = fake_profile_transport(monkeypatch)
    report = compare_encodings(transport, pairs=2, image_hw=(8, 8))
    assert [request["diagnostic_seed"] for request in transport.requests] == [0, 0, 1, 1]
    assert [request["images"]["top"]["encoding"] for request in transport.requests] == ["rgb8", "jpeg"] * 2
    assert transport.requests[0]["images"]["top"]["data"] == transport.requests[2]["images"]["top"]["data"]
    assert transport.requests[1]["images"]["top"]["data"] == transport.requests[3]["images"]["top"]["data"]
    assert all(pair["action_max_absolute_delta"] == 0 for pair in report["pairs"])
    assert transport.cancelled


def test_quality_sweep_uses_common_noise_all_cameras_and_smallest_passing_payload(monkeypatch):
    transport = fake_profile_transport(monkeypatch)
    predict = transport.predict_chunk

    def response_with_codec_delta(request, timeout_s):
        response = predict(request, timeout_s)
        quality = request["images"]["top"].get("quality")
        for row in response["chunk"]:
            row[6] = {85: 0.03, 90: 0.015, 95: 0.005}.get(quality, 0)
            row[0] = 0.005 if quality is not None else 0
        return response

    transport.predict_chunk = response_with_codec_delta
    report = compare_quality_sweep(transport, pairs=2, image_hw=(64, 64))
    assert report["selected_encoding"] == "jpeg" and report["selected_jpeg_quality"] == 90
    assert report["raw_repeat_deterministic"] and report["experiment_only"]
    assert report["gripper_delta_limit"] == 0.02 and report["joint_delta_limit_rad"] == 0.01
    assert [request["diagnostic_seed"] for request in transport.requests] == [0] * 5 + [1] * 5
    for request in transport.requests:
        assert len(request["images"]) == 3
        assert len({(frame["encoding"], frame.get("quality")) for frame in request["images"].values()}) == 1
        assert "diagnostic_num_inference_steps" not in request and "diagnostic_cuda_graph" not in request
    assert transport.requests[0]["images"] == transport.requests[1]["images"] == transport.requests[5]["images"]
    sample = report["pairs"][0]["variants"]["jpeg_q90"]
    assert sample["gripper_max_absolute_delta"] == 0.015
    assert sample["joint_max_absolute_delta_rad"] == 0.005
    assert all("encode_s" in camera and "diagnostic_decode_s" in camera and camera["payload_bytes"] > 0
               for camera in sample["per_camera"].values())
    assert transport.cancelled


@pytest.mark.parametrize("rejection", ["gripper", "joint", "nondeterministic_raw"])
def test_quality_selection_falls_back_to_raw_when_output_limits_fail(monkeypatch, rejection):
    transport = fake_profile_transport(monkeypatch)
    predict = transport.predict_chunk

    def divergent(request, timeout_s):
        response = predict(request, timeout_s)
        if request["images"]["top"]["encoding"] == "jpeg":
            response["chunk"][0][6 if rejection == "gripper" else 0] = 0.03 if rejection != "nondeterministic_raw" else 0
        elif rejection == "nondeterministic_raw" and request["sequence_id"] % 5 == 1:
            response["chunk"][0][0] = 0.001
        return response

    transport.predict_chunk = divergent
    report = compare_quality_sweep(transport, pairs=1, image_hw=(8, 8))
    assert report["selected_encoding"] == "rgb8" and report["selected_jpeg_quality"] is None


def test_denoising_compares_fixed_noise_separates_first_forward_and_keeps_overrides_diagnostic(monkeypatch):
    transport = fake_profile_transport(monkeypatch)
    predict = transport.predict_chunk

    def measured(request, timeout_s):
        response = predict(request, timeout_s)
        response["timing"]["inference_s"] = 0.1 if request.get("diagnostic_num_inference_steps") == 5 else 0.2
        return response

    transport.predict_chunk = measured
    report = compare_denoising(transport, pairs=11, image_hw=(8, 8), include_graph=True)
    assert len(transport.requests) == 33 and report["experiment_only"]
    for seed in range(11):
        default, half, graph = transport.requests[seed * 3:seed * 3 + 3]
        assert [request["diagnostic_seed"] for request in (default, half, graph)] == [seed] * 3
        assert default["images"] == half["images"] == graph["images"]
        assert all(image["encoding"] == "rgb8" for image in default["images"].values())
        assert "diagnostic_num_inference_steps" not in default and "diagnostic_cuda_graph" not in default
        assert half["diagnostic_num_inference_steps"] == 5 and "diagnostic_cuda_graph" not in half
        assert graph["diagnostic_cuda_graph"] is True and "diagnostic_num_inference_steps" not in graph
    assert all(variant["warm_sample_count"] == 10 for variant in report["variants"].values())
    assert report["variants"]["raw_default_steps"]["warm_model_forward_s"]["p50"] == 0.2
    assert report["variants"]["raw_half_steps"]["warm_model_forward_s"]["p95"] == 0.1


def test_explicit_profile_experiments_are_reported_and_default_requests_have_no_override(monkeypatch):
    transport = fake_profile_transport(monkeypatch)
    report = profile_modal(transport, warm_samples=1, image_hw=(8, 8),
                           diagnostic_num_inference_steps=5, diagnostic_cuda_graph=True)
    assert report["experiment_only"] and report["sample_count"] == 2
    assert all(request["diagnostic_num_inference_steps"] == 5 and request["diagnostic_cuda_graph"] is True
               for request in transport.requests)
    assert all(sample["model_execution"]["effective_num_inference_steps"] == 5 for sample in report["samples"])
    default = fake_profile_transport(monkeypatch)
    report = profile_modal(default, warm_samples=1, image_hw=(8, 8))
    assert not report["experiment_only"]
    assert report["image_encoding"] == "rgb8" and report["jpeg_quality"] is None
    assert all(image["encoding"] == "rgb8" for request in default.requests for image in request["images"].values())
    assert all("diagnostic_num_inference_steps" not in request and "diagnostic_cuda_graph" not in request
               for request in default.requests)


def test_paired_sweep_wall_budget_retains_partial_evidence_and_does_not_select_jpeg(monkeypatch):
    transport = fake_profile_transport(monkeypatch)
    report = compare_quality_sweep(transport, pairs=4, image_hw=(8, 8), max_wall_s=1.5)
    assert report["terminated"] == "wall_budget" and len(transport.requests) == 1
    assert report["selected_encoding"] == "rgb8" and transport.cancelled


def test_modal_profile_excludes_container_change_and_retains_bounded_partial_results(monkeypatch):
    transport = fake_profile_transport(monkeypatch, changed_instance_at=2)
    report = profile_modal(transport, warm_samples=100, image_hw=(8, 8))
    assert report["terminated"] == "failure"
    assert report["sample_count"] == 2
    assert report["container_instance_count"] == 1
    assert not report["warm_sample_requirement_met"]
    assert report["failures"][0]["reason"] == "ValueError"
    assert transport.cancelled


def test_modal_profile_wall_budget_stops_before_extra_dispatch(monkeypatch):
    transport = fake_profile_transport(monkeypatch)
    report = profile_modal(transport, warm_samples=100, max_wall_s=6.5, image_hw=(8, 8))
    assert report["terminated"] == "wall_budget"
    assert report["sample_count"] == 1 and not report["warm_sample_requirement_met"]
    assert len(transport.requests) == 1


def test_sdk_instrumentation_observes_one_serialization_and_restores_original_hooks(monkeypatch):
    sdk = pytest.importorskip("modal._utils.function_utils")
    serializations = []

    def serialize(value, data_format):
        serializations.append(value)
        return b"serialized request"

    async def upload(data, stub):
        return "fake-blob", False, 100

    async def download(blob, stub):
        return b"serialized response"

    monkeypatch.setattr(sdk, "_serialize_data_format", serialize)
    monkeypatch.setattr(sdk, "deserialize_data_format", lambda data, *args: {"decoded": True})
    monkeypatch.setattr(sdk, "blob_upload_with_r2_failure_info", upload)
    monkeypatch.setattr(sdk, "blob_download", download)
    with modal_sdk_measurements() as metrics:
        encoded = sdk._serialize_data_format({"fixture": True}, 1)
        assert asyncio.run(sdk.blob_upload_with_r2_failure_info(encoded, None))[0] == "fake-blob"
        result = asyncio.run(sdk.blob_download("fake-blob", None))
        assert sdk.deserialize_data_format(result, 1, None) == {"decoded": True}
        assert metrics["request_serialization_calls"] == 1
        assert metrics["serialized_request_bytes"] == metrics["input_blob_bytes"] == len(encoded)
        assert metrics["serialized_response_bytes"] == metrics["output_blob_bytes"] == len(result)
    assert len(serializations) == 1
    assert sdk._serialize_data_format is serialize
    assert sdk.blob_upload_with_r2_failure_info is upload


def test_compact_queue_report_preserves_unknown_initial_deadline_margin():
    scenario = dict.fromkeys(("name", "injected_delay_cycle_s", "elapsed_s", "sample_count", "warm_sample_count",
                             "warm_round_trip_s", "failed", "underruns", "executed_actions",
                             "expired_prefix_dropped", "overlap_prefix_dropped", "expired_chunks",
                             "peak_queue_depth", "all_fake_robots_released", "error"), 0)
    scenario.update(prediction_samples=[{"next_action_margin_at_start_s": None},
                                       {"next_action_margin_at_start_s": 0.1}],
                    image_hw=[480, 640], sdk_commands_during_completed_rpc=[])
    result = compact_report({"measurement": "synthetic regression fixture", "scenarios": [scenario]})
    assert result["scenarios"][0]["next_action_margin_at_start_s_range"] == [0.1, 0.1]
    assert result["scenarios"][0]["next_action_margin_at_return_s_range"] is None
