"""Pure recorded-array hypotheses never certify a physical OpenPI YAM mapping."""

import ast
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from yamkit.inference.mapping import YAM_NAMES
from yamkit.openpi import contract
from yamkit.openpi import yam_candidate as candidate

CORPUS = "a" * 64


def state():
    values = np.arange(14, dtype=np.float64) / 10
    values[[6, 13]] = [0.1, 0.8]
    return values


def commands():
    values = np.tile(state(), (50, 1))
    values[:, list(candidate.JOINTS)] += np.arange(50)[:, None] * 0.02
    values[:, 6] = np.linspace(0, 1, 50)
    values[:, 13] = np.linspace(1, 0, 50)
    return values


def quantiles(low=-1, high=1, **kwargs):
    return candidate.CandidateQuantiles(np.full(14, low, dtype=float), np.full(14, high, dtype=float),
                                        kwargs.get("provenance", "saved paired corpus, not pretrained assets"),
                                        kwargs.get("corpus_sha256", CORPUS))


def statistics():
    return candidate.CandidateStatistics(quantiles(-2, 2), quantiles(-0.5, 0.5))


def episode(name="episode-0", count=70, offset=0, sample_hz=30):
    measured = np.tile(state(), (count, 1))
    measured[:, list(candidate.JOINTS)] += offset + np.arange(count)[:, None] * 0.01
    measured[:, 6] = np.linspace(0.1, 0.7, count)
    measured[:, 13] = np.linspace(0.8, 0.2, count)
    sent = measured.copy()
    sent[:, list(candidate.JOINTS)] += 0.002
    sent[:, 6] = np.linspace(0, 1, count)
    sent[:, 13] = np.linspace(1, 0, count)
    return candidate.PairedEpisode(name, measured, sent, YAM_NAMES, sample_hz)


def test_named_state_order_joint_frame_and_gripper_closure_are_explicit():
    original = state()
    result = candidate.candidate_state(original)
    assert YAM_NAMES == tuple(f"{side}_{name}.pos" for side in ("left", "right")
                              for name in ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5",
                                           "joint_6", "gripper"))
    np.testing.assert_array_equal(result[list(candidate.JOINTS)], original[list(candidate.JOINTS)])
    np.testing.assert_allclose(result[[6, 13]], [0.9, 0.2])
    np.testing.assert_array_equal(original, state())


def test_every_action_is_relative_to_one_measured_anchor_not_previous_command_or_dt():
    actual = candidate.candidate_action_chunk(state(), commands())
    np.testing.assert_allclose(actual[:, 0], np.arange(50) * 0.02)
    assert actual[-1, 0] == pytest.approx(0.98)
    assert actual[-1, 0] != pytest.approx(0.02)
    np.testing.assert_allclose(actual[:, 6], 1 - commands()[:, 6])
    np.testing.assert_allclose(actual[:, 13], 1 - commands()[:, 13])
    reconstructed = candidate.reconstruct_yam_chunk(state(), actual)
    np.testing.assert_allclose(reconstructed, commands(), atol=1e-15)
    changed_anchor = state()
    changed_anchor[0] += 0.7
    assert candidate.reconstruct_yam_chunk(changed_anchor, actual)[-1, 0] == pytest.approx(1.68)


@pytest.mark.parametrize("shape", [(7,), (32,), (1, 14), (14, 1)])
def test_measured_state_shape_never_pads_or_truncates(shape):
    with pytest.raises(ValueError, match="exact shape"):
        candidate.candidate_state(np.zeros(shape))


@pytest.mark.parametrize("shape", [(49, 14), (51, 14), (50, 32), (1, 50, 14)])
def test_action_chunk_requires_exact_full_horizon(shape):
    with pytest.raises(ValueError, match="exact shape"):
        candidate.candidate_action_chunk(state(), np.zeros(shape))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_input_output_and_stats_are_not_repaired(bad):
    observed = state()
    observed[0] = bad
    with pytest.raises(ValueError, match="finite"):
        candidate.candidate_state(observed)
    raw = np.zeros((50, 32))
    raw[10, 31] = bad  # even unused model columns are validated
    with pytest.raises(ValueError, match="finite"):
        candidate.decode_chunk(raw, state(), statistics())
    with pytest.raises(ValueError, match="finite"):
        quantiles(low=bad)


@pytest.mark.parametrize("bad", [-0.000001, 1.000001])
def test_source_grippers_and_inverse_closure_are_rejected_not_clipped(bad):
    observed = state()
    observed[6] = bad
    with pytest.raises(ValueError, match="no clipping"):
        candidate.candidate_state(observed)
    sent = commands()
    sent[0, 13] = bad
    with pytest.raises(ValueError, match="no clipping"):
        candidate.candidate_action_chunk(state(), sent)
    with pytest.raises(ValueError, match="no clipping"):
        candidate.reconstruct_yam_chunk(state(), sent)


@pytest.mark.parametrize("values", [np.full(14, "1"), np.zeros(14, dtype=bool),
                                    np.ones(14, dtype=complex)])
def test_nonreal_or_coerced_strings_rejected(values):
    with pytest.raises(ValueError, match="finite real"):
        candidate.candidate_state(values)


def test_native_quantile_epsilon_order_and_no_unit_range_clipping():
    q = quantiles(-0.3, 0.7)
    values = np.linspace(-2, 3, 14)
    actual = candidate.normalize_14(values, q)
    expected = (values - np.asarray(q.q01)) / (np.asarray(q.q99) - q.q01 + 1e-6) * 2.0 - 1.0
    np.testing.assert_array_equal(actual, expected)
    assert actual.min() < -1 and actual.max() > 1
    reconstructed = candidate.unnormalize_14(actual, q)
    expected_inverse = (actual + 1.0) / 2.0 * (np.asarray(q.q99) - q.q01 + 1e-6) + q.q01
    np.testing.assert_array_equal(reconstructed, expected_inverse)
    np.testing.assert_allclose(reconstructed, values, atol=1e-15)
    # The epsilon means normalized +1 maps slightly ABOVE q99; don't pretend it is clipped.
    np.testing.assert_allclose(candidate.unnormalize_14(np.ones(14), q), np.asarray(q.q99) + 1e-6)


def test_encoded_state_remains14_for_native_tokenization_before_padding():
    encoded = candidate.encode_state(state(), statistics())
    assert encoded.shape == (14,)
    np.testing.assert_array_equal(encoded,
                                  candidate.normalize_14(candidate.candidate_state(state()), statistics().state))


def test_decode_preserves_raw32_unused18_dtype_and_explicit_unqualified_identity():
    stats = statistics()
    original = np.arange(1600, dtype=np.float32).reshape(50, 32) / 200 - 3
    raw = original.copy()
    decoded = candidate.decode_chunk(raw, state(), stats)
    raw[:] = 0
    assert decoded["raw_normalized_model_chunk"].dtype == np.float32
    np.testing.assert_array_equal(decoded["raw_normalized_model_chunk"], original)
    np.testing.assert_array_equal(decoded["unused_normalized_model_dimensions"], original[:, 14:])
    assert decoded["proposed_yam_absolute_commands"].shape == (50, 14)
    assert not decoded["clipping_applied"]
    assert decoded["out_of_range_gripper_scalars"] > 0
    assert not decoded["proposed_gripper_range_passes"]
    assert not decoded["joint_bounds_checked"]
    assert "not proven YAM padding" in decoded["unused_dimensions_assumption"]
    for field in ("qualified_for_yam", "physical_ready", "hardware_tested"):
        assert decoded[field] is False
    assert decoded["model_rows_sent_to_robot"] == 0
    with pytest.raises(ValueError, match="physical rollout is blocked"):
        contract.require_yam_contract()


def test_synthetic_normalized_chunk_roundtrips_candidate_without_dispatch():
    stats = statistics()
    transformed = candidate.candidate_action_chunk(state(), commands())
    raw = np.zeros((50, 32))
    raw[:, :14] = candidate.normalize_14(transformed, stats.actions)
    decoded = candidate.decode_chunk(raw, state(), stats)
    np.testing.assert_allclose(decoded["proposed_yam_absolute_commands"], commands(), atol=1e-15)
    assert not decoded["physical_ready"]


@pytest.mark.parametrize("low,high", [(0, 0), (1, -1), (0, 0.000001), (0, 0.00009)])
def test_degenerate_or_unstable_spans_rejected_without_auto_expansion(low, high):
    with pytest.raises(ValueError, match="declared 1e-4 floor"):
        quantiles(low, high)


@pytest.mark.parametrize("shape", [(13,), (15,), (1, 14), (32,)])
def test_statistics_dimensions_are_exact(shape):
    with pytest.raises(ValueError, match="exact shape"):
        candidate.CandidateQuantiles(np.zeros(shape), np.ones(shape), "source", CORPUS)


@pytest.mark.parametrize("kwargs", [{"provenance": ""}, {"provenance": None},
                                     {"corpus_sha256": "x" * 64}, {"corpus_sha256": "A" * 64},
                                     {"corpus_sha256": ""}])
def test_statistics_require_real_provenance_and_corpus_identity(kwargs):
    with pytest.raises(ValueError):
        quantiles(**kwargs)


def test_statistics_are_copied_immutable_and_hash_sensitive():
    low, high = np.zeros(14), np.ones(14)
    q = candidate.CandidateQuantiles(low, high, "first", CORPUS)
    low[:] = -20
    assert q.q01 == (0,) * 14
    one = candidate.CandidateStatistics(q, q).metadata()
    two = candidate.CandidateStatistics(quantiles(0, 1, provenance="second"), q).metadata()
    assert one["candidate_statistics_sha256"] != two["candidate_statistics_sha256"]
    json.dumps(one, allow_nan=False)
    with pytest.raises(ValueError, match="same paired corpus"):
        candidate.CandidateStatistics(q, quantiles(corpus_sha256="b" * 64))
    with pytest.raises(TypeError, match="Explicit state and action"):
        candidate.CandidateStatistics(None, None)


def test_complete_windows_use_paired_commands_and_do_not_cross_episode_boundaries():
    episodes = [episode(count=53), episode("episode-1", count=52, offset=10)]
    estimated = candidate.estimate_statistics(episodes, provenance="paired original logs", corpus_sha256=CORPUS)
    manual_states, manual_actions = [], []
    for ep in episodes:
        for start in range(len(ep.measured_states) - 49):
            manual_states.append(candidate.candidate_state(ep.measured_states[start]))
            manual_actions.extend(candidate.candidate_action_chunk(ep.measured_states[start],
                                                                   ep.absolute_commands[start:start + 50]))
    expected = np.quantile(manual_actions, [0.01, 0.99], axis=0, method="linear")
    np.testing.assert_array_equal(estimated.actions.q01, expected[0])
    np.testing.assert_array_equal(estimated.actions.q99, expected[1])
    report = candidate.audit_heldout(episodes, estimated)
    assert report["complete_windows"] == 7
    assert report["action_support"]["sample_count"] == 350
    assert report["state_support"]["sample_count"] == 7
    assert not report["cross_episode_padding"]
    assert not report["timestamp_alignment_checked"]
    assert report["state_support"]["max_abs_round_trip_error"] < 1e-12
    assert report["action_support"]["max_abs_round_trip_error"] < 1e-12
    assert not report["support_is_compatibility_proof"]
    assert not report["physical_ready"]


def test_statistics_use_observed_chunk_origin_not_adjacent_measured_differences():
    ep = episode()
    stats = candidate.estimate_statistics([ep], provenance="paired", corpus_sha256=CORPUS)
    # Adjacent measured deltas are constant .01; command chunk deltas span ~0..49*.01.
    assert stats.actions.q99[0] > 0.47
    assert stats.actions.q01[0] < 0.02


def test_constant_recording_cannot_fabricate_a_normalization_scale():
    ep = episode()
    observed = ep.measured_states.copy()
    observed[:, 1] = 0
    bad = candidate.PairedEpisode("constant-joint", observed, ep.absolute_commands, YAM_NAMES, 30)
    with pytest.raises(ValueError, match="quantile spans"):
        candidate.estimate_statistics([bad], provenance="constant joint", corpus_sha256=CORPUS)


def test_heldout_outliers_are_counted_not_clipped_or_called_ready():
    ep = episode()
    stats = candidate.estimate_statistics([ep], provenance="first recording", corpus_sha256=CORPUS)
    shifted = episode("heldout", offset=100)
    report = candidate.audit_heldout([shifted], stats)
    assert report["state_support"]["outside_normalized_unit_range_count"][0] == 21
    assert report["state_support"]["normalized_max"][0] > 1
    assert not report["physical_ready"]


@pytest.mark.parametrize("rate", [0, -1, np.inf, np.nan, True, "30", 1j])
def test_invalid_declared_sample_rate_rejected(rate):
    with pytest.raises(ValueError, match="sample rate"):
        episode(sample_hz=rate)


def test_episode_order_pairing_window_budget_and_provenance_fail_closed():
    ep = episode()
    with pytest.raises(ValueError, match="exact YAM order"):
        candidate.PairedEpisode("bad order", ep.measured_states, ep.absolute_commands, tuple(reversed(YAM_NAMES)), 30)
    with pytest.raises(ValueError, match="exact shape"):
        candidate.PairedEpisode("unpaired", ep.measured_states, ep.absolute_commands[:-1], YAM_NAMES, 30)
    with pytest.raises(ValueError, match="complete 50-row"):
        episode(count=49)
    with pytest.raises(ValueError, match="Mixed episode timebases"):
        candidate.estimate_statistics([ep, episode("other", sample_hz=50)], provenance="paired", corpus_sha256=CORPUS)
    with pytest.raises(ValueError, match="identities must be unique"):
        candidate.audit_heldout([ep, ep], statistics())
    with pytest.raises(ValueError, match="20000"):
        candidate.audit_heldout([episode(count=20_050)], statistics())
    with pytest.raises(ValueError, match="state-only"):
        candidate.estimate_statistics([state()], provenance="npz state only", corpus_sha256=CORPUS)
    with pytest.raises(ValueError, match="provenance"):
        candidate.estimate_statistics([ep], provenance=None, corpus_sha256=CORPUS)


def test_module_is_pure_arithmetic_with_no_hardware_runtime_or_file_entrypoint():
    tree = ast.parse(Path(candidate.__file__).read_text())
    imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    imports |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    assert imports == {"__future__", "hashlib", "json", "re", "collections.abc", "dataclasses",
                       "numpy", "yamkit.inference.mapping"}
    forbidden_calls = {"open", "connect", "send_action", "send", "recv", "run", "Popen", "load", "save"}
    called = {node.func.id if isinstance(node.func, ast.Name) else node.func.attr
              for node in ast.walk(tree) if isinstance(node, ast.Call)
              and isinstance(node.func, (ast.Name, ast.Attribute))}
    assert not called & forbidden_calls
    assert "absolute_commands" in inspect.signature(candidate.PairedEpisode).parameters
    with pytest.raises(TypeError):
        candidate.PairedEpisode("state-only", np.zeros((60, 14)))
