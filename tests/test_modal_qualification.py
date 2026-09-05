"""Qualification collection never needs actual hardware or paid test resources."""

import json
from types import SimpleNamespace

import pytest

from yamkit.config import RigConfig
from yamkit.inference import qualification
from yamkit.modal_qualification import collect_qualification


def test_failed_direct_collection_saves_invalid_record_without_more_remote_requests(monkeypatch, tmp_path):
    path = tmp_path / "rig.yaml"
    rig = RigConfig(cameras={name: {"type": "opencv", "index_or_path": index,
                                   "height": 480, "width": 640, "fps": 30}
                            for index, name in enumerate(("top", "left_wrist", "right_wrist"))})
    rig.save(path)
    calls = []
    monkeypatch.setattr(qualification, "DATA_DIR", tmp_path)
    # A new failed attempt must retire the previous success atomically.
    prior_path = qualification.save_qualification({"settings": {"profile": "molmoact2"},
                                                  "assessment": {"qualified": True}, "old_record": True})

    def transport(*args, **kwargs):
        calls.append(kwargs)
        return object()

    benchmark = SimpleNamespace(make_benchmark_transport=transport,
                                profile_modal=lambda *a, **k: {"terminated": "failure", "warm_sample_count": 0,
                                                              "readiness": {}},
                                run_scenario=lambda *a, **k: pytest.fail("Failed profile must not dispatch again"))
    monkeypatch.setattr("yamkit.modal_qualification._benchmark_module", lambda: benchmark)
    monkeypatch.setattr("yamkit.modal_ops.owned_service", lambda: {
        "app_name": "fake-qualified-service", "region": "us-west", "routing_region": "us-west"})
    result = collect_qualification(rig_path=path, requests=50)
    assert result["assessment"]["qualified"] is False
    assert result["status"] == "QUALIFICATION_FAILED"
    assert result["hardware_tested"] is False
    saved = json.loads(prior_path.read_text())
    assert saved["assessment"]["qualified"] is False and "old_record" not in saved
    assert saved["settings"]["image_hw"] == [480, 640]
    assert saved["settings"]["requested_region"] == saved["settings"]["routing_region"] == "us-west"
    assert saved["settings"]["observed_region"] is None
    assert saved["direct"]["readiness"] == {} and saved["integrated"] == {}
    assert result["qualification_path"] == str(prior_path)
    assert len(calls) == 1 and calls[0]["call_mode"] == "remote"


def test_qualification_rejects_short_sample_budget_before_service_lookup(monkeypatch):
    monkeypatch.setattr("yamkit.modal_ops.owned_service", lambda: pytest.fail("Invalid input must not use a service"))
    with pytest.raises(ValueError, match="50–200"):
        collect_qualification(requests=49)
