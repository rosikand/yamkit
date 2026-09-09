"""Render generated saved data only; no hardware, credentials or cloud services."""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/render_rollout_trace.py"
spec = importlib.util.spec_from_file_location("render_rollout_trace", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    summary = {"synthetic_fixture": True, "phase_started_monotonic_s": 100.0,
               "action_names": [f"{side}_{joint}" for side in module.SIDES for joint in module.JOINTS],
               "task": '<script>alert("test")</script>', "counts": {"<img onerror=test>": 2},
               "resources_released": True, "overflow": False}
    trace = {"events": [
        {"kind": "observation", "monotonic_s": 100.0, "positions": [0.0] * 14},
        {"kind": "send_start", "monotonic_s": 100.1, "arm": "left_follower",
         "requested": dict.fromkeys(module.JOINTS, .2)},
        {"kind": "send_end", "monotonic_s": 100.2, "arm": "left_follower",
         "postclamp": dict.fromkeys(module.JOINTS, .05)},
        {"kind": "chunk_merge", "monotonic_s": 100.3},
        {"kind": "send_error", "monotonic_s": 100.4, "arm": "right_follower"},
        {"kind": "observation", "monotonic_s": 101.0, "positions": [.1] * 14}], "chunks": []}
    return summary, trace


def saved(directory):
    summary, trace = fixture()
    (directory / "summary.json").write_text(json.dumps(summary))
    (directory / "trace.json").write_text(json.dumps(trace))
    return summary, trace


def test_series_preserves_missing_samples_and_failed_sends():
    values, merges, errors = module.series(*fixture())
    assert values["left"]["joint_1.pos"]["measured"] == [(0.0, 0.0), (1.0, .1)]
    assert len(values["left"]["joint_1.pos"]["requested"]) == 1
    assert values["right"]["joint_1.pos"]["sent"] == []
    assert merges == pytest.approx([.3]) and errors == pytest.approx([.4])


@pytest.mark.parametrize("count", [8192, 10802, 12288, 12289])
def test_renderer_accepts_collector_event_capacity_but_rejects_overflow(count):
    summary, trace = fixture()
    trace["events"] = [{"kind": "observation", "monotonic_s": 100 + index / count * 45,
                        "positions": [0.0] * 14} for index in range(count)]
    if count > 12288:
        with pytest.raises(ValueError, match="event count"):
            module.series(summary, trace)
    else:
        values, _, _ = module.series(summary, trace)
        assert len(values["left"]["joint_1.pos"]["measured"]) == count


@pytest.mark.parametrize("change", [
    lambda s, t: s.update(phase_started_monotonic_s=float("nan")),
    lambda s, t: s.update(action_names=list(reversed(s["action_names"]))),
    lambda s, t: t.update(events=[None]),
    lambda s, t: t["events"][0].update(positions=[0] * 13),
    lambda s, t: t["events"][0].update(positions=[float("inf")] * 14),
    lambda s, t: t["events"][0].update(positions=[True] * 14),
    lambda s, t: t["events"][1].update(requested={"joint_1.pos": .2}),
    lambda s, t: t["events"][1].update(arm="unexpected"),
    lambda s, t: t.update(chunks=[{"actions": [[0] * 14] * 29}]),
    lambda s, t: t.update(chunks=[{"actions": [[float("nan")] * 14] * 30}]),
])
def test_invalid_shapes_ordering_and_nonfinite_values_are_rejected(change):
    summary, trace = fixture()
    change(summary, trace)
    with pytest.raises(ValueError):
        module.series(summary, trace)


def test_json_bounded_regular_nonsymlink_and_finite(tmp_path, monkeypatch):
    source = tmp_path / "trace.json"
    source.write_text('{"value": NaN}')
    with pytest.raises(ValueError):
        module.read_json(source)
    source.write_text('{"value": 1e999}')
    with pytest.raises(ValueError):
        module.read_json(source)
    source.write_text('{}')
    link = tmp_path / "link.json"
    link.symlink_to(source)
    with pytest.raises(ValueError):
        module.read_json(link)
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError):
        module.read_json(fifo)
    monkeypatch.setattr(module, "MAX_JSON_BYTES", 1)
    with pytest.raises(ValueError):
        module.read_json(source)


def test_output_symlink_and_parent_escape_rejected_before_writing(tmp_path, monkeypatch):
    saved(tmp_path)
    target = tmp_path / "unchanged.txt"
    target.write_text("unchanged")
    (tmp_path / "joints-right.png").symlink_to(target)
    with pytest.raises(ValueError):
        module.render(tmp_path)
    assert target.read_text() == "unchanged" and not (tmp_path / "joints-left.png").exists()
    with pytest.raises(ValueError):
        module.repo_path(Path("/tmp/outside-yamkit-render"))
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        module.repo_path(alias / "report.html")
    monkeypatch.setattr(module, "ROOT", tmp_path / "different-root")
    with pytest.raises(ValueError):
        module.render(tmp_path)


@pytest.mark.parametrize("fps", [5, 30])
def test_html_escaped_synthetic_labeled_and_samples_never_connected(tmp_path, monkeypatch, fps):
    summary, _ = saved(tmp_path)
    summary["video_fps"] = fps
    if fps == 30:
        summary.update(video_timing='Original PTS, gaps retained <img onerror=bad>',
                       video_quality='Near-lossless <script>bad</script>',
                       capture_scope='Policy phase only; startup and return-home movement are not recorded')
        (tmp_path / "video_timeline.json").write_text('{}')
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    plots = []
    closed = []

    class Axis:
        def plot(self, *args, **kwargs):
            plots.append((args, kwargs))
        def axvline(self, *args, **kwargs):
            pass
        def set_ylabel(self, *args):
            pass
        def set_xlabel(self, *args):
            pass
        def grid(self, **kwargs):
            pass
        def legend(self, **kwargs):
            pass

    figure = SimpleNamespace(suptitle=lambda *_: None,
                             savefig=lambda path, **kw: path.write_bytes(b"synthetic-test-placeholder"))
    cache = module.ROOT / ".context" / "matplotlib"
    fake = SimpleNamespace(use=lambda *_: None, get_cachedir=lambda: str(cache), get_configdir=lambda: str(cache),
        pyplot=SimpleNamespace(subplots=lambda *a, **kw: (figure, [Axis() for _ in range(7)]),
                               close=lambda fig: closed.append(fig)))
    monkeypatch.setitem(sys.modules, "matplotlib", fake)
    report = module.render(tmp_path)
    document = report.read_text()
    assert "<script>" not in document and "&lt;script&gt;" in document
    assert "<img onerror=test>" not in document and "&lt;img onerror=test&gt;" in document
    assert "Synthetic software fixture" in document and "No arms were connected" in document
    assert "yam_upstream_literal_v1" in document and "Historical reference runs" in document
    assert "disables the policy target-speed clamp" in document
    assert "Returned SDK targets are commands, not proof of movement" in document
    assert "Targets after the yamkit clamp" not in document
    assert "extra timing and endpoint holds for existing limits" not in document
    assert f"Nominal video capture rate: {fps} fps" in document
    assert '<img onerror=bad>' not in document and '<script>bad</script>' not in document
    if fps == 30:
        assert 'href="video_timeline.json"' in document and 'Original PTS, gaps retained' in document
        assert 'return-home movement are not recorded' in document
    else:
        assert 'Legacy sampled video' in document and 'href="video_timeline.json"' not in document
    assert plots and all(kwargs["linestyle"] == "none" for _, kwargs in plots)
    assert len(closed) == 2
    assert os.environ["MPLCONFIGDIR"] == str(cache)


def test_fresh_renderer_generates_real_pngs_and_repository_cache(tmp_path):
    saved(tmp_path)
    env = {**os.environ, "MPLCONFIGDIR": str(tmp_path / "ignored-cache-preference")}
    result = subprocess.run([sys.executable, str(SCRIPT), str(tmp_path)],
                            capture_output=True, timeout=45, env=env, check=False)
    assert result.returncode == 0, result.stderr.decode()
    assert (tmp_path / "report.html").is_file()
    for side in module.SIDES:
        assert (tmp_path / f"joints-{side}.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert not (tmp_path / "ignored-cache-preference").exists()
    assert (module.ROOT / ".context" / "matplotlib").is_dir()
