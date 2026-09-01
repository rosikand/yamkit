"""Read-only filesystem catalogs for the web UI.

Datasets are LeRobot v3 directories under ``data/datasets`` (written by `yamkit record`);
models are checkpoint directories under ``outputs/``; deployments are the run records written by
`yamkit.ui.sessions.DeploymentLog`. Everything here only reads files.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def _read_json(p: Path) -> dict[str, Any] | None:
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def _read_parquet(path: Path) -> Any | None:  # -> pyarrow.Table
    try:
        import pyarrow.parquet as pq

        return pq.read_table(path)
    except Exception as e:  # noqa: BLE001 — pyarrow missing or file unreadable
        log.debug("could not read %s: %s", path, e)
        return None


# ---------------------------------------------------------------------------------- datasets --
def _camera_keys(info: dict[str, Any]) -> list[str]:
    return [k for k, f in (info.get("features") or {}).items() if f.get("dtype") in ("video", "image")]


def _tasks(ds: Path) -> list[str]:
    t = _read_parquet(ds / "meta" / "tasks.parquet")
    if t is None:
        return []
    for col in ("task", "__index_level_0__"):
        if col in t.column_names:
            return [str(x) for x in t.column(col).to_pylist()]
    # tasks are the index in lerobot v3; fall back to the first string column
    for col in t.column_names:
        vals = t.column(col).to_pylist()
        if vals and isinstance(vals[0], str):
            return [str(x) for x in vals]
    return []


def dataset_summary(ds: Path) -> dict[str, Any] | None:
    info = _read_json(ds / "meta" / "info.json")
    if info is None:
        return None
    return {
        "name": ds.name,
        "path": str(ds),
        "episodes": info.get("total_episodes"),
        "frames": info.get("total_frames"),
        "fps": info.get("fps"),
        "robot_type": info.get("robot_type"),
        "tasks": _tasks(ds),
        "cameras": _camera_keys(info),
        "size_bytes": _dir_size(ds),
        "modified": ds.stat().st_mtime,
    }


def list_datasets(datasets_dir: Path) -> list[dict[str, Any]]:
    out = []
    if datasets_dir.is_dir():
        for ds in sorted(datasets_dir.iterdir()):
            s = dataset_summary(ds) if ds.is_dir() else None
            if s:
                out.append(s)
    return out


def _episode_rows(ds: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for f in sorted((ds / "meta" / "episodes").rglob("*.parquet")):
        t = _read_parquet(f)
        if t is not None:
            rows.extend(t.to_pylist())
    return rows


def dataset_detail(ds: Path) -> dict[str, Any] | None:
    summary = dataset_summary(ds)
    if summary is None:
        return None
    info = _read_json(ds / "meta" / "info.json") or {}
    episodes = []
    for r in _episode_rows(ds):
        videos = {}
        for cam in summary["cameras"]:
            pre = f"videos/{cam}/"
            if pre + "chunk_index" in r:
                videos[cam] = {
                    "chunk_index": r.get(pre + "chunk_index"),
                    "file_index": r.get(pre + "file_index"),
                    "from_timestamp": r.get(pre + "from_timestamp"),
                    "to_timestamp": r.get(pre + "to_timestamp"),
                }
        episodes.append(
            {
                "episode_index": r.get("episode_index"),
                "length": r.get("length"),
                "tasks": r.get("tasks"),
                "videos": videos,
            }
        )
    features = {
        k: {"dtype": f.get("dtype"), "shape": f.get("shape"), "names": f.get("names")}
        for k, f in (info.get("features") or {}).items()
    }
    return {**summary, "features": features, "video_path": info.get("video_path"), "episode_list": episodes}


def episode_video_file(ds: Path, camera: str, episode_index: int) -> tuple[Path, dict[str, Any]] | None:
    """Resolve the mp4 that contains an episode (LeRobot v3 concatenates episodes per file)."""
    detail = dataset_detail(ds)
    if detail is None or not detail.get("video_path"):
        return None
    for ep in detail["episode_list"]:
        if ep["episode_index"] == episode_index and camera in ep["videos"]:
            v = ep["videos"][camera]
            if v.get("chunk_index") is None:
                return None
            rel = detail["video_path"].format(
                video_key=camera, chunk_index=int(v["chunk_index"]), file_index=int(v["file_index"])
            )
            p = ds / rel
            return (p, v) if p.is_file() else None
    return None


def episode_series(ds: Path, episode_index: int, max_points: int = 1500) -> dict[str, Any] | None:
    """timestamp / observation.state / action arrays for one episode (downsampled for the UI)."""
    info = _read_json(ds / "meta" / "info.json")
    if info is None:
        return None
    row = next((r for r in _episode_rows(ds) if r.get("episode_index") == episode_index), None)
    if row is None or row.get("data/chunk_index") is None:
        return None
    data_path = info["data_path"].format(
        chunk_index=int(row["data/chunk_index"]), file_index=int(row["data/file_index"])
    )
    t = _read_parquet(ds / data_path)
    if t is None:
        return None
    import pyarrow.compute as pc

    t = t.filter(pc.equal(t.column("episode_index"), episode_index))
    n = t.num_rows
    step = max(1, n // max_points)
    idx = list(range(0, n, step))
    out: dict[str, Any] = {"episode_index": episode_index, "n_frames": n}
    for key in ("timestamp", "observation.state", "action"):
        if key in t.column_names:
            vals = t.column(key).to_pylist()
            out[key] = [vals[i] for i in idx]
    feats = info.get("features") or {}
    out["names"] = (feats.get("observation.state") or {}).get("names")
    return out


# ------------------------------------------------------------------------------------ models --
CHECKPOINT_MARKERS = ("config.json", "model.safetensors", "train_config.json")


def list_models(outputs_dir: Path, max_depth: int = 6) -> list[dict[str, Any]]:
    """Find checkpoint directories under outputs/ (anything with a config.json or safetensors)."""
    found: list[dict[str, Any]] = []
    if not outputs_dir.is_dir():
        return found

    def walk(d: Path, depth: int) -> None:
        try:
            children = sorted(d.iterdir())
        except OSError:
            return
        files = {c.name for c in children if c.is_file()}
        if files & set(CHECKPOINT_MARKERS) or any(f.endswith(".safetensors") for f in files):
            cfg = _read_json(d / "config.json") or {}
            train_cfg = _read_json(d / "train_config.json") or _read_json(d.parent / "train_config.json") or {}
            found.append(
                {
                    "path": str(d.relative_to(outputs_dir)),
                    "policy_type": cfg.get("type"),
                    "files": sorted(files),
                    "size_bytes": _dir_size(d),
                    "modified": d.stat().st_mtime,
                    "steps": (train_cfg.get("steps") if isinstance(train_cfg, dict) else None),
                    "dataset": (train_cfg.get("dataset") or {}).get("repo_id")
                    if isinstance(train_cfg.get("dataset"), dict)
                    else None,
                }
            )
            return  # do not descend into a checkpoint dir
        if depth < max_depth:
            for c in children:
                if c.is_dir() and c.name != "ui":
                    walk(c, depth + 1)

    walk(outputs_dir, 0)
    return found


def model_detail(outputs_dir: Path, d: Path) -> dict[str, Any] | None:
    """Full metadata for one checkpoint directory (config, train config, per-file sizes)."""
    files = sorted(c for c in d.iterdir() if c.is_file())
    names = {f.name for f in files}
    if not (names & set(CHECKPOINT_MARKERS) or any(n.endswith(".safetensors") for n in names)):
        return None
    cfg = _read_json(d / "config.json") or {}
    train_cfg = _read_json(d / "train_config.json") or _read_json(d.parent / "train_config.json") or {}
    return {
        "path": str(d.relative_to(outputs_dir)),
        "policy_type": cfg.get("type"),
        "config": cfg,
        "train_config": train_cfg,
        "files": [{"name": f.name, "size_bytes": f.stat().st_size} for f in files],
        "size_bytes": _dir_size(d),
        "modified": d.stat().st_mtime,
    }


# ------------------------------------------------------------------------------- deployments --
def list_deployments(depl_dir: Path) -> list[dict[str, Any]]:
    out = []
    if depl_dir.is_dir():
        for d in sorted(depl_dir.iterdir(), reverse=True):
            meta = _read_json(d / "meta.json")
            if meta is not None:
                meta["has_log"] = (d / "log.txt").is_file()
                meta["videos"] = sorted(p.name for p in d.glob("*.mp4"))
                out.append(meta)
    return out


def deployment_detail(depl_dir: Path, run_id: str) -> dict[str, Any] | None:
    d = depl_dir / run_id
    meta = _read_json(d / "meta.json")
    if meta is None:
        return None
    log_file = d / "log.txt"
    meta["log"] = log_file.read_text().splitlines()[-300:] if log_file.is_file() else []
    meta["videos"] = sorted(p.name for p in d.glob("*.mp4"))
    return meta
