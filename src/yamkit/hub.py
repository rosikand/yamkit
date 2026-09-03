"""Hugging Face Hub: sign-in, and listing / pushing / pulling LeRobot datasets and policies.

Every network call of yamkit lives here. The token is stored where the `huggingface_hub` library
keeps it, which yamkit's environment redirects into ``./data/hf/token`` (git-ignored), so plain
``lerobot-*`` commands are signed in too and nothing secret ever enters the rig file.
Listings are cached for a minute so the web UI stays snappy; everything degrades to "local only"
when offline or signed out.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .paths import DATASETS_DIR

log = logging.getLogger(__name__)

CACHE_TTL_S = 60.0
_CACHE: dict[str, tuple[float, Any]] = {}
DESTINATIONS = ("local", "hub", "both")


def _cached(key: str, fn, ttl: float = CACHE_TTL_S):
    now = time.monotonic()
    hit = _CACHE.get(key)
    if hit is not None and now - hit[0] < ttl:
        return hit[1]
    val = fn()
    _CACHE[key] = (now, val)
    return val


def clear_cache() -> None:
    _CACHE.clear()


def _short(e: BaseException) -> str:
    s = str(e).strip().splitlines()
    return (s[0] if s else type(e).__name__)[:200]


# ----- sign-in -----------------------------------------------------------------------------
def token_path() -> Path:
    from huggingface_hub import constants

    return Path(constants.HF_TOKEN_PATH)


def get_token() -> str | None:
    from huggingface_hub import get_token

    return get_token()


def login(token: str) -> str:
    """Check the token against the Hub, store it, return the account name."""
    from huggingface_hub import HfApi
    from huggingface_hub import login as hf_login

    token = token.strip()
    if not token:
        raise ValueError("empty token")
    name = HfApi(token=token).whoami()["name"]
    hf_login(token=token, add_to_git_credential=False)
    clear_cache()
    return name


def logout() -> None:
    from huggingface_hub import logout as hf_logout

    try:
        hf_logout()
    except Exception as e:  # noqa: BLE001
        log.debug("hf logout: %s", e)
    p = token_path()
    if p.exists():
        p.unlink()
    clear_cache()


def status() -> dict[str, Any]:
    """{'logged_in', 'username', 'token_path', 'online', 'error'}; the account lookup is cached a minute."""

    def _s() -> dict[str, Any]:
        tok = get_token()
        out: dict[str, Any] = {"logged_in": bool(tok), "username": None, "token_path": str(token_path()), "online": None, "error": None}
        if not tok:
            return out
        try:
            from huggingface_hub import HfApi

            out["username"] = HfApi(token=tok).whoami()["name"]
            out["online"] = True
        except Exception as e:  # noqa: BLE001 — offline or bad token: the UI shows local only
            out["online"] = False
            out["error"] = _short(e)
        return out

    return _cached("status", _s)


def username(rig_username: str | None = None) -> str | None:
    """The account datasets/models are pushed under: the rig's `hub.username`, else the signed-in one."""
    return rig_username or status().get("username")


def repo_id(name: str, rig_username: str | None = None, kind: str = "dataset") -> str:
    """`<username>/<name>` for a bare name; a name that already has a `/` is used as is."""
    if "/" in name:
        return name
    user = username(rig_username)
    if not user:
        raise RuntimeError(f"cannot name the {kind} on the Hub: sign in first (`yamkit hub login`) or set hub.username in the rig")
    return f"{user}/{name}"


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def _tasks_from_parquet(path: str | Path) -> list[str]:
    try:
        import pyarrow.parquet as pq

        t = pq.read_table(path)
        col = "task" if "task" in t.column_names else t.column_names[0]
        return [str(x) for x in t.column(col).to_pylist()][:10]
    except Exception:  # noqa: BLE001
        return []


# ----- listings --------------------------------------------------------------------------------
def list_datasets(rig_username: str | None = None) -> list[dict[str, Any]]:
    """LeRobot datasets in the account (those with meta/info.json), like the local catalog rows."""
    user = username(rig_username)
    if not user or not get_token():
        return []

    def _one(api, d) -> dict[str, Any] | None:
        from huggingface_hub import hf_hub_download

        try:
            info = api.dataset_info(d.id, files_metadata=True)
            files = {s.rfilename: (s.size or 0) for s in (info.siblings or [])}
            if "meta/info.json" not in files:
                return None
            meta = _read_json(hf_hub_download(d.id, "meta/info.json", repo_type="dataset"))
            tasks = _tasks_from_parquet(hf_hub_download(d.id, "meta/tasks.parquet", repo_type="dataset")) if "meta/tasks.parquet" in files else []
        except Exception as e:  # noqa: BLE001
            log.debug("hub dataset %s skipped: %s", d.id, e)
            return None
        return {
            "name": d.id.split("/", 1)[1],
            "repo_id": d.id,
            "private": bool(getattr(info, "private", False)),
            "episodes": meta.get("total_episodes"),
            "frames": meta.get("total_frames"),
            "fps": meta.get("fps"),
            "robot_type": meta.get("robot_type"),
            "tasks": tasks,
            "cameras": [k for k in (meta.get("features") or {}) if k.startswith("observation.images")],
            "size_bytes": sum(files.values()),
            "modified": info.last_modified.timestamp() if getattr(info, "last_modified", None) else None,
            "url": f"https://huggingface.co/datasets/{d.id}",
        }

    def _l() -> list[dict[str, Any]]:
        from huggingface_hub import HfApi

        api = HfApi()
        try:
            found = list(api.list_datasets(author=user, limit=500))
        except Exception as e:  # noqa: BLE001
            log.warning("could not list Hub datasets: %s", _short(e))
            return []
        with ThreadPoolExecutor(max_workers=8) as pool:
            rows = [r for r in pool.map(lambda d: _one(api, d), found) if r]
        return sorted(rows, key=lambda r: r["name"])

    return _cached(f"datasets:{user}", _l)


def list_models(rig_username: str | None = None) -> list[dict[str, Any]]:
    """LeRobot policies in the account (config.json + weights), like the local catalog rows."""
    user = username(rig_username)
    if not user or not get_token():
        return []

    def _one(api, m) -> dict[str, Any] | None:
        from huggingface_hub import hf_hub_download

        try:
            info = api.model_info(m.id, files_metadata=True)
            files = {s.rfilename: (s.size or 0) for s in (info.siblings or [])}
            if "config.json" not in files or not any(f.endswith(".safetensors") for f in files):
                return None
            cfg = _read_json(hf_hub_download(m.id, "config.json"))
            train_cfg = _read_json(hf_hub_download(m.id, "train_config.json")) if "train_config.json" in files else {}
        except Exception as e:  # noqa: BLE001
            log.debug("hub model %s skipped: %s", m.id, e)
            return None
        ds = train_cfg.get("dataset") if isinstance(train_cfg, dict) else None
        return {
            "name": m.id.split("/", 1)[1],
            "repo_id": m.id,
            "path": m.id,
            "private": bool(getattr(info, "private", False)),
            "policy_type": cfg.get("type"),
            "files": sorted(files),
            "size_bytes": sum(files.values()),
            "modified": info.last_modified.timestamp() if getattr(info, "last_modified", None) else None,
            "steps": train_cfg.get("steps") if isinstance(train_cfg, dict) else None,
            "dataset": ds.get("repo_id") if isinstance(ds, dict) else None,
            "url": f"https://huggingface.co/{m.id}",
        }

    def _l() -> list[dict[str, Any]]:
        from huggingface_hub import HfApi

        api = HfApi()
        try:
            found = list(api.list_models(author=user, limit=500))
        except Exception as e:  # noqa: BLE001
            log.warning("could not list Hub models: %s", _short(e))
            return []
        with ThreadPoolExecutor(max_workers=8) as pool:
            rows = [r for r in pool.map(lambda m: _one(api, m), found) if r]
        return sorted(rows, key=lambda r: r["name"])

    return _cached(f"models:{user}", _l)


def model_detail(repo: str) -> dict[str, Any] | None:
    """Files + config.json of one Hub policy (None if it is not a LeRobot policy)."""
    from huggingface_hub import HfApi, hf_hub_download

    try:
        info = HfApi().model_info(repo, files_metadata=True)
    except Exception as e:  # noqa: BLE001
        log.debug("hub model %s: %s", repo, e)
        return None
    files = [{"name": s.rfilename, "size_bytes": s.size or 0} for s in (info.siblings or [])]
    names = {f["name"] for f in files}
    if "config.json" not in names:
        return None
    cfg = _read_json(hf_hub_download(repo, "config.json"))
    train_cfg = _read_json(hf_hub_download(repo, "train_config.json")) if "train_config.json" in names else {}
    return {
        "path": repo,
        "repo_id": repo,
        "where": "cloud",
        "url": f"https://huggingface.co/{repo}",
        "policy_type": cfg.get("type"),
        "files": files,
        "size_bytes": sum(f["size_bytes"] for f in files),
        "modified": info.last_modified.timestamp() if getattr(info, "last_modified", None) else None,
        "config": cfg,
        "train_config": train_cfg,
    }


# ----- transfers --------------------------------------------------------------------------------
def push_dataset(name: str, *, private: bool = True, rig_username: str | None = None, root: Path | None = None) -> str:
    """Upload a local LeRobot dataset (LeRobot's own push: card, tags and the version tag its loader needs)."""
    root = root or DATASETS_DIR / name
    if not (root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"{root} is not a LeRobot dataset (no meta/info.json)")
    rid = repo_id(name, rig_username)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(rid, root=root)
    ds.push_to_hub(private=private, tags=["yamkit"])
    clear_cache()
    return f"https://huggingface.co/datasets/{rid}"


def push_model(path: Path, name: str | None = None, *, private: bool = True, rig_username: str | None = None) -> str:
    """Upload a checkpoint directory (`.../pretrained_model`) as a Hub model repo."""
    path = Path(path)
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"{path} is not a policy checkpoint (no config.json)")
    if name is None:  # outputs/train/<job>/checkpoints/last/pretrained_model -> <job>
        parts = path.resolve().parts
        name = parts[parts.index("train") + 1] if "train" in parts and parts.index("train") + 1 < len(parts) else path.name
    rid = repo_id(name, rig_username, kind="model")
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(rid, private=private, exist_ok=True)
    api.upload_folder(folder_path=str(path), repo_id=rid, commit_message="yamkit push-model")
    clear_cache()
    return f"https://huggingface.co/{rid}"


def pull_dataset(repo: str, *, rig_username: str | None = None, dest: Path | None = None) -> Path:
    """Download a Hub dataset into data/datasets/<name> so the local viewer and trainer can use it."""
    rid = repo_id(repo, rig_username)
    dest = dest or DATASETS_DIR / rid.split("/", 1)[1]
    from huggingface_hub import snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(rid, repo_type="dataset", local_dir=str(dest))
    return dest


def remove_local_dataset(name: str, root: Path | None = None) -> None:
    root = root or DATASETS_DIR / name
    if (root / "meta" / "info.json").is_file():
        shutil.rmtree(root)
