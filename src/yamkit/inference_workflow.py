"""The simple CLI and UI share the existing reference prompt preparation helper.

This layer selects a backend, then delegates to the frozen qualification and
rollout interfaces. Preparing inference never grants motion permission.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path

from .backend_workflow import WorkflowError, assert_ui_idle, ensure_backend, load_target
from .deployment import InferenceOptions
from .paths import DEFAULT_RIG, ROOT


def recovery_context(options: InferenceOptions, rig) -> dict:
    """Local-only UI eligibility for configured lifecycle recovery, never cached readiness."""
    from .inference.profiles import get_profile
    from .inference.qualification import is_cloud_host
    from .probes import preflight_live_probe

    options.validate()
    if (options.backend != "external" or options.controller_mode != "reference"
            or options.policy not in ("molmoact2", "lerobot/MolmoAct2-BimanualYAM-LeRobot")
            or options.call_mode != "http" or options.execution_mode != "cuda_graph10"
            or options.image_encoding != "rgb8" or options.jpeg_quality != 85 or options.center_crop
            or options.rtc or options.prediction_queue_threshold is not None
            or options.supervised_confirmed or options.mapping_accepted or is_cloud_host()):
        raise WorkflowError("Configured recovery requires the unchanged reference selection on the robot host, without motion approval")
    profile = get_profile("molmoact2")
    specs, _ = preflight_live_probe(rig, options.arms or None, expected_state_names=profile.state_names)
    if (len(specs) != 2 or any(spec.side != side or spec.arm_type != "yam" or spec.gripper != "linear_4310"
                              for side, spec in zip(("left", "right"), specs))
            or set(rig.cameras) != set(profile.image_keys) or options.fps != 30
            or any((rig.cameras[name].get("height"), rig.cameras[name].get("width")) != (480, 640)
                   for name in profile.image_keys)):
        raise WorkflowError("Configured recovery requires the reviewed left/right YAM mapping and full three-camera shape")
    target = load_target("lambda", "molmoact2")
    if target.service != options.external_service or target.token_file is None:
        raise WorkflowError("Configure this exact service and its private token-file path before lifecycle recovery")
    fingerprint = hashlib.sha256(json.dumps(asdict(target), sort_keys=True, default=str).encode()).hexdigest()
    return {"recovery": True, "service": target.service, "backend_config_sha256": fingerprint,
            "selection_key": options.operation_key}


def recover_reference_backend(options, rig, expected, *, directory, progress=lambda _value: None):
    if recovery_context(options, rig) != expected:
        raise WorkflowError("Configured backend or prompt selection changed before recovery")
    target = load_target("lambda", "molmoact2")
    return ensure_backend(target, options.task, progress=progress, own_preparation_dir=directory)


def reference_options(*, policy, task, service, rig=DEFAULT_RIG, duration=60, arms=()):
    """Defaults for the reviewed production policy, without touching legacy defaults."""
    if policy not in ("molmoact2", "lerobot/MolmoAct2-BimanualYAM-LeRobot"):
        raise WorkflowError("Select the separate native workflow for this policy; it cannot use MolmoAct2 reference execution")
    return InferenceOptions(policy="molmoact2", task=task, backend="external", external_service=service,
                            controller_mode="reference", call_mode="http", execution_mode="cuda_graph10",
                            image_encoding="rgb8", jpeg_quality=85, center_crop=False,
                            duration=duration, arms=tuple(arms), rig_path=str(Path(rig).resolve())).validate()


def require_prepared_current(options: InferenceOptions) -> None:
    """Recheck local exact proof and lease margin after the operator's terminal wait."""
    from .config import RigConfig
    from .inference.qualification import MAX_AGE_S, settings_from_rig, validate_qualification
    from .ui.server import _prompt_preparation_context

    try:
        options = replace(options, mapping_accepted=False, supervised_confirmed=False)
        context = _prompt_preparation_context(options, RigConfig.load(options.rig_path))
        record = validate_qualification(settings_from_rig(options))
        expires = min(context["expires_at"], record["created_unix_s"] + MAX_AGE_S)
    except (OSError, KeyError, TypeError):
        raise WorkflowError("Prepared inference files or qualification changed after confirmation; prepare again before a new supervised command") from None
    if time.time() + options.duration + 60 >= expires:
        raise WorkflowError("Qualification expires too soon after confirmation; prepare again before a new supervised command")


def prepare_reference(options: InferenceOptions, *, progress=lambda _value: None, force=False) -> dict:
    """Reuse current proof or execute exactly the same hardware-free helper as UI Start."""
    from .config import RigConfig
    from .inference.qualification import MAX_AGE_S, settings_from_rig, validate_qualification
    from .ui.server import _prompt_preparation_context

    options = replace(options, mapping_accepted=False, supervised_confirmed=False)
    assert_ui_idle()
    context = _prompt_preparation_context(options, RigConfig.load(options.rig_path))
    if not force:
        try:
            record = validate_qualification(settings_from_rig(options))
            expires = min(context["expires_at"], record["created_unix_s"] + MAX_AGE_S)
            if time.time() + options.duration + 60 < expires:
                return {"ready": True, "reused": True, "hardware_tested": False,
                        "selection_key": options.operation_key, "expires_at": expires,
                        "reason": "Current task and service qualification reused; no motion was started",
                        "assessment": record.get("assessment", {})}
        except ValueError:
            pass
    assert_ui_idle()
    directory = ROOT / ".context/inference-preparation" / uuid.uuid4().hex
    directory.mkdir(mode=0o700, parents=True)
    request = directory / "request.json"
    with request.open("x") as output:
        json.dump({"options": asdict(options), "expected": context, "capture_trace": False}, output, allow_nan=False)
        output.write("\n")
    progress("Warming this task and qualifying with generated images and fake arms (50 warm samples)")
    source = ROOT / "scripts/prepare_inference_prompt.py"
    spec = importlib.util.spec_from_file_location("_yamkit_workflow_prepare", source)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    code = helper.execute(request)
    try:
        result = json.loads((directory / "result.json").read_text())
    except (OSError, ValueError):
        raise WorkflowError("Prompt preparation did not save a result; no motion was started") from None
    if code != 0 or result.get("ready") is not True or result.get("hardware_tested") is not False:
        raise WorkflowError(result.get("reason", "Prompt qualification failed; no motion was started"))
    return {**result, "reused": False}


def prepare_inference(*, backend, policy, task, rig=DEFAULT_RIG, duration=60, arms=(),
                      config=None, progress=lambda _value: None, force=False):
    """Software-only public orchestration API. Returns options with approval explicitly false."""
    from .workflow_lock import workflow_lock

    with workflow_lock(root=ROOT):
        return _prepare_inference(backend=backend, policy=policy, task=task, rig=rig, duration=duration,
                                  arms=arms, config=config, progress=progress, force=force)


def _prepare_inference(*, backend, policy, task, rig, duration, arms, config, progress, force):
    from .policy_selection import canonical_policy

    policy = canonical_policy(policy)
    if policy == "pi05-base":
        from .openpi.workflow import prepare

        return prepare(backend=backend, task=task, rig=rig, duration=duration, arms=arms,
                       config=config, progress=progress, force=force)
    if policy == "pi05-yam":
        from .pi05_workflow import prepare_pi05

        return prepare_pi05(backend=backend, task=task, rig=rig, duration=duration, arms=arms,
                            config=config, progress=progress, force=force)
    # Validate form errors before connecting or starting any software service.
    reference_options(policy=policy, task=task, service="validation", rig=rig, duration=duration, arms=arms)
    target = load_target(backend, policy, config=config)
    ensure_backend(target, task, progress=progress)
    options = reference_options(policy=policy, task=task, service=target.service, rig=rig, duration=duration, arms=arms)
    result = prepare_reference(options, progress=progress, force=force)
    return options, {**result, "backend": backend, "policy": policy, "service": target.service,
                     "controller_mode": "molmoact2_reference", "motion_approval_received": False}
