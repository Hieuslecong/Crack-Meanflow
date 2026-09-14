from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


RUN_COMPLETION_SCHEMA = "CRACKMEANFLOW_RUN_COMPLETION_V1"


def assert_training_split_allowed(split_name: str) -> None:
    """Reject target/test access from the training data path at runtime."""
    name = str(split_name).lower()
    if name not in {"train", "val"}:
        raise RuntimeError(
            f"training TEST firewall rejected split={split_name!r}; only train and val are allowed"
        )


def training_split_view(splits: Mapping[str, Any]) -> dict[str, Any]:
    """Return the only source splits that the training loader may consume."""
    if not isinstance(splits, Mapping) or any(name not in splits for name in ("train", "val", "test")):
        raise RuntimeError("training TEST firewall requires train, val, and test source splits")
    for name in ("train", "val"):
        assert_training_split_allowed(name)
    return {"train": splits["train"], "val": splits["val"]}


def _finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def validate_execution_identity(
    *,
    expected_config_hash: str | None,
    actual_config_hash: str,
    expected_source_tree_hash: str | None,
    actual_source_tree_hash: str,
    expected_protocol_bundle_hash: str | None,
    actual_protocol_bundle_hash: str,
) -> None:
    """Reject execution when a queue-bound identity no longer matches.

    A queue is an immutable scientific plan, so its effective config and the
    execution/protocol source hashes must be checked immediately before any
    dataset or model work begins.
    """
    checks = (
        ("effective config", expected_config_hash, actual_config_hash),
        ("source tree", expected_source_tree_hash, actual_source_tree_hash),
        ("protocol bundle", expected_protocol_bundle_hash, actual_protocol_bundle_hash),
    )
    for label, expected, actual in checks:
        if expected is not None and str(expected) != str(actual):
            raise RuntimeError(
                f"queue-bound {label} mismatch: expected={expected} actual={actual}"
            )


def require_complete_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    eligibility_class: str,
    expected_optimizer_steps: int | None = None,
    exact_budget: bool = True,
) -> dict[str, Any]:
    """Validate checkpoint completion metadata before scientific evaluation."""
    if eligibility_class not in {"headline", "diagnostic"}:
        raise ValueError("eligibility_class must be headline or diagnostic")
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("checkpoint completion metadata is missing")
    extra = checkpoint.get("extra_state")
    if not isinstance(extra, Mapping):
        raise RuntimeError("checkpoint completion metadata is missing extra_state")
    fairness = extra.get("fairness")
    if not isinstance(fairness, Mapping):
        raise RuntimeError("checkpoint completion metadata is missing fairness budget")
    planned_value = expected_optimizer_steps
    if planned_value is None:
        planned_value = fairness.get("planned_optimizer_steps")
    try:
        planned = int(planned_value)
        completed = int(checkpoint.get("global_optimizer_step"))
    except (TypeError, ValueError):
        raise RuntimeError("checkpoint completion metadata has invalid optimizer-step values") from None
    if planned < 1:
        raise RuntimeError("checkpoint completion metadata has no positive planned optimizer steps")
    if exact_budget and completed != planned:
        raise RuntimeError(
            f"checkpoint has {completed} optimizer steps; expected the planned optimizer steps {planned}"
        )
    if not exact_budget and not (0 < completed <= planned):
        raise RuntimeError(f"checkpoint optimizer step {completed} is outside the planned budget {planned}")
    if extra.get("epoch_complete") is not True and extra.get("budget_reached") is not True:
        raise RuntimeError("checkpoint is an incomplete partial epoch")
    if not _finite_number(checkpoint.get("best_val_metric")):
        raise RuntimeError("checkpoint has no finite source-validation best metric")
    if not _finite_number(checkpoint.get("best_val_threshold")):
        raise RuntimeError("checkpoint has no finite source-validation threshold")
    for key in ("config_hash", "source_tree_sha256", "protocol_bundle_sha256"):
        if not checkpoint.get(key):
            raise RuntimeError(f"checkpoint provenance is missing {key}")
    expected_eligibility = {
        "headline": (False, True, True),
        "diagnostic": (True, False, False),
    }[eligibility_class]
    actual_eligibility = (
        extra.get("diagnostic_only"),
        extra.get("research_metric_valid"),
        extra.get("eligible_for_paper"),
    )
    if actual_eligibility != expected_eligibility:
        raise RuntimeError(f"checkpoint does not satisfy {eligibility_class} eligibility")
    return {
        "planned_optimizer_steps": planned,
        "completed_optimizer_steps": completed,
        "epoch_complete": bool(extra.get("epoch_complete")),
        "budget_reached": bool(extra.get("budget_reached")),
        "eligibility_class": eligibility_class,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_run_completion_artifact(
    checkpoint_path: str | Path,
    checkpoint: Mapping[str, Any],
    *,
    eligibility_class: str,
) -> dict[str, Any]:
    """Verify the immutable record binding a selected checkpoint to run end."""
    selected_path = Path(checkpoint_path).resolve()
    record_path = selected_path.parent / "RUN_COMPLETE.json"
    if not record_path.is_file():
        raise RuntimeError(f"run completion artifact is missing: {record_path}")
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"run completion artifact is unreadable: {record_path}") from exc
    if not isinstance(record, Mapping) or record.get("schema") != RUN_COMPLETION_SCHEMA:
        raise RuntimeError("run completion artifact has an invalid schema")
    if record.get("status") != "PASS":
        raise RuntimeError("run completion artifact is not PASS")
    if not _finite_number(record.get("training_runtime_seconds")):
        raise RuntimeError("run completion artifact has missing or non-finite training runtime")
    if record.get("best_checkpoint") != selected_path.name:
        raise RuntimeError("selected checkpoint is not the immutable validation-selected best checkpoint")
    if record.get("best_checkpoint_sha256") != _file_sha256(selected_path):
        raise RuntimeError("run completion artifact best checkpoint hash mismatch")
    selected_status = require_complete_checkpoint(
        checkpoint,
        eligibility_class=eligibility_class,
        expected_optimizer_steps=record.get("planned_optimizer_steps"),
        exact_budget=False,
    )
    if record.get("best_optimizer_step") != selected_status["completed_optimizer_steps"]:
        raise RuntimeError("run completion artifact best optimizer-step mismatch")
    if record.get("best_val_metric") != checkpoint.get("best_val_metric"):
        raise RuntimeError("run completion artifact best validation metric mismatch")
    if record.get("best_val_threshold") != checkpoint.get("best_val_threshold"):
        raise RuntimeError("run completion artifact best validation threshold mismatch")
    final_name = record.get("final_checkpoint")
    final_path = selected_path.parent / str(final_name) if final_name else None
    if final_path is None or not final_path.is_file():
        raise RuntimeError("run completion artifact final checkpoint is missing")
    if record.get("final_checkpoint_sha256") != _file_sha256(final_path):
        raise RuntimeError("run completion artifact final checkpoint hash mismatch")
    import torch

    final_checkpoint = torch.load(final_path, map_location="cpu", weights_only=False)
    final_status = require_complete_checkpoint(
        final_checkpoint,
        eligibility_class=eligibility_class,
        expected_optimizer_steps=record.get("planned_optimizer_steps"),
    )
    if final_checkpoint.get("extra_state", {}).get("run_complete") is not True:
        raise RuntimeError("final checkpoint is not marked run_complete")
    if record.get("completed_optimizer_steps") != final_status["completed_optimizer_steps"]:
        raise RuntimeError("run completion artifact optimizer-step mismatch")
    expected_record_eligibility = {
        "headline": (False, True, True),
        "diagnostic": (True, False, False),
    }[eligibility_class]
    actual_record_eligibility = (
        record.get("diagnostic_only"),
        record.get("research_metric_valid"),
        record.get("eligible_for_paper"),
    )
    if actual_record_eligibility != expected_record_eligibility:
        raise RuntimeError(f"run completion artifact does not satisfy {eligibility_class} eligibility")
    for key in ("config_hash", "source_tree_sha256", "protocol_bundle_sha256"):
        if record.get(key) != checkpoint.get(key) or record.get(key) != final_checkpoint.get(key):
            raise RuntimeError(f"run completion artifact provenance mismatch for {key}")
    required_artifacts = record.get("required_artifacts")
    if not isinstance(required_artifacts, list) or any(
        not (selected_path.parent / str(name)).is_file() for name in required_artifacts
    ):
        raise RuntimeError("run completion artifact required output is missing")
    return dict(record)


PRIMARY_RUNTIME_LAYOUTS = {
    "CONFERENCE_CRACKMEANFLOW_U": (4, 2),
    "A2B_HYBRID_IMF_MASK_ENDPOINT_AWARE_015_CAPACITY_MATCHED": (1, 8),
    "A5_GEOCRACK_IMF_ENDPOINT_AWARE_015_CANDIDATE": (1, 8),
}


def runtime_policy_status(experiment, batch_size, grad_accum_steps):
    """Record whether an effective runtime layout is eligible for the primary comparison."""
    actual = (int(batch_size), int(grad_accum_steps))
    expected = PRIMARY_RUNTIME_LAYOUTS.get(str(experiment))
    overridden = expected is not None and actual != expected
    return {
        "canonical_batch_size": None if expected is None else expected[0],
        "canonical_grad_accum_steps": None if expected is None else expected[1],
        "runtime_batch_size": actual[0],
        "runtime_grad_accum_steps": actual[1],
        "RUNTIME_POLICY_OVERRIDE": overridden,
        "INVALID_FOR_PRIMARY_COMPARISON": overridden,
    }


def resume_taint_state(parent_checkpoint, current_config_mismatch):
    """Keep an allowed resume configuration change tainted in every descendant."""
    extra = (parent_checkpoint or {}).get("extra_state") or {}
    inherited = bool(extra.get("resume_config_mismatch", False))
    current = bool(current_config_mismatch)
    return {
        "resume_config_mismatch_current": current,
        "resume_config_mismatch_inherited": inherited,
        "resume_config_mismatch": bool(current or inherited),
    }
