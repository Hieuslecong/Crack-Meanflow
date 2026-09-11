"""Fail-closed provenance, queue, runtime, and paired-comparison helpers.

This module deliberately does not start training or evaluation.  It makes the
scientific contract executable so that a queue can be inspected before any GPU
work and a comparison cannot silently mix identities or seeds.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from .checkpointing import (
    config_hash,
    environment_info,
    file_sha256,
    protocol_bundle_hash,
    source_tree_hash,
)


CONCLUSION_STATES = (
    "CORRECT_AND_BETTER",
    "CORRECT_BUT_NOT_BETTER",
    "INVALID_OR_INCOMPLETE",
)
MATRIX_SCHEMA = "CRACKMEANFLOW_V1_V2_EXPERIMENT_MATRIX_V1"
RUN_IDENTITY_SCHEMA = "CRACKMEANFLOW_RUN_IDENTITY_V1"
CONFIG_LOCK_SCHEMA = "CRACKMEANFLOW_CONFIG_LOCK_V1"
RUNTIME_PROFILE_SCHEMA = "CRACKMEANFLOW_RUNTIME_PROFILE_V1"
ACCESS_POLICY = {"CFD_TEST": "CLOSED", "GAPS384": "CLOSED"}
PRIMARY_METRICS = ("f1", "iou")
GEOMETRY_METRICS = (
    "assd",
    "assd_px",
    "centerline_assd_px",
    "edt_mae",
    "edt_radius_mae_px",
    "skeleton_length_rel_error",
    "skeleton_error",
)
REQUIRED_GEOMETRY_METRICS = (
    "centerline_assd_px",
    "edt_radius_mae_px",
    "skeleton_length_rel_error",
)


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), default=str) + "\n").encode()


def _slug(value: Any) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value)).strip("_")


def _git_state(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, stderr=subprocess.DEVNULL, text=True
        )
        return {
            "git_commit": commit,
            "workspace_dirty": bool(status.strip()),
            "dirty_paths": [line[3:] for line in status.splitlines() if len(line) >= 4],
        }
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": "NO_GIT_METADATA", "workspace_dirty": None, "dirty_paths": []}


def _load_config(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_run_identity(
    *,
    root: str | os.PathLike[str],
    variant: str,
    lane: str,
    arm: str,
    seed: int,
    config_path: str | os.PathLike[str],
    protocol_path: str | os.PathLike[str],
    data_identity: Mapping[str, Any],
    training_budget: Mapping[str, Any],
) -> dict[str, Any]:
    """Create a deterministic identity for one planned or executed run.

    The identity includes both normalized configuration hashes and exact source
    file hashes.  A dirty worktree is not hidden: it is recorded explicitly so
    a caller can reject it or cite the captured state.
    """
    variant = str(variant).upper()
    if variant not in {"V1", "V2"}:
        raise ValueError("variant must be V1 or V2")
    root_path = Path(root).resolve()
    cfg_path = Path(config_path).resolve()
    proto_path = Path(protocol_path).resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"config does not exist: {cfg_path}")
    if not proto_path.is_file():
        raise FileNotFoundError(f"protocol does not exist: {proto_path}")
    data = copy.deepcopy(dict(data_identity))
    split_id = data.get("split_id")
    if not split_id:
        raise ValueError("data_identity must include split_id")
    cfg = _load_config(cfg_path)
    run_id = "{}_{}_{}_s{}".format(_slug(lane), variant.lower(), _slug(arm), int(seed))
    identity = {
        "schema": RUN_IDENTITY_SCHEMA,
        "run_id": run_id,
        "variant": variant,
        "lane": str(lane),
        "arm": str(arm),
        "training_seed": int(seed),
        "split_id": str(split_id),
        "dataset_identity": data,
        "config_path": str(cfg_path.relative_to(root_path)),
        "config_sha256": config_hash(cfg),
        "config_file_sha256": file_sha256(cfg_path),
        "protocol_path": str(proto_path.relative_to(root_path)),
        "protocol_sha256": file_sha256(proto_path),
        "source_tree_sha256": source_tree_hash(root_path),
        "protocol_bundle_sha256": protocol_bundle_hash(root_path),
        "training_budget": copy.deepcopy(dict(training_budget)),
        "access_policy": copy.deepcopy(ACCESS_POLICY),
        "environment": environment_info(),
        "source_snapshot": _git_state(root_path),
    }
    identity["identity_sha256"] = hashlib.sha256(_canonical_bytes(identity)).hexdigest()
    return identity


def build_config_lock(
    identity: Mapping[str, Any], config: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Build the immutable config lock associated with a run identity."""
    lock = {
        "schema": CONFIG_LOCK_SCHEMA,
        "run_id": identity.get("run_id"),
        "variant": identity.get("variant"),
        "arm": identity.get("arm"),
        "training_seed": identity.get("training_seed"),
        "config_path": identity.get("config_path"),
        "config_sha256": identity.get("config_sha256"),
        "protocol_path": identity.get("protocol_path"),
        "protocol_sha256": identity.get("protocol_sha256"),
        "source_tree_sha256": identity.get("source_tree_sha256"),
        "protocol_bundle_sha256": identity.get("protocol_bundle_sha256"),
        "split_id": identity.get("split_id"),
        "training_budget": copy.deepcopy(identity.get("training_budget") or {}),
        "config": copy.deepcopy(dict(config)) if config is not None else None,
    }
    lock["lock_sha256"] = hashlib.sha256(_canonical_bytes(lock)).hexdigest()
    return lock


def write_immutable_json(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> None:
    """Write an artifact once; a changed existing artifact is always rejected."""
    target = Path(path)
    content = _canonical_bytes(dict(payload))
    if target.exists():
        if target.read_bytes() != content:
            raise RuntimeError(f"immutable artifact already exists with different content: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)


def verify_immutable_json(path: str | os.PathLike[str], expected: Mapping[str, Any]) -> bool:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(target)
    if target.read_bytes() != _canonical_bytes(dict(expected)):
        raise RuntimeError(f"immutable artifact verification failed: {target}")
    return True


def _gate_passes(gates: Mapping[str, Any], name: str) -> bool:
    value = gates.get(name, False)
    return value is True or str(value).upper() in {"PASS", "PASSED", "OPEN", "TRUE"}


def assert_dataset_access_allowed(
    dataset: str, gates: Mapping[str, Any], *, purpose: str = "evaluation"
) -> None:
    """Enforce the CFD TEST/GAPS384 firewall at the evaluation boundary."""
    del purpose  # The gate is intentionally independent of caller-selected labels.
    name = str(dataset).upper()
    if name == "CFD_TEST":
        missing = [key for key in ("fix_gate", "threshold_lock", "target_lock") if not _gate_passes(gates, key)]
        if missing:
            raise RuntimeError(f"CFD_TEST is CLOSED; missing gate(s): {', '.join(missing)}")
        return
    if name in {"GAPS384", "GAPS384_TEST", "GAPS384_OFFICIAL_TEST"}:
        missing = [key for key in ("fix_gate", "threshold_lock") if not _gate_passes(gates, key)]
        if missing:
            raise RuntimeError(f"GAPS384 is CLOSED; missing gate(s): {', '.join(missing)}")
        return


def load_experiment_matrix(path: str | os.PathLike[str]) -> dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        matrix = yaml.safe_load(handle)
    if not isinstance(matrix, dict) or matrix.get("schema") != MATRIX_SCHEMA:
        raise RuntimeError(f"invalid experiment matrix schema in {path}")
    if set(matrix.get("seeds", [])) != {0, 1, 2}:
        raise RuntimeError("experiment matrix must contain exactly training seeds 0, 1, 2")
    source = matrix.get("source")
    if not isinstance(source, Mapping):
        raise RuntimeError("experiment matrix source identity must be a mapping")
    missing_source = [key for key in ("dataset_name", "dataset_version", "split_id") if not source.get(key)]
    if missing_source:
        raise RuntimeError(f"experiment matrix source identity missing: {', '.join(missing_source)}")
    protocols = matrix.get("variant_protocols") or {}
    missing_protocols = [variant for variant in ("V1", "V2") if not protocols.get(variant)]
    if missing_protocols:
        raise RuntimeError(f"experiment matrix missing protocol path(s): {', '.join(missing_protocols)}")
    arms = matrix.get("arms") or {}
    if set(arms) != {"Conference", "A2B_ENDPOINT", "A5_ENDPOINT"}:
        raise RuntimeError("matrix must contain Conference, A2B_ENDPOINT, and A5_ENDPOINT")
    lanes = matrix.get("lanes") or {}
    if set(lanes) != {"matched_optimizer_steps", "practical_protocol"}:
        raise RuntimeError("matrix must contain matched_optimizer_steps and practical_protocol lanes")
    for arm, values in arms.items():
        for key in ("v1_config", "v2_config"):
            if not values.get(key):
                raise RuntimeError(f"matrix arm {arm} is missing {key}")
    for lane, values in lanes.items():
        budgets = values.get("planned_optimizer_steps") or {}
        if "V2" not in budgets:
            raise RuntimeError(f"matrix lane {lane} is missing V2 optimizer budget")
    return matrix


def expand_experiment_matrix(matrix: Mapping[str, Any], *, root: str | os.PathLike[str] | None = None) -> list[dict[str, Any]]:
    """Expand lanes x variants x arms x paired training seeds into queue rows."""
    root_path = Path(root).resolve() if root is not None else None
    rows: list[dict[str, Any]] = []
    source = matrix.get("source") or {}
    protocol_paths = matrix.get("variant_protocols") or {}
    for lane, lane_cfg in (matrix.get("lanes") or {}).items():
        planned = lane_cfg.get("planned_optimizer_steps") or {}
        for variant in ("V1", "V2"):
            protocol_path = protocol_paths.get(variant)
            if not protocol_path:
                raise RuntimeError(f"matrix is missing protocol path for {variant}")
            for arm, arm_cfg in (matrix.get("arms") or {}).items():
                config_path = arm_cfg[f"{variant.lower()}_config"]
                if root_path is not None:
                    for required_path in (config_path, protocol_path):
                        if not (root_path / required_path).is_file():
                            raise FileNotFoundError(root_path / required_path)
                for seed in matrix["seeds"]:
                    rows.append(
                        {
                            "run_id": "{}_{}_{}_s{}".format(_slug(lane), variant.lower(), _slug(arm), int(seed)),
                            "variant": variant,
                            "lane": lane,
                            "arm": arm,
                            "training_seed": int(seed),
                            "config_path": str(config_path),
                            "protocol_path": str(protocol_path),
                            "dataset_name": source.get("dataset_name", "CFD"),
                            "dataset_version": source.get("dataset_version"),
                            "split_id": source.get("split_id"),
                            "dataset_identity": copy.deepcopy(source.get("dataset_identity") or {}),
                            "training_budget": {
                                "budget_protocol": lane_cfg.get("budget_protocol"),
                                "planned_optimizer_steps": planned.get(variant),
                            },
                            "access_policy": copy.deepcopy(ACCESS_POLICY),
                        }
                    )
    return rows


def build_queue(
    rows: Iterable[Mapping[str, Any]],
    *,
    root: str | os.PathLike[str],
    output_root: str | os.PathLike[str] | None = None,
) -> list[dict[str, Any]]:
    root_path = Path(root).resolve()
    output_base = Path(output_root).resolve() if output_root is not None else root_path / "outputs" / "v1_v2_matrix"
    bound_source_tree = source_tree_hash(root_path)
    bound_protocol_bundle = protocol_bundle_hash(root_path)
    queue = []
    for row in rows:
        dataset_name = row.get("dataset_name")
        dataset_version = row.get("dataset_version")
        if not dataset_name or not dataset_version or not row.get("split_id"):
            raise RuntimeError("queue row is missing dataset name, dataset version, or split identity")
        config_path = root_path / str(row["config_path"])
        protocol_path = root_path / str(row["protocol_path"])
        cfg = _load_config(config_path)
        effective_cfg = copy.deepcopy(cfg)
        effective_cfg.setdefault("train", {})["seed"] = int(row["training_seed"])
        command = [
            sys.executable,
            str(root_path / "scripts/train_journal.py"),
            "--config",
            str(config_path),
            "--data",
            "${DATA_ROOT}",
            "--dataset-name",
            str(dataset_name),
            "--dataset-version",
            str(dataset_version),
            "--seed",
            str(row["training_seed"]),
            "--out",
            str(output_base / row["run_id"]),
        ]
        steps = (row.get("training_budget") or {}).get("planned_optimizer_steps")
        if steps is not None:
            effective_cfg["train"]["max_optimizer_steps"] = int(steps)
            command.extend(["--max-optimizer-steps", str(int(steps))])
        config_file_hash = file_sha256(config_path)
        protocol_file_hash = file_sha256(protocol_path)
        expected_source_identity = json.dumps(
            row.get("dataset_identity") or {}, sort_keys=True, separators=(",", ":")
        )
        command.extend(
            [
                "--expected-config-sha256",
                config_hash(effective_cfg),
                "--expected-config-file-sha256",
                config_file_hash,
                "--expected-protocol-path",
                str(protocol_path),
                "--expected-protocol-sha256",
                protocol_file_hash,
                "--expected-source-tree-sha256",
                bound_source_tree,
                "--expected-protocol-bundle-sha256",
                bound_protocol_bundle,
                "--expected-split-id",
                str(row["split_id"]),
                "--expected-source-identity-json",
                expected_source_identity,
            ]
        )
        queue.append(
            {
                **dict(row),
                "execution": "TRAIN_ONLY",
                "effective_config_sha256": config_hash(effective_cfg),
                "config_file_sha256": config_file_hash,
                "protocol_file_sha256": protocol_file_hash,
                "source_tree_sha256": bound_source_tree,
                "protocol_bundle_sha256": bound_protocol_bundle,
                "output_dir": str(output_base / row["run_id"]),
                "command": command,
            }
        )
    return queue


class RuntimeProfile:
    """Low-overhead stage and optimizer-step timing collector."""

    def __init__(self, *, device: str | None = None) -> None:
        self.device = device or ("cuda" if _cuda_available() else "cpu")
        self._started = time.perf_counter()
        self._stage_started: dict[str, float] = {}
        self._stages: dict[str, dict[str, float]] = {}
        self._step_seconds: list[float] = []
        self._samples = 0

    def start_stage(self, name: str) -> None:
        if name in self._stage_started:
            raise RuntimeError(f"stage already started: {name}")
        self._stage_started[str(name)] = time.perf_counter()

    def stop_stage(self, name: str, *, elapsed_seconds: float | None = None) -> None:
        name = str(name)
        if elapsed_seconds is None:
            if name not in self._stage_started:
                raise RuntimeError(f"stage was not started: {name}")
            elapsed_seconds = time.perf_counter() - self._stage_started.pop(name)
        else:
            self._stage_started.pop(name, None)
        record = self._stages.setdefault(name, {"count": 0, "total_seconds": 0.0})
        record["count"] += 1
        record["total_seconds"] += float(elapsed_seconds)

    def record_optimizer_step(self, *, samples: int, elapsed_seconds: float) -> None:
        self._step_seconds.append(float(elapsed_seconds))
        self._samples += int(samples)

    def finalize(self) -> dict[str, Any]:
        steps = len(self._step_seconds)
        measured = sum(self._step_seconds)
        stages = {
            name: {
                **record,
                "mean_seconds": record["total_seconds"] / record["count"],
            }
            for name, record in sorted(self._stages.items())
        }
        result = {
            "schema": RUNTIME_PROFILE_SCHEMA,
            "device": self.device,
            "wall_clock_seconds": time.perf_counter() - self._started,
            "optimizer_steps": steps,
            "samples": self._samples,
            "seconds_per_optimizer_step": measured / steps if steps else None,
            "samples_per_second": self._samples / measured if measured > 0 else None,
            "stages": stages,
            "gpu_peak_memory_bytes": _cuda_peak_memory(self.device),
        }
        return result


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _cuda_peak_memory(device: str) -> int | None:
    if not str(device).startswith("cuda"):
        return None
    try:
        import torch

        return int(torch.cuda.max_memory_allocated())
    except Exception:
        return None


def _metric(report: Mapping[str, Any], name: str) -> float | None:
    value: Any = report.get(name)
    if value is None and isinstance(report.get("metrics"), Mapping):
        value = report["metrics"].get(name)
    if isinstance(value, Mapping):
        value = value.get("mean")
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _student_t_critical(n: int) -> float:
    try:
        from scipy.stats import t

        return float(t.ppf(0.975, df=n - 1))
    except Exception:
        return {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}.get(n, 1.96)


def _paired_metric(v1: list[Mapping[str, Any]], v2: list[Mapping[str, Any]], name: str) -> dict[str, Any] | None:
    left = [_metric(row, name) for row in v1]
    right = [_metric(row, name) for row in v2]
    if any(value is None for value in left + right):
        return None
    left_f = [float(value) for value in left]
    right_f = [float(value) for value in right]
    deltas = [b - a for a, b in zip(left_f, right_f)]
    mean_delta = statistics.mean(deltas)
    std_delta = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    critical = _student_t_critical(len(deltas)) if len(deltas) > 1 else math.nan
    half = critical * std_delta / math.sqrt(len(deltas)) if len(deltas) > 1 else math.nan
    return {
        "v1_values": left_f,
        "v2_values": right_f,
        "paired_deltas": deltas,
        "mean_v1": statistics.mean(left_f),
        "mean_v2": statistics.mean(right_f),
        "mean_delta": mean_delta,
        "std_delta": std_delta,
        "ci95_delta": [mean_delta - half, mean_delta + half] if len(deltas) > 1 else None,
        "win_rate_v2": sum(delta > 0 for delta in deltas) / len(deltas),
        "v1_std": statistics.stdev(left_f) if len(left_f) > 1 else 0.0,
        "v2_std": statistics.stdev(right_f) if len(right_f) > 1 else 0.0,
    }


REPORT_REQUIRED_FIELDS = (
    "artifact_complete",
    "scientific_validity",
    "requested_nfe",
    "actual_nfe",
    "nfe_contract_pass",
    "source_identity",
    "target_identity",
    "checkpoint_source_tree_sha256",
    "checkpoint_protocol_bundle_sha256",
    "method_config_hash",
    "training_budget",
    "runtime_seconds",
)


def _report_completion_reasons(rows: list[Mapping[str, Any]], label: str) -> list[str]:
    reasons: list[str] = []
    for index, row in enumerate(rows):
        missing = [key for key in REPORT_REQUIRED_FIELDS if key not in row]
        if missing:
            reasons.append(f"{label}[{index}] missing required completion/provenance field(s): {', '.join(missing)}")
            continue
        if row.get("artifact_complete") is not True:
            reasons.append(f"{label}[{index}] artifact_complete is not true")
        if row.get("scientific_validity") != "VALID_HEADLINE_PROTOCOL":
            reasons.append(f"{label}[{index}] scientific_validity is not VALID_HEADLINE_PROTOCOL")
        if row.get("requested_nfe") != 1 or row.get("actual_nfe") != 1 or row.get("nfe_contract_pass") is not True:
            reasons.append(f"{label}[{index}] NFE=1 contract did not pass")
        for key in ("source_identity", "target_identity", "training_budget"):
            if not isinstance(row.get(key), Mapping) or not row[key]:
                reasons.append(f"{label}[{index}] {key} provenance is missing")
        for key in ("checkpoint_source_tree_sha256", "checkpoint_protocol_bundle_sha256", "method_config_hash"):
            if not isinstance(row.get(key), str) or not row[key]:
                reasons.append(f"{label}[{index}] {key} provenance is missing")
        try:
            if not math.isfinite(float(row.get("runtime_seconds"))):
                raise ValueError
        except (TypeError, ValueError):
            reasons.append(f"{label}[{index}] runtime_seconds is missing or non-finite")
        if str(row.get("backbone", "")).lower() == "geocrack_imf":
            missing_geometry = [name for name in REQUIRED_GEOMETRY_METRICS if _metric(row, name) is None]
            if missing_geometry:
                reasons.append(f"{label}[{index}] missing required geometry metric(s): {', '.join(missing_geometry)}")
    return reasons


def _same_identity(rows: list[Mapping[str, Any]], key: str) -> bool:
    values = [json.dumps(row.get(key), sort_keys=True, separators=(",", ":")) for row in rows]
    return bool(values) and len(set(values)) == 1


def compare_paired_reports(
    v1_reports: Iterable[Mapping[str, Any]],
    v2_reports: Iterable[Mapping[str, Any]],
    *,
    correctness_gates: Mapping[str, Any] | None,
    min_seeds: int = 3,
    noninferiority_margin: float = 0.01,
    geometry_regression_fraction: float = 0.05,
    runtime_regression_fraction: float = 0.20,
) -> dict[str, Any]:
    """Compare paired training seeds and emit one of the three allowed states."""
    left = list(v1_reports)
    right = list(v2_reports)
    left_seeds = [row.get("training_seed") for row in left]
    right_seeds = [row.get("training_seed") for row in right]
    if set(left_seeds) != set(right_seeds) or len(set(left_seeds)) != len(left_seeds) or len(set(right_seeds)) != len(right_seeds):
        raise RuntimeError("training seeds must match uniquely between V1 and V2")
    right_by_seed = {row["training_seed"]: row for row in right}
    left = sorted(left, key=lambda row: row["training_seed"])
    right = [right_by_seed[row["training_seed"]] for row in left]
    result: dict[str, Any] = {
        "schema": "CRACKMEANFLOW_PAIRED_COMPARISON_V1",
        "n_training_seeds": len(left),
        "training_seeds": [row["training_seed"] for row in left],
        "correctness_gates": dict(correctness_gates or {}),
        "metrics": {},
        "conclusion": "INVALID_OR_INCOMPLETE",
    }
    if len(left) < int(min_seeds):
        result["invalid_reasons"] = [f"need at least {min_seeds} paired training seeds"]
        return result
    if not correctness_gates or not _gate_passes(correctness_gates, "all_pass"):
        result["invalid_reasons"] = ["correctness gates did not pass"]
        return result
    completion_reasons = _report_completion_reasons(left, "V1") + _report_completion_reasons(right, "V2")
    if completion_reasons:
        result["invalid_reasons"] = completion_reasons
        return result
    all_reports = left + right
    identity_reasons = []
    for key in ("source_identity", "target_identity", "checkpoint_source_tree_sha256", "checkpoint_protocol_bundle_sha256"):
        if not _same_identity(all_reports, key):
            identity_reasons.append(f"{key} differs across paired reports")
    for label, rows_for_method in (("V1", left), ("V2", right)):
        for key in ("method_config_hash", "training_budget"):
            if not _same_identity(rows_for_method, key):
                identity_reasons.append(f"{key} differs across {label} training seeds")
    if identity_reasons:
        result["invalid_reasons"] = identity_reasons
        return result

    for name in PRIMARY_METRICS + GEOMETRY_METRICS + ("runtime_seconds",):
        stats = _paired_metric(left, right, name)
        if stats is not None:
            result["metrics"][name] = stats
    missing = [name for name in PRIMARY_METRICS if name not in result["metrics"]]
    if missing:
        result["invalid_reasons"] = [f"missing primary metric(s): {', '.join(missing)}"]
        return result

    primary_checks = {
        name: result["metrics"][name]["mean_delta"] >= -float(noninferiority_margin)
        for name in PRIMARY_METRICS
    }
    geometry_checks = {}
    for name in GEOMETRY_METRICS:
        if name not in result["metrics"]:
            continue
        base = abs(result["metrics"][name]["mean_v1"])
        geometry_checks[name] = (
            result["metrics"][name]["mean_delta"] <= float(geometry_regression_fraction) * max(base, 1e-12)
        )
    result["noninferiority"] = {"primary": primary_checks, "geometry_lower_is_better": geometry_checks}
    primary_benefit = any(result["metrics"][name]["mean_delta"] > 0 for name in PRIMARY_METRICS)
    geometry_benefit = any(
        result["metrics"][name]["mean_delta"] <= -float(geometry_regression_fraction) * max(abs(result["metrics"][name]["mean_v1"]), 1e-12)
        for name in geometry_checks
    )
    stability_benefit = any(
        result["metrics"][name]["v1_std"] > 0
        and result["metrics"][name]["v2_std"] <= 0.75 * result["metrics"][name]["v1_std"]
        for name in PRIMARY_METRICS
    )
    runtime_benefit = False
    runtime_within_budget = False
    runtime_stats = result["metrics"].get("runtime_seconds")
    if runtime_stats is not None:
        runtime_within_budget = runtime_stats["mean_delta"] <= float(runtime_regression_fraction) * max(abs(runtime_stats["mean_v1"]), 1e-12)
        runtime_benefit = runtime_stats["mean_delta"] < 0
    benefits = {
        "primary_metric_increase": primary_benefit,
        "geometry_improvement_5_percent": geometry_benefit,
        "seed_stability_improvement_25_percent": stability_benefit,
        "runtime_faster": runtime_benefit,
        "runtime_within_20_percent": runtime_within_budget,
    }
    result["benefits"] = benefits
    all_noninferior = all(primary_checks.values()) and all(geometry_checks.values())
    if not all_noninferior:
        result["conclusion"] = "CORRECT_BUT_NOT_BETTER"
    elif any(benefits.values()):
        result["conclusion"] = "CORRECT_AND_BETTER"
    else:
        result["conclusion"] = "CORRECT_BUT_NOT_BETTER"
    return result


# Descriptive aliases make the intended API discoverable without duplicating logic.
make_run_identity = build_run_identity
expand_matrix = expand_experiment_matrix
