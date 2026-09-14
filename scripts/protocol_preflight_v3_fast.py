from __future__ import annotations

"""Static preflight for the V4 FAST, partition-only protocol bundle."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crackmeanflow.common import (  # noqa: E402
    PairedCrackDataset,
    audit_content_split_integrity,
    build_dataset_identity,
    config_hash,
    file_sha256,
    optimizer_steps_per_epoch,
    protocol_bundle_hash,
    source_splits_for_config,
    source_tree_hash,
)
from crackmeanflow.common.training_protocol import training_split_view  # noqa: E402
from crackmeanflow.common.v3_provenance import (  # noqa: E402
    active_v3_bundle_hash,
    nvidia_driver_info,
    paper_v3_worktree_blockers,
    validate_fast_config_pair,
)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], stderr=subprocess.STDOUT, text=True).strip()


def _resolved(root: Path, path_value: str | Path) -> Path:
    path = Path(path_value)
    return (path if path.is_absolute() else root / path).resolve()


def _assert_conference_curriculum(cfg: dict, protocol: dict) -> dict:
    actual = cfg.get("loss", {}).get("time_curriculum") or {}
    expected = protocol["conference_curriculum"]
    checks = {
        "enabled": bool(actual.get("enabled")) is True,
        "basis": actual.get("basis") == expected["basis"],
        "effective_batch_size": int(actual.get("effective_batch_size", -1)) == int(expected["effective_batch_size"]),
        "stages": actual.get("stages") == expected["stages"],
        "covers_full_budget": int(actual.get("stages", [])[-1]["optimizer_steps"][1]) == int(protocol["optimization"]["matched_optimizer_steps"]),
        "endpoint_stage_reachable": any(
            float(stage.get("boundary_prob", 0.0)) > 0
            and int(stage["optimizer_steps"][0]) < int(protocol["optimization"]["matched_optimizer_steps"])
            for stage in actual.get("stages", [])
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Conference optimizer-step curriculum mismatch: {checks}")
    return checks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="configs/protocol/post_repair_protocol_v3_fast.yaml")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="reports/FAST_GENERALIZATION_V3/03_FAST_PREFLIGHT/FAST_PREFLIGHT.json")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    protocol_path = _resolved(root, args.protocol)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != "CRACKMEANFLOW_POST_REPAIR_PROTOCOL_V3":
        raise RuntimeError("wrong base protocol version")
    if protocol.get("protocol_variant") != "FAST_PARTITION_ONLY":
        raise RuntimeError("FAST protocol must declare protocol_variant=FAST_PARTITION_ONLY")
    if not protocol.get("fast_execution_partition_only"):
        raise RuntimeError("FAST protocol must declare partition-only execution semantics")

    base_protocol_path = _resolved(root, protocol["base_protocol"])
    if file_sha256(base_protocol_path) != protocol.get("base_protocol_sha256"):
        raise RuntimeError("base protocol file hash does not match FAST protocol lock")
    base_protocol = yaml.safe_load(base_protocol_path.read_text(encoding="utf-8"))
    if base_protocol.get("protocol_version") != protocol["protocol_version"]:
        raise RuntimeError("FAST/base protocol version mismatch")

    blockers = paper_v3_worktree_blockers()
    clean = not blockers
    if not clean and not args.allow_dirty:
        raise RuntimeError(f"paper-v3 worktree is not provenance-clean: {blockers}")

    provenance = {
        "branch": _git("branch", "--show-current"),
        "commit": _git("rev-parse", "HEAD"),
        "worktree_clean_for_paper_v3": clean,
        "worktree_blockers": blockers,
        "source_tree_sha256": source_tree_hash(),
        "protocol_bundle_sha256_legacy_global": protocol_bundle_hash(),
        "protocol_bundle_sha256_v3_active": active_v3_bundle_hash(protocol_path, protocol),
        "protocol_file_sha256": file_sha256(protocol_path),
        "base_protocol_file_sha256": file_sha256(base_protocol_path),
        "gpu_driver": nvidia_driver_info(),
    }

    expected_train = int(base_protocol["source_dataset"]["expected_train_samples"])
    expected_steps = int(base_protocol["optimization"]["expected_optimizer_steps_per_epoch"])
    expected_effective_batch = int(base_protocol["optimization"]["effective_batch_size"])
    expected_usable = int(base_protocol["optimization"]["expected_usable_train_samples_per_epoch"])
    expected_omitted = int(base_protocol["optimization"]["expected_omitted_train_samples_per_epoch"])
    expected_total = int(base_protocol["optimization"]["matched_optimizer_steps"])
    expected_warm = int(base_protocol["optimization"]["expected_warmup_optimizer_steps"])

    arms = {}
    reference_identity = None
    for arm, cfg_rel in protocol["primary_arms"].items():
        cfg_path = _resolved(root, cfg_rel)
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        base_cfg_path = _resolved(root, base_protocol["primary_arms"][arm])
        base_cfg = yaml.safe_load(base_cfg_path.read_text(encoding="utf-8"))
        partition = validate_fast_config_pair(base_cfg, cfg, expected_effective_batch)
        semantic_hash = config_hash(cfg)
        file_hash = file_sha256(cfg_path)
        lock = protocol["config_locks"][arm]
        if semantic_hash != lock["semantic_sha256"]:
            raise RuntimeError(f"{arm} FAST semantic config hash mismatch")
        if file_hash != lock["file_sha256"]:
            raise RuntimeError(f"{arm} FAST config file hash mismatch")

        splits = source_splits_for_config(str(Path(args.data).resolve()), cfg)
        training = training_split_view(splits)
        leakage = audit_content_split_integrity(splits)
        identity = build_dataset_identity(splits, include_rows=False)
        if reference_identity is None:
            reference_identity = identity
        elif identity != reference_identity:
            raise RuntimeError(f"dataset identity differs across FAST arms at {arm}")

        train_ds = PairedCrackDataset(
            training["train"], cfg["model"]["img_size"], True,
            bool(cfg["train"].get("photometric_augment", False)),
            cfg["train"].get("mask_resize_mode", "nearest"),
            cfg["train"].get("mask_binarization", "auto_binary_safe"),
        )
        batch = int(cfg["train"]["batch_size"])
        accum = int(cfg["train"]["grad_accum_steps"])
        drop_last = bool(cfg["train"].get("drop_last", True))
        num_batches = len(train_ds) // batch if drop_last else (len(train_ds) + batch - 1) // batch
        steps = optimizer_steps_per_epoch(num_batches, accum, bool(cfg["train"].get("drop_incomplete_accumulation", True)))
        usable = steps * expected_effective_batch
        omitted = max(0, len(train_ds) - usable)
        total = int(cfg["train"].get("max_optimizer_steps", steps * int(cfg["train"]["epochs"])))
        warm = int(cfg["train"].get("warmup_epochs", 0)) * int(steps)
        checks = {
            "config_semantic_lock": True,
            "config_file_lock": True,
            "partition_only": bool(partition["scientific_semantics_unchanged"]),
            "train_samples": len(train_ds) == expected_train,
            "optimizer_steps_per_epoch": int(steps) == expected_steps,
            "effective_batch": batch * accum == expected_effective_batch,
            "usable_samples_per_epoch": usable == expected_usable,
            "omitted_samples_per_epoch": omitted == expected_omitted,
            "matched_optimizer_steps": total == expected_total,
            "warmup_optimizer_steps": warm == expected_warm,
            "eval_nfe": int(cfg.get("eval", {}).get("num_steps", -1)) == int(base_protocol["validation"]["headline_nfe"]),
            "checkpoint_selection_seeds": [int(x) for x in cfg.get("eval", {}).get("checkpoint_selection_seeds", [])] == [int(x) for x in base_protocol["validation"]["checkpoint_selection_seeds"]],
            "final_threshold_calibration_seeds": [int(x) for x in cfg.get("eval", {}).get("final_threshold_calibration_seeds", [])] == [int(x) for x in base_protocol["validation"]["final_threshold_calibration_seeds"]],
        }
        curriculum_checks = _assert_conference_curriculum(cfg, base_protocol) if arm == "Conference" else None
        if not all(checks.values()):
            raise RuntimeError(f"V3 FAST protocol mismatch for {arm}: {checks}")
        arms[arm] = {
            "config": str(cfg_path),
            "base_config": str(base_cfg_path),
            "config_file_sha256": file_hash,
            "config_semantic_sha256": semantic_hash,
            "base_config_file_sha256": file_sha256(base_cfg_path),
            "base_config_semantic_sha256": config_hash(base_cfg),
            "backbone": cfg["backbone"],
            "train_samples": len(train_ds),
            "batch_size": batch,
            "grad_accum_steps": accum,
            "effective_batch": batch * accum,
            "optimizer_steps_per_epoch": int(steps),
            "usable_samples_per_epoch": usable,
            "omitted_samples_per_epoch": omitted,
            "matched_optimizer_steps": total,
            "warmup_optimizer_steps": warm,
            "partition_override": partition,
            "checks": checks,
            "curriculum_checks": curriculum_checks,
            "content_leakage_audit": leakage,
        }

    report = {
        "schema": "CRACKMEANFLOW_PROTOCOL_PREFLIGHT_V3_FAST",
        "status": "PASS",
        "training_performed": False,
        "test_metrics_accessed": False,
        "ood_metrics_accessed": False,
        "diagnostic_only": True,
        "research_metric_valid": False,
        "eligible_for_paper": False,
        "protocol": str(protocol_path),
        "base_protocol": str(base_protocol_path),
        "fast_execution_partition_only": True,
        "provenance": provenance,
        "dataset_identity": reference_identity,
        "arms": arms,
    }
    out = _resolved(root, args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "PASS", "out": str(out), "commit": provenance["commit"]}, sort_keys=True))


if __name__ == "__main__":
    main()
