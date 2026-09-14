from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import yaml

from .checkpointing import config_hash, file_sha256, protocol_bundle_hash, source_tree_hash


def paper_v3_worktree_blockers() -> list[str]:
    raw = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        stderr=subprocess.STDOUT,
        text=True,
    ).strip()
    allowed = ("reports/", "outputs/", "_data/")
    blockers = []
    for row in [x for x in raw.splitlines() if x.strip()]:
        path = (row[3:] if len(row) >= 4 else row).strip().strip('"')
        if row.startswith("?? ") and path.startswith(allowed):
            continue
        blockers.append(row)
    return blockers


def active_v3_bundle_hash(protocol_path, protocol: dict) -> str:
    """Hash only the active V3 protocol and locked primary config files.

    The historical global protocol bundle remains available for backward
    compatibility, but this active hash is not perturbed by edits to superseded
    V1/V2 protocol files.
    """
    protocol_path = Path(protocol_path).resolve()
    rows = [{"path": str(protocol_path.as_posix()), "sha256": file_sha256(protocol_path)}]
    for arm, cfg_path in sorted(protocol["primary_arms"].items()):
        rows.append({"arm": arm, "path": str(Path(cfg_path).as_posix()), "sha256": file_sha256(cfg_path)})
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def nvidia_driver_info() -> dict:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        ).strip()
        rows = []
        for line in out.splitlines():
            parts = [x.strip() for x in line.split(",")]
            rows.append({
                "name": parts[0] if len(parts) > 0 else None,
                "driver_version": parts[1] if len(parts) > 1 else None,
                "memory_total_mib": int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else (parts[2] if len(parts) > 2 else None),
            })
        return {"available": True, "gpus": rows}
    except Exception as exc:
        return {"available": False, "error": type(exc).__name__}


def verify_v3_provenance(
    *,
    protocol_path,
    config_path,
    preflight_path,
    dataset_name: str,
    dataset_version: str,
    research_total_optimizer_steps: int,
    diagnostic_stop_optimizer_steps: int | None = None,
    actual_dataset_identity: dict | None = None,
    root=None,
) -> dict:
    """Verify every mutable input bound by the active V3 preflight."""
    root = Path(root or Path(__file__).resolve().parents[2]).resolve()
    protocol_hash_path = protocol_path
    protocol_path = Path(protocol_path).resolve()
    config_path = Path(config_path).resolve()
    preflight_path = Path(preflight_path).resolve()
    if not preflight_path.is_file():
        raise RuntimeError("V3 protocol preflight is missing")
    try:
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("V3 protocol preflight is unreadable") from exc
    if preflight.get("schema") != "CRACKMEANFLOW_PROTOCOL_PREFLIGHT_V3" or preflight.get("status") != "PASS":
        raise RuntimeError("V3 protocol preflight is not PASS")
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != "CRACKMEANFLOW_POST_REPAIR_PROTOCOL_V3":
        raise RuntimeError("wrong active V3 protocol")

    def resolved(path_value):
        path = Path(path_value)
        return (path if path.is_absolute() else root / path).resolve()

    arm = next((name for name, path in protocol.get("primary_arms", {}).items() if resolved(path) == config_path), None)
    if arm is None:
        raise RuntimeError("config is not a locked V3 primary arm")
    lock = protocol["config_locks"][arm]
    if file_sha256(config_path) != lock.get("file_sha256") or config_hash(cfg) != lock.get("semantic_sha256"):
        raise RuntimeError("canonical V3 config lock mismatch")
    preflight_arm = (preflight.get("arms") or {}).get(arm)
    if not isinstance(preflight_arm, dict):
        raise RuntimeError("preflight locked arm record is missing")
    if (
        resolved(preflight_arm.get("config", "")) != config_path
        or preflight_arm.get("config_file_sha256") != lock.get("file_sha256")
        or preflight_arm.get("config_semantic_sha256") != lock.get("semantic_sha256")
        or not isinstance(preflight_arm.get("checks"), dict)
        or not all(preflight_arm["checks"].values())
    ):
        raise RuntimeError("preflight config lock does not match the active V3 arm")
    horizon = int(protocol["optimization"]["matched_optimizer_steps"])
    if int(research_total_optimizer_steps) != horizon or horizon != 21000:
        raise RuntimeError("V3 research horizon must remain exactly 21000 optimizer steps")
    if int(cfg.get("train", {}).get("max_optimizer_steps", -1)) != horizon:
        raise RuntimeError("config research horizon does not match locked V3 protocol")
    if int(cfg.get("eval", {}).get("num_steps", -1)) != 1:
        raise RuntimeError("V3 primary execution requires NFE=1")
    if diagnostic_stop_optimizer_steps is not None and not (0 < int(diagnostic_stop_optimizer_steps) < horizon):
        raise RuntimeError("diagnostic stop must be positive and shorter than research horizon")
    source_dataset = protocol["source_dataset"]
    if (
        str(dataset_name) != str(source_dataset["name"])
        or str(dataset_version) != str(source_dataset["split_id"])
    ):
        raise RuntimeError("source dataset name/version does not match the locked V3 source")
    expected_protocol_path = Path(preflight.get("protocol", ""))
    expected_protocol_path = (expected_protocol_path if expected_protocol_path.is_absolute() else root / expected_protocol_path).resolve()
    if expected_protocol_path != protocol_path:
        raise RuntimeError("preflight protocol path does not match the active V3 protocol")

    provenance = preflight.get("provenance") or {}
    current = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "source_tree_sha256": source_tree_hash(root),
        "protocol_file_sha256": file_sha256(protocol_path),
        "protocol_bundle_sha256_legacy_global": protocol_bundle_hash(root),
        "protocol_bundle_sha256_v3_active": active_v3_bundle_hash(protocol_hash_path, protocol),
    }
    labels = {
        "commit": "commit",
        "source_tree_sha256": "source tree",
        "protocol_file_sha256": "protocol file",
        "protocol_bundle_sha256_legacy_global": "legacy protocol bundle",
        "protocol_bundle_sha256_v3_active": "active V3 bundle",
    }
    for key, value in current.items():
        if provenance.get(key) != value:
            raise RuntimeError(f"preflight {labels[key]} does not match current execution")
    expected_identity = preflight.get("dataset_identity")
    if not isinstance(expected_identity, dict) or not expected_identity:
        raise RuntimeError("preflight dataset identity is missing")
    if actual_dataset_identity is not None and json.dumps(actual_dataset_identity, sort_keys=True, separators=(",", ":")) != json.dumps(expected_identity, sort_keys=True, separators=(",", ":")):
        raise RuntimeError("actual dataset identity does not match preflight dataset identity")
    return {
        "arm": arm,
        "protocol": protocol,
        "config": cfg,
        "preflight": preflight,
        "dataset_identity": expected_identity,
        "research_total_optimizer_steps": horizon,
        **current,
    }
