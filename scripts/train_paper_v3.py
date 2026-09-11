from __future__ import annotations

"""Locked launcher for paper-grade V3 training.

This launcher verifies the canonical V3 config lock, binds the effective
seed-specific config plus source/protocol/dataset identities, writes a launch
record, then delegates to the canonical ``train_journal.py`` research path.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crackmeanflow.common import config_hash, file_sha256, protocol_bundle_hash, source_tree_hash
from crackmeanflow.common.v3_provenance import active_v3_bundle_hash, nvidia_driver_info, paper_v3_worktree_blockers


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], stderr=subprocess.STDOUT, text=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--protocol", default="configs/protocol/post_repair_protocol_v3.yaml")
    ap.add_argument("--preflight", default="reports/PROTOCOL_PREFLIGHT_V3.json")
    ap.add_argument("--data", required=True)
    ap.add_argument("--dataset-name", default="CFD")
    ap.add_argument("--dataset-version", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    blockers = paper_v3_worktree_blockers()
    if blockers:
        raise RuntimeError(f"paper V3 launch refused: dirty worktree {blockers}")

    protocol = yaml.safe_load(open(args.protocol, "r", encoding="utf-8"))
    canonical_cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    config_path = str(Path(args.config).as_posix())
    arm = next((name for name, path in protocol["primary_arms"].items() if str(Path(path).as_posix()) == config_path), None)
    if arm is None:
        raise RuntimeError("config is not a locked V3 primary arm")
    lock = protocol["config_locks"][arm]
    if file_sha256(args.config) != lock["file_sha256"] or config_hash(canonical_cfg) != lock["semantic_sha256"]:
        raise RuntimeError("canonical V3 config lock mismatch")

    preflight = json.loads(Path(args.preflight).read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS":
        raise RuntimeError("V3 protocol preflight is not PASS")

    commit = _git("rev-parse", "HEAD")
    source_hash = source_tree_hash()
    protocol_file_hash = file_sha256(args.protocol)
    legacy_bundle_hash = protocol_bundle_hash()
    active_bundle_hash = active_v3_bundle_hash(args.protocol, protocol)
    prov = preflight["provenance"]
    if prov["commit"] != commit:
        raise RuntimeError("preflight commit does not match current HEAD")
    if prov["source_tree_sha256"] != source_hash:
        raise RuntimeError("preflight source tree does not match current source")
    if prov["protocol_file_sha256"] != protocol_file_hash:
        raise RuntimeError("preflight protocol file does not match current protocol")
    if prov["protocol_bundle_sha256_legacy_global"] != legacy_bundle_hash:
        raise RuntimeError("preflight legacy protocol bundle does not match current bundle")
    if prov["protocol_bundle_sha256_v3_active"] != active_bundle_hash:
        raise RuntimeError("preflight active V3 bundle does not match current V3 bundle")

    effective_cfg = json.loads(json.dumps(canonical_cfg))
    effective_cfg.setdefault("train", {})["seed"] = int(args.seed)
    effective_hash = config_hash(effective_cfg)
    source_identity_json = json.dumps(preflight["dataset_identity"], sort_keys=True, separators=(",", ":"))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    launch = {
        "schema": "CRACKMEANFLOW_PAPER_V3_LAUNCH_IDENTITY",
        "arm": arm,
        "branch": _git("branch", "--show-current"),
        "commit": commit,
        "seed": int(args.seed),
        "canonical_config_file_sha256": lock["file_sha256"],
        "canonical_config_semantic_sha256": lock["semantic_sha256"],
        "effective_seeded_config_semantic_sha256": effective_hash,
        "protocol_file_sha256": protocol_file_hash,
        "protocol_bundle_sha256_legacy_global": legacy_bundle_hash,
        "protocol_bundle_sha256_v3_active": active_bundle_hash,
        "source_tree_sha256": source_hash,
        "dataset_identity": preflight["dataset_identity"],
        "gpu_driver": nvidia_driver_info(),
        "paper_resume_policy": "epoch_boundary_only",
        "dry_run": bool(args.dry_run),
    }
    (out / "V3_LAUNCH_IDENTITY.json").write_text(json.dumps(launch, indent=2, sort_keys=True), encoding="utf-8")

    cmd = [
        sys.executable, "scripts/train_journal.py",
        "--config", args.config,
        "--data", args.data,
        "--dataset-name", args.dataset_name,
        "--dataset-version", args.dataset_version,
        "--out", str(out),
        "--seed", str(args.seed),
        "--expected-config-sha256", effective_hash,
        "--expected-config-file-sha256", lock["file_sha256"],
        "--expected-protocol-path", args.protocol,
        "--expected-protocol-sha256", protocol_file_hash,
        "--expected-source-tree-sha256", source_hash,
        "--expected-protocol-bundle-sha256", legacy_bundle_hash,
        "--expected-split-id", protocol["source_dataset"]["split_id"],
        "--expected-source-identity-json", source_identity_json,
    ]
    if args.dry_run:
        print(json.dumps({"status": "DRY_RUN_PASS", "launch": launch, "command": cmd}, indent=2))
        return
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
