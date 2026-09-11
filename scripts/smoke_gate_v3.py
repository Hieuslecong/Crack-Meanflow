from __future__ import annotations

"""Paper-V3 hardened smoke gate.

This wrapper binds a diagnostic smoke to the exact git/config/protocol/source
identity that passed the V3 preflight. It delegates GPU timing/training to
``smoke_preflight_v3.py`` and writes a provenance envelope around that report.
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


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], stderr=subprocess.STDOUT, text=True).strip()


def _worktree_blockers() -> list[str]:
    raw = _git("status", "--porcelain=v1", "--untracked-files=all")
    allowed = ("reports/", "outputs/", "_data/")
    blockers = []
    for row in [x for x in raw.splitlines() if x.strip()]:
        path = (row[3:] if len(row) >= 4 else row).strip().strip('"')
        if row.startswith("?? ") and path.startswith(allowed):
            continue
        blockers.append(row)
    return blockers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--protocol", default="configs/protocol/post_repair_protocol_v3.yaml")
    ap.add_argument("--data", required=True)
    ap.add_argument("--research-total-optimizer-steps", type=int, default=21000)
    ap.add_argument("--diagnostic-stop-steps", type=int, default=20)
    ap.add_argument("--validation-mode", choices=["none", "bounded"], default="none")
    ap.add_argument("--validation-max-batches", type=int, default=4)
    ap.add_argument("--validation-seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    protocol = yaml.safe_load(open(args.protocol, "r", encoding="utf-8"))
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    config_path = str(Path(args.config).as_posix())
    arm = None
    for name, path in protocol["primary_arms"].items():
        if str(Path(path).as_posix()) == config_path:
            arm = name
            break
    if arm is None:
        raise RuntimeError(f"config is not a V3 primary arm: {config_path}")

    blockers = _worktree_blockers()
    if blockers:
        raise RuntimeError(f"worktree is not provenance-clean: {blockers}")

    cfg_sem = config_hash(cfg)
    cfg_file = file_sha256(args.config)
    lock = protocol["config_locks"][arm]
    if cfg_sem != lock["semantic_sha256"] or cfg_file != lock["file_sha256"]:
        raise RuntimeError(
            f"config lock mismatch for {arm}: semantic={cfg_sem}, file={cfg_file}, expected={lock}"
        )
    if int(args.research_total_optimizer_steps) != int(protocol["optimization"]["scheduler_total_optimizer_steps"]):
        raise RuntimeError("research scheduler horizon differs from V3 protocol")

    provenance_before = {
        "branch": _git("branch", "--show-current"),
        "commit": _git("rev-parse", "HEAD"),
        "source_tree_sha256": source_tree_hash(),
        "protocol_file_sha256": file_sha256(args.protocol),
        "protocol_bundle_sha256": protocol_bundle_hash(),
        "config_file_sha256": cfg_file,
        "config_semantic_sha256": cfg_sem,
        "arm": arm,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw_out = out.with_suffix(out.suffix + ".raw.json")
    cmd = [
        sys.executable,
        "scripts/smoke_preflight_v3.py",
        "--config", args.config,
        "--protocol", args.protocol,
        "--data", args.data,
        "--research-total-optimizer-steps", str(args.research_total_optimizer_steps),
        "--diagnostic-stop-steps", str(args.diagnostic_stop_steps),
        "--validation-mode", args.validation_mode,
        "--validation-max-batches", str(args.validation_max_batches),
        "--validation-seed", str(args.validation_seed),
        "--out", str(raw_out),
    ]
    completed = subprocess.run(cmd, check=False, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"underlying V3 smoke failed with exit code {completed.returncode}")
    raw_report = json.loads(raw_out.read_text(encoding="utf-8"))

    provenance_after = {
        "branch": _git("branch", "--show-current"),
        "commit": _git("rev-parse", "HEAD"),
        "source_tree_sha256": source_tree_hash(),
        "protocol_file_sha256": file_sha256(args.protocol),
        "protocol_bundle_sha256": protocol_bundle_hash(),
        "config_file_sha256": file_sha256(args.config),
        "config_semantic_sha256": config_hash(yaml.safe_load(open(args.config, "r", encoding="utf-8"))),
        "arm": arm,
    }
    if provenance_before != provenance_after:
        raise RuntimeError("execution identity changed during smoke run")

    envelope = {
        "schema": "CRACKMEANFLOW_DIAGNOSTIC_SMOKE_GATE_V3",
        "status": raw_report.get("status"),
        "diagnostic_only": True,
        "research_metric_valid": False,
        "eligible_for_paper": False,
        "provenance": provenance_before,
        "raw_smoke": raw_report,
    }
    out.write_text(json.dumps(envelope, indent=2, sort_keys=True), encoding="utf-8")
    try:
        raw_out.unlink()
    except OSError:
        pass
    print(json.dumps({"status": envelope["status"], "arm": arm, "out": str(out)}, sort_keys=True))


if __name__ == "__main__":
    main()
