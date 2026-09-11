from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from .checkpointing import file_sha256


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
    rows = [{"path": str(Path(protocol_path).as_posix()), "sha256": file_sha256(protocol_path)}]
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
