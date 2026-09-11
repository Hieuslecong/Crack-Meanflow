"""Execute an immutable training queue sequentially with fail-closed checks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crackmeanflow.common import build_dataset_identity, discover_required_splits
from crackmeanflow.common.checkpointing import protocol_bundle_hash, source_tree_hash
from crackmeanflow.common.training_protocol import require_complete_checkpoint, verify_run_completion_artifact


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--lane", default="matched_optimizer_steps")
    parser.add_argument("--log", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    queue_path = Path(args.queue).resolve()
    payload = json.loads(queue_path.read_text(encoding="utf-8"))
    if payload.get("execution") != "QUEUE_ONLY_NO_GPU_RUNS":
        raise RuntimeError("queue execution marker is not QUEUE_ONLY_NO_GPU_RUNS")
    rows = [row for row in payload.get("rows", []) if row.get("lane") == args.lane]
    if not rows:
        raise RuntimeError(f"queue has no rows for lane={args.lane}")
    if any(row.get("access_policy") != {"CFD_TEST": "CLOSED", "GAPS384": "CLOSED"} for row in rows):
        raise RuntimeError("queue contains a row with an invalid target access policy")
    actual_identity = build_dataset_identity(discover_required_splits(args.data_root), include_rows=False)
    current_source = source_tree_hash(root)
    current_protocol = protocol_bundle_hash(root)
    for row in rows:
        if row.get("dataset_identity") != actual_identity:
            raise RuntimeError(f"dataset identity mismatch for {row.get('run_id')}")
        if row.get("source_tree_sha256") != current_source:
            raise RuntimeError(f"source tree changed since queue creation for {row.get('run_id')}")
        if row.get("protocol_bundle_sha256") != current_protocol:
            raise RuntimeError(f"protocol bundle changed since queue creation for {row.get('run_id')}")

    log_path = Path(args.log).resolve() if args.log else queue_path.with_name(queue_path.stem + "_execution.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows, 1):
        out = Path(row["output_dir"])
        best = out / "best.pt"
        last = out / "last.pt"
        complete = out / "RUN_COMPLETE.json"
        if complete.is_file():
            import torch
            record = verify_run_completion_artifact(best, torch.load(best, map_location="cpu", weights_only=False))
            require_complete_checkpoint(torch.load(last, map_location="cpu", weights_only=False), expected_optimizer_steps=int(row["training_budget"]["planned_optimizer_steps"]))
            status = "SKIP_ALREADY_COMPLETE"
        else:
            command = [str(args.data_root) if token == "${DATA_ROOT}" else token for token in row["command"]]
            if last.is_file():
                command.extend(["--resume", str(last)])
                status = "RESUME"
            else:
                status = "START"
            if args.dry_run:
                print(json.dumps({"index": index, "run_id": row["run_id"], "status": status, "command": command}, sort_keys=True))
                continue
            out.mkdir(parents=True, exist_ok=True)
            started = time.time()
            with log_path.open("a", encoding="utf-8") as journal:
                journal.write(json.dumps({"run_id": row["run_id"], "status": status, "started_at": started}) + "\n")
                journal.flush()
                result = subprocess.run(command, cwd=root, stdout=journal, stderr=subprocess.STDOUT, check=False)
            if result.returncode != 0:
                raise RuntimeError(f"run {row['run_id']} failed with return code {result.returncode}; see {out}")
            import torch
            record = verify_run_completion_artifact(best, torch.load(best, map_location="cpu", weights_only=False))
            require_complete_checkpoint(torch.load(last, map_location="cpu", weights_only=False), expected_optimizer_steps=int(row["training_budget"]["planned_optimizer_steps"]))
            status = "PASS"
        with log_path.open("a", encoding="utf-8") as journal:
            journal.write(json.dumps({"run_id": row["run_id"], "status": status, "completed_optimizer_steps": record["completed_optimizer_steps"]}) + "\n")
        print(json.dumps({"index": index, "total": len(rows), "run_id": row["run_id"], "status": status}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
