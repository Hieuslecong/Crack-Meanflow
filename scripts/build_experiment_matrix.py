"""Expand the preregistered matrix into an inspectable, train-only queue."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crackmeanflow.common.experiment_harness import (
    build_queue,
    expand_experiment_matrix,
    load_experiment_matrix,
    write_immutable_json,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default=str(root / "configs/protocol/v1_v2_experiment_matrix.yaml"))
    parser.add_argument("--root", default=str(root))
    parser.add_argument("--output-root", default=str(root / "outputs/v1_v2_matrix"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()
    matrix = load_experiment_matrix(args.matrix)
    rows = expand_experiment_matrix(matrix, root=args.root)
    queue = build_queue(rows, root=args.root, output_root=args.output_root)
    payload = {
        "schema": "CRACKMEANFLOW_EXPERIMENT_QUEUE_V1",
        "execution": "QUEUE_ONLY_NO_GPU_RUNS",
        "matrix": str(Path(args.matrix).resolve()),
        "n_runs": len(queue),
        "target_access_policy": {"CFD_TEST": "CLOSED", "GAPS384": "CLOSED"},
        "rows": queue,
    }
    if args.allow_overwrite:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        write_immutable_json(args.out, payload)
    print(json.dumps({"n_runs": len(queue), "execution": payload["execution"], "out": str(Path(args.out).resolve())}, indent=2))


if __name__ == "__main__":
    main()
