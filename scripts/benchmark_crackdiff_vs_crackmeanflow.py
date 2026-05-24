import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml
from crackmeanflow.paths import ensure_paths
ensure_paths()
from crackmeanflow.test import evaluate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("benchmark")


def run_ablations(cfg, ckpt_path, output_dir):
    ablations = [
        {"num_steps": 1, "threshold": 0.0},
        {"num_steps": 4, "threshold": 0.0},
        {"num_steps": 1, "threshold": -0.1},
        {"num_steps": 1, "threshold": 0.1},
    ]
    results = []
    for i, kwargs in enumerate(ablations):
        out = Path(output_dir) / f"ablation_{i}"
        logger.info("Running ablation %s: %s", i, kwargs)
        m = evaluate(cfg, ckpt_path, out, **kwargs, use_ema=True)
        m["ablation_id"] = i
        m["ablation_config"] = kwargs
        results.append(m)
        logger.info("F1=%.4f", m["f1"])
    (Path(output_dir) / "ablation_results.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "crackmeanflow_default.yaml"))
    parser.add_argument("--ckpt", type=str, default=str(ROOT / "checkpoints" / "best.pt"))
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "outputs" / "benchmark"))
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    results = run_ablations(cfg, args.ckpt, args.output_dir)
    logger.info("Best F1: %.4f", max(r["f1"] for r in results))


if __name__ == "__main__":
    main()
