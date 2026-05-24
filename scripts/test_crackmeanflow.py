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
logger = logging.getLogger("test_cli")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "crackmeanflow_default.yaml"))
    parser.add_argument("--ckpt", type=str, default=str(ROOT / "checkpoints" / "best.pt"))
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "outputs" / "test_best"))
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--use-ema", action="store_true", default=True)
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    metrics = evaluate(cfg, args.ckpt, args.output_dir, num_steps=args.num_steps, threshold=args.threshold, use_ema=args.use_ema, split=args.split)
    logger.info("F1=%.4f  IoU=%.4f  Dice=%.4f  Precision=%.4f  Recall=%.4f  Latency=%.4fs",
                metrics["f1"], metrics["iou"], metrics["dice"], metrics["precision"], metrics["recall"], metrics["latency_seconds"])


if __name__ == "__main__":
    main()
