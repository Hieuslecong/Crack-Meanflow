import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml
from crackmeanflow.paths import ensure_paths
ensure_paths()
from crackmeanflow.train import train

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("train_cli")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "crackmeanflow_default.yaml"))
    args, extra = parser.parse_known_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    for arg in extra:
        if "=" in arg:
            k, v = arg.split("=", 1)
            keys = k.lstrip("-").split(".")
            d = cfg
            for key in keys[:-1]:
                d = d.setdefault(key, {})
            try:
                d[keys[-1]] = float(v)
            except ValueError:
                d[keys[-1]] = v

    train(cfg)


if __name__ == "__main__":
    main()
