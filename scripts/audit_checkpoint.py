import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crackmeanflow.checkpointing import audit_checkpoint


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path", type=str)
    args = parser.parse_args()
    report = audit_checkpoint(args.ckpt_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
