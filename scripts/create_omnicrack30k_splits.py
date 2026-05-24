#!/usr/bin/env python3
"""Create stratified OmniCrack30K dev/full splits."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_pairs(image_dir: Path, mask_dir: Path) -> list[tuple[str, str, str]]:
    images = {p.stem: p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS}
    masks = {p.stem: p for p in mask_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS}
    names = sorted(set(images) & set(masks))
    return [(name, str(images[name]), str(masks[name])) for name in names]


def prefix_of(name: str) -> str:
    return name.split("_")[0] if "_" in name else name.split("-")[0]


def is_empty(mask_path: str) -> bool:
    with Image.open(mask_path) as img:
        arr = np.asarray(img.convert("L"))
    return bool((arr > 0).sum() == 0)


def largest_remainder_counts(n: int, ratios: dict[str, float]) -> dict[str, int]:
    raw = {k: n * v for k, v in ratios.items()}
    counts = {k: int(np.floor(v)) for k, v in raw.items()}
    remaining = n - sum(counts.values())
    order = sorted(ratios, key=lambda k: raw[k] - counts[k], reverse=True)
    for key in order[:remaining]:
        counts[key] += 1
    return counts


def split_bucket(items: list[dict[str, Any]], rng: random.Random, ratios: dict[str, float]) -> dict[str, list[dict[str, Any]]]:
    shuffled = list(items)
    rng.shuffle(shuffled)
    counts = largest_remainder_counts(len(shuffled), ratios)
    out = {}
    start = 0
    for split in ("train", "val", "test"):
        end = start + counts[split]
        out[split] = shuffled[start:end]
        start = end
    return out


def make_splits(records: list[dict[str, Any]], subset_size: int, seed: int) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        buckets[(rec["prefix"], rec["empty_label"])].append(rec)

    selected = []
    if subset_size and subset_size < len(records):
        bucket_sizes = {k: len(v) for k, v in buckets.items()}
        total = sum(bucket_sizes.values())
        raw = {k: subset_size * (size / total) for k, size in bucket_sizes.items()}
        counts = {k: min(bucket_sizes[k], int(np.floor(raw[k]))) for k in bucket_sizes}
        remaining = subset_size - sum(counts.values())
        order = sorted(bucket_sizes, key=lambda k: raw[k] - counts[k], reverse=True)
        idx = 0
        while remaining > 0:
            key = order[idx % len(order)]
            if counts[key] < bucket_sizes[key]:
                counts[key] += 1
                remaining -= 1
            idx += 1
        for key, items in buckets.items():
            sampled = list(items)
            rng.shuffle(sampled)
            selected.extend(sampled[: counts[key]])
    else:
        selected = list(records)

    split_ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
    selected_buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for rec in selected:
        selected_buckets[(rec["prefix"], rec["empty_label"])].append(rec)

    splits = {"train": [], "val": [], "test": []}
    for items in selected_buckets.values():
        bucket_split = split_bucket(items, rng, split_ratios)
        for split, split_items in bucket_split.items():
            splits[split].extend(split_items)

    for split in splits:
        splits[split] = sorted(splits[split], key=lambda r: r["name"])
    return splits


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "image_path", "mask_path", "prefix", "empty_label", "crack_ratio"])
        writer.writeheader()
        writer.writerows(rows)


def summarize(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    out = {}
    for split, rows in splits.items():
        out[split] = {
            "count": len(rows),
            "empty": sum(r["empty_label"] == "empty" for r in rows),
            "nonempty": sum(r["empty_label"] == "nonempty" for r in rows),
            "prefix_counts": dict(Counter(r["prefix"] for r in rows).most_common()),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--mask-dir", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("splits/omnicrack30k"), type=Path)
    parser.add_argument("--subset-size", default=5000, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--report", default=Path("reports/OMNICRACK30K_SPLIT_REPORT.md"), type=Path)
    args = parser.parse_args()

    pairs = list_pairs(args.image_dir, args.mask_dir)
    records = []
    for idx, (name, image_path, mask_path) in enumerate(pairs, start=1):
        with Image.open(mask_path) as img:
            arr = np.asarray(img.convert("L"))
        ratio = float((arr > 0).sum() / arr.size)
        records.append(
            {
                "name": name,
                "image_path": image_path,
                "mask_path": mask_path,
                "prefix": prefix_of(name),
                "empty_label": "empty" if ratio == 0 else "nonempty",
                "crack_ratio": ratio,
            }
        )
        if idx % 5000 == 0:
            print(f"indexed {idx}/{len(pairs)}")

    splits = make_splits(records, args.subset_size, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        write_csv(args.output_dir / f"omnicrack30k_5k_{split}.csv", rows)
    write_csv(args.output_dir / "omnicrack30k_5k_all.csv", [row for split in ("train", "val", "test") for row in splits[split]])

    summary = {
        "image_dir": str(args.image_dir),
        "mask_dir": str(args.mask_dir),
        "seed": args.seed,
        "subset_size": args.subset_size,
        "total_pairs": len(records),
        "splits": summarize(splits),
    }
    (args.output_dir / "omnicrack30k_5k_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# OmniCrack30K Split Report",
        "",
        f"Seed: {args.seed}",
        f"Subset size: {args.subset_size}",
        "Policy: stratified random by `prefix + empty/nonempty`, ratios 70/15/15.",
        "",
        "| Split | Count | Empty | Nonempty |",
        "|---|---:|---:|---:|",
    ]
    for split in ("train", "val", "test"):
        s = summary["splits"][split]
        lines.append(f"| {split} | {s['count']} | {s['empty']} | {s['nonempty']} |")
    lines.extend(["", "## Files"])
    for split in ("train", "val", "test", "all"):
        lines.append(f"- `{args.output_dir / f'omnicrack30k_5k_{split}.csv'}`")
    lines.extend(["", "## Prefix counts"])
    for split in ("train", "val", "test"):
        lines.append(f"### {split}")
        lines.append(str(summary["splits"][split]["prefix_counts"]))
        lines.append("")
    args.report.write_text("\n".join(lines))

    print(json.dumps({k: summary["splits"][k]["count"] for k in ("train", "val", "test")}, indent=2))
    print(f"report={args.report}")


if __name__ == "__main__":
    main()
