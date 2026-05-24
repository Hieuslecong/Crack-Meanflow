#!/usr/bin/env python3
"""Audit OmniCrack30K image/mask pairs for CrackMeanFlow."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_pairs(image_dir: Path, mask_dir: Path) -> list[tuple[str, str, str]]:
    images = {p.stem: p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS}
    masks = {p.stem: p for p in mask_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS}
    names = sorted(set(images) & set(masks))
    return [(name, str(images[name]), str(masks[name])) for name in names]


def _prefix(name: str) -> str:
    return name.split("_")[0] if "_" in name else name.split("-")[0]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _image_inventory(path: Path) -> dict[str, Any]:
    files = [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return {
        "count": len(files),
        "suffixes": dict(sorted(Counter(p.suffix.lower() for p in files).items())),
        "stems_sample": [p.stem for p in sorted(files)[:10]],
    }


def _save_montage(groups: dict[str, list[tuple[str, str, str, float]]], output_path: Path, image_size: int = 128) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    labels = []
    for group_name in ("empty", "thin", "thick"):
        rows.append(groups.get(group_name, [])[:5])
        labels.append(group_name)
    if not any(rows):
        return

    label_w = 80
    cols = 5
    cell_w = image_size * 2
    cell_h = image_size + 28
    canvas = Image.new("RGB", (label_w + cols * cell_w, len(rows) * cell_h), "white")
    draw = ImageDraw.Draw(canvas)

    for r, (row, label) in enumerate(zip(rows, labels)):
        y0 = r * cell_h
        draw.text((8, y0 + 8), label, fill=(0, 0, 0))
        for c, item in enumerate(row):
            name, image_path, mask_path, ratio = item
            x0 = label_w + c * cell_w
            image = Image.open(image_path).convert("RGB").resize((image_size, image_size))
            mask = Image.open(mask_path).convert("L").resize((image_size, image_size))
            mask_rgb = Image.merge("RGB", (mask, mask, mask))
            canvas.paste(image, (x0, y0))
            canvas.paste(mask_rgb, (x0 + image_size, y0))
            draw.text((x0 + 2, y0 + image_size + 2), f"{name[:24]} {ratio:.4f}", fill=(0, 0, 0))
    canvas.save(output_path)


def audit(image_dir: Path, mask_dir: Path, output_json: Path, report: Path) -> dict[str, Any]:
    pairs = list_pairs(image_dir, mask_dir)
    image_inv = _image_inventory(image_dir)
    mask_inv = _image_inventory(mask_dir)

    summary: dict[str, Any] = {
        "image_dir": str(image_dir),
        "mask_dir": str(mask_dir),
        "image_inventory": image_inv,
        "mask_inventory": mask_inv,
        "matched_pairs": len(pairs),
        "unmatched_images": image_inv["count"] - len(pairs),
        "unmatched_masks": mask_inv["count"] - len(pairs),
    }

    image_modes = Counter()
    mask_modes = Counter()
    image_sizes = Counter()
    mask_sizes = Counter()
    mask_values = Counter()
    prefix_counts = Counter()
    prefix_empty = Counter()
    crack_ratios = []
    bad_masks = []
    groups: dict[str, list[tuple[str, str, str, float]]] = defaultdict(list)

    for idx, (name, image_path, mask_path) in enumerate(pairs, start=1):
        prefix = _prefix(name)
        prefix_counts[prefix] += 1

        with Image.open(image_path) as image:
            image_modes[image.mode] += 1
            image_sizes[f"{image.size[0]}x{image.size[1]}"] += 1

        with Image.open(mask_path) as mask_img:
            mask = mask_img.convert("L")
            mask_modes[mask_img.mode] += 1
            mask_sizes[f"{mask_img.size[0]}x{mask_img.size[1]}"] += 1
            arr = np.asarray(mask)

        vals = np.unique(arr)
        for value in vals.tolist():
            mask_values[int(value)] += 1
        if not set(vals.tolist()).issubset({0, 255}):
            bad_masks.append({"name": name, "values": vals[:20].astype(int).tolist()})

        positive = int((arr > 0).sum())
        ratio = positive / float(arr.size)
        crack_ratios.append(ratio)
        if positive == 0:
            prefix_empty[prefix] += 1
            if len(groups["empty"]) < 12:
                groups["empty"].append((name, image_path, mask_path, ratio))
        elif ratio <= 0.01:
            if len(groups["thin"]) < 12:
                groups["thin"].append((name, image_path, mask_path, ratio))
        elif ratio >= 0.05:
            if len(groups["thick"]) < 12:
                groups["thick"].append((name, image_path, mask_path, ratio))

        if idx % 5000 == 0:
            print(f"audited {idx}/{len(pairs)}")

    ratios = np.asarray(crack_ratios, dtype=np.float64)
    empty_count = int((ratios == 0).sum()) if len(ratios) else 0
    near_empty_count = int(((ratios > 0) & (ratios <= 0.001)).sum()) if len(ratios) else 0
    nonempty_count = len(ratios) - empty_count
    montage_path = output_json.parent / "audit_montage.png"
    _save_montage(groups, montage_path)

    summary.update(
        {
            "image_modes": dict(sorted(image_modes.items())),
            "mask_modes": dict(sorted(mask_modes.items())),
            "top_image_sizes": dict(image_sizes.most_common(20)),
            "top_mask_sizes": dict(mask_sizes.most_common(20)),
            "mask_unique_values": dict(sorted(mask_values.items())),
            "bad_mask_value_examples": bad_masks[:50],
            "empty_masks": empty_count,
            "near_empty_masks_positive_le_0_001": near_empty_count,
            "nonempty_masks": nonempty_count,
            "empty_ratio": empty_count / len(ratios) if len(ratios) else 0.0,
            "crack_pixel_ratio": {
                "min": float(ratios.min()) if len(ratios) else 0.0,
                "p01": float(np.quantile(ratios, 0.01)) if len(ratios) else 0.0,
                "p05": float(np.quantile(ratios, 0.05)) if len(ratios) else 0.0,
                "p50": float(np.quantile(ratios, 0.50)) if len(ratios) else 0.0,
                "p95": float(np.quantile(ratios, 0.95)) if len(ratios) else 0.0,
                "p99": float(np.quantile(ratios, 0.99)) if len(ratios) else 0.0,
                "max": float(ratios.max()) if len(ratios) else 0.0,
                "mean": float(ratios.mean()) if len(ratios) else 0.0,
            },
            "prefix_counts": dict(prefix_counts.most_common()),
            "prefix_empty_counts": dict(prefix_empty.most_common()),
            "montage_path": str(montage_path),
        }
    )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, default=_json_default) + "\n")

    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# OmniCrack30K Data Audit",
        "",
        f"Image dir: `{image_dir}`",
        f"Mask dir: `{mask_dir}`",
        f"Matched pairs: **{len(pairs)}**",
        f"Images: {image_inv['count']} | Masks: {mask_inv['count']}",
        f"Unmatched images: {summary['unmatched_images']} | Unmatched masks: {summary['unmatched_masks']}",
        "",
        "## Mask summary",
        f"- Empty masks: {empty_count}",
        f"- Nonempty masks: {nonempty_count}",
        f"- Near-empty positive <=0.001: {near_empty_count}",
        f"- Empty ratio: {summary['empty_ratio']:.4f}",
        f"- Unique values: `{summary['mask_unique_values']}`",
        f"- Bad mask value examples: {len(bad_masks)}",
        "",
        "## Ratio stats",
        *[f"- {k}: {v:.6f}" for k, v in summary["crack_pixel_ratio"].items()],
        "",
        "## Modes / sizes",
        f"- Image modes: `{summary['image_modes']}`",
        f"- Mask modes: `{summary['mask_modes']}`",
        f"- Top image sizes: `{summary['top_image_sizes']}`",
        f"- Top mask sizes: `{summary['top_mask_sizes']}`",
        "",
        "## Source prefixes",
        "| Prefix | Count | Empty |",
        "|---|---:|---:|",
    ]
    for prefix, count in prefix_counts.most_common():
        lines.append(f"| {prefix} | {count} | {prefix_empty.get(prefix, 0)} |")
    lines.extend(["", "## Samples", f"Montage: `{montage_path}`", ""])
    report.write_text("\n".join(lines))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--mask-dir", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = audit(args.image_dir, args.mask_dir, args.output_json, args.report)
    print(f"matched_pairs={summary['matched_pairs']}")
    print(f"empty_masks={summary['empty_masks']}")
    print(f"report={args.report}")


if __name__ == "__main__":
    main()
