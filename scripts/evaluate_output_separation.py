import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crackmeanflow.adapter import CrackMeanFlowModel  # noqa: E402
from crackmeanflow.checkpointing import load_checkpoint_strict  # noqa: E402
from crackmeanflow.data import PairedCrackDataset, load_split_csv  # noqa: E402
from crackmeanflow.metrics import compute_segmentation_metrics  # noqa: E402
from crackmeanflow.paths import ensure_paths  # noqa: E402
from crackmeanflow.sampler import crack_meanflow_sampler  # noqa: E402
from crackmeanflow.train import _merge_global_metrics  # noqa: E402

ensure_paths()
from multi_task.mlt_unet import UNet  # noqa: E402


def _to_uint8(x):
    x = x.detach().cpu().float().squeeze()
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    return (x.numpy() * 255).clip(0, 255).astype("uint8")


def _save_overlay(path, image, mask, pred):
    base = image.detach().cpu().float().squeeze()
    if base.ndim == 3:
        base = base.mean(0)
    base = _to_uint8(base)
    import numpy as np
    rgb = np.stack([base, base, base], axis=-1)
    gt = (mask.detach().cpu().squeeze().numpy() > 0.5)
    pr = (pred.detach().cpu().squeeze().numpy() > 0.5)
    rgb[gt] = [0, 255, 0]
    rgb[pr] = [255, 0, 0]
    rgb[gt & pr] = [255, 255, 0]
    Image.fromarray(rgb.astype("uint8")).save(path)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--overlays", type=int, default=20)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

    torch.manual_seed(args.eval_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.eval_seed)

    pairs = load_split_csv(cfg["paths"][f"{args.split}_csv"])
    loader = DataLoader(PairedCrackDataset(pairs, image_size=cfg["model"]["img_size"]), batch_size=1, shuffle=False, num_workers=0)

    unet = UNet(
        T=cfg["model"]["T"],
        ch=cfg["model"]["ch"],
        ch_mult=cfg["model"]["ch_mult"],
        attn=cfg["model"]["attn"],
        num_res_blocks=cfg["model"]["num_res_blocks"],
        dropout=cfg["model"]["dropout"],
    )
    model = CrackMeanFlowModel(unet, T=cfg["model"]["T"]).to(device)
    load_checkpoint_strict(model, args.ckpt, map_location=device)
    model.eval()

    out_dir = Path(args.output_dir)
    overlay_dir = out_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    crack_vals, bg_vals, metrics = [], [], []
    pred_ratios, gt_ratios = [], []
    hist_bins = torch.linspace(-1.0, 1.0, 101, device=device)
    crack_hist = torch.zeros(100, device=device)
    bg_hist = torch.zeros(100, device=device)

    for i, batch in enumerate(loader):
        image = batch["crack"].to(device)
        mask = batch["mask"].to(device)
        z = torch.randn_like(mask)
        sampled, _seg = crack_meanflow_sampler(model, z, image, num_steps=1)
        pred = (sampled > args.threshold).float()
        gt = (mask > 0.5).float()
        cp = sampled[gt > 0.5]
        bp = sampled[gt <= 0.5]
        if cp.numel() > 0:
            crack_vals.append(cp.detach().float().cpu())
            crack_hist += torch.histc(cp.float(), bins=100, min=-1.0, max=1.0)
        if bp.numel() > 0:
            bg_vals.append(bp.detach().float().cpu())
            bg_hist += torch.histc(bp.float(), bins=100, min=-1.0, max=1.0)
        metrics.append(compute_segmentation_metrics(pred, mask))
        pred_ratios.append(float(pred.mean().item()))
        gt_ratios.append(float(gt.mean().item()))
        if i < args.overlays:
            _save_overlay(overlay_dir / f"{i:03d}_{batch['name'][0]}.png", image[0], mask[0], pred[0])

    crack_cat = torch.cat(crack_vals) if crack_vals else torch.empty(0)
    bg_cat = torch.cat(bg_vals) if bg_vals else torch.empty(0)
    merged = _merge_global_metrics(metrics)
    gt_ratio = sum(gt_ratios) / max(1, len(gt_ratios))
    pred_ratio = sum(pred_ratios) / max(1, len(pred_ratios))
    report = {
        "split": args.split,
        "threshold": args.threshold,
        "eval_seed": args.eval_seed,
        "metrics": merged,
        "pred_pos_ratio": pred_ratio,
        "gt_pos_ratio": gt_ratio,
        "pred_gt_ratio": pred_ratio / max(gt_ratio, 1e-12),
        "crack_mean": float(crack_cat.mean().item()) if crack_cat.numel() else 0.0,
        "crack_std": float(crack_cat.std().item()) if crack_cat.numel() > 1 else 0.0,
        "crack_median": float(crack_cat.median().item()) if crack_cat.numel() else 0.0,
        "background_mean": float(bg_cat.mean().item()) if bg_cat.numel() else 0.0,
        "background_std": float(bg_cat.std().item()) if bg_cat.numel() > 1 else 0.0,
        "background_median": float(bg_cat.median().item()) if bg_cat.numel() else 0.0,
        "separation_score": float(crack_cat.mean().item() - bg_cat.mean().item()) if crack_cat.numel() and bg_cat.numel() else 0.0,
        "crack_px_count": int(crack_cat.numel()),
        "background_px_count": int(bg_cat.numel()),
        "histogram": {
            "bins": [float(x) for x in hist_bins.detach().cpu().tolist()],
            "crack_counts": [float(x) for x in crack_hist.detach().cpu().tolist()],
            "background_counts": [float(x) for x in bg_hist.detach().cpu().tolist()],
        },
        "overlay_dir": str(overlay_dir),
    }
    (out_dir / "output_separation.json").write_text(json.dumps(report, indent=2) + "\n")
    summary = {k: report[k] for k in ["split", "threshold", "pred_gt_ratio", "crack_mean", "background_mean", "separation_score"]}
    summary["f1"] = merged["f1"]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
