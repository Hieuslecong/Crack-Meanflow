import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from .adapter import CrackMeanFlowModel
from .checkpointing import has_nan_weights, load_checkpoint_strict, save_checkpoint
from .data import PairedCrackDataset, list_pairs, load_split_csv, split_pairs, write_split_report
from .loss import CrackSILoss
from .paths import CRACKDIFF_ROOT, ensure_paths
from .sampler import crack_meanflow_sampler
from .metrics import compute_segmentation_metrics

DEFAULT_EVAL_THRESHOLDS = [
    -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
]

ensure_paths()
from multi_task.mlt_unet import UNet  # noqa: E402

logger = logging.getLogger("crackmeanflow.train")


class DirectSegWrapper(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, image):
        b, _, h, w = image.shape
        x_dummy = torch.zeros(b, 1, h, w, device=image.device, dtype=image.dtype)
        t_dummy = torch.zeros(b, device=image.device, dtype=torch.long)
        _, seg_logits = self.unet(x_dummy, t_dummy, image)
        return seg_logits


class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}
        self.num_updates = 0

    def update(self, model):
        self.num_updates += 1
        d = self.decay
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k not in self.shadow:
                    self.shadow[k] = v.clone().detach()
                    continue
                if torch.is_floating_point(v):
                    self.shadow[k].mul_(d).add_(v, alpha=1.0 - d)
                else:
                    self.shadow[k].copy_(v)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state):
        self.shadow = state


def _merge_global_metrics(metrics_list):
    tp = sum(m["tp"] for m in metrics_list)
    fp = sum(m["fp"] for m in metrics_list)
    fn = sum(m["fn"] for m in metrics_list)
    tn = sum(m["tn"] for m in metrics_list)
    eps = 1e-7
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    return {
        "f1": float(f1), "dice": float(f1), "iou": float(iou),
        "precision": float(precision), "recall": float(recall), "accuracy": float(accuracy),
        "tp": float(tp), "fp": float(fp), "fn": float(fn), "tn": float(tn),
    }


def _build_teacher_model(cfg, device):
    teacher_path = cfg["loss"].get("teacher_checkpoint")
    if not teacher_path:
        raise ValueError("loss.teacher_checkpoint is required when distill_weight > 0.")
    teacher_cfg = cfg["loss"].get("teacher_model", {})
    teacher_unet = UNet(
        T=teacher_cfg.get("T", 1000),
        ch=teacher_cfg.get("ch", 32),
        ch_mult=teacher_cfg.get("ch_mult", [1, 2]),
        attn=teacher_cfg.get("attn", []),
        num_res_blocks=teacher_cfg.get("num_res_blocks", 2),
        dropout=teacher_cfg.get("dropout", 0.1),
    )
    teacher = DirectSegWrapper(teacher_unet).to(device)
    ckpt = torch.load(teacher_path, map_location=device)
    state = ckpt.get("model", ckpt)
    teacher.load_state_dict(state, strict=True)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    logger.info("Loaded teacher checkpoint: %s  params=%.2fM", teacher_path, sum(p.numel() for p in teacher.parameters()) / 1e6)
    return teacher


@torch.no_grad()
def _full_val_eval(model, val_pairs, cfg, device, eval_seed=0):
    model.eval()
    ds = PairedCrackDataset(val_pairs, image_size=cfg["model"]["img_size"])
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    thresholds = [float(x) for x in cfg.get("eval", {}).get("thresholds", DEFAULT_EVAL_THRESHOLDS)]
    if eval_seed is not None:
        torch.manual_seed(int(eval_seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(eval_seed))
    by_th = {th: [] for th in thresholds}
    sampled_min, sampled_max, sampled_abs_max = [], [], []
    seg_abs_max = []
    pred_ratios = {th: [] for th in thresholds}
    gt_ratios = []
    crack_vals_all, bg_vals_all = [], []
    for batch in loader:
        image = batch["crack"].to(device)
        mask = batch["mask"].to(device)
        z = torch.randn_like(mask)
        sampled, seg_logits = crack_meanflow_sampler(model, z, image, num_steps=1)
        if not torch.isfinite(sampled).all():
            raise RuntimeError("Non-finite sampled_mask during full validation.")
        if sampled.abs().max().item() > float(cfg.get("eval", {}).get("max_sampled_abs", 50.0)):
            raise RuntimeError(f"sampled_mask abs max too high: {sampled.abs().max().item():.6f}")
        if seg_logits is not None and seg_logits.abs().max().item() > float(cfg.get("eval", {}).get("max_seg_logits_abs", 80.0)):
            raise RuntimeError(f"seg_logits abs max too high: {seg_logits.abs().max().item():.6f}")
        sampled_min.append(float(sampled.min().item()))
        sampled_max.append(float(sampled.max().item()))
        sampled_abs_max.append(float(sampled.abs().max().item()))
        if seg_logits is not None:
            seg_abs_max.append(float(seg_logits.abs().max().item()))
        gt_ratios.append(float((mask > 0.5).float().mean().item()))
        # separation diagnostics: sampled values on crack vs background pixels
        gt_binary = (mask > 0.5).float()
        crack_px = sampled[gt_binary > 0.5]
        bg_px = sampled[gt_binary <= 0.5]
        if crack_px.numel() > 0:
            crack_vals_all.append(crack_px)
        if bg_px.numel() > 0:
            bg_vals_all.append(bg_px)
        for th in thresholds:
            pred = (sampled > th).float()
            pred_ratios[th].append(float(pred.mean().item()))
            by_th[th].append(compute_segmentation_metrics(pred, mask))
    sweep = {str(th): _merge_global_metrics(ms) for th, ms in by_th.items()}
    best_th = max(thresholds, key=lambda th: (sweep[str(th)]["f1"], sweep[str(th)]["recall"]))
    best = dict(sweep[str(best_th)])
    best["best_th"] = float(best_th)
    best["best_f1"] = float(best["f1"])
    gt_ratio = sum(gt_ratios) / max(1, len(gt_ratios))
    pred_ratio = sum(pred_ratios[best_th]) / max(1, len(pred_ratios[best_th]))
    best["gt_pos_ratio"] = float(gt_ratio)
    best["pred_pos_ratio"] = float(pred_ratio)
    best["pred_gt_ratio"] = float(pred_ratio / max(gt_ratio, 1e-12))
    best["eval_seed"] = int(eval_seed)
    stats = {
        "sampled_min": min(sampled_min) if sampled_min else 0.0,
        "sampled_max": max(sampled_max) if sampled_max else 0.0,
        "sampled_abs_max": max(sampled_abs_max) if sampled_abs_max else 0.0,
        "seg_logits_abs_max": max(seg_abs_max) if seg_abs_max else 0.0,
    }
    # separation diagnostics
    separation = {}
    if crack_vals_all and bg_vals_all:
        crack_cat = torch.cat(crack_vals_all)
        bg_cat = torch.cat(bg_vals_all)
        separation = {
            "crack_mean": float(crack_cat.mean().item()),
            "crack_std": float(crack_cat.std().item()),
            "crack_median": float(crack_cat.median().item()),
            "bg_mean": float(bg_cat.mean().item()),
            "bg_std": float(bg_cat.std().item()),
            "bg_median": float(bg_cat.median().item()),
            "separation_score": float(crack_cat.mean().item() - bg_cat.mean().item()),
            "crack_px_count": int(crack_cat.numel()),
            "bg_px_count": int(bg_cat.numel()),
        }
    model.train()
    return best, sweep, stats, separation


def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
    out_dir = Path(cfg["paths"]["output_dir"])
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    report_dir = Path(cfg["paths"]["reports_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # --- data split ---
    paths_cfg = cfg["paths"]
    if all(k in paths_cfg for k in ("train_csv", "val_csv", "test_csv")):
        train_pairs = load_split_csv(paths_cfg["train_csv"])
        val_pairs = load_split_csv(paths_cfg["val_csv"])
        test_pairs = load_split_csv(paths_cfg["test_csv"])
        pairs = train_pairs + val_pairs + test_pairs
        split_policy = "configured_csv_split"
    else:
        pairs = list_pairs(paths_cfg["image_dir"], paths_cfg["mask_dir"])
        train_ratio = cfg["train"].get("train_ratio", 0.8)
        val_ratio = cfg["train"].get("val_ratio", 0.1)
        train_pairs, val_pairs, test_pairs, split_policy = split_pairs(pairs, train_ratio=train_ratio, val_ratio=val_ratio, seed=42)
    assert len(pairs) >= 5, f"Too few pairs ({len(pairs)}), need >= 5"
    write_split_report(report_dir / "DATA_SPLIT_REPORT.md", pairs, train_pairs, val_pairs, test_pairs, paths_cfg["image_dir"], paths_cfg["mask_dir"], split_policy)
    logger.info("Pairs=%d  Train=%d  Val=%d  Test=%d  policy=%s", len(pairs), len(train_pairs), len(val_pairs), len(test_pairs), split_policy)

    train_ds = PairedCrackDataset(train_pairs, image_size=cfg["model"]["img_size"], augment=cfg["train"].get("augment", True))
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=0, pin_memory=True)

    # --- model ---
    unet = UNet(
        T=cfg["model"]["T"],
        ch=cfg["model"]["ch"],
        ch_mult=cfg["model"]["ch_mult"],
        attn=cfg["model"]["attn"],
        num_res_blocks=cfg["model"]["num_res_blocks"],
        dropout=cfg["model"]["dropout"],
    )
    model = CrackMeanFlowModel(unet, T=cfg["model"]["T"]).to(device)
    logger.info("Model params: %.2fM", sum(p.numel() for p in model.parameters()) / 1e6)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    ema = EMA(model, decay=cfg["train"]["ema_decay"])
    scaler = GradScaler()
    distill_weight = cfg["loss"].get("distill_weight", 0.0)
    teacher_model = _build_teacher_model(cfg, device) if distill_weight > 0.0 else None
    criterion = CrackSILoss(
        si_loss_kwargs=cfg["loss"]["si_loss_kwargs"],
        seg_loss_weight=cfg["loss"]["seg_loss_weight"],
        endpoint_loss_weight=cfg["loss"]["endpoint_loss_weight"],
        thin_loss_weight=cfg["loss"]["thin_loss_weight"],
        mode=cfg["loss"].get("mode", "hybrid"),
        endpoint_mode=cfg["loss"].get("endpoint_mode", "l1"),
        tversky_alpha=cfg["loss"].get("tversky_alpha", 0.3),
        tversky_beta=cfg["loss"].get("tversky_beta", 0.7),
        si_loss_weight=cfg["loss"].get("si_loss_weight", 1.0),
        distill_weight=distill_weight,
        teacher_model=teacher_model,
    )

    # --- resume ---
    global_step = 0
    best_f1 = 0.0
    resume_path = ckpt_dir / "last.pt"
    if resume_path.exists():
        try:
            ckpt_data, _ = load_checkpoint_strict(model, resume_path, map_location=device)
            if ckpt_data.get("optimizer"):
                optimizer.load_state_dict(ckpt_data["optimizer"])
            if ckpt_data.get("global_step"):
                global_step = ckpt_data["global_step"]
            if ckpt_data.get("best_metrics", {}).get("f1"):
                best_f1 = ckpt_data["best_metrics"]["f1"]
            if ckpt_data.get("ema"):
                ema.load_state_dict(ckpt_data["ema"])
            logger.info("Resumed from step %d  best_f1=%.4f", global_step, best_f1)
        except Exception as e:
            logger.warning("Resume failed (%s), starting fresh.", e)

    epochs = cfg["train"]["epochs"]
    log_interval = cfg["train"]["log_interval"]
    save_interval = cfg["train"]["save_interval"]
    max_grad_norm = cfg["train"]["max_grad_norm"]
    max_batches = cfg["train"].get("max_train_batches", 0)
    grad_accum_steps = int(cfg["train"].get("grad_accum_steps", 1))
    best_ckpt_path = ckpt_dir / "best.pt"

    # --- train loop ---
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(train_loader):
            if max_batches and batch_idx >= max_batches:
                break
            image = batch["crack"].to(device)
            mask = batch["mask"].to(device)
            x0 = mask * 2.0 - 1.0

            with autocast():
                total_loss, loss_dict = criterion(model, x0, {"y": image, "mask_gt": mask})

            if not torch.isfinite(total_loss):
                logger.warning("Non-finite loss at step %d, skipping.", global_step)
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(total_loss / grad_accum_steps).backward()
            should_step = ((batch_idx + 1) % grad_accum_steps == 0)
            if max_batches and (batch_idx + 1) >= max_batches:
                should_step = True
            elif (batch_idx + 1) == len(train_loader):
                should_step = True

            if should_step:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)
                global_step += 1

                if global_step % log_interval == 0:
                    ld = loss_dict
                    logger.info("[step %d] total=%.4f seg=%.4f %s", global_step, ld["total_loss"], ld.get("seg_loss", 0.0),
                                " ".join(f"{k}={v:.4f}" for k, v in ld.items() if k not in ("total_loss", "seg_loss", "nan_flags", "mode")))

                    if global_step % save_interval == 0:
                        save_checkpoint(ckpt_dir / "last.pt", model, optimizer=optimizer, ema=ema, epoch=epoch, global_step=global_step, best_metrics={"f1": best_f1}, config=cfg)

            epoch_losses.append(loss_dict)

        if not epoch_losses:
            logger.warning("Epoch %d: no valid batches.", epoch)
            continue

        numeric_keys = [k for k, v in epoch_losses[0].items() if isinstance(v, (int, float))]
        avg = {k: sum(d[k] for d in epoch_losses) / len(epoch_losses) for k in numeric_keys}
        elapsed = time.time() - t0
        logger.info("Epoch %d done in %.1fs  avg_total=%.4f", epoch, elapsed, avg["total_loss"])

        # --- full validation gate; deterministic one-step flow output only ---
        eval_seed = int(cfg.get("eval", {}).get("eval_seed", 0))
        val_metrics, val_sweep, val_stats, val_separation = _full_val_eval(model, val_pairs if val_pairs else test_pairs, cfg, device, eval_seed=eval_seed)
        logger.info(
            "Full val F1=%.4f th=%.3f pred/gt=%.3f sampled_abs=%.3f seg_abs=%.3f (best=%.4f)",
            val_metrics["f1"], val_metrics["best_th"], val_metrics["pred_gt_ratio"],
            val_stats["sampled_abs_max"], val_stats["seg_logits_abs_max"], best_f1,
        )
        if val_metrics["f1"] > best_f1:
            best_f1 = float(val_metrics["f1"])
            best_payload = dict(val_metrics)
            best_payload["checkpoint_saved_by"] = "full_val"
            save_checkpoint(best_ckpt_path, model, optimizer=optimizer, ema=ema, epoch=epoch, global_step=global_step, best_metrics=best_payload, config=cfg)
            (out_dir / "val_threshold_sweep.json").write_text(json.dumps(val_sweep, indent=2) + "\n")
            (out_dir / "sampled_mask_stats.json").write_text(json.dumps(val_stats, indent=2) + "\n")
            (out_dir / "sampled_mask_separation.json").write_text(json.dumps(val_separation, indent=2) + "\n")
            (out_dir / "prediction_ratio_report.json").write_text(json.dumps({
                "eval_seed": eval_seed,
                "threshold": val_metrics["best_th"],
                "pred_pos_ratio": val_metrics["pred_pos_ratio"],
                "gt_pos_ratio": val_metrics["gt_pos_ratio"],
                "pred_gt_ratio": val_metrics["pred_gt_ratio"],
            }, indent=2) + "\n")
            (out_dir / "full_val_metrics.json").write_text(json.dumps(best_payload, indent=2) + "\n")
            logger.info("New best full-val F1=%.4f saved.", best_f1)

        save_checkpoint(ckpt_dir / "last.pt", model, optimizer=optimizer, ema=ema, epoch=epoch, global_step=global_step, best_metrics={"f1": best_f1}, config=cfg)

    logger.info("Training complete. best_f1=%.4f", best_f1)
    return best_ckpt_path


@torch.no_grad()
def _quick_eval(model, test_pairs, cfg, device):
    model.eval()
    from .data import PairedCrackDataset
    from torch.utils.data import DataLoader
    ds = PairedCrackDataset(test_pairs, image_size=cfg["model"]["img_size"])
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    f1s = []
    max_eval = cfg["train"].get("max_eval_batches", 20)
    for i, batch in enumerate(loader):
        if i >= max_eval:
            break
        image = batch["crack"].to(device)
        mask = batch["mask"].to(device)
        z = torch.randn_like(mask)
        sampled, _seg_logits = crack_meanflow_sampler(model, z, image, num_steps=1)
        threshold = float(cfg.get("eval", {}).get("threshold", 0.0))
        pred_flow = (sampled > threshold).float()
        m_flow = compute_segmentation_metrics(pred_flow, mask)
        f1s.append(m_flow["f1"])
    model.train()
    return sum(f1s) / len(f1s) if f1s else 0.0
