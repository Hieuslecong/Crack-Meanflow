from __future__ import annotations

"""Paper-V3 diagnostic smoke runner.

This runner is intentionally NOT a research-training entry point.  It reuses the
canonical model/loss/data/scheduler components while keeping the scientific
scheduler horizon independent from the short diagnostic stop.  It never writes
RUN_COMPLETE.json or a paper-valid checkpoint.
"""

import argparse
import hashlib
import itertools
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crackmeanflow.common import (
    EMA,
    EpochRandomSampler,
    PairedCrackDataset,
    audit_content_split_integrity,
    build_dataset_identity,
    environment_info,
    make_warmup_cosine_scheduler,
    optimizer_steps_per_epoch,
    resolve_thresholds,
    source_splits_for_config,
)
from crackmeanflow.common.evaluation import calibrate_threshold_on_validation
from crackmeanflow.common.training_protocol import training_split_view
from crackmeanflow.factory import build_training_components
from crackmeanflow.journal import calibrate_geometry_threshold_on_validation
from crackmeanflow.journal.engine.dataset import GeometryDataset
from crackmeanflow.sampler import crack_meanflow_sampler


def seed_all(seed: int, deterministic: bool, warn_only: bool = False) -> None:
    seed = int(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=bool(warn_only))


def _seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _limited_loader(loader, max_batches: int):
    class Limited:
        def __iter__(self_inner):
            return itertools.islice(iter(loader), int(max_batches))

        def __len__(self_inner):
            return min(len(loader), int(max_batches))

    return Limited()


def _diagnostic_thresholds(eval_cfg: dict, max_count: int = 5) -> list[float]:
    values = [float(x) for x in resolve_thresholds(eval_cfg, final=False)]
    if len(values) <= max_count:
        return values
    idx = np.linspace(0, len(values) - 1, num=max_count, dtype=int)
    return sorted({values[int(i)] for i in idx})


def _tensor_digest(state: dict) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if torch.is_tensor(value):
            x = value.detach().cpu().contiguous()
            h.update(key.encode("utf-8"))
            h.update(str(tuple(x.shape)).encode("ascii"))
            h.update(str(x.dtype).encode("ascii"))
            h.update(x.numpy().tobytes())
    return h.hexdigest()


def _checkpoint_roundtrip(model, optimizer, scheduler, ema, global_step: int) -> dict:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "ema": ema.shadow,
        "global_step": int(global_step),
    }
    before_model = _tensor_digest(payload["model"])
    before_ema = _tensor_digest(payload["ema"])
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as fh:
        path = fh.name
    try:
        torch.save(payload, path)
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        result = {
            "pass": (
                int(loaded["global_step"]) == int(global_step)
                and _tensor_digest(loaded["model"]) == before_model
                and _tensor_digest(loaded["ema"]) == before_ema
                and isinstance(loaded.get("optimizer"), dict)
                and isinstance(loaded.get("scheduler"), dict)
            ),
            "bytes": int(os.path.getsize(path)),
            "model_sha256": before_model,
            "ema_sha256": before_ema,
        }
        return result
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--protocol", default="configs/protocol/post_repair_protocol_v3.yaml")
    ap.add_argument("--data", required=True)
    ap.add_argument("--research-total-optimizer-steps", type=int, default=None)
    ap.add_argument("--diagnostic-stop-steps", type=int, default=20)
    ap.add_argument("--validation-mode", choices=["none", "bounded"], default="none")
    ap.add_argument("--validation-max-batches", type=int, default=4)
    ap.add_argument("--validation-seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    started = time.perf_counter()
    timings = {
        "startup_seconds": 0.0,
        "provenance_seconds": 0.0,
        "dataset_seconds": 0.0,
        "first_batch_seconds": 0.0,
        "data_wait_seconds": 0.0,
        "h2d_seconds": 0.0,
        "model_loss_forward_seconds": 0.0,
        "backward_seconds": 0.0,
        "optimizer_seconds": 0.0,
        "ema_seconds": 0.0,
        "validation_seconds": 0.0,
        "checkpoint_roundtrip_seconds": 0.0,
    }

    t = time.perf_counter()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    protocol = yaml.safe_load(open(args.protocol, "r", encoding="utf-8"))
    timings["startup_seconds"] = time.perf_counter() - t

    research_total = int(
        args.research_total_optimizer_steps
        if args.research_total_optimizer_steps is not None
        else cfg.get("train", {}).get("max_optimizer_steps", protocol["optimization"]["scheduler_total_optimizer_steps"])
    )
    stop_steps = int(args.diagnostic_stop_steps)
    if stop_steps < 1 or research_total < 2 or stop_steps >= research_total:
        raise ValueError("diagnostic_stop_steps must be >=1 and strictly smaller than research scheduler horizon")
    cfg_budget = int(cfg.get("train", {}).get("max_optimizer_steps", research_total))
    if cfg_budget != research_total:
        raise RuntimeError(f"config/research horizon mismatch: config={cfg_budget} requested={research_total}")

    seed = int(cfg["train"].get("seed", 42))
    seed_all(seed, bool(cfg["train"].get("deterministic", False)), bool(cfg["train"].get("deterministic_warn_only", False)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Paper-V3 GPU smoke requires CUDA; refusing a misleading CPU performance smoke")

    t = time.perf_counter()
    splits = source_splits_for_config(args.data, cfg)
    training_splits = training_split_view(splits)
    leakage = audit_content_split_integrity(splits)
    identities = build_dataset_identity(splits, include_rows=False)
    timings["provenance_seconds"] = time.perf_counter() - t

    t = time.perf_counter()
    base = lambda split, aug: PairedCrackDataset(
        training_splits[split],
        cfg["model"]["img_size"],
        aug,
        aug and cfg["train"].get("photometric_augment", False),
        cfg["train"].get("mask_resize_mode", "nearest"),
        cfg["train"].get("mask_binarization", "auto_binary_safe"),
    )
    if cfg["backbone"] == "geocrack_imf":
        train_ds = GeometryDataset(base("train", True), cfg["model"].get("max_radius", 16), cfg["model"].get("representation", "centerline_radius"), cfg["model"].get("distance_encoding", "linear"))
        val_ds = GeometryDataset(base("val", False), cfg["model"].get("max_radius", 16), cfg["model"].get("representation", "centerline_radius"), cfg["model"].get("distance_encoding", "linear"))
    else:
        train_ds = base("train", True)
        val_ds = base("val", False)

    g = torch.Generator().manual_seed(seed)
    sampler = EpochRandomSampler(len(train_ds), seed=seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        sampler=sampler,
        drop_last=bool(cfg["train"].get("drop_last", True)),
        num_workers=int(cfg["train"].get("num_workers", 0)),
        worker_init_fn=_seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.get("eval", {}).get("batch_size", 1)),
        shuffle=False,
        num_workers=int(cfg["train"].get("num_workers", 0)),
    )
    timings["dataset_seconds"] = time.perf_counter() - t

    grad_accum = int(cfg["train"]["grad_accum_steps"])
    drop_incomplete = bool(cfg["train"].get("drop_incomplete_accumulation", True))
    steps_per_epoch = optimizer_steps_per_epoch(len(train_loader), grad_accum, drop_incomplete)
    expected_train = int(protocol["source_dataset"]["expected_train_samples"])
    expected_steps = int(protocol["optimization"]["expected_optimizer_steps_per_epoch"])
    expected_effective_batch = int(protocol["optimization"]["effective_batch_size"])
    effective_batch = int(cfg["train"]["batch_size"]) * grad_accum
    static_checks = {
        "train_samples": {"actual": len(train_ds), "expected": expected_train, "pass": len(train_ds) == expected_train},
        "optimizer_steps_per_epoch": {"actual": int(steps_per_epoch), "expected": expected_steps, "pass": int(steps_per_epoch) == expected_steps},
        "effective_batch": {"actual": effective_batch, "expected": expected_effective_batch, "pass": effective_batch == expected_effective_batch},
        "scheduler_total_steps": {"actual": research_total, "expected": int(protocol["optimization"]["scheduler_total_optimizer_steps"]), "pass": research_total == int(protocol["optimization"]["scheduler_total_optimizer_steps"])},
    }
    if not all(v["pass"] for v in static_checks.values()):
        raise RuntimeError(f"Paper-V3 static protocol gate failed: {static_checks}")

    model, rasterizer, lossfn = build_training_components(cfg, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))
    scheduler = make_warmup_cosine_scheduler(
        optimizer,
        int(cfg["train"]["epochs"]),
        int(steps_per_epoch),
        int(cfg["train"].get("warmup_epochs", 0)),
        total_optimizer_steps=research_total,
    )
    ema = EMA(model, float(cfg["train"]["ema_decay"]))
    if int(getattr(scheduler, "_cmf_total_steps", -1)) != research_total:
        raise RuntimeError("scheduler horizon is not the requested research horizon")
    expected_warm = int(protocol["optimization"]["expected_warmup_optimizer_steps"])
    if int(getattr(scheduler, "_cmf_warmup_steps", -1)) != expected_warm:
        raise RuntimeError(f"warmup mismatch: {getattr(scheduler, '_cmf_warmup_steps', None)} != {expected_warm}")

    if hasattr(lossfn, "set_epoch"):
        lossfn.set_epoch(0)
    if hasattr(train_loader.sampler, "set_epoch"):
        train_loader.sampler.set_epoch(0)

    first_param = next(p for p in model.parameters() if p.requires_grad)
    first_param_before = first_param.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    global_sample_offset = 0
    losses = []
    grad_norms = []
    samples = 0
    microbatches = 0
    data_iter = iter(train_loader)
    first_batch_seen = False
    training_started = time.perf_counter()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    while global_step < stop_steps:
        t_wait = time.perf_counter()
        try:
            batch = next(data_iter)
        except StopIteration:
            if hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(1)
            data_iter = iter(train_loader)
            batch = next(data_iter)
        wait = time.perf_counter() - t_wait
        timings["data_wait_seconds"] += wait
        if not first_batch_seen:
            timings["first_batch_seconds"] = wait
            first_batch_seen = True

        _sync(device)
        th = time.perf_counter()
        image = batch["crack"].to(device)
        if cfg["backbone"] == "geocrack_imf":
            geometry = batch["geometry"].to(device)
            radius_valid = batch["radius_valid"].to(device)
            mask = batch["mask"].to(device)
        else:
            mask = batch["mask"].to(device)
        _sync(device)
        timings["h2d_seconds"] += time.perf_counter() - th

        bs = int(image.shape[0])
        samples += bs
        microbatches += 1
        _sync(device)
        tf = time.perf_counter()
        if cfg["backbone"] == "geocrack_imf":
            loss, logs = lossfn(model, geometry, image, radius_valid, mask_gt=mask, sample_offset=global_sample_offset)
        elif cfg["backbone"] in {"sit_imf_mask", "hybrid_imf_mask"}:
            loss, logs = lossfn(model, mask, image, sample_offset=global_sample_offset)
        else:
            loss, logs = lossfn(model, mask, {"y": image, "sample_offset": global_sample_offset})
        _sync(device)
        timings["model_loss_forward_seconds"] += time.perf_counter() - tf
        if not bool(torch.isfinite(loss).all()):
            raise FloatingPointError(f"non-finite diagnostic loss at optimizer_step={global_step}")
        global_sample_offset += bs
        losses.append(float(logs["total_loss"]))

        group_index = (microbatches - 1) % grad_accum
        _sync(device)
        tb = time.perf_counter()
        (loss / grad_accum).backward()
        _sync(device)
        timings["backward_seconds"] += time.perf_counter() - tb

        if group_index == grad_accum - 1:
            _sync(device)
            to = time.perf_counter()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["max_grad_norm"]), error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            _sync(device)
            timings["optimizer_seconds"] += time.perf_counter() - to
            grad_norms.append(float(grad_norm.detach().cpu()))

            _sync(device)
            te = time.perf_counter()
            ema.update(model)
            _sync(device)
            timings["ema_seconds"] += time.perf_counter() - te
            global_step += 1

    pure_training_seconds = time.perf_counter() - training_started
    parameter_delta = float((first_param.detach() - first_param_before).abs().mean().cpu())
    if parameter_delta <= 0.0:
        raise RuntimeError("diagnostic training produced no parameter change")

    validation = None
    if args.validation_mode == "bounded":
        limited = _limited_loader(val_loader, int(args.validation_max_batches))
        thresholds = _diagnostic_thresholds(cfg.get("eval", {}), max_count=5)
        _sync(device)
        tv = time.perf_counter()
        ema.apply(model)
        try:
            if cfg["backbone"] == "geocrack_imf":
                sweep, best_th = calibrate_geometry_threshold_on_validation(
                    model, limited, device, rasterizer, thresholds, int(args.validation_seed), cfg["model"].get("max_radius", 16)
                )
            else:
                sweep, best_th = calibrate_threshold_on_validation(
                    model, limited, device, crack_meanflow_sampler, thresholds, 1, int(args.validation_seed), cfg.get("eval", {}).get("cfg_scale", 1.0)
                )
        finally:
            ema.restore(model)
        _sync(device)
        timings["validation_seconds"] = time.perf_counter() - tv
        validation = {
            "mode": "bounded",
            "max_batches": int(args.validation_max_batches),
            "seed": int(args.validation_seed),
            "thresholds": thresholds,
            "best_threshold": float(best_th),
            "best_f1": float(sweep[best_th]["f1"]),
            "research_metric_valid": False,
        }

    tc = time.perf_counter()
    checkpoint_roundtrip = _checkpoint_roundtrip(model, optimizer, scheduler, ema, global_step)
    timings["checkpoint_roundtrip_seconds"] = time.perf_counter() - tc
    if not checkpoint_roundtrip["pass"]:
        raise RuntimeError("diagnostic checkpoint roundtrip failed")

    total_seconds = time.perf_counter() - started
    cuda_memory = {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "total_device_bytes": int(torch.cuda.get_device_properties(device).total_memory),
    }
    report = {
        "schema": "CRACKMEANFLOW_DIAGNOSTIC_SMOKE_V3",
        "status": "PASS",
        "diagnostic_only": True,
        "research_metric_valid": False,
        "eligible_for_paper": False,
        "config": str(args.config),
        "protocol": str(args.protocol),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "environment": environment_info(),
        "dataset_identity": identities,
        "content_leakage_audit": leakage,
        "static_checks": static_checks,
        "training": {
            "research_scheduler_total_steps": research_total,
            "diagnostic_stop_steps": stop_steps,
            "scheduler_warmup_steps": int(getattr(scheduler, "_cmf_warmup_steps", -1)),
            "optimizer_steps_per_epoch": int(steps_per_epoch),
            "completed_optimizer_steps": int(global_step),
            "microbatches": int(microbatches),
            "samples_processed": int(samples),
            "loss_mean": float(np.mean(losses)),
            "loss_first": float(losses[0]),
            "loss_last": float(losses[-1]),
            "grad_norm_mean": float(np.mean(grad_norms)),
            "lr_end": float(optimizer.param_groups[0]["lr"]),
            "parameter_mean_abs_delta": parameter_delta,
            "pure_training_seconds": pure_training_seconds,
            "seconds_per_optimizer_step": pure_training_seconds / max(global_step, 1),
            "samples_per_second": samples / max(pure_training_seconds, 1e-12),
        },
        "validation": validation or {"mode": "none", "research_metric_valid": False},
        "checkpoint_roundtrip": checkpoint_roundtrip,
        "cuda_memory": cuda_memory,
        "timings": {**timings, "total_seconds": total_seconds},
        "artifact_policy": {
            "writes_run_complete": False,
            "writes_best_checkpoint": False,
            "writes_last_checkpoint": False,
            "paper_valid": False,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "PASS", "diagnostic_only": True, "out": str(out), "completed_optimizer_steps": global_step}, sort_keys=True))


if __name__ == "__main__":
    main()
