from __future__ import annotations

"""Diagnostic V4 benchmark for execution-partition-only micro-batch changes.

The benchmark validates the canonical V3 configuration before making an
in-memory train partition override.  It uses the same source split, sampler,
model, loss, optimizer, scheduler horizon, EMA, and deterministic setup as the
source screen.  Its output is never paper-eligible.
"""

import argparse
import copy
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crackmeanflow.common import (  # noqa: E402
    EMA,
    PairedCrackDataset,
    audit_content_split_integrity,
    audit_group_integrity,
    build_dataset_identity,
    config_hash,
    environment_info,
    file_sha256,
    optimizer_steps_per_epoch,
    protocol_bundle_hash,
    source_splits_for_config,
    source_tree_hash,
    write_split_manifest,
)
from crackmeanflow.common.training_protocol import training_split_view  # noqa: E402
from crackmeanflow.common.v3_provenance import verify_v3_provenance  # noqa: E402
from crackmeanflow.journal.engine.dataset import GeometryDataset  # noqa: E402
from scripts.train_generalization_screen_v3 import build_research_scheduler  # noqa: E402
from scripts.train_journal import (  # noqa: E402
    _group_fn,
    _set_loader_epoch,
    _train_loader,
    _with_optional_normals,
    build_track,
    seed_all,
)


def validate_partition(
    micro_batch_size: int,
    grad_accumulation_steps: int,
    effective_batch: int = 8,
) -> dict[str, int]:
    """Validate the only allowed V4 training override."""
    batch = int(micro_batch_size)
    accum = int(grad_accumulation_steps)
    expected = int(effective_batch)
    if batch < 1 or accum < 1:
        raise ValueError("micro batch and gradient accumulation must be positive")
    if batch * accum != expected:
        raise ValueError(f"effective batch must equal {expected}: got {batch}*{accum}")
    return {
        "micro_batch_size": batch,
        "grad_accum_steps": accum,
        "effective_batch_size": batch * accum,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V4 diagnostic micro-batch benchmark")
    parser.add_argument("--config", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--dataset-name", default="CFD")
    parser.add_argument("--dataset-version", default="CFD_FROZEN_HISTORICAL_V1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--micro-batch-size", type=int, required=True)
    parser.add_argument("--grad-accumulation-steps", type=int, required=True)
    parser.add_argument("--research-total-optimizer-steps", type=int, default=21000)
    parser.add_argument("--diagnostic-stop-optimizer-steps", type=int, default=200)
    parser.add_argument("--out", required=True)
    return parser


def build_benchmark_identity(
    *,
    config_path: str,
    config_sha256: str,
    protocol_path: str,
    protocol_sha256: str,
    source_tree_sha256: str,
    dataset_identity: dict,
    micro_batch_size: int,
    grad_accumulation_steps: int,
    research_total_optimizer_steps: int,
    diagnostic_stop_optimizer_steps: int,
    seed: int,
) -> dict:
    partition = validate_partition(micro_batch_size, grad_accumulation_steps)
    return {
        "schema": "CRACKMEANFLOW_FAST_MICROBATCH_BENCHMARK_IDENTITY_V1",
        "diagnostic_only": True,
        "research_metric_valid": False,
        "eligible_for_paper": False,
        "target_metrics_seen_before_lock": False,
        "config_path": str(config_path),
        "config_sha256": str(config_sha256),
        "protocol_path": str(protocol_path),
        "protocol_sha256": str(protocol_sha256),
        "source_tree_sha256": str(source_tree_sha256),
        "dataset_identity": dataset_identity,
        "execution_partition_override": {
            "micro_batch_size": partition["micro_batch_size"],
            "grad_accumulation_steps": partition["grad_accum_steps"],
            "effective_batch_size": partition["effective_batch_size"],
        },
        "research_scheduler_total_steps": int(research_total_optimizer_steps),
        "diagnostic_stop_optimizer_steps": int(diagnostic_stop_optimizer_steps),
        "seed": int(seed),
        "scientific_semantics": "canonical V3 unchanged; execution partition only",
    }


def _nvidia_sample() -> dict:
    query = "utilization.gpu,power.draw,temperature.gpu,memory.used,memory.total"
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=5,
        ).strip()
        rows = []
        for line in raw.splitlines():
            values = [x.strip() for x in line.split(",")]
            if len(values) != 5:
                continue
            rows.append(
                {
                    "gpu_utilization_percent": float(values[0]),
                    "power_draw_watts": float(values[1]),
                    "temperature_celsius": float(values[2]),
                    "memory_used_mib": float(values[3]),
                    "memory_total_mib": float(values[4]),
                }
            )
        return {"available": bool(rows), "gpus": rows}
    except Exception as exc:  # pragma: no cover - depends on workstation driver
        return {"available": False, "error": type(exc).__name__}


class _TelemetrySampler(threading.Thread):
    def __init__(self, interval_seconds: float = 1.0):
        super().__init__(daemon=True)
        self.interval_seconds = float(interval_seconds)
        self.stop_event = threading.Event()
        self.samples: list[dict] = []

    def run(self) -> None:  # pragma: no cover - timing/driver dependent
        while not self.stop_event.is_set():
            sample = _nvidia_sample()
            if sample.get("available"):
                self.samples.append({"time": time.time(), **sample})
            self.stop_event.wait(self.interval_seconds)

    def stop(self) -> None:
        self.stop_event.set()
        self.join(timeout=6)


def _training_inputs(cfg: dict, data_root: str, seed: int):
    splits = source_splits_for_config(str(Path(data_root).resolve()), cfg)
    training_splits = _with_optional_normals(str(Path(data_root).resolve()), training_split_view(splits), cfg)
    group_regex = cfg["train"].get("parent_group_regex")
    if group_regex:
        audit_group_integrity(splits, _group_fn(group_regex))
    leakage = audit_content_split_integrity(splits)
    identities = build_dataset_identity(splits, include_rows=False)

    def base(split: str, augment: bool):
        return PairedCrackDataset(
            training_splits[split],
            cfg["model"]["img_size"],
            augment,
            augment and bool(cfg["train"].get("photometric_augment", False)),
            cfg["train"].get("mask_resize_mode", "nearest"),
            cfg["train"].get("mask_binarization", "auto_binary_safe"),
        )

    if cfg["backbone"] == "geocrack_imf":
        train_ds = GeometryDataset(
            base("train", True),
            cfg["model"].get("max_radius", 16),
            cfg["model"].get("representation", "centerline_radius"),
            cfg["model"].get("distance_encoding", "linear"),
        )
    else:
        train_ds = base("train", True)
    loader = _train_loader(train_ds, training_splits["train"], cfg, seed)
    return splits, identities, leakage, train_ds, loader


def _loss_call(cfg, lossfn, model, batch, device, sample_offset: int):
    image = batch["crack"].to(device)
    if cfg["backbone"] == "geocrack_imf":
        return lossfn(
            model,
            batch["geometry"].to(device),
            image,
            batch["radius_valid"].to(device),
            mask_gt=batch["mask"].to(device),
            sample_offset=sample_offset,
        )
    if cfg["backbone"] in {"sit_imf_mask", "hybrid_imf_mask"}:
        return lossfn(model, batch["mask"].to(device), image, sample_offset=sample_offset)
    return lossfn(model, batch["mask"].to(device), {"y": image, "sample_offset": sample_offset})


def _finite(value: float) -> bool:
    return bool(np.isfinite(float(value)))


def _run(args: argparse.Namespace) -> dict:
    if int(args.research_total_optimizer_steps) != 21000:
        raise ValueError("research scheduler horizon must remain exactly 21000")
    if not 100 <= int(args.diagnostic_stop_optimizer_steps) <= 200:
        raise ValueError("diagnostic benchmark stop must be between 100 and 200 updates")
    partition = validate_partition(args.micro_batch_size, args.grad_accumulation_steps)
    cfg_path = Path(args.config).resolve()
    protocol_path = Path(args.protocol).resolve()
    preflight_path = Path(args.preflight).resolve()
    data_root = Path(args.data).resolve()
    if not cfg_path.is_file() or not protocol_path.is_file() or not preflight_path.is_file():
        raise FileNotFoundError("config, protocol, and PASS preflight must exist")
    cfg_base = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    verified = verify_v3_provenance(
        protocol_path=str(protocol_path),
        config_path=str(cfg_path),
        preflight_path=str(preflight_path),
        dataset_name=args.dataset_name,
        dataset_version=args.dataset_version,
        research_total_optimizer_steps=args.research_total_optimizer_steps,
        diagnostic_stop_optimizer_steps=args.diagnostic_stop_optimizer_steps,
    )
    cfg = copy.deepcopy(cfg_base)
    cfg["train"]["batch_size"] = int(args.micro_batch_size)
    cfg["train"]["grad_accum_steps"] = int(args.grad_accumulation_steps)
    seed_all(
        args.seed,
        bool(cfg["train"].get("deterministic", False)),
        bool(cfg["train"].get("deterministic_warn_only", False)),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("FAST benchmark requires a real CUDA device")
    if int(cfg.get("eval", {}).get("num_steps", 1)) != 1:
        raise RuntimeError("FAST benchmark requires NFE=1")
    if int(cfg["train"].get("max_optimizer_steps", -1)) != 21000:
        raise RuntimeError("FAST benchmark config must retain max_optimizer_steps=21000")
    if cfg["train"].get("resize_policy", "stretch_square") != "stretch_square":
        raise RuntimeError("FAST benchmark requires canonical stretch_square resize policy")

    splits, identities, leakage, train_ds, loader = _training_inputs(cfg, str(data_root), args.seed)
    verified = verify_v3_provenance(
        protocol_path=str(protocol_path),
        config_path=str(cfg_path),
        preflight_path=str(preflight_path),
        dataset_name=args.dataset_name,
        dataset_version=args.dataset_version,
        actual_dataset_identity=identities,
        research_total_optimizer_steps=args.research_total_optimizer_steps,
        diagnostic_stop_optimizer_steps=args.diagnostic_stop_optimizer_steps,
    )
    if len(loader) == 0:
        raise RuntimeError("training loader is empty")
    drop_incomplete = bool(cfg["train"].get("drop_incomplete_accumulation", True))
    steps_per_epoch = optimizer_steps_per_epoch(len(loader), int(args.grad_accumulation_steps), drop_incomplete)
    if steps_per_epoch != 825:
        raise RuntimeError(f"FAST partition changed optimizer steps/epoch: {steps_per_epoch}")

    model, _rasterizer, lossfn = build_track(cfg, device)
    initial_state = {
        key: value.detach().cpu().float().clone()
        for key, value in model.state_dict().items()
        if torch.is_tensor(value) and value.dtype.is_floating_point
    }
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )
    scheduler = build_research_scheduler(
        optimizer,
        epochs=int(cfg["train"]["epochs"]),
        optimizer_steps_per_epoch=steps_per_epoch,
        warmup_epochs=int(cfg["train"].get("warmup_epochs", 0)),
        research_total_optimizer_steps=args.research_total_optimizer_steps,
    )
    ema = EMA(model, cfg["train"]["ema_decay"])
    out = Path(args.out).resolve()
    if out.exists():
        raise RuntimeError(f"benchmark artifact path already exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    telemetry = _TelemetrySampler()
    telemetry.start()
    timing = {
        "data_wait_seconds": 0.0,
        "h2d_seconds": 0.0,
        "forward_loss_seconds": 0.0,
        "backward_seconds": 0.0,
        "optimizer_seconds": 0.0,
        "ema_seconds": 0.0,
    }
    update_seconds: list[float] = []
    losses: list[float] = []
    grad_norms: list[float] = []
    branch_counts = {
        "fm_samples": 0,
        "boundary_samples": 0,
        "endpoint_samples": 0,
        "thin_samples": 0,
        "gic_active_samples": 0,
        "gic_active_batches": 0,
        "seen_samples": 0,
        "seen_micro_batches": 0,
    }
    global_sample_offset = 0
    previous_batch_end = time.perf_counter()
    run_started = time.perf_counter()
    last_epoch = 0
    last_batch = -1
    try:
        iterator = iter(loader)
        for optimizer_step in range(int(args.diagnostic_stop_optimizer_steps)):
            update_started = time.perf_counter()
            if hasattr(lossfn, "set_epoch"):
                lossfn.set_epoch(last_epoch)
            optimizer.zero_grad(set_to_none=True)
            update_losses = []
            for micro_index in range(int(args.grad_accumulation_steps)):
                try:
                    batch = next(iterator)
                except StopIteration:
                    last_epoch += 1
                    _set_loader_epoch(loader, last_epoch, args.seed)
                    iterator = iter(loader)
                    batch = next(iterator)
                now = time.perf_counter()
                timing["data_wait_seconds"] += now - previous_batch_end
                last_batch = micro_index
                h2d_started = time.perf_counter()
                image = batch["crack"].to(device)
                timing["h2d_seconds"] += time.perf_counter() - h2d_started
                batch_size = int(image.shape[0])
                branch_counts["seen_samples"] += batch_size
                branch_counts["seen_micro_batches"] += 1
                forward_started = time.perf_counter()
                loss, logs = _loss_call(cfg, lossfn, model, batch, device, global_sample_offset)
                timing["forward_loss_seconds"] += time.perf_counter() - forward_started
                if not bool(torch.isfinite(loss).all()):
                    raise FloatingPointError(f"non-finite loss at optimizer step {optimizer_step + 1}")
                global_sample_offset += batch_size
                branch_counts["fm_samples"] += int(logs.get("fm_count", 0))
                branch_counts["boundary_samples"] += int(logs.get("boundary_count", logs.get("near_deployment_count", 0)))
                branch_counts["endpoint_samples"] += int(logs.get("exact_deployment_count", 0))
                branch_counts["gic_active_samples"] += int(logs.get("gic_active_samples", 0))
                branch_counts["gic_active_batches"] += int(logs.get("gic_active_batches", 0))
                backward_started = time.perf_counter()
                (loss / int(args.grad_accumulation_steps)).backward()
                timing["backward_seconds"] += time.perf_counter() - backward_started
                update_losses.append(float(logs["total_loss"]))
                previous_batch_end = time.perf_counter()
            optimizer_started = time.perf_counter()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg["train"]["max_grad_norm"], error_if_nonfinite=True
            )
            grad_norm_value = float(grad_norm.detach().cpu())
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            timing["optimizer_seconds"] += time.perf_counter() - optimizer_started
            ema_started = time.perf_counter()
            ema.update(model)
            timing["ema_seconds"] += time.perf_counter() - ema_started
            elapsed = time.perf_counter() - update_started
            update_seconds.append(float(elapsed))
            losses.append(float(np.mean(update_losses)))
            grad_norms.append(grad_norm_value)
            if not _finite(grad_norm_value):
                raise FloatingPointError(f"non-finite gradient norm at optimizer step {optimizer_step + 1}")
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            torch.cuda.empty_cache()
            raise RuntimeError(f"CUDA_OOM: {exc}") from exc
        raise
    finally:
        telemetry.stop()

    pure_training_seconds = float(sum(timing.values()))
    total_seconds = float(time.perf_counter() - run_started)
    warm_updates = update_seconds[:30]
    steady_updates = update_seconds[30:]
    allocated = int(torch.cuda.max_memory_allocated())
    reserved = int(torch.cuda.max_memory_reserved())
    total_device = int(torch.cuda.get_device_properties(device).total_memory)
    parameter_delta = 0.0
    for name, value in model.state_dict().items():
        if name in initial_state and torch.is_tensor(value) and value.dtype.is_floating_point:
            parameter_delta += float((value.detach().cpu().float() - initial_state[name]).pow(2).sum())
    parameter_delta = float(parameter_delta**0.5)
    cfg_sha = config_hash(cfg_base)
    identity = build_benchmark_identity(
        config_path=str(cfg_path),
        config_sha256=cfg_sha,
        protocol_path=str(protocol_path),
        protocol_sha256=file_sha256(protocol_path),
        source_tree_sha256=source_tree_hash(),
        dataset_identity=identities,
        micro_batch_size=args.micro_batch_size,
        grad_accumulation_steps=args.grad_accumulation_steps,
        research_total_optimizer_steps=args.research_total_optimizer_steps,
        diagnostic_stop_optimizer_steps=args.diagnostic_stop_optimizer_steps,
        seed=args.seed,
    )
    identity["arm"] = cfg_base.get("name", cfg_base.get("experiment", cfg_base.get("backbone")))
    identity["backbone"] = cfg_base.get("backbone")
    identity["verified_v3_provenance"] = {
        key: value
        for key, value in verified.items()
        if key not in {"protocol", "config", "preflight", "dataset_identity"}
    }
    identity["environment"] = environment_info()
    identity["content_leakage_audit"] = leakage
    identity["steps_per_epoch"] = int(steps_per_epoch)
    identity["warmup_optimizer_steps"] = int(cfg["train"].get("warmup_epochs", 0)) * int(steps_per_epoch)
    identity["protocol_bundle_sha256"] = protocol_bundle_hash()
    report = {
        **identity,
        "status": "PASS",
        "completed_optimizer_steps": len(update_seconds),
        "finite_loss": all(_finite(value) for value in losses),
        "finite_gradient_norm": all(_finite(value) for value in grad_norms),
        "parameter_delta_l2": parameter_delta,
        "optimizer_executed": len(update_seconds) == int(args.diagnostic_stop_optimizer_steps),
        "scheduler_executed": int(scheduler.last_epoch) == len(update_seconds),
        "ema_executed": len(update_seconds) > 0,
        "nan_detected": False,
        "inf_detected": False,
        "oom_detected": False,
        "loss_summary": {
            "mean": float(np.mean(losses)),
            "first": float(losses[0]),
            "last": float(losses[-1]),
            "slope_last_minus_first": float(losses[-1] - losses[0]),
            "min": float(np.min(losses)),
            "max": float(np.max(losses)),
        },
        "gradient_norm_summary": {
            "mean": float(np.mean(grad_norms)),
            "min": float(np.min(grad_norms)),
            "max": float(np.max(grad_norms)),
        },
        "branch_counts": branch_counts,
        "timing": {
            **timing,
            "pure_training_seconds": pure_training_seconds,
            "total_seconds": total_seconds,
            "warmup_updates": 30,
            "warmup_seconds": float(sum(warm_updates)),
            "steady_state_updates": len(steady_updates),
            "steady_state_seconds": float(sum(steady_updates)),
            "seconds_per_optimizer_step": pure_training_seconds / len(update_seconds),
            "steady_state_seconds_per_optimizer_step": float(np.mean(steady_updates)) if steady_updates else None,
            "samples_per_second": branch_counts["seen_samples"] / pure_training_seconds,
            "update_seconds": update_seconds,
        },
        "gpu_memory": {
            "peak_allocated_bytes": allocated,
            "peak_reserved_bytes": reserved,
            "total_device_bytes": total_device,
            "allocated_fraction": allocated / total_device,
            "reserved_fraction": reserved / total_device,
        },
        "gpu_telemetry": {
            "samples": telemetry.samples,
            "max_utilization_percent": max(
                (item["gpu_utilization_percent"] for sample in telemetry.samples for item in sample["gpus"]),
                default=None,
            ),
            "mean_utilization_percent": (
                float(np.mean([item["gpu_utilization_percent"] for sample in telemetry.samples for item in sample["gpus"]]))
                if telemetry.samples
                else None
            ),
            "max_power_draw_watts": max(
                (item["power_draw_watts"] for sample in telemetry.samples for item in sample["gpus"]),
                default=None,
            ),
            "max_temperature_celsius": max(
                (item["temperature_celsius"] for sample in telemetry.samples for item in sample["gpus"]),
                default=None,
            ),
        },
    }
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    args = build_parser().parse_args()
    out = Path(args.out).resolve()
    try:
        report = _run(args)
    except Exception as exc:
        out.parent.mkdir(parents=True, exist_ok=True)
        failure = {
            "schema": "CRACKMEANFLOW_FAST_MICROBATCH_BENCHMARK_V1",
            "status": "FAIL",
            "diagnostic_only": True,
            "research_metric_valid": False,
            "eligible_for_paper": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "micro_batch_size": int(args.micro_batch_size),
            "grad_accumulation_steps": int(args.grad_accumulation_steps),
            "research_scheduler_total_steps": int(args.research_total_optimizer_steps),
            "diagnostic_stop_optimizer_steps": int(args.diagnostic_stop_optimizer_steps),
        }
        out.write_text(json.dumps(failure, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(failure, sort_keys=True), flush=True)
        raise
    print(
        json.dumps(
            {
                "status": report["status"],
                "out": str(out),
                "seconds_per_optimizer_step": report["timing"]["seconds_per_optimizer_step"],
                "steady_state_seconds_per_optimizer_step": report["timing"]["steady_state_seconds_per_optimizer_step"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
