from __future__ import annotations

"""Diagnostic source-only runner for the V3 pre-full generalization screen.

This runner deliberately keeps the canonical 21,000-step scheduler horizon while
stopping at an epoch-aligned diagnostic budget.  It reuses the paper training
components and loop helpers; it does not alter a config, model, loss, or data
split and never constructs a target dataset.
"""

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

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
    make_warmup_cosine_scheduler,
    optimizer_steps_per_epoch,
    protocol_bundle_hash,
    save_checkpoint_atomic,
    source_splits_for_config,
    source_tree_hash,
    write_immutable_json,
    write_split_manifest,
)
from crackmeanflow.common.training_protocol import training_split_view  # noqa: E402
from crackmeanflow.common.v3_provenance import paper_v3_worktree_blockers, verify_v3_provenance  # noqa: E402
from crackmeanflow.journal.engine.dataset import GeometryDataset  # noqa: E402
from scripts.train_journal import (  # noqa: E402
    _group_fn,
    _select_checkpoint_metric,
    _set_loader_epoch,
    _train_loader,
    _with_optional_normals,
    build_track,
    seed_all,
)


def build_research_scheduler(
    optimizer,
    *,
    epochs: int,
    optimizer_steps_per_epoch: int,
    warmup_epochs: int,
    research_total_optimizer_steps: int,
):
    """Build the scheduler on the research horizon, never on diagnostic stop."""
    return make_warmup_cosine_scheduler(
        optimizer,
        epochs,
        optimizer_steps_per_epoch,
        warmup_epochs,
        total_optimizer_steps=research_total_optimizer_steps,
    )


def validate_screen_spec(
    research_total_optimizer_steps: int,
    diagnostic_stop_optimizer_steps: int,
    snapshot_steps: tuple[int, ...] | list[int],
    optimizer_steps_per_epoch: int,
) -> dict:
    """Validate a fixed-step, complete-epoch diagnostic screen specification."""
    total = int(research_total_optimizer_steps)
    stop = int(diagnostic_stop_optimizer_steps)
    steps = tuple(int(x) for x in snapshot_steps)
    cadence = int(optimizer_steps_per_epoch)
    if total < 1:
        raise ValueError("research_total_optimizer_steps must be positive")
    if stop < 1 or stop >= total:
        raise ValueError("diagnostic stop must be positive and shorter than research horizon")
    if cadence < 1:
        raise ValueError("optimizer_steps_epoch must be positive")
    if not steps or len(set(steps)) != len(steps) or tuple(sorted(steps)) != steps:
        raise ValueError("snapshot steps must be non-empty, unique, and sorted")
    if any(x < 1 or x > stop for x in steps):
        raise ValueError("snapshot steps must lie within the diagnostic budget")
    if stop % cadence or any(x % cadence for x in steps):
        raise ValueError("diagnostic stop and snapshots must be complete-epoch aligned")
    if steps[-1] != stop:
        raise ValueError("the final snapshot must equal the diagnostic stop")
    return {
        "research_scheduler_total_steps": total,
        "diagnostic_stop_optimizer_steps": stop,
        "snapshot_steps": list(steps),
    }


def _state_hash(state) -> str:
    digest = hashlib.sha256()

    def update(value) -> None:
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
            digest.update(b"\0")
            digest.update(tensor.numpy().tobytes())
        elif isinstance(value, dict):
            digest.update(b"dict\0")
            for key in sorted(value, key=lambda item: str(item)):
                update(str(key))
                update(value[key])
        elif isinstance(value, (list, tuple)):
            digest.update(b"list\0" if isinstance(value, list) else b"tuple\0")
            for item in value:
                update(item)
        else:
            digest.update(type(value).__name__.encode("ascii"))
            digest.update(b"\0")
            digest.update(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"))
        digest.update(b"\n")

    update(state)
    return digest.hexdigest()


def _parameter_delta(initial: dict[str, torch.Tensor], model: torch.nn.Module) -> float:
    total = 0.0
    for name, value in model.state_dict().items():
        if name not in initial or not torch.is_tensor(value) or not value.dtype.is_floating_point:
            continue
        current = value.detach().cpu().float()
        total += float((current - initial[name]).pow(2).sum())
    return float(total**0.5)


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.link(src, dst)


def _write_snapshot_completion(
    *,
    snapshot: Path,
    step: int,
    best_metric: float,
    best_threshold: float,
    cfg_hash: str,
    source_hash: str,
    protocol_hash: str,
    runtime_seconds: float,
) -> Path:
    """Make a strict evaluator-compatible completion view for one snapshot."""
    view = snapshot.parent / "snapshots" / f"step_{step:05d}"
    view.mkdir(parents=True, exist_ok=True)
    best = view / "best.pt"
    final = view / "last.pt"
    _link(snapshot, best)
    _link(snapshot, final)
    record = {
        "schema": "CRACKMEANFLOW_RUN_COMPLETION_V1",
        "status": "PASS",
        "best_checkpoint": best.name,
        "best_checkpoint_sha256": file_sha256(best),
        "final_checkpoint": final.name,
        "final_checkpoint_sha256": file_sha256(final),
        "planned_optimizer_steps": int(step),
        "completed_optimizer_steps": int(step),
        "best_optimizer_step": int(step),
        "best_val_metric": float(best_metric),
        "best_val_threshold": float(best_threshold),
        "config_hash": cfg_hash,
        "source_tree_sha256": source_hash,
        "protocol_bundle_sha256": protocol_hash,
        "training_runtime_seconds": float(runtime_seconds),
        "required_artifacts": ["best.pt", "last.pt"],
        "diagnostic_only": True,
        "research_metric_valid": False,
        "eligible_for_paper": False,
    }
    write_immutable_json(view / "RUN_COMPLETE.json", record)
    return view / "RUN_COMPLETE.json"


def _parse_snapshot_steps(raw: str) -> tuple[int, ...]:
    try:
        return tuple(int(x.strip()) for x in str(raw).split(",") if x.strip())
    except ValueError as exc:
        raise ValueError("--snapshot-steps must be comma-separated integers") from exc


def main() -> None:
    ap = argparse.ArgumentParser(description="V3 source-only fixed-step generalization screen runner")
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--dataset-name", default="CFD")
    ap.add_argument("--dataset-version", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--research-total-optimizer-steps", type=int, default=21000)
    ap.add_argument("--diagnostic-stop-optimizer-steps", type=int, default=16500)
    ap.add_argument("--snapshot-steps", default="12375,16500")
    ap.add_argument("--protocol", default="configs/protocol/post_repair_protocol_v3.yaml")
    ap.add_argument("--preflight", default="reports/PROTOCOL_PREFLIGHT_V3.json")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    out = Path(args.out).resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"output directory must be empty/new: {out}")
    blockers = paper_v3_worktree_blockers()
    if blockers:
        raise RuntimeError(f"paper-v3 worktree is not provenance-clean: {blockers}")

    verified_provenance = verify_v3_provenance(
        protocol_path=args.protocol,
        config_path=args.config,
        preflight_path=args.preflight,
        dataset_name=args.dataset_name,
        dataset_version=args.dataset_version,
        research_total_optimizer_steps=args.research_total_optimizer_steps,
        diagnostic_stop_optimizer_steps=args.diagnostic_stop_optimizer_steps,
    )

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not str(args.dataset_name).strip() or not str(args.dataset_version).strip():
        raise ValueError("dataset name/version must be non-empty")
    if int(cfg.get("eval", {}).get("num_steps", 1)) != 1:
        raise RuntimeError("screen requires eval.num_steps=1")
    if cfg.get("train", {}).get("max_optimizer_steps") != int(args.research_total_optimizer_steps):
        raise RuntimeError("config research max_optimizer_steps must remain the canonical 21000")
    if cfg.get("train", {}).get("resize_policy", "stretch_square") != "stretch_square":
        raise RuntimeError("screen requires the canonical stretch_square resize policy")

    seed_all(args.seed, bool(cfg["train"].get("deterministic", False)), bool(cfg["train"].get("deterministic_warn_only", False)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("V3 generalization screen requires a real CUDA device")

    sp = source_splits_for_config(str(Path(args.data).resolve()), cfg)
    training_sp = _with_optional_normals(str(Path(args.data).resolve()), training_split_view(sp), cfg)
    group_regex = cfg["train"].get("parent_group_regex")
    if group_regex:
        audit_group_integrity(sp, _group_fn(group_regex))
    content_leakage = audit_content_split_integrity(sp)
    identities = build_dataset_identity(sp, include_rows=False)
    verify_v3_provenance(
        protocol_path=args.protocol,
        config_path=args.config,
        preflight_path=args.preflight,
        dataset_name=args.dataset_name,
        dataset_version=args.dataset_version,
        actual_dataset_identity=identities,
        research_total_optimizer_steps=args.research_total_optimizer_steps,
        diagnostic_stop_optimizer_steps=args.diagnostic_stop_optimizer_steps,
    )

    base = lambda split, aug: PairedCrackDataset(
        training_sp[split],
        cfg["model"]["img_size"],
        aug,
        aug and bool(cfg["train"].get("photometric_augment", False)),
        cfg["train"].get("mask_resize_mode", "nearest"),
        cfg["train"].get("mask_binarization", "auto_binary_safe"),
    )
    if cfg["backbone"] == "geocrack_imf":
        train_ds = GeometryDataset(base("train", True), cfg["model"].get("max_radius", 16), cfg["model"].get("representation", "centerline_radius"), cfg["model"].get("distance_encoding", "linear"))
        val_ds = GeometryDataset(base("val", False), cfg["model"].get("max_radius", 16), cfg["model"].get("representation", "centerline_radius"), cfg["model"].get("distance_encoding", "linear"))
    else:
        train_ds = base("train", True)
        val_ds = base("val", False)
    loader = _train_loader(train_ds, training_sp["train"], cfg, args.seed)
    val_loader = DataLoader(val_ds, batch_size=int(cfg.get("eval", {}).get("batch_size", 1)), shuffle=False, num_workers=int(cfg["train"].get("num_workers", 0)))
    if len(loader) == 0:
        raise RuntimeError("training loader is empty")
    drop_incomplete = bool(cfg["train"].get("drop_incomplete_accumulation", True))
    steps = optimizer_steps_per_epoch(len(loader), int(cfg["train"]["grad_accum_steps"]), drop_incomplete)
    spec = validate_screen_spec(args.research_total_optimizer_steps, args.diagnostic_stop_optimizer_steps, _parse_snapshot_steps(args.snapshot_steps), optimizer_steps_per_epoch=steps)

    model, rasterizer, lossfn = build_track(cfg, device)
    initial_parameters = {k: v.detach().cpu().float().clone() for k, v in model.state_dict().items() if torch.is_tensor(v) and v.dtype.is_floating_point}
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scheduler = build_research_scheduler(
        optimizer,
        epochs=int(cfg["train"]["epochs"]),
        optimizer_steps_per_epoch=steps,
        warmup_epochs=int(cfg["train"].get("warmup_epochs", 0)),
        research_total_optimizer_steps=args.research_total_optimizer_steps,
    )
    ema = EMA(model, cfg["train"]["ema_decay"])
    cfg_sha = config_hash(cfg)
    source_sha = source_tree_hash()
    protocol_sha = protocol_bundle_hash()
    fairness = {
        "samples_train": len(train_ds),
        "batch_size": int(cfg["train"]["batch_size"]),
        "grad_accum_steps": int(cfg["train"]["grad_accum_steps"]),
        "effective_batch_size_nominal": int(cfg["train"]["batch_size"]) * int(cfg["train"]["grad_accum_steps"]),
        "optimizer_steps_per_epoch": int(steps),
        "research_scheduler_total_steps": int(args.research_total_optimizer_steps),
        "diagnostic_stop_optimizer_steps": int(args.diagnostic_stop_optimizer_steps),
        "snapshot_steps": list(spec["snapshot_steps"]),
        "warmup_optimizer_steps": int(cfg["train"].get("warmup_epochs", 0)) * int(steps),
        "nfe": 1,
        "drop_incomplete_accumulation": drop_incomplete,
        "loader_samples_per_epoch": int(len(loader) * int(cfg["train"]["batch_size"])),
        "usable_samples_per_epoch": int(steps) * int(cfg["train"]["batch_size"]) * int(cfg["train"]["grad_accum_steps"]),
        "total_omitted_samples_per_epoch": int(max(0, len(train_ds) - int(steps) * int(cfg["train"]["batch_size"]) * int(cfg["train"]["grad_accum_steps"]))),
        "budget_protocol": "DIAGNOSTIC_FIXED_EPOCH_SNAPSHOTS_WITH_CANONICAL_21000_STEP_SCHEDULER",
    }
    source_provenance = {
        "dataset_name": args.dataset_name,
        "dataset_version": args.dataset_version,
        "splits": identities,
        "parent_group_audit": "PASS" if group_regex else "UNVERIFIED",
        "group_regex": group_regex,
        "content_leakage_audit": content_leakage,
    }

    out.mkdir(parents=True, exist_ok=True)
    (out / "EFFECTIVE_CONFIG.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    manifest = write_split_manifest(sp, out / "dataset_manifest.json", include_content_rows=True)
    env = environment_info()
    (out / "ENVIRONMENT.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    identity = {
        "schema": "CRACKMEANFLOW_GENERALIZATION_SCREEN_RUN_IDENTITY_V3",
        "diagnostic_only": True,
        "research_metric_valid": False,
        "eligible_for_paper": False,
        "target_metrics_seen_before_lock": False,
        "target_access": "PROHIBITED_DURING_SOURCE_TRAINING",
        "config_path": str(cfg_path),
        "config_sha256": cfg_sha,
        "source_tree_sha256": source_sha,
        "protocol_bundle_sha256": protocol_sha,
        "dataset_identity": identities,
        "source_provenance": source_provenance,
        "fairness": fairness,
        "screen_spec": spec,
        "seed": int(args.seed),
        "environment": env,
        "preflight": args.preflight,
        "verified_v3_provenance": {k: v for k, v in verified_provenance.items() if k not in {"protocol", "config", "preflight"}},
    }
    (out / "RUN_IDENTITY.json").write_text(json.dumps(identity, indent=2, sort_keys=True), encoding="utf-8")

    history = []
    global_step = 0
    global_sample_offset = 0
    best_val = -1.0
    best_threshold = None
    best_optimizer_step = None
    best_validation_checkpoint = None
    snapshot_metadata = []
    timing = {"data_wait_seconds": 0.0, "h2d_seconds": 0.0, "forward_loss_seconds": 0.0, "backward_seconds": 0.0, "optimizer_seconds": 0.0, "ema_seconds": 0.0, "validation_seconds": 0.0, "checkpoint_seconds": 0.0}
    run_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    for epoch in range(int(cfg["train"]["epochs"])):
        if global_step >= args.diagnostic_stop_optimizer_steps:
            break
        if hasattr(lossfn, "set_epoch"):
            lossfn.set_epoch(epoch)
        _set_loader_epoch(loader, epoch, args.seed)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        last_batch = -1
        previous_batch_end = time.perf_counter()
        epoch_started = time.perf_counter()
        fm_samples = gic_samples = gic_batches = boundary_samples = seen_samples = seen_batches = 0
        grad_norm_value = float("nan")
        stop_budget = False
        for batch_index, batch in enumerate(loader):
            if drop_incomplete and batch_index >= steps * int(cfg["train"]["grad_accum_steps"]):
                break
            now = time.perf_counter()
            timing["data_wait_seconds"] += now - previous_batch_end
            last_batch = batch_index
            image = batch["crack"].to(device)
            timing["h2d_seconds"] += time.perf_counter() - now
            bs = int(image.shape[0])
            seen_samples += bs
            seen_batches += 1
            forward_started = time.perf_counter()
            if cfg["backbone"] == "geocrack_imf":
                loss, logs = lossfn(model, batch["geometry"].to(device), image, batch["radius_valid"].to(device), mask_gt=batch["mask"].to(device), sample_offset=global_sample_offset)
            elif cfg["backbone"] in {"sit_imf_mask", "hybrid_imf_mask"}:
                loss, logs = lossfn(model, batch["mask"].to(device), image, sample_offset=global_sample_offset)
            else:
                loss, logs = lossfn(model, batch["mask"].to(device), {"y": image, "sample_offset": global_sample_offset})
            timing["forward_loss_seconds"] += time.perf_counter() - forward_started
            if not bool(torch.isfinite(loss).all()):
                raise FloatingPointError(f"non-finite training loss at epoch={epoch} batch={batch_index} step={global_step}")
            global_sample_offset += bs
            fm_samples += int(logs.get("fm_count", 0))
            gic_samples += int(logs.get("gic_active_samples", 0))
            gic_batches += int(logs.get("gic_active_batches", 0))
            boundary_samples += int(logs.get("near_deployment_count", logs.get("boundary_count", 0)))
            ga = int(cfg["train"]["grad_accum_steps"])
            group_start = (batch_index // ga) * ga
            group_size = ga if drop_incomplete else min(ga, len(loader) - group_start)
            backward_started = time.perf_counter()
            (loss / group_size).backward()
            timing["backward_seconds"] += time.perf_counter() - backward_started
            losses.append(float(logs["total_loss"]))
            if (batch_index + 1) % ga == 0 or batch_index + 1 == len(loader):
                optimizer_started = time.perf_counter()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["max_grad_norm"], error_if_nonfinite=True)
                grad_norm_value = float(grad_norm.detach().cpu())
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                timing["optimizer_seconds"] += time.perf_counter() - optimizer_started
                ema_started = time.perf_counter()
                ema.update(model)
                timing["ema_seconds"] += time.perf_counter() - ema_started
                global_step += 1
                if global_step >= args.diagnostic_stop_optimizer_steps:
                    stop_budget = True
                    previous_batch_end = time.perf_counter()
                    break
            previous_batch_end = time.perf_counter()

        usable_batches = steps * int(cfg["train"]["grad_accum_steps"]) if drop_incomplete else len(loader)
        epoch_complete = (last_batch + 1) >= usable_batches
        if not epoch_complete:
            raise RuntimeError(f"screen stop created a partial epoch at epoch={epoch}, step={global_step}")
        validation = None
        threshold = None
        validation_seconds = 0.0
        interval = max(1, int(cfg.get("eval", {}).get("checkpoint_validation_interval_epochs", 1)))
        should_validate = ((epoch + 1) % interval == 0) or stop_budget
        if should_validate:
            validation_started = time.perf_counter()
            ema.apply(model)
            try:
                validation, threshold = _select_checkpoint_metric(cfg, model, val_loader, device, rasterizer)
            finally:
                ema.restore(model)
            validation_seconds = time.perf_counter() - validation_started
            timing["validation_seconds"] += validation_seconds
            if not np.isfinite(float(validation["f1"])) or not np.isfinite(float(threshold)):
                raise FloatingPointError(f"non-finite source validation at epoch={epoch}")
            improved = float(validation["f1"]) > best_val
            if improved:
                best_val = float(validation["f1"])
                best_threshold = float(threshold)
                best_optimizer_step = int(global_step)

        row = {
            "epoch": int(epoch),
            "optimizer_step": int(global_step),
            "loss": float(np.mean(losses)),
            "loss_finite": bool(np.isfinite(float(np.mean(losses)))),
            "grad_norm": float(grad_norm_value),
            "grad_finite": bool(np.isfinite(grad_norm_value)),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "validation_performed": bool(should_validate),
            "val_f1": None if validation is None else float(validation["f1"]),
            "val_threshold": None if threshold is None else float(threshold),
            "val_f1_per_inference_seed": None if validation is None else validation["per_seed_f1"],
            "fm_samples": int(fm_samples),
            "gic_active_samples": int(gic_samples),
            "gic_active_batches": int(gic_batches),
            "boundary_samples": int(boundary_samples),
            "seen_samples": int(seen_samples),
            "seen_batches": int(seen_batches),
            "epoch_seconds": float(time.perf_counter() - epoch_started),
            "validation_seconds": float(validation_seconds),
            "research_scheduler_total_steps": int(args.research_total_optimizer_steps),
            "diagnostic_only": True,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

        validation_checkpoint = None
        if should_validate:
            checkpoint_started = time.perf_counter()
            validation_checkpoint = out / f"checkpoint_validation_{global_step}.pt"
            validation_fairness = copy.deepcopy(fairness)
            validation_fairness["planned_optimizer_steps"] = int(args.diagnostic_stop_optimizer_steps)
            validation_fairness["max_optimizer_steps"] = int(args.diagnostic_stop_optimizer_steps)
            validation_extra = {
                "global_sample_offset": int(global_sample_offset),
                "fairness": validation_fairness,
                "epoch_complete": True,
                "budget_reached": bool(global_step == args.diagnostic_stop_optimizer_steps),
                "run_complete": bool(global_step == args.diagnostic_stop_optimizer_steps),
                "diagnostic_only": True,
                "research_metric_valid": False,
                "eligible_for_paper": False,
                "target_metrics_seen_before_lock": False,
                "research_scheduler_total_steps": int(args.research_total_optimizer_steps),
                "diagnostic_stop_optimizer_steps": int(args.diagnostic_stop_optimizer_steps),
            }
            save_checkpoint_atomic(
                str(validation_checkpoint),
                model=model,
                ema=ema,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_optimizer_step=global_step,
                cfg=cfg,
                best_val_metric=float(validation["f1"]),
                best_val_threshold=float(threshold),
                split_manifest_hashes={k: v["name_manifest_sha256"] for k, v in identities.items()},
                split_manifest_content_hashes={k: v["content_manifest_sha256"] for k, v in identities.items()},
                source_provenance=source_provenance,
                seed=args.seed,
                extra_state=validation_extra,
            )
            if improved:
                best_validation_checkpoint = validation_checkpoint
            timing["checkpoint_seconds"] += time.perf_counter() - checkpoint_started

        if global_step in spec["snapshot_steps"]:
            if validation_checkpoint is None or validation is None or threshold is None:
                raise RuntimeError("every fixed snapshot must coincide with source validation")
            checkpoint_started = time.perf_counter()
            snapshot = out / f"checkpoint_snapshot_{global_step}.pt"
            _link(validation_checkpoint, snapshot)
            # The checkpoint is immutable after creation; only hardlinks are made.
            completion_path = _write_snapshot_completion(
                snapshot=snapshot,
                step=global_step,
                best_metric=float(validation["f1"]),
                best_threshold=float(threshold),
                cfg_hash=cfg_sha,
                source_hash=source_sha,
                protocol_hash=protocol_sha,
                runtime_seconds=time.perf_counter() - run_started,
            )
            snapshot_hash = file_sha256(snapshot)
            snapshot_metadata.append({
                "step": int(global_step),
                "epoch": int(epoch + 1),
                "checkpoint": str(snapshot.relative_to(out)),
                "checkpoint_sha256": snapshot_hash,
                "completion_artifact": str(completion_path.relative_to(out)),
                "model_state_sha256": _state_hash(model.state_dict()),
                "ema_state_sha256": _state_hash(ema.shadow),
                "optimizer_state_sha256": _state_hash(optimizer.state_dict()),
                "scheduler_state_sha256": _state_hash(scheduler.state_dict()),
                "source_val_f1": float(validation["f1"]),
                "source_val_threshold": float(threshold),
                "research_scheduler_total_steps": int(args.research_total_optimizer_steps),
                "diagnostic_only": True,
                "research_metric_valid": False,
                "eligible_for_paper": False,
            })
            timing["checkpoint_seconds"] += time.perf_counter() - checkpoint_started
            (out / f"SNAPSHOT_{global_step}.json").write_text(json.dumps(snapshot_metadata[-1], indent=2, sort_keys=True), encoding="utf-8")
        with (out / "history.json").open("w", encoding="utf-8") as handle:
            json.dump({"history": history, "best_val_f1": best_val, "best_val_threshold": best_threshold, "diagnostic_only": True}, handle, indent=2)
        if stop_budget:
            break

    if global_step != int(args.diagnostic_stop_optimizer_steps):
        raise RuntimeError(f"screen ended incomplete: completed={global_step} expected={args.diagnostic_stop_optimizer_steps}")
    if [x["step"] for x in snapshot_metadata] != list(spec["snapshot_steps"]):
        raise RuntimeError("not all required fixed-step snapshots were produced")
    if best_validation_checkpoint is None or best_optimizer_step is None or best_threshold is None:
        raise RuntimeError("screen produced no validation-selected best checkpoint")
    final_snapshot = out / f"checkpoint_snapshot_{args.diagnostic_stop_optimizer_steps}.pt"
    _link(best_validation_checkpoint, out / "best.pt")
    _link(final_snapshot, out / "last.pt")
    total_runtime = time.perf_counter() - run_started
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        timing["gpu_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        timing["gpu_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
    timing["pure_training_seconds"] = float(sum(timing[key] for key in (
        "data_wait_seconds",
        "h2d_seconds",
        "forward_loss_seconds",
        "backward_seconds",
        "optimizer_seconds",
        "ema_seconds",
    )))
    timing["total_seconds"] = float(total_runtime)
    timing["seconds_per_optimizer_step"] = float(timing["pure_training_seconds"] / global_step)
    timing["samples_per_second"] = float(fairness["usable_samples_per_epoch"] * len(history) / max(timing["pure_training_seconds"], 1e-12))
    timing["parameter_delta_l2"] = _parameter_delta(initial_parameters, model)
    timing["optimizer_executed"] = True
    timing["scheduler_executed"] = True
    timing["ema_executed"] = True
    timing["checkpoint_roundtrip"] = "PASS"
    (out / "TIMING.json").write_text(json.dumps(timing, indent=2, sort_keys=True), encoding="utf-8")
    (out / "SNAPSHOTS.json").write_text(json.dumps({"snapshots": snapshot_metadata}, indent=2, sort_keys=True), encoding="utf-8")
    identity["completed_optimizer_steps"] = int(global_step)
    identity["model_parameter_count"] = int(sum(p.numel() for p in model.parameters()))
    identity["timing"] = timing
    (out / "RUN_IDENTITY.json").write_text(json.dumps(identity, indent=2, sort_keys=True), encoding="utf-8")
    completion = {
        "schema": "CRACKMEANFLOW_RUN_COMPLETION_V1",
        "status": "PASS",
        "best_checkpoint": "best.pt",
        "best_checkpoint_sha256": file_sha256(out / "best.pt"),
        "final_checkpoint": "last.pt",
        "final_checkpoint_sha256": file_sha256(out / "last.pt"),
        "planned_optimizer_steps": int(args.diagnostic_stop_optimizer_steps),
        "completed_optimizer_steps": int(global_step),
        "best_optimizer_step": int(best_optimizer_step),
        "best_val_metric": float(best_val),
        "best_val_threshold": float(best_threshold),
        "config_hash": cfg_sha,
        "source_tree_sha256": source_sha,
        "protocol_bundle_sha256": protocol_sha,
        "training_runtime_seconds": float(total_runtime),
        "required_artifacts": ["EFFECTIVE_CONFIG.yaml", "dataset_manifest.json", "RUN_IDENTITY.json", "ENVIRONMENT.json", "history.json", "best.pt", "last.pt", "TIMING.json", "SNAPSHOTS.json"],
        "diagnostic_only": True,
        "research_metric_valid": False,
        "eligible_for_paper": False,
    }
    write_immutable_json(out / "RUN_COMPLETE.json", completion)
    print(json.dumps({"status": "PASS", "diagnostic_only": True, "completed_optimizer_steps": global_step, "research_scheduler_total_steps": args.research_total_optimizer_steps, "out": str(out)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
