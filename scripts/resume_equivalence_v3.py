from __future__ import annotations

"""Fast diagnostic 10+10 vs 20 state-equivalence test.

The canonical research trainer intentionally prohibits partial-epoch scientific
resume. This diagnostic therefore tests the checkpoint/RNG/optimizer/scheduler/
EMA machinery itself on a deterministic toy optimization trajectory. It does
not authorize partial-epoch paper training.
"""

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from crackmeanflow.common import EMA, make_warmup_cosine_scheduler
from crackmeanflow.common.checkpointing import capture_rng_state, restore_rng_state


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _nested_equal(a, b):
    if torch.is_tensor(a) and torch.is_tensor(b):
        return torch.equal(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_nested_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, type(a)) and len(a) == len(b):
        return all(_nested_equal(x, y) for x, y in zip(a, b))
    return a == b


def _make(seed: int):
    seed_all(seed)
    model = nn.Sequential(nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, 4))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = make_warmup_cosine_scheduler(opt, epochs=1, optimizer_steps_epoch=20, warmup_epochs=0, total_optimizer_steps=20)
    ema = EMA(model, 0.99)
    return model, opt, sched, ema


def _step(model, opt, sched, ema):
    x = torch.randn(6, 8)
    y = torch.randn(6, 4)
    opt.zero_grad(set_to_none=True)
    pred = model(x)
    loss = (pred - y).square().mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    opt.step()
    sched.step()
    ema.update(model)
    return float(loss.detach())


def _run_continuous(seed: int):
    model, opt, sched, ema = _make(seed)
    losses = [_step(model, opt, sched, ema) for _ in range(20)]
    return model, opt, sched, ema, losses


def _run_split(seed: int):
    model, opt, sched, ema = _make(seed)
    losses = [_step(model, opt, sched, ema) for _ in range(10)]
    checkpoint = {
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(opt.state_dict()),
        "scheduler": copy.deepcopy(sched.state_dict()),
        "ema": copy.deepcopy(ema.shadow),
        "rng": capture_rng_state(),
    }

    # Reconstruct objects exactly as a resume would, then load all mutable state.
    resumed_model, resumed_opt, resumed_sched, _ = _make(seed + 999)
    resumed_model.load_state_dict(checkpoint["model"])
    resumed_opt.load_state_dict(checkpoint["optimizer"])
    resumed_sched.load_state_dict(checkpoint["scheduler"])
    resumed_ema = EMA(resumed_model, 0.99, checkpoint["ema"])
    restore_rng_state(checkpoint["rng"])
    losses.extend(_step(resumed_model, resumed_opt, resumed_sched, resumed_ema) for _ in range(10))
    return resumed_model, resumed_opt, resumed_sched, resumed_ema, losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="reports/RESUME_EQUIVALENCE_V3.json")
    args = ap.parse_args()

    a_model, a_opt, a_sched, a_ema, a_losses = _run_continuous(args.seed)
    b_model, b_opt, b_sched, b_ema, b_losses = _run_split(args.seed)

    checks = {
        "model_exact": _nested_equal(a_model.state_dict(), b_model.state_dict()),
        "ema_exact": _nested_equal(a_ema.shadow, b_ema.shadow),
        "optimizer_exact": _nested_equal(a_opt.state_dict(), b_opt.state_dict()),
        "scheduler_exact": _nested_equal(a_sched.state_dict(), b_sched.state_dict()),
        "loss_trajectory_exact": a_losses == b_losses,
    }
    report = {
        "schema": "CRACKMEANFLOW_RESUME_EQUIVALENCE_V3",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "diagnostic_only": True,
        "canonical_partial_epoch_resume_supported": False,
        "paper_resume_policy": "epoch_boundary_only",
        "trajectory": "20_continuous_vs_10_checkpoint_restore_10",
        "checks": checks,
        "continuous_losses": a_losses,
        "split_losses": b_losses,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if report["status"] != "PASS":
        raise RuntimeError(f"resume equivalence failed: {checks}")
    print(json.dumps({"status": "PASS", "out": str(out)}, sort_keys=True))


if __name__ == "__main__":
    main()
