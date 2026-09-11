import math

import torch

from scripts.train_generalization_screen_v3 import (
    build_research_scheduler,
    validate_screen_spec,
)


def test_screen_scheduler_keeps_canonical_research_horizon():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    scheduler = build_research_scheduler(
        optimizer,
        epochs=100,
        optimizer_steps_per_epoch=825,
        warmup_epochs=10,
        research_total_optimizer_steps=21000,
    )

    assert scheduler._cmf_total_steps == 21000
    assert scheduler._cmf_warmup_steps == 8250

    expected = 0.5 * (1.0 + math.cos(math.pi * ((16500 - 8250) / (21000 - 8250))))
    assert math.isclose(scheduler.lr_lambdas[0](16500), expected, rel_tol=0.0, abs_tol=1e-12)


def test_screen_spec_requires_epoch_aligned_snapshots_and_shorter_stop():
    assert validate_screen_spec(
        research_total_optimizer_steps=21000,
        diagnostic_stop_optimizer_steps=16500,
        snapshot_steps=(12375, 16500),
        optimizer_steps_per_epoch=825,
    ) == {
        "research_scheduler_total_steps": 21000,
        "diagnostic_stop_optimizer_steps": 16500,
        "snapshot_steps": [12375, 16500],
    }
