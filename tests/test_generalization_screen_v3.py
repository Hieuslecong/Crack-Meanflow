import math
import os
import subprocess
import sys

import torch

from scripts.train_generalization_screen_v3 import (
    _state_hash,
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


def test_screen_spec_rejects_stop_at_research_horizon():
    import pytest

    with pytest.raises(ValueError, match="shorter than research horizon"):
        validate_screen_spec(21000, 21000, (21000,), 825)


def test_state_hash_is_mapping_order_independent_and_nested():
    left = {"z": {"b": torch.tensor([2]), "a": [1, 2]}, "a": 3.0}
    right = {"a": 3.0, "z": {"a": [1, 2], "b": torch.tensor([2])}}
    assert _state_hash(left) == _state_hash(right)


def test_state_hash_is_stable_across_process_hash_seeds():
    code = (
        "import torch; from scripts.train_generalization_screen_v3 import _state_hash; "
        "print(_state_hash({'state': {2: {'step': 7, 'tensor': torch.tensor([1., 2.])}, "
        "1: {'values': (3, 4)}}}))"
    )
    outputs = []
    for seed in ("1", "98765"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        outputs.append(subprocess.check_output([sys.executable, "-c", code], text=True, env=env).strip())
    assert outputs[0] == outputs[1]
