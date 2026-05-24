import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_supervised_baseline.py"

spec = importlib.util.spec_from_file_location("train_supervised_baseline", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_supervised_threshold_grid_matches_required_contract():
    assert mod.DEFAULT_THRESHOLDS == [
        -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
    ]


def test_global_micro_aggregation_uses_counts_not_per_image_mean():
    metrics = [
        {"tp": 1.0, "fp": 0.0, "fn": 0.0, "tn": 9.0, "iou": 1.0, "dice": 1.0, "f1": 1.0, "precision": 1.0, "recall": 1.0, "accuracy": 1.0},
        {"tp": 0.0, "fp": 9.0, "fn": 1.0, "tn": 0.0, "iou": 0.0, "dice": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0},
    ]
    out = mod.aggregate_metrics(metrics)
    assert out["tp"] == 1.0
    assert out["fp"] == 9.0
    assert out["fn"] == 1.0
    assert abs(out["precision"] - 0.1) < 1e-6
    assert abs(out["recall"] - 0.5) < 1e-6
    assert abs(out["f1"] - (2 * 0.1 * 0.5 / (0.1 + 0.5))) < 1e-6


def test_make_loss_supports_requested_teacher_variants():
    logits = torch.tensor([[[[0.0, 1.0], [-1.0, 2.0]]]])
    targets = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
    for name in ["bce_dice", "bce_dice_tversky", "focal_bce_dice", "bce_dice_thin"]:
        loss_fn = mod.make_loss({"loss": name, "tversky_alpha": 0.2, "tversky_beta": 0.8, "thin_weight": 0.5})
        loss = loss_fn(logits, targets)
        assert torch.isfinite(loss), name
        assert loss.item() > 0, name


if __name__ == "__main__":
    test_supervised_threshold_grid_matches_required_contract()
    test_global_micro_aggregation_uses_counts_not_per_image_mean()
    test_make_loss_supports_requested_teacher_variants()
    print("SUPERVISED_TEACHER_CONTRACT_TESTS_PASS")
