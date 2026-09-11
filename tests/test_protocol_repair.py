import pytest
import torch
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml
from PIL import Image


def test_resume_taint_is_sticky_across_descendants():
    from crackmeanflow.common.training_protocol import resume_taint_state

    generation_one = resume_taint_state(parent_checkpoint={}, current_config_mismatch=True)
    generation_two = resume_taint_state(
        parent_checkpoint={"extra_state": generation_one}, current_config_mismatch=False
    )
    generation_three = resume_taint_state(
        parent_checkpoint={"extra_state": generation_two}, current_config_mismatch=False
    )
    assert generation_one == {
        "resume_config_mismatch_current": True,
        "resume_config_mismatch_inherited": False,
        "resume_config_mismatch": True,
    }
    assert generation_two["resume_config_mismatch_current"] is False
    assert generation_two["resume_config_mismatch_inherited"] is True
    assert generation_two["resume_config_mismatch"] is True
    assert generation_three["resume_config_mismatch"] is True


def test_training_split_view_has_a_runtime_test_firewall():
    from crackmeanflow.common.training_protocol import training_split_view

    splits = {"train": ["train"], "val": ["val"], "test": ["test"]}
    assert training_split_view(splits) == {"train": ["train"], "val": ["val"]}
    with pytest.raises(RuntimeError, match="TEST firewall"):
        training_split_view({"train": ["train"], "test": ["test"]})


def test_checkpoint_completion_is_fail_closed_for_missing_or_short_runs():
    from crackmeanflow.common.training_protocol import require_complete_checkpoint

    incomplete = {
        "global_optimizer_step": 19,
        "best_val_metric": 0.5,
        "best_val_threshold": 0.5,
        "config_hash": "config",
        "source_tree_sha256": "source",
        "protocol_bundle_sha256": "protocol",
        "extra_state": {
            "fairness": {"planned_optimizer_steps": 20},
            "epoch_complete": True,
        },
    }
    with pytest.raises(RuntimeError, match="planned optimizer steps"):
        require_complete_checkpoint(incomplete)

    complete = {
        "global_optimizer_step": 20,
        "best_val_metric": 0.5,
        "best_val_threshold": 0.5,
        "config_hash": "config",
        "source_tree_sha256": "source",
        "protocol_bundle_sha256": "protocol",
        "extra_state": {
            "fairness": {"planned_optimizer_steps": 20},
            "epoch_complete": True,
            "budget_reached": True,
        },
    }
    assert require_complete_checkpoint(complete)["completed_optimizer_steps"] == 20


def test_run_completion_rejects_nonfinite_training_runtime(tmp_path):
    from crackmeanflow.common.training_protocol import verify_run_completion_artifact

    best = tmp_path / "best.pt"
    last = tmp_path / "last.pt"
    common = {
        "global_optimizer_step": 20,
        "best_val_metric": 0.5,
        "best_val_threshold": 0.5,
        "config_hash": "config",
        "source_tree_sha256": "source",
        "protocol_bundle_sha256": "protocol",
        "extra_state": {
            "fairness": {"planned_optimizer_steps": 20},
            "epoch_complete": True,
            "budget_reached": True,
            "run_complete": True,
        },
    }
    torch.save(common, best)
    torch.save(common, last)
    record = {
        "schema": "CRACKMEANFLOW_RUN_COMPLETION_V1",
        "status": "PASS",
        "best_checkpoint": "best.pt",
        "best_checkpoint_sha256": hashlib.sha256(best.read_bytes()).hexdigest(),
        "final_checkpoint": "last.pt",
        "final_checkpoint_sha256": hashlib.sha256(last.read_bytes()).hexdigest(),
        "planned_optimizer_steps": 20,
        "completed_optimizer_steps": 20,
        "config_hash": "config",
        "source_tree_sha256": "source",
        "protocol_bundle_sha256": "protocol",
        "required_artifacts": [],
        "training_runtime_seconds": "nan",
    }
    (tmp_path / "RUN_COMPLETE.json").write_text(json.dumps(record))

    with pytest.raises(RuntimeError, match="runtime"):
        verify_run_completion_artifact(best, common)


def test_execution_identity_guard_rejects_a_changed_source_or_effective_config():
    from crackmeanflow.common.training_protocol import validate_execution_identity

    with pytest.raises(RuntimeError, match="source tree"):
        validate_execution_identity(
            expected_config_hash="config",
            actual_config_hash="config",
            expected_source_tree_hash="expected-source",
            actual_source_tree_hash="actual-source",
            expected_protocol_bundle_hash="protocol",
            actual_protocol_bundle_hash="protocol",
        )
    with pytest.raises(RuntimeError, match="effective config"):
        validate_execution_identity(
            expected_config_hash="expected-config",
            actual_config_hash="actual-config",
            expected_source_tree_hash="source",
            actual_source_tree_hash="source",
            expected_protocol_bundle_hash="protocol",
            actual_protocol_bundle_hash="protocol",
        )


@pytest.mark.parametrize(
    ("experiment", "batch", "accum", "valid"),
    [
        ("CONFERENCE_CRACKMEANFLOW_U", 4, 2, True),
        ("A2B_HYBRID_IMF_MASK_ENDPOINT_AWARE_015_CAPACITY_MATCHED", 1, 8, True),
        ("A5_GEOCRACK_IMF_ENDPOINT_AWARE_015_CANDIDATE", 1, 8, True),
        ("A2B_HYBRID_IMF_MASK_ENDPOINT_AWARE_015_CAPACITY_MATCHED", 2, 4, False),
        ("A2B_HYBRID_IMF_MASK_ENDPOINT_AWARE_015_CAPACITY_MATCHED", 4, 2, False),
        ("A2B_HYBRID_IMF_MASK_ENDPOINT_AWARE_015_CAPACITY_MATCHED", 8, 1, False),
        ("A5_GEOCRACK_IMF_ENDPOINT_AWARE_015_CANDIDATE", 2, 4, False),
        ("A5_GEOCRACK_IMF_ENDPOINT_AWARE_015_CANDIDATE", 4, 2, False),
        ("A5_GEOCRACK_IMF_ENDPOINT_AWARE_015_CANDIDATE", 8, 1, False),
    ],
)
def test_primary_runtime_policy_requires_exact_microbatch_layout(experiment, batch, accum, valid):
    from crackmeanflow.common.training_protocol import runtime_policy_status

    status = runtime_policy_status(experiment, batch, accum)
    assert status["RUNTIME_POLICY_OVERRIDE"] is (not valid)
    assert status["INVALID_FOR_PRIMARY_COMPARISON"] is (not valid)


def test_geometry_aggregation_is_batch_partition_invariant():
    from crackmeanflow.journal.engine.evaluation import aggregate_geometry

    gt = -torch.ones(3, 1, 5, 5)
    gt[1, 0, 2, 2] = 1
    gt[2, 0, 1:4, 2] = 1
    geometry = -torch.ones(3, 2, 5, 5)
    probability = torch.zeros(3, 1, 5, 5)
    kwargs = dict(threshold=0.5, max_radius=16, representation="centerline_edt", distance_encoding="sqrt")
    whole = aggregate_geometry([(geometry, probability, gt)], **kwargs)
    partitioned = aggregate_geometry(
        [(geometry[:2], probability[:2], gt[:2]), (geometry[2:], probability[2:], gt[2:])], **kwargs
    )
    for key in ("edt_radius_mae_px", "centerline_assd_px", "skeleton_length_rel_error", "cldice", "boundary_f1", "f1", "iou"):
        assert partitioned[key] == pytest.approx(whole[key], abs=1e-10), key


def test_rejected_resume_does_not_mutate_existing_run_metadata(tmp_path):
    """A resume provenance failure must leave a prior run directory byte-identical."""
    repo = Path(__file__).resolve().parents[1]
    data = tmp_path / "data"
    for index, split in enumerate(("train", "val", "test")):
        images = data / split / "cracked" / "images"
        masks = data / split / "cracked" / "masks"
        images.mkdir(parents=True)
        masks.mkdir(parents=True)
        Image.new("RGB", (8, 8), color=(index + 1, 0, 0)).save(images / f"{split}_sample.png")
        Image.new("L", (8, 8), color=255).save(masks / f"{split}_sample.png")

    config = yaml.safe_load((repo / "configs/conference/crackmeanflow_unet.yaml").read_text())
    config["model"]["img_size"] = 32
    config["train"].update({"batch_size": 1, "grad_accum_steps": 1, "epochs": 1, "num_workers": 0})
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    out = tmp_path / "prior_run"
    out.mkdir()
    originals = {
        "EFFECTIVE_CONFIG.yaml": b"prior-config\n",
        "dataset_manifest.json": b"prior-manifest\n",
        "RUN_IDENTITY.json": b"prior-identity\n",
        "ENVIRONMENT.json": b"prior-environment\n",
    }
    for name, content in originals.items():
        (out / name).write_bytes(content)
    before = {name: hashlib.sha256((out / name).read_bytes()).hexdigest() for name in originals}
    resume = tmp_path / "missing_provenance.pt"
    torch.save({}, resume)

    result = subprocess.run(
        [sys.executable, "scripts/train_journal.py", "--config", str(config_path), "--data", str(data),
         "--dataset-name", "fixture", "--dataset-version", "v1", "--out", str(out), "--resume", str(resume)],
        cwd=repo, text=True, capture_output=True,
    )

    assert result.returncode != 0
    assert "checkpoint has no source provenance" in result.stderr
    after = {name: hashlib.sha256((out / name).read_bytes()).hexdigest() for name in originals}
    assert after == before


def test_threshold_calibration_is_partition_invariant():
    from crackmeanflow.common.evaluation import _micro_threshold_sweep_from_score_gt

    scores = torch.tensor([[[[0.1, 0.4], [0.6, 0.9]]], [[[0.2, 0.5], [0.7, 0.8]]], [[[0.3, 0.45], [0.55, 0.95]]]])
    gt = torch.tensor([[[[0., 0.], [1., 1.]]], [[[0., 1.], [1., 0.]]], [[[0., 0.], [1., 1.]]]])
    thresholds = [0.2, 0.4, 0.5, 0.7]
    whole = _micro_threshold_sweep_from_score_gt([(scores, gt)], thresholds)
    split_21 = _micro_threshold_sweep_from_score_gt([(scores[:2], gt[:2]), (scores[2:], gt[2:])], thresholds)
    split_111 = _micro_threshold_sweep_from_score_gt([(scores[i:i + 1], gt[i:i + 1]) for i in range(3)], thresholds)
    for threshold in thresholds:
        for key in ("f1", "iou", "precision", "recall", "tp", "fp", "fn", "tn"):
            assert split_21[threshold][key] == pytest.approx(whole[threshold][key])
            assert split_111[threshold][key] == pytest.approx(whole[threshold][key])
    assert max(whole, key=lambda t: whole[t]["f1"]) == max(split_21, key=lambda t: split_21[t]["f1"]) == max(split_111, key=lambda t: split_111[t]["f1"])


def test_radius_loss_and_gradient_depend_on_microbatch_partition():
    """The locked radius reduction is per microbatch, so equal effective batches differ."""
    prediction = torch.tensor([1.0, 3.0, 10.0, 20.0], requires_grad=True)
    target = torch.zeros(4)
    valid = torch.tensor([1.0, 1.0, 1.0, 0.0])

    def radius_loss(indices):
        return ((prediction[indices] - target[indices]).abs() * valid[indices]).sum() / valid[indices].sum().clamp_min(1.0)

    full = radius_loss(slice(None))
    full_grad, = torch.autograd.grad(full, prediction, retain_graph=True)
    micro = (radius_loss(slice(0, 2)) + radius_loss(slice(2, 4))) / 2
    micro_grad, = torch.autograd.grad(micro, prediction)
    assert float(full) != pytest.approx(float(micro))
    assert not torch.allclose(full_grad, micro_grad)


def test_post_repair_v2_scheduler_is_shared_and_has_postwarmup_fast_budget():
    from crackmeanflow.common.scheduler import make_warmup_cosine_scheduler

    protocol = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/protocol/post_repair_protocol_v2.yaml").read_text())
    assert protocol["protocol_version"] == "CRACKMEANFLOW_POST_REPAIR_PROTOCOL_V2"
    assert protocol["scheduler"]["total_optimizer_steps"] == 21_000
    assert protocol["scheduler"]["warmup_epochs"] == 10
    assert protocol["fast_screen"]["optimizer_steps"] == 12_375
    assert protocol["fast_screen"]["post_warmup_steps"] == 4_125
    opt = torch.optim.SGD([torch.nn.Parameter(torch.tensor(1.0))], lr=1.0)
    sched = make_warmup_cosine_scheduler(opt, epochs=100, optimizer_steps_epoch=825, warmup_epochs=10, total_optimizer_steps=21_000)
    assert sched._cmf_warmup_steps == 8_250
    assert sched._cmf_total_steps == 21_000
