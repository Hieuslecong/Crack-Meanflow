from pathlib import Path

import torch
import yaml

from crackmeanflow.common import config_hash, file_sha256
from crackmeanflow.common.scheduler import make_warmup_cosine_scheduler
from crackmeanflow.conference.losses import ConferenceMeanFlowLoss


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/protocol/post_repair_protocol_v3.yaml"
CONFIG_DIR = ROOT / "configs/post_repair_v3"
SMOKE = ROOT / "scripts/smoke_preflight_v3.py"
SMOKE_GATE = ROOT / "scripts/smoke_gate_v3.py"
BRANCH_PROBE = ROOT / "scripts/branch_coverage_probe_v3.py"
RESUME_PROBE = ROOT / "scripts/resume_equivalence_v3.py"
FAST_BENCHMARK = ROOT / "scripts/benchmark_microbatch_v3.py"
FAST_PREFLIGHT = ROOT / "scripts/protocol_preflight_v3_fast.py"


def _load(path):
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_v3_protocol_locks_observed_training_geometry():
    p = _load(PROTOCOL)
    assert p["source_dataset"]["expected_train_samples"] == 6606
    assert p["optimization"]["expected_optimizer_steps_per_epoch"] == 825
    assert p["optimization"]["effective_batch_size"] == 8
    assert p["optimization"]["expected_usable_train_samples_per_epoch"] == 6600
    assert p["optimization"]["expected_omitted_train_samples_per_epoch"] == 6
    assert p["optimization"]["matched_optimizer_steps"] == 21000
    assert p["optimization"]["expected_warmup_optimizer_steps"] == 8250


def test_all_v3_primary_configs_are_semantically_locked():
    p = _load(PROTOCOL)
    for arm, cfg_rel in p["primary_arms"].items():
        path = ROOT / cfg_rel
        cfg = _load(path)
        assert cfg["train"]["max_optimizer_steps"] == 21000
        assert cfg["train"]["batch_size"] * cfg["train"]["grad_accum_steps"] == 8
        assert cfg["eval"]["num_steps"] == 1
        rationale = cfg["train"]["sample_balance"]["rationale"]
        assert "1682" not in rationale
        assert "6606" in rationale
        assert config_hash(cfg) == p["config_locks"][arm]["semantic_sha256"]
        assert file_sha256(path) == p["config_locks"][arm]["file_sha256"]


def test_conference_curriculum_is_optimizer_step_based_and_reaches_endpoint_stage():
    p = _load(PROTOCOL)
    cfg = _load(ROOT / p["primary_arms"]["Conference"])
    tc = cfg["loss"]["time_curriculum"]
    assert tc["basis"] == "optimizer_step"
    assert tc["effective_batch_size"] == 8
    stages = tc["stages"]
    assert stages == p["conference_curriculum"]["stages"]
    assert stages[0]["optimizer_steps"][0] == 0
    assert stages[-1]["optimizer_steps"][1] == 21000
    assert stages[-1]["boundary_prob"] == 0.15
    assert 0 < stages[-1]["optimizer_steps"][0] < 21000
    assert 21000 - stages[-1]["optimizer_steps"][0] == p["conference_curriculum"]["endpoint_stage_optimizer_steps"]

    loss = ConferenceMeanFlowLoss(**cfg["loss"])
    # Position the sampler inside the final deployment stage.
    offset = stages[-1]["optimizer_steps"][0] * tc["effective_batch_size"]
    _, _, logs = loss._sample_r_t(128, torch.device("cpu"), return_stage=True, sample_offset=offset)
    assert logs["curriculum_basis"] == "optimizer_step"
    assert logs["curriculum_position"] == stages[-1]["optimizer_steps"][0]
    assert logs["boundary_prob"] == 0.15
    assert logs["boundary_count"] > 0


def test_diagnostic_scheduler_keeps_research_horizon():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4)
    scheduler = make_warmup_cosine_scheduler(
        optimizer,
        epochs=100,
        optimizer_steps_epoch=825,
        warmup_epochs=10,
        total_optimizer_steps=21000,
    )
    assert scheduler._cmf_total_steps == 21000
    assert scheduler._cmf_warmup_steps == 8250
    for _ in range(20):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] > 0.0
    assert scheduler._cmf_total_steps == 21000
    assert scheduler._cmf_warmup_steps == 8250


def test_smoke_runner_is_explicitly_non_paper_artifact():
    source = SMOKE.read_text(encoding="utf-8")
    assert '"diagnostic_only": True' in source
    assert '"research_metric_valid": False' in source
    assert '"eligible_for_paper": False' in source
    assert '"writes_run_complete": False' in source
    assert '"writes_best_checkpoint": False' in source
    assert '"writes_last_checkpoint": False' in source
    assert "RUN_COMPLETE.json" in source
    assert "write_immutable_json" not in source
    assert "save_checkpoint_atomic" not in source


def test_smoke_runner_has_independent_stop_and_scheduler_controls():
    source = SMOKE.read_text(encoding="utf-8")
    assert "--research-total-optimizer-steps" in source
    assert "--diagnostic-stop-steps" in source
    assert "total_optimizer_steps=research_total" in source
    assert "while global_step < stop_steps" in source


def test_hardened_smoke_gate_binds_execution_identity():
    source = SMOKE_GATE.read_text(encoding="utf-8")
    for token in (
        "source_tree_sha256",
        "protocol_file_sha256",
        "protocol_bundle_sha256",
        "config_file_sha256",
        "config_semantic_sha256",
        "provenance_before",
        "provenance_after",
    ):
        assert token in source


def test_branch_and_resume_gates_exist_and_are_diagnostic_only():
    branch = BRANCH_PROBE.read_text(encoding="utf-8")
    resume = RESUME_PROBE.read_text(encoding="utf-8")
    assert "endpoint_loss" in branch and "thin_loss" in branch and "gic_count" in branch
    assert '"research_metric_valid": False' in branch
    assert "20_continuous_vs_10_checkpoint_restore_10" in resume
    assert '"canonical_partial_epoch_resume_supported": False' in resume
    assert '"paper_resume_policy": "epoch_boundary_only"' in resume


def test_fast_preflight_exports_partition_only_validator():
    source = FAST_PREFLIGHT.read_text(encoding="utf-8")
    assert "validate_fast_config_pair" in source
    assert "batch_size" in source
    assert "grad_accum_steps" in source
    assert "effective_batch" in source


def test_fast_benchmark_contract_is_diagnostic_and_horizon_locked():
    source = FAST_BENCHMARK.read_text(encoding="utf-8")
    assert '"diagnostic_only": True' in source
    assert '"research_metric_valid": False' in source
    assert '"eligible_for_paper": False' in source
    assert "research scheduler horizon must remain exactly 21000" in source
