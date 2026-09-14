import json
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def _identity_kwargs(tmp_path):
    data_identity = {
        "dataset_name": "CFD",
        "dataset_version": "frozen-v1",
        "split_id": "CFD_FROZEN_HISTORICAL_V1",
        "splits": {
            "train": {"count": 2, "content_manifest_sha256": "train"},
            "val": {"count": 1, "content_manifest_sha256": "val"},
            "test": {"count": 1, "content_manifest_sha256": "test"},
        },
    }
    return dict(
        root=ROOT,
        variant="V2",
        lane="matched_optimizer_steps",
        arm="Conference",
        seed=0,
        config_path=ROOT / "configs/post_repair_v2/conference.yaml",
        protocol_path=ROOT / "configs/protocol/post_repair_protocol_v2.yaml",
        data_identity=data_identity,
        training_budget={"planned_optimizer_steps": 21000},
    )


def test_run_identity_is_stable_and_records_all_provenance(tmp_path):
    from crackmeanflow.common.experiment_harness import build_run_identity

    identity = build_run_identity(**_identity_kwargs(tmp_path))
    assert identity["schema"] == "CRACKMEANFLOW_RUN_IDENTITY_V1"
    assert identity["training_seed"] == 0
    assert identity["source_tree_sha256"]
    assert identity["protocol_bundle_sha256"]
    assert identity["config_sha256"]
    assert identity["split_id"] == "CFD_FROZEN_HISTORICAL_V1"
    assert identity["access_policy"] == {"CFD_TEST": "CLOSED", "GAPS384": "CLOSED"}
    assert identity == build_run_identity(**_identity_kwargs(tmp_path))


def test_immutable_artifact_rejects_mutation(tmp_path):
    from crackmeanflow.common.experiment_harness import write_immutable_json

    path = tmp_path / "RUN_IDENTITY.json"
    write_immutable_json(path, {"run_id": "a"})
    write_immutable_json(path, {"run_id": "a"})
    with pytest.raises(RuntimeError, match="immutable"):
        write_immutable_json(path, {"run_id": "b"})


def test_evaluation_gate_keeps_test_and_gaps_closed_until_prerequisites():
    from crackmeanflow.common.experiment_harness import assert_dataset_access_allowed

    with pytest.raises(RuntimeError, match="CFD_TEST"):
        assert_dataset_access_allowed("CFD_TEST", {"fix_gate": False, "threshold_lock": False})
    with pytest.raises(RuntimeError, match="GAPS384"):
        assert_dataset_access_allowed("GAPS384", {"fix_gate": True, "threshold_lock": False})
    assert_dataset_access_allowed(
        "CFD_TEST", {"fix_gate": True, "threshold_lock": True, "target_lock": True}
    )
    assert_dataset_access_allowed("GAPS384", {"fix_gate": True, "threshold_lock": True})


def test_matrix_expansion_is_complete_and_never_opens_targets():
    from crackmeanflow.common.experiment_harness import expand_experiment_matrix, load_experiment_matrix

    matrix = load_experiment_matrix(ROOT / "configs/protocol/v1_v2_experiment_matrix.yaml")
    rows = expand_experiment_matrix(matrix, root=ROOT)
    assert len(rows) == 36
    assert {r["lane"] for r in rows} == {"matched_optimizer_steps", "practical_protocol"}
    assert {r["variant"] for r in rows} == {"V1", "V2"}
    assert {r["training_seed"] for r in rows} == {0, 1, 2}
    assert all(r["access_policy"] == {"CFD_TEST": "CLOSED", "GAPS384": "CLOSED"} for r in rows)
    assert len({r["run_id"] for r in rows}) == len(rows)


def test_matrix_requires_dataset_and_split_identity(tmp_path):
    from crackmeanflow.common.experiment_harness import load_experiment_matrix

    matrix = yaml.safe_load((ROOT / "configs/protocol/v1_v2_experiment_matrix.yaml").read_text())
    matrix["source"].pop("split_id")
    path = tmp_path / "invalid_matrix.yaml"
    path.write_text(yaml.safe_dump(matrix))
    with pytest.raises(RuntimeError, match="source identity.*split_id"):
        load_experiment_matrix(path)


def test_runtime_profile_reports_stage_and_throughput():
    from crackmeanflow.common.experiment_harness import RuntimeProfile

    profile = RuntimeProfile(device="cpu")
    profile.start_stage("forward")
    profile.stop_stage("forward")
    profile.record_optimizer_step(samples=8, elapsed_seconds=2.0)
    result = profile.finalize()
    assert result["schema"] == "CRACKMEANFLOW_RUNTIME_PROFILE_V1"
    assert result["optimizer_steps"] == 1
    assert result["samples_per_second"] == pytest.approx(4.0)
    assert result["stages"]["forward"]["count"] == 1


def test_runtime_profile_training_only_excludes_validation_and_checkpoint():
    from crackmeanflow.common.experiment_harness import RuntimeProfile

    profile = RuntimeProfile(device="cpu")
    profile.stop_stage("train", elapsed_seconds=2.0)
    profile.stop_stage("validation", elapsed_seconds=7.0)
    profile.stop_stage("checkpoint", elapsed_seconds=11.0)
    assert profile.finalize()["training_only_seconds"] == 2.0


def test_paired_comparison_requires_matching_seeds_and_classifies_better():
    from crackmeanflow.common.experiment_harness import compare_paired_reports

    def report(seed, f1, method):
        return {
            "training_seed": seed,
            "f1": f1,
            "iou": f1 - 0.1,
            "runtime_seconds": 100,
            "artifact_complete": True,
            "scientific_validity": "VALID_HEADLINE_PROTOCOL",
            "requested_nfe": 1,
            "actual_nfe": 1,
            "nfe_contract_pass": True,
            "source_identity": {"split_id": "CFD_FROZEN_HISTORICAL_V1"},
            "target_identity": {"dataset_name": "CFD_TEST"},
            "checkpoint_source_tree_sha256": "source-tree",
            "checkpoint_protocol_bundle_sha256": "protocol-bundle",
            "method_config_hash": method,
            "training_budget": {"planned_optimizer_steps": 21000},
            "backbone": "unet",
        }

    v1 = [report(s, value, "v1-method") for s, value in enumerate((0.50, 0.51, 0.49))]
    v2 = [report(s, value, "v2-method") for s, value in enumerate((0.55, 0.56, 0.54))]
    result = compare_paired_reports(v1, v2, correctness_gates={"all_pass": True})
    assert result["conclusion"] == "CORRECT_AND_BETTER"
    assert result["metrics"]["f1"]["paired_deltas"] == pytest.approx([0.05, 0.05, 0.05])
    assert result["metrics"]["f1"]["mean_delta"] == pytest.approx(0.05)

    with pytest.raises(RuntimeError, match="training seeds"):
        compare_paired_reports(v1, v2[:-1], correctness_gates={"all_pass": True})


def test_incomplete_or_gate_failed_comparison_is_invalid():
    from crackmeanflow.common.experiment_harness import compare_paired_reports

    v1 = [{"training_seed": s, "f1": 0.5, "iou": 0.4} for s in range(3)]
    v2 = [{"training_seed": s, "f1": 0.5, "iou": 0.4} for s in range(3)]
    result = compare_paired_reports(v1, v2, correctness_gates={"all_pass": False})
    assert result["conclusion"] == "INVALID_OR_INCOMPLETE"


def test_paired_comparison_rejects_missing_completion_and_provenance():
    from crackmeanflow.common.experiment_harness import compare_paired_reports

    rows = [{"training_seed": s, "f1": 0.5, "iou": 0.4} for s in range(3)]
    result = compare_paired_reports(rows, rows, correctness_gates={"all_pass": True})
    assert result["conclusion"] == "INVALID_OR_INCOMPLETE"
    assert any("artifact_complete" in reason for reason in result["invalid_reasons"])


def test_paired_comparison_requires_geometry_for_geometry_backbone():
    from crackmeanflow.common.experiment_harness import compare_paired_reports

    def report(seed, method):
        return {
            "training_seed": seed,
            "f1": 0.5,
            "iou": 0.4,
            "runtime_seconds": 100,
            "artifact_complete": True,
            "scientific_validity": "VALID_HEADLINE_PROTOCOL",
            "requested_nfe": 1,
            "actual_nfe": 1,
            "nfe_contract_pass": True,
            "source_identity": {"split_id": "CFD_FROZEN_HISTORICAL_V1"},
            "target_identity": {"dataset_name": "CFD_TEST"},
            "checkpoint_source_tree_sha256": "source-tree",
            "checkpoint_protocol_bundle_sha256": "protocol-bundle",
            "method_config_hash": method,
            "training_budget": {"planned_optimizer_steps": 21000},
            "backbone": "geocrack_imf",
        }

    result = compare_paired_reports(
        [report(s, "v1") for s in range(3)],
        [report(s, "v2") for s in range(3)],
        correctness_gates={"all_pass": True},
    )
    assert result["conclusion"] == "INVALID_OR_INCOMPLETE"
    assert any("geometry" in reason for reason in result["invalid_reasons"])


def test_cli_matrix_writer_emits_replayable_json(tmp_path):
    import subprocess
    import sys

    out = tmp_path / "queue.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/build_experiment_matrix.py"),
            "--matrix",
            str(ROOT / "configs/protocol/v1_v2_experiment_matrix.yaml"),
            "--root",
            str(ROOT),
            "--out",
            str(out),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(out.read_text())
    assert payload["n_runs"] == 36
    assert payload["execution"] == "QUEUE_ONLY_NO_GPU_RUNS"
    first = payload["rows"][0]
    assert first["dataset_name"] == "CFD"
    assert first["dataset_version"] == "frozen-historical-v1"
    assert first["command"][first["command"].index("--dataset-version") + 1] == "frozen-historical-v1"


def test_queue_rows_bind_effective_config_protocol_and_source_hashes():
    from crackmeanflow.common.experiment_harness import build_queue, expand_experiment_matrix, load_experiment_matrix

    matrix = load_experiment_matrix(ROOT / "configs/protocol/v1_v2_experiment_matrix.yaml")
    row = next(
        r for r in expand_experiment_matrix(matrix, root=ROOT)
        if r["lane"] == "matched_optimizer_steps" and r["variant"] == "V2" and r["arm"] == "Conference" and r["training_seed"] == 0
    )
    queued = build_queue([row], root=ROOT, output_root=ROOT / "outputs" / "queue-test")
    bound = queued[0]
    for key in (
        "effective_config_sha256",
        "config_file_sha256",
        "protocol_file_sha256",
        "source_tree_sha256",
        "protocol_bundle_sha256",
    ):
        assert isinstance(bound[key], str) and len(bound[key]) == 64
    command = bound["command"]
    assert "--expected-config-sha256" in command
    assert "--expected-source-identity-json" in command
    assert command[command.index("--out") + 1].endswith("outputs/queue-test/matched_optimizer_steps_v2_conference_s0")
