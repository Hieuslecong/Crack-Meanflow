from __future__ import annotations

import importlib

import pytest


benchmark = importlib.import_module("scripts.benchmark_microbatch_v3")


def test_validate_partition_accepts_effective_batch_eight():
    assert benchmark.validate_partition(4, 2) == {"micro_batch_size": 4, "grad_accum_steps": 2, "effective_batch_size": 8}
    assert benchmark.validate_partition(8, 1) == {"micro_batch_size": 8, "grad_accum_steps": 1, "effective_batch_size": 8}


def test_validate_partition_rejects_noncanonical_effective_batch():
    with pytest.raises(ValueError, match="effective batch"):
        benchmark.validate_partition(3, 2)


def test_parser_preserves_canonical_horizon_for_diagnostic_benchmark():
    parser = benchmark.build_parser()
    args = parser.parse_args(
        [
            "--config",
            "config.yaml",
            "--protocol",
            "protocol.yaml",
            "--preflight",
            "preflight.json",
            "--data",
            "data",
            "--out",
            "result.json",
            "--micro-batch-size",
            "4",
            "--grad-accumulation-steps",
            "2",
            "--research-total-optimizer-steps",
            "21000",
            "--diagnostic-stop-optimizer-steps",
            "200",
        ]
    )
    assert args.research_total_optimizer_steps == 21000
    assert args.diagnostic_stop_optimizer_steps == 200


def test_identity_is_diagnostic_and_records_partition_override():
    identity = benchmark.build_benchmark_identity(
        config_path="configs/post_repair_v3/a2b_endpoint.yaml",
        config_sha256="config-hash",
        protocol_path="configs/protocol/post_repair_protocol_v3.yaml",
        protocol_sha256="protocol-hash",
        source_tree_sha256="source-hash",
        dataset_identity={"split_id": "CFD_FROZEN_HISTORICAL_V1"},
        micro_batch_size=4,
        grad_accumulation_steps=2,
        research_total_optimizer_steps=21000,
        diagnostic_stop_optimizer_steps=200,
        seed=0,
    )
    assert identity["diagnostic_only"] is True
    assert identity["research_metric_valid"] is False
    assert identity["eligible_for_paper"] is False
    assert identity["execution_partition_override"] == {
        "micro_batch_size": 4,
        "grad_accumulation_steps": 2,
        "effective_batch_size": 8,
    }
    assert identity["research_scheduler_total_steps"] == 21000
