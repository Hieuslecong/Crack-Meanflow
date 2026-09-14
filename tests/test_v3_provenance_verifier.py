import copy
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from crackmeanflow.common import config_hash, file_sha256, protocol_bundle_hash, source_tree_hash
from crackmeanflow.common.v3_provenance import active_v3_bundle_hash, verify_v3_provenance


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/protocol/post_repair_protocol_v3.yaml"
CONFIG = ROOT / "configs/post_repair_v3/conference.yaml"


def _valid_preflight(tmp_path):
    protocol = yaml.safe_load(PROTOCOL.read_text())
    identity = {"train": {"count": 6606}, "val": {"count": 1447}, "test": {"count": 1384}}
    payload = {
        "schema": "CRACKMEANFLOW_PROTOCOL_PREFLIGHT_V3",
        "status": "PASS",
        "protocol": "configs/protocol/post_repair_protocol_v3.yaml",
        "dataset_identity": identity,
        "arms": {
            "Conference": {
                "config": "configs/post_repair_v3/conference.yaml",
                "config_file_sha256": protocol["config_locks"]["Conference"]["file_sha256"],
                "config_semantic_sha256": protocol["config_locks"]["Conference"]["semantic_sha256"],
                "checks": {"config_file_lock": True, "config_semantic_lock": True},
            }
        },
        "provenance": {
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "source_tree_sha256": source_tree_hash(ROOT),
            "protocol_file_sha256": file_sha256(PROTOCOL),
            "protocol_bundle_sha256_legacy_global": protocol_bundle_hash(ROOT),
            "protocol_bundle_sha256_v3_active": active_v3_bundle_hash(PROTOCOL, protocol),
        },
    }
    path = tmp_path / "preflight.json"
    path.write_text(json.dumps(payload))
    return path, identity


def _verify(path, identity):
    return verify_v3_provenance(
        protocol_path=PROTOCOL,
        config_path=CONFIG,
        preflight_path=path,
        dataset_name="CFD",
        dataset_version="CFD_FROZEN_HISTORICAL_V1",
        actual_dataset_identity=identity,
        research_total_optimizer_steps=21000,
        diagnostic_stop_optimizer_steps=16500,
        root=ROOT,
    )


def test_v3_verifier_accepts_only_locked_primary_identity(tmp_path):
    path, identity = _valid_preflight(tmp_path)
    result = _verify(path, identity)
    assert result["arm"] == "Conference"
    assert result["research_total_optimizer_steps"] == 21000


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda p: p.update(status="FAIL"), "not PASS"),
        (lambda p: p["provenance"].update(commit="wrong"), "commit"),
        (lambda p: p["provenance"].update(source_tree_sha256="wrong"), "source tree"),
        (lambda p: p["provenance"].update(protocol_file_sha256="wrong"), "protocol file"),
        (lambda p: p["provenance"].update(protocol_bundle_sha256_legacy_global="wrong"), "legacy protocol bundle"),
        (lambda p: p["provenance"].update(protocol_bundle_sha256_v3_active="wrong"), "active V3 bundle"),
    ],
)
def test_v3_verifier_rejects_preflight_and_provenance_mismatch(tmp_path, mutation, message):
    path, identity = _valid_preflight(tmp_path)
    payload = json.loads(path.read_text())
    mutation(payload)
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match=message):
        _verify(path, identity)


def test_v3_verifier_rejects_dataset_mismatch_and_non_primary_config(tmp_path):
    path, identity = _valid_preflight(tmp_path)
    changed = copy.deepcopy(identity)
    changed["train"]["count"] -= 1
    with pytest.raises(RuntimeError, match="dataset identity"):
        _verify(path, changed)
    with pytest.raises(RuntimeError, match="primary arm"):
        verify_v3_provenance(
            protocol_path=PROTOCOL,
            config_path=ROOT / "configs/post_repair_v2/conference.yaml",
            preflight_path=path,
            dataset_name="CFD",
            dataset_version="CFD_FROZEN_HISTORICAL_V1",
            actual_dataset_identity=identity,
            research_total_optimizer_steps=21000,
            diagnostic_stop_optimizer_steps=16500,
            root=ROOT,
        )


def test_v3_verifier_rejects_bad_arm_record_horizon_stop_and_version(tmp_path):
    path, identity = _valid_preflight(tmp_path)
    payload = json.loads(path.read_text())
    payload["arms"]["Conference"]["checks"]["config_file_lock"] = False
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="preflight config lock"):
        _verify(path, identity)

    path, identity = _valid_preflight(tmp_path)
    common = dict(
        protocol_path=PROTOCOL,
        config_path=CONFIG,
        preflight_path=path,
        dataset_name="CFD",
        dataset_version="CFD_FROZEN_HISTORICAL_V1",
        actual_dataset_identity=identity,
        root=ROOT,
    )
    with pytest.raises(RuntimeError, match="exactly 21000"):
        verify_v3_provenance(**common, research_total_optimizer_steps=20000)
    with pytest.raises(RuntimeError, match="shorter than research horizon"):
        verify_v3_provenance(
            **common,
            research_total_optimizer_steps=21000,
            diagnostic_stop_optimizer_steps=21000,
        )
    common["dataset_version"] = "unlocked-version"
    with pytest.raises(RuntimeError, match="dataset name/version"):
        verify_v3_provenance(**common, research_total_optimizer_steps=21000)
