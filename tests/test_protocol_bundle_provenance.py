from pathlib import Path
import json, torch, yaml, pytest
from crackmeanflow.common import protocol_bundle_hash, protocol_bundle_manifest

ROOT=Path(__file__).resolve().parents[1]

def test_protocol_bundle_is_nonempty_and_covers_protocol_and_fairness():
    rows=protocol_bundle_manifest(ROOT); paths={r['path'] for r in rows}
    assert 'configs/protocol/paper_protocol.yaml' in paths
    assert 'configs/fairness/common_recipe_matrix.yaml' in paths
    assert 'configs/fairness/data_sampling_matrix.yaml' in paths
    assert len(protocol_bundle_hash(ROOT))==64

def test_release_preflight_has_no_cli_license_bypass():
    text=(ROOT/'scripts/release_preflight.py').read_text()
    assert 'allow-unconfirmed-conference-unet-rights' not in text
    assert "redistribution_license_status')!='CONFIRMED'" in text

def test_evaluator_requires_protocol_bundle_and_cross_dataset_contamination_audit():
    text=(ROOT/'scripts/evaluate_journal.py').read_text()
    assert 'protocol_bundle_sha256' in text
    assert 'audit_cross_dataset_image_content_overlap' in text
