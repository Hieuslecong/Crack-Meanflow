from pathlib import Path
import copy
import yaml
ROOT=Path(__file__).resolve().parents[1]

def _load(name):return yaml.safe_load((ROOT/name).read_text())

def _strip_identity(cfg):
    x=copy.deepcopy(cfg)
    for k in ('track','experiment','protocol_role'): x.pop(k,None)
    return x

def _strip_endpoint(loss):
    x=dict(loss);x.pop('endpoint_probability',None);x.pop('endpoint_sampling',None);return x

def test_a5_final_is_baseline_not_unvalidated_endpoint_candidate():
    base=_load('configs/journal/a5_geocrack_imf_baseline.yaml')
    assert base['track']=='journal'
    assert 'endpoint_probability' not in base['loss']
    assert base['protocol_role']=='journal_baseline_pre_endpoint'

def test_no_gic_control_differs_only_in_gic_from_endpoint_candidate():
    base=_load('configs/journal/a5_geocrack_imf_endpoint_candidate.yaml');ctrl=_load('configs/journal/a5_geocrack_imf_no_gic_control.yaml')
    assert base['model']==ctrl['model'];assert base['train']==ctrl['train'];assert base['eval']==ctrl['eval']
    b=dict(base['loss']);c=dict(ctrl['loss'])
    for k in ['gic_weight','gic_probability']:b.pop(k,None);c.pop(k,None)
    assert b==c

def test_a5_endpoint_candidate_differs_from_baseline_only_by_endpoint_and_identity():
    base=_load('configs/journal/a5_geocrack_imf_baseline.yaml');cand=_load('configs/journal/a5_geocrack_imf_endpoint_candidate.yaml')
    assert base['model']==cand['model'];assert base['train']==cand['train'];assert base['eval']==cand['eval']
    assert _strip_endpoint(cand['loss'])==base['loss']
    assert cand['loss']['endpoint_probability']==0.15 and cand['loss']['endpoint_sampling']=='stratified_disjoint'

def test_a2b_original_mismatch_control_differs_only_by_endpoint_and_identity():
    base=_load('configs/journal/a2b_hybrid_imf_mask_capacity_matched.yaml');orig=_load('configs/journal/a2b_original_mismatch_control.yaml')
    assert base['model']==orig['model'];assert base['train']==orig['train'];assert base['eval']==orig['eval']
    assert _strip_endpoint(base['loss'])==_strip_endpoint(orig['loss'])
    assert base['loss']['endpoint_probability']==0.15 and base['loss']['endpoint_sampling']=='stratified_disjoint'
    assert 'endpoint_probability' not in orig['loss']
