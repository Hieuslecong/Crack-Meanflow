from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]

def test_two_paper_design_lock():
    c=yaml.safe_load((ROOT/'configs/conference/crackmeanflow_unet.yaml').read_text())
    a=yaml.safe_load((ROOT/'configs/journal/a5_geocrack_imf_baseline.yaml').read_text())
    b=yaml.safe_load((ROOT/'configs/journal/a2b_hybrid_imf_mask_capacity_matched.yaml').read_text())
    assert c['track']=='conference' and c['backbone']=='unet' and c['eval']['num_steps']==1
    assert a['track']=='journal' and a['backbone']=='geocrack_imf' and a['model']['representation']=='centerline_edt' and a['eval']['num_steps']==1
    assert b['backbone']=='hybrid_imf_mask' and b['eval']['num_steps']==1


def test_all_model_configs_are_nfe1():
    for p in (ROOT/'configs').rglob('*.yaml'):
        c=yaml.safe_load(p.read_text())
        if not isinstance(c, dict) or 'track' not in c or 'backbone' not in c:
            continue  # meta-config (e.g. fairness matrix), not an executable model config
        assert c['eval']['num_steps']==1, p
