import yaml
from pathlib import Path
from scripts.gpu_preflight import _stress_sample_offset
from crackmeanflow.journal.flow.improved_meanflow import _independent_stratified_mask
import torch
ROOT=Path(__file__).resolve().parents[1]

def test_gpu_preflight_selects_gic_active_offset_for_a5():
    cfg=yaml.safe_load((ROOT/'configs/journal/a5_geocrack_imf_baseline.yaml').read_text())
    off=_stress_sample_offset(cfg); b=int(cfg['train']['batch_size']);p=float(cfg['loss']['gic_probability'])
    m=_independent_stratified_mask(b,p,off,torch.device('cpu'),'gic')
    assert bool(m.any())

def test_gpu_preflight_no_gic_control_needs_no_special_offset():
    cfg=yaml.safe_load((ROOT/'configs/journal/a5_geocrack_imf_no_gic_control.yaml').read_text())
    assert _stress_sample_offset(cfg)==0
