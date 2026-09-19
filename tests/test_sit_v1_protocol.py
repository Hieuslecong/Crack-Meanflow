from pathlib import Path
import copy
import yaml
import torch

from crackmeanflow.common.sit_v1_provenance import validate_sit_v1_config, validate_sit_v1_pair
from crackmeanflow.common.sit_v1_schedule import resolve_sit_v1_budget, resolve_sit_v1_run_class, MILESTONE_STEPS
from crackmeanflow.journal.models.shared_sit_flow import SharedCrackSiT
from crackmeanflow.journal.losses.matched_sit_flow import MatchedSiTFlowLoss

ROOT=Path(__file__).resolve().parents[1]
FILES={'S0':'s0_sit_mf.yaml','S1':'s1_sit_imf.yaml'}

def _configs():
    return {k:yaml.safe_load((ROOT/'configs'/'sit_v1'/v).read_text()) for k,v in FILES.items()}

def test_sit_v1_pair_is_objective_only():
    assert validate_sit_v1_pair(_configs())['status']=='PASS'

def test_sit_v1_configs_obey_lock():
    for variant,cfg in _configs().items():
        assert validate_sit_v1_config(cfg,variant)['status']=='PASS'

def test_sit_v1_rejects_auxiliary_loss_creep():
    cfg=copy.deepcopy(_configs()['S1']);cfg['loss']['gic_weight']=0.1
    try:validate_sit_v1_config(cfg,'S1')
    except ValueError as exc:assert 'forbidden' in str(exc)
    else:raise AssertionError('GIC must be forbidden in SiT-V1')

def test_sit_v1_budget_and_run_classes_are_fail_closed():
    cfg=_configs()['S0'];total,stop=resolve_sit_v1_budget(cfg['train'],stop_after_step=16500)
    assert (total,stop)==(21000,16500)
    assert resolve_sit_v1_run_class('screen',stop,total)=='screen'
    assert tuple(cfg['train']['milestone_steps'])==MILESTONE_STEPS

def test_shared_sit_model_is_direct_mask_and_nfe_compatible():
    model=SharedCrackSiT(img_size=32,patch=8,dim=32,depth=2,heads=4)
    z=torch.randn(2,1,32,32);image=torch.randn(2,3,32,32);r=torch.zeros(2);t=torch.ones(2)
    out=model(z,r,t,y=image)
    assert out.shape==z.shape
    assert model.get_seg_logits() is None

def test_mf_and_imf_losses_share_model_and_have_finite_gradients():
    x0=torch.where(torch.rand(2,1,32,32)>.9,torch.ones(1),-torch.ones(1));image=torch.randn(2,3,32,32)
    param_counts=[]
    for mode in ('mf','imf'):
        torch.manual_seed(123)
        model=SharedCrackSiT(img_size=32,patch=8,dim=32,depth=2,heads=4)
        param_counts.append(sum(p.numel() for p in model.parameters()))
        loss,_=MatchedSiTFlowLoss(mode=mode)(model,x0,image,sample_offset=0)
        assert torch.isfinite(loss)
        loss.backward()
        grads=[p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
    assert param_counts[0]==param_counts[1]
