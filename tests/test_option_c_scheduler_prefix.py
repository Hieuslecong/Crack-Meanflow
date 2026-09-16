import pytest
import torch
from crackmeanflow.common.scheduler import make_warmup_cosine_scheduler
from crackmeanflow.common.option_c_schedule import resolve_option_c_budget,resolve_option_c_run_class,milestone_checkpoint_name

def _trace(stop,total=21000):
    p=torch.nn.Parameter(torch.tensor(1.));opt=torch.optim.SGD([p],lr=1.);sched=make_warmup_cosine_scheduler(opt,epochs=200,optimizer_steps_epoch=825,warmup_epochs=10,total_optimizer_steps=total);vals=[]
    for _ in range(stop):opt.step();sched.step();vals.append(opt.param_groups[0]['lr'])
    return vals,sched

def test_200_step_diagnostic_is_exact_prefix_of_21k_schedule():
    total,stop=resolve_option_c_budget({'research_total_steps':21000},stop_after_step=200);short,s1=_trace(stop,total);long,s2=_trace(400,total);assert short==long[:200];assert s1._cmf_total_steps==s2._cmf_total_steps==21000;assert s1._cmf_warmup_steps==s2._cmf_warmup_steps==8250

def test_16500_screen_uses_same_21k_horizon():
    total,stop=resolve_option_c_budget({'research_total_steps':21000},stop_after_step=16500);assert total==21000 and stop==16500

def test_run_class_is_fail_closed_and_semantically_bound():
    with pytest.raises(ValueError):resolve_option_c_run_class(None,200,21000)
    assert resolve_option_c_run_class('diagnostic',200,21000)=='diagnostic'
    assert resolve_option_c_run_class('screen',16500,21000)=='screen'
    assert resolve_option_c_run_class('headline',21000,21000)=='headline'
    with pytest.raises(ValueError):resolve_option_c_run_class('diagnostic',16500,21000)
    with pytest.raises(ValueError):resolve_option_c_run_class('screen',200,21000)
    with pytest.raises(ValueError):resolve_option_c_run_class('headline',16500,21000)

def test_milestone_checkpoint_names_are_stable():
    assert milestone_checkpoint_name(4125)=='step_004125.pt'
    assert milestone_checkpoint_name(21000)=='step_021000.pt'
    with pytest.raises(ValueError):milestone_checkpoint_name(200)
