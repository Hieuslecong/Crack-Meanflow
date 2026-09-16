from pathlib import Path
import yaml
from crackmeanflow.common.option_c_provenance import validate_option_c_config,validate_option_c_ladder,option_c_design_lock_sha256
from crackmeanflow.common.option_c_schedule import resolve_option_c_budget,milestone_steps_from_config,CANONICAL_MILESTONE_STEPS

ROOT=Path(__file__).resolve().parents[1]
FILES={'J0':'j0_direct_mask.yaml','J1':'j1_centerline_edt_noncausal.yaml','J2':'j2_centerline_radius_causal.yaml','J3':'j3_centerline_radius_gic.yaml','J4':'j4_centerline_radius_gic_endpoint.yaml'}

def _configs():return {v:yaml.safe_load((ROOT/'configs'/'option_c_v1'/n).read_text()) for v,n in FILES.items()}

def test_all_variant_configs_obey_design_lock():
    for variant,cfg in _configs().items():assert validate_option_c_config(cfg,variant)['status']=='PASS'

def test_j1_to_j4_ladder_has_only_preregistered_differences():
    assert validate_option_c_ladder(_configs())['status']=='PASS'

def test_short_run_does_not_change_research_scheduler_horizon():
    cfg=_configs()['J2'];total,stop=resolve_option_c_budget(cfg['train'],stop_after_step=200);assert total==21000 and stop==200

def test_stop_cannot_exceed_research_horizon():
    cfg=_configs()['J2']
    try:resolve_option_c_budget(cfg['train'],stop_after_step=21001)
    except ValueError:pass
    else:raise AssertionError('expected fail-closed stop/horizon validation')

def test_milestones_are_exactly_preregistered():
    for cfg in _configs().values():assert milestone_steps_from_config(cfg['train'])==CANONICAL_MILESTONE_STEPS

def test_design_lock_is_bound_from_protocol_directory():
    assert len(option_c_design_lock_sha256(ROOT))==64
