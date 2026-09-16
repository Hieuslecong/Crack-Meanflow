from pathlib import Path
import yaml
from crackmeanflow.common.option_c_provenance import validate_option_c_config
from crackmeanflow.common.option_c_schedule import resolve_option_c_budget

ROOT=Path(__file__).resolve().parents[1]
FILES={'J0':'j0_direct_mask.yaml','J1':'j1_centerline_edt_noncausal.yaml','J2':'j2_centerline_radius_causal.yaml','J3':'j3_centerline_radius_gic.yaml','J4':'j4_centerline_radius_gic_endpoint.yaml'}

def test_all_variant_configs_obey_design_lock():
    for variant,name in FILES.items():
        cfg=yaml.safe_load((ROOT/'configs'/'option_c_v1'/name).read_text());assert validate_option_c_config(cfg,variant)['status']=='PASS'

def test_short_run_does_not_change_research_scheduler_horizon():
    cfg=yaml.safe_load((ROOT/'configs'/'option_c_v1'/'j2_centerline_radius_causal.yaml').read_text());total,stop=resolve_option_c_budget(cfg['train'],stop_after_step=200);assert total==21000 and stop==200

def test_stop_cannot_exceed_research_horizon():
    cfg=yaml.safe_load((ROOT/'configs'/'option_c_v1'/'j2_centerline_radius_causal.yaml').read_text())
    try:resolve_option_c_budget(cfg['train'],stop_after_step=21001)
    except ValueError:pass
    else:raise AssertionError('expected fail-closed stop/horizon validation')
