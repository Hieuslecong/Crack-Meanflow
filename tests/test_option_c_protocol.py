from pathlib import Path
import copy
import hashlib
import subprocess
import yaml
from crackmeanflow.common.option_c_provenance import validate_option_c_config,validate_option_c_ladder,option_c_design_lock_sha256,option_c_factorial_addendum_sha256
from crackmeanflow.common import protocol_bundle_hash,protocol_bundle_manifest
from crackmeanflow.common.option_c_schedule import resolve_option_c_budget,milestone_steps_from_config,CANONICAL_MILESTONE_STEPS

ROOT=Path(__file__).resolve().parents[1]
FILES={'J0':'j0_direct_mask.yaml','J1':'j1_centerline_edt_noncausal.yaml','J2':'j2_centerline_radius_causal.yaml','J2E':'j2e_centerline_radius_endpoint_only.yaml','J3':'j3_centerline_radius_gic.yaml','J4':'j4_centerline_radius_gic_endpoint.yaml'}

def _configs():return {v:yaml.safe_load((ROOT/'configs'/'option_c_v1'/n).read_text()) for v,n in FILES.items()}

def test_all_variant_configs_obey_design_lock():
    for variant,cfg in _configs().items():assert validate_option_c_config(cfg,variant)['status']=='PASS'

def test_j1_to_j4_ladder_has_only_preregistered_differences():
    assert validate_option_c_ladder(_configs())['status']=='PASS'

def test_j2e_is_registered_and_passes_config_validation():
    configs=_configs();assert 'J2E' in configs;assert validate_option_c_config(configs['J2E'],'J2E')['status']=='PASS'

def test_j2e_uses_endpoint_only_loss_contract():
    loss=_configs()['J2E']['loss'];assert loss['endpoint_probability']==0.15;assert loss['gic_weight']==0.0;assert loss['endpoint_sampling']=='stratified_disjoint'

def test_option_c_ladder_includes_factorial_addendum():
    assert validate_option_c_ladder(_configs())['status']=='PASS'

def test_j2e_rejects_random_endpoint_sampling():
    cfg=copy.deepcopy(_configs()['J2E']);cfg['loss']['endpoint_sampling']='random'
    try:validate_option_c_config(cfg,'J2E')
    except ValueError as exc:assert 'stratified_disjoint' in str(exc)
    else:raise AssertionError('J2E random endpoint sampling must fail closed')

def test_factorial_variants_have_symmetric_contracts():
    configs=_configs()
    for variant in ('J2','J2E','J3','J4'):assert configs[variant]['model']['representation']=='centerline_radius'
    assert configs['J2']['loss']['gic_weight']==0.0 and configs['J2E']['loss']['gic_weight']==0.0
    assert configs['J3']['loss']['gic_weight']==0.1 and configs['J4']['loss']['gic_weight']==0.1
    assert configs['J2']['loss']['endpoint_probability']==0.0 and configs['J3']['loss']['endpoint_probability']==0.0
    assert configs['J2E']['loss']['endpoint_probability']==0.15 and configs['J4']['loss']['endpoint_probability']==0.15

def _flatten(config,prefix=''):
    flattened={}
    for key,value in config.items():
        path=f'{prefix}.{key}' if prefix else key
        if isinstance(value,dict):flattened.update(_flatten(value,path))
        else:flattened[path]=value
    return flattened

def _difference_set(left,right):
    left_flat,right_flat=_flatten(left),_flatten(right)
    return {key for key in left_flat.keys()|right_flat.keys() if left_flat.get(key)!=right_flat.get(key)}

def test_factorial_config_differences_are_exactly_preregistered():
    configs=_configs()
    assert _difference_set(configs['J2'],configs['J2E'])=={'option_c_variant','experiment','loss.endpoint_probability','protocol_role'}
    assert _difference_set(configs['J2E'],configs['J4'])=={'option_c_variant','experiment','loss.gic_weight','protocol_role'}

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

def test_factorial_addendum_hash_is_sha256():
    digest=option_c_factorial_addendum_sha256(ROOT);assert len(digest)==64;assert all(char in '0123456789abcdef' for char in digest)

def test_factorial_addendum_is_bound_into_protocol_bundle():
    paths={row['path'] for row in protocol_bundle_manifest(ROOT)}
    assert 'configs/protocol/option_c_factorial_addendum_v1.yaml' in paths
    assert len(protocol_bundle_hash(ROOT))==64

def test_factorial_addendum_changes_bundle_hash_from_frozen_v1():
    frozen='fb2cf43d719941b41024fbb078c777c485b1a490'
    paths=subprocess.check_output(['git','ls-tree','-r','--name-only',frozen,'configs/protocol','configs/fairness'],cwd=ROOT,text=True).splitlines()
    original=hashlib.sha256()
    for path in sorted(path for path in paths if path.endswith('.yaml')):
        blob=subprocess.check_output(['git','rev-parse',f'{frozen}:{path}'],cwd=ROOT,text=True).strip()
        original.update(path.encode());original.update(b'\\0');original.update(blob.encode());original.update(b'\\n')
    assert protocol_bundle_hash(ROOT)!=original.hexdigest()

def test_runtime_microbatch_override_is_forbidden_for_option_c():
    cfg=copy.deepcopy(_configs()['J2'])
    cfg['train']['runtime_batch_override']={'original_batch_size':2,'original_grad_accum_steps':4,'effective_batch_size_preserved':True}
    try:validate_option_c_config(cfg,'J2')
    except ValueError as exc:assert 'runtime microbatch override is forbidden' in str(exc)
    else:raise AssertionError('Option-C runtime microbatch override must fail closed')
