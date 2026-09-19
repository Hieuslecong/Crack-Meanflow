from __future__ import annotations
import hashlib
from pathlib import Path

PROTOCOL_ID='CRACKMEANFLOW_SIT_V1'
VARIANTS={'S0':'mf','S1':'imf'}
LOCKED_MODEL={'img_size':256,'patch':8,'dim':384,'depth':10,'heads':6,'mlp_ratio':4.0}
LOCKED_TRAIN={'epochs':200,'research_total_steps':21000,'milestone_steps':[4125,8250,12375,16500,21000],'batch_size':2,'grad_accum_steps':4,'drop_incomplete_accumulation':True,'lr':1e-4,'weight_decay':0.0,'adam_beta1':0.9,'adam_beta2':0.95,'lr_schedule':'constant','warmup_epochs':0,'ema_decay':0.9999,'max_grad_norm':1.0,'augment':True,'photometric_augment':True,'resize_policy':'stretch_square','mask_resize_mode':'nearest','mask_binarization':'auto_binary_safe','drop_last':True,'deterministic':True,'deterministic_warn_only':False,'num_workers':0}
ALLOWED_TRAINING_SEEDS={0,1,2}
LOCKED_EVAL={'num_steps':1,'eval_seeds':[0,1,2,3,4],'checkpoint_selection_seeds':[0,1,2],'checkpoint_use_final_threshold_grid':True,'batch_size':2,'checkpoint_validation_interval_epochs':5}
LOCKED_FINAL_THRESHOLD_GRID={'start':-2.0,'stop':2.0,'step':0.05}
LOCKED_THRESHOLDS=[-2.0,-1.6,-1.2,-1.0,-0.8,-0.6,-0.4,-0.2,0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.6,2.0]
LOCKED_LOSS_COMMON={'fm_fraction':0.5,'time_mu':-0.4,'time_sigma':1.0,'norm_p':1.0,'norm_eps':0.01,'fm_sampling':'stratified'}
ALLOWED_PAIR_DIFF={'sit_variant','experiment','loss.mode','protocol_role'}

def _flatten(d,prefix=''):
    out={}
    for k,v in d.items():
        p=f'{prefix}.{k}' if prefix else k
        if isinstance(v,dict): out.update(_flatten(v,p))
        else: out[p]=v
    return out

def _diff(a,b):
    aa,bb=_flatten(a),_flatten(b)
    return {k for k in aa.keys()|bb.keys() if aa.get(k)!=bb.get(k)}

def validate_sit_v1_config(cfg,variant=None):
    variant=variant or cfg.get('sit_variant')
    if variant not in VARIANTS: raise ValueError(f'unknown SiT-V1 variant={variant!r}')
    if cfg.get('protocol_id')!=PROTOCOL_ID: raise ValueError('protocol_id mismatch')
    if cfg.get('backbone')!='sit_shared_mask': raise ValueError('SiT-V1 requires backbone=sit_shared_mask')
    if cfg.get('track')!='journal': raise ValueError('SiT-V1 requires track=journal')
    for k,v in LOCKED_MODEL.items():
        if cfg.get('model',{}).get(k)!=v: raise ValueError(f'locked model field mismatch: model.{k}')
    tr=cfg.get('train',{})
    for k,v in LOCKED_TRAIN.items():
        if tr.get(k)!=v: raise ValueError(f'locked train field mismatch: train.{k}')
    if int(tr.get('seed',-1)) not in ALLOWED_TRAINING_SEEDS: raise ValueError(f'SiT-V1 seed must be one of {sorted(ALLOWED_TRAINING_SEEDS)}')
    normals=tr.get('normal_negatives') or {}
    if normals!={'train':False,'val':False,'test':False}: raise ValueError('SiT-V1 forbids source normal-negative augmentation')
    balance=tr.get('sample_balance') or {}
    expected_balance={'enabled':False,'regex':'^([^_]+)','power':0.5,'cap_ratio':4.0,'unit':'uniform_crop_without_replacement'}
    if balance!=expected_balance: raise ValueError('SiT-V1 sample_balance contract mismatch')
    if tr.get('parent_group_regex')!='^([^_]+)': raise ValueError('SiT-V1 parent_group_regex mismatch')
    if 'max_optimizer_steps' in tr: raise ValueError('SiT-V1 forbids legacy max_optimizer_steps')
    if 'runtime_batch_override' in tr: raise ValueError('SiT-V1 forbids runtime microbatch override')
    if int(tr['batch_size'])*int(tr['grad_accum_steps'])!=8: raise ValueError('effective batch must equal 8')
    loss=cfg.get('loss',{})
    if loss.get('mode')!=VARIANTS[variant]: raise ValueError('loss.mode does not match variant')
    for k,v in LOCKED_LOSS_COMMON.items():
        if loss.get(k)!=v: raise ValueError(f'locked loss field mismatch: loss.{k}')
    forbidden={'endpoint_probability','endpoint_loss_weight','thin_loss_weight','seg_loss_weight','clean_weight','gic_weight','geometry_weight','mask_weight','radius_weight'}
    present=sorted(k for k in forbidden if k in loss)
    if present: raise ValueError(f'forbidden SiT-V1 auxiliary loss fields: {present}')
    ev=cfg.get('eval',{})
    for k,v in LOCKED_EVAL.items():
        if ev.get(k)!=v: raise ValueError(f'locked eval field mismatch: eval.{k}')
    if ev.get('final_threshold_calibration_seeds')!=[0,1,2,3,4]: raise ValueError('final threshold calibration seeds mismatch')
    if ev.get('final_threshold_grid')!=LOCKED_FINAL_THRESHOLD_GRID: raise ValueError('final threshold grid mismatch')
    if ev.get('thresholds')!=LOCKED_THRESHOLDS: raise ValueError('threshold candidates mismatch')
    return {'status':'PASS','variant':variant,'objective':VARIANTS[variant]}

def validate_sit_v1_pair(configs):
    if set(configs)!={'S0','S1'}: raise ValueError('pair must contain exactly S0 and S1')
    for v,cfg in configs.items(): validate_sit_v1_config(cfg,v)
    diffs=_diff(configs['S0'],configs['S1'])
    if diffs!=ALLOWED_PAIR_DIFF: raise ValueError(f'S0/S1 config diff violates objective-only lock: {sorted(diffs)}')
    return {'status':'PASS','differences':sorted(diffs)}

def sit_v1_protocol_sha256(root=None):
    root=Path(root) if root else Path(__file__).resolve().parents[2]
    p=root/'configs'/'protocol'/'crackmeanflow_sit_v1.yaml'
    if not p.is_file(): raise RuntimeError(f'missing SiT-V1 protocol: {p}')
    return hashlib.sha256(p.read_bytes()).hexdigest()
