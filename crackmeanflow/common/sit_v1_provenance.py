from __future__ import annotations
import hashlib
from pathlib import Path

PROTOCOL_ID='CRACKMEANFLOW_SIT_V1'
VARIANTS={'S0':'mf','S1':'imf'}
LOCKED_MODEL={'img_size':256,'patch':8,'dim':384,'depth':10,'heads':6,'mlp_ratio':4.0,'background_init':-0.95}
LOCKED_TRAIN={'research_total_steps':21000,'milestone_steps':[4125,8250,12375,16500,21000],'batch_size':2,'grad_accum_steps':4,'lr':7.5e-5,'weight_decay':0.0,'warmup_epochs':10,'ema_decay':0.999,'max_grad_norm':1.0,'resize_policy':'stretch_square','mask_resize_mode':'nearest'}
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
    if int(cfg.get('eval',{}).get('num_steps',1))!=1: raise ValueError('SiT-V1 requires NFE=1')
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
