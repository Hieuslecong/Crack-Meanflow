from __future__ import annotations
import hashlib,json

BASE_V4_COMMIT='f873ec2351dbd07cacd84e7ee7394a9bd5527b07'
PROTOCOL_ID='OPTION_C_V1'
VARIANTS={
    'J0':{'backbone':'hybrid_imf_mask','representation':None,'gic':False,'endpoint':False},
    'J1':{'backbone':'geocrack_imf','representation':'centerline_edt','gic':False,'endpoint':False},
    'J2':{'backbone':'geocrack_imf','representation':'centerline_radius','gic':False,'endpoint':False},
    'J3':{'backbone':'geocrack_imf','representation':'centerline_radius','gic':True,'endpoint':False},
    'J4':{'backbone':'geocrack_imf','representation':'centerline_radius','gic':True,'endpoint':True},
}

def semantic_sha256(obj):
    blob=json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()
    return hashlib.sha256(blob).hexdigest()

def validate_option_c_config(cfg,variant):
    variant=str(variant).upper()
    if variant not in VARIANTS:raise ValueError(f'unknown Option-C variant={variant!r}')
    exp=VARIANTS[variant];train=cfg.get('train') or {};loss=cfg.get('loss') or {};model=cfg.get('model') or {};ev=cfg.get('eval') or {}
    errors=[]
    if cfg.get('option_c_variant')!=variant:errors.append('option_c_variant mismatch')
    if cfg.get('protocol_id')!=PROTOCOL_ID:errors.append('protocol_id must be OPTION_C_V1')
    if cfg.get('backbone')!=exp['backbone']:errors.append('backbone mismatch')
    if int(train.get('batch_size',0))*int(train.get('grad_accum_steps',0))!=8:errors.append('effective batch must be 8')
    if int(train.get('research_total_steps',0))!=21000:errors.append('research_total_steps must be 21000')
    if int(ev.get('num_steps',0))!=1:errors.append('NFE/eval.num_steps must be 1')
    if exp['representation'] is not None and model.get('representation')!=exp['representation']:errors.append('representation mismatch')
    gic=bool(float(loss.get('gic_weight',0))>0)
    endpoint=bool(float(loss.get('endpoint_probability',0))>0)
    if gic!=exp['gic']:errors.append('GIC state mismatch')
    if endpoint!=exp['endpoint']:errors.append('endpoint state mismatch')
    if variant=='J4' and loss.get('endpoint_sampling')!='stratified_disjoint':errors.append('J4 requires stratified_disjoint endpoint sampling')
    if variant in {'J1','J2','J3','J4'}:
        for key,val in [('size','S'),('patch',8),('img_size',256),('local_refine',True)]:
            if model.get(key)!=val:errors.append(f'J1-J4 shared backbone invariant failed: {key}')
    if errors:raise ValueError('; '.join(errors))
    return {'variant':variant,'status':'PASS','semantic_sha256':semantic_sha256(cfg)}
