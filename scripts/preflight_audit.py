from __future__ import annotations
import argparse,json,re,sys
from pathlib import Path
import torch,yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from crackmeanflow.common import build_dataset_identity,audit_group_integrity,PairedCrackDataset,source_splits_for_config,audit_content_split_integrity,source_tree_hash,audit_mask_encoding
from crackmeanflow.factory import build_training_components

CANON={
    'Conference':'configs/conference/crackmeanflow_unet.yaml',
    'A2B_BASELINE':'configs/journal/a2b_original_mismatch_control.yaml',
    'A5_BASELINE':'configs/journal/a5_geocrack_imf_baseline.yaml',
    'A2B_ENDPOINT':'configs/journal/a2b_hybrid_imf_mask_capacity_matched.yaml',
    'A5_ENDPOINT':'configs/journal/a5_geocrack_imf_endpoint_candidate.yaml',
    'A5_ENDPOINT_NO_GIC':'configs/journal/a5_geocrack_imf_no_gic_control.yaml',
}

def _group_fn(pattern):
    rx=re.compile(pattern)
    def f(stem):
        if stem.startswith('NORMAL::'): return 'NORMAL'
        m=rx.search(stem)
        if not m:raise RuntimeError(f'group regex does not match {stem}')
        return m.group(1) if m.groups() else m.group(0)
    return f

def _finite_grads(model):return all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)

def _validate_cfg(name,cfg):
    errs=[];tr=cfg.get('train') or {};ev=cfg.get('eval') or {};loss=cfg.get('loss') or {}
    if ev.get('num_steps')!=1:errs.append('eval.num_steps must equal 1')
    if tr.get('mask_binarization')!='auto_binary_safe':errs.append('mask_binarization must be auto_binary_safe')
    if tr.get('resize_policy')!='stretch_square':errs.append('resize_policy must be stretch_square')
    if not tr.get('deterministic',False):errs.append('deterministic must be true')
    bal=tr.get('sample_balance') or {}
    if bal.get('enabled'):errs.append('canonical CFD configs must use uniform no-replacement sampling; parent-balanced replacement is sensitivity-only')
    if bal.get('unit')!='uniform_crop_without_replacement':errs.append('canonical sample_balance.unit must be uniform_crop_without_replacement')
    if 'source_balance' in tr:errs.append('legacy source_balance must not appear in canonical configs')
    if name in {'A2B_ENDPOINT','A5_ENDPOINT','A5_ENDPOINT_NO_GIC'}:
        if float(loss.get('endpoint_probability',-1))!=0.15 or loss.get('endpoint_sampling')!='stratified_disjoint':errs.append('endpoint-aware 15% deployment coverage is required for endpoint candidate/control')
    if name in {'A2B_BASELINE','A5_BASELINE'} and 'endpoint_probability' in loss: errs.append(f'{name} must not silently include the endpoint candidate intervention')
    if errs:raise RuntimeError(f'{name} config contract failed: {errs}')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default=None);ap.add_argument('--out',default='reports/PREFLIGHT_AUDIT.json');a=ap.parse_args();root=Path(__file__).resolve().parents[1];device=torch.device('cpu')
    report={'source_tree_sha256':source_tree_hash(),'checks':{},'gates':{},'warnings':[]}
    cfgs={k:yaml.safe_load((root/v).read_text()) for k,v in CANON.items()}
    for k,c in cfgs.items():_validate_cfg(k,c)
    report['checks']['design_lock']={k:{'track':c['track'],'backbone':c['backbone'],'nfe':c['eval']['num_steps'],'protocol_role':c.get('protocol_role'),'endpoint_probability':(c.get('loss') or {}).get('endpoint_probability',(c.get('loss') or {}).get('boundary_prob'))} for k,c in cfgs.items()}
    # Build everything through the one canonical factory. Use 32px copies only for CPU smoke.
    smoke={};params={}
    for name,cfg0 in cfgs.items():
        cfg=yaml.safe_load(yaml.safe_dump(cfg0)); cfg['model']['img_size']=32
        model,rast,lossfn=build_training_components(cfg,device);params[name]=sum(p.numel() for p in build_training_components(cfg0,device)[0].parameters())
        model.train();img=torch.rand(1,3,32,32);mask=torch.where(torch.rand(1,1,32,32)>.9,torch.tensor(1.),torch.tensor(-1.))
        if cfg['backbone']=='geocrack_imf':
            from crackmeanflow.journal.geometry.targets import mask_to_geometry_state
            g,rv=mask_to_geometry_state(mask, cfg['model'].get('max_radius',16), cfg['model'].get('representation','centerline_radius'), cfg['model'].get('distance_encoding','linear'))
            loss,logs=lossfn(model,g,img,rv,mask_gt=mask,sample_offset=0)
        elif cfg['backbone'] in {'sit_imf_mask','hybrid_imf_mask'}: loss,logs=lossfn(model,mask,img,sample_offset=0)
        else: loss,logs=lossfn(model,mask,{'y':img,'sample_offset':0})
        loss.backward();smoke[name]={'loss':float(loss.detach()),'grad_finite':_finite_grads(model)}
        if not torch.isfinite(loss) or not smoke[name]['grad_finite']:raise RuntimeError(f'{name} synthetic forward/backward is non-finite')
    report['checks']['parameter_counts_256']=params;report['checks']['synthetic_forward_backward_32']=smoke
    # Scientific claim guards.
    from crackmeanflow.journal.geometry import GeometryRasterizer
    rast=GeometryRasterizer(16,8,representation='centerline_edt',distance_encoding='sqrt');state=torch.rand(2,2,16,16)*2-1;other=state.clone();other[:,0]=torch.rand_like(other[:,0])*2-1;diff=float((rast(state)-rast(other)).abs().max())
    report['checks']['centerline_edt_direct_mask_causality_max_abs_diff']=diff
    if diff==0:report['warnings'].append('centerline_edt centerline is auxiliary/shared-representation supervision only; do not claim direct mask rasterization causality.')
    report['warnings'].append('GIC is transport/interval consistency with the same RGB condition; do not claim appearance invariance without a separate mechanism/experiment.')
    report['gates'].update({'Code_contract':'PASS','Conference_runnable':'PASS','Journal_runnable':'PASS','NFE1_static':'PASS','Fairness_protocol_defined':'PASS'})
    if a.data:
        # All canonical methods must reconstruct the same source identity under current settings.
        ids=[]
        for name,cfg in cfgs.items():
            sp=source_splits_for_config(a.data,cfg);audit_content_split_integrity(sp);rx=cfg['train'].get('parent_group_regex');
            if rx:audit_group_integrity(sp,_group_fn(rx))
            ids.append((name,build_dataset_identity(sp,include_rows=False)))
        canon=json.dumps(ids[0][1],sort_keys=True)
        if any(json.dumps(x[1],sort_keys=True)!=canon for x in ids[1:]):raise RuntimeError('canonical models do not see identical source split identity')
        sp=source_splits_for_config(a.data,cfgs['Conference']);sample=PairedCrackDataset(sp['train'][:1],256,False,False,'nearest','auto_binary_safe')[0]
        mask_audit={k:audit_mask_encoding(v) for k,v in sp.items()}
        ambiguous={k:v['mask_encoding_convention'] for k,v in mask_audit.items() if v['mask_encoding_convention'] not in {'binary_0_1','binary_0_255','empty-mask-files-only'}}
        if ambiguous: raise RuntimeError(f'ambiguous source mask encoding: {ambiguous}')
        report['checks']['dataset_identity']=ids[0][1];report['checks']['source_mask_encoding']=mask_audit;report['checks']['loader_contract']={'image_shape':list(sample['crack'].shape),'mask_shape':list(sample['mask'].shape),'finite':bool(torch.isfinite(sample['crack']).all() and torch.isfinite(sample['mask']).all()),'mask_values':sorted(float(x) for x in torch.unique(sample['mask']))}
        report['gates'].update({'Data_integrity':'PASS','Content_provenance':'PASS','Parent_group_leakage':'PASS'})
    else:report['gates'].update({'Data_integrity':'NOT_RUN','Content_provenance':'NOT_RUN','Parent_group_leakage':'NOT_RUN'})
    report['gates']['GPU_256']='NOT_RUN_WORKSTATION';report['gates']['Micro_overfit']='NOT_RUN_WORKSTATION';report['gates']['Short_CFD']='NOT_RUN_WORKSTATION'
    report['code_ready_for_workstation']=all(v=='PASS' for k,v in report['gates'].items() if k not in {'GPU_256','Micro_overfit','Short_CFD'})
    report['full_paper_training']='NO-GO_UNTIL_GPU_MICRO_SHORT_PASS'
    out=Path(a.out);out=root/out if not out.is_absolute() else out;out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2))
    md=out.with_suffix('.md');md.write_text('# PRE-FLIGHT AUDIT\n\n'+f"- Code ready for workstation: **{report['code_ready_for_workstation']}**\n- Full paper training: **{report['full_paper_training']}**\n\n## Gates\n"+'\n'.join(f'- {k}: **{v}**' for k,v in report['gates'].items())+'\n\n## Warnings\n'+'\n'.join(f'- {w}' for w in report['warnings'])+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
