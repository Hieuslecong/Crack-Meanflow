from __future__ import annotations
import argparse,gc,json,sys,traceback
from pathlib import Path
import torch,yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.train_journal import seed_all
from crackmeanflow.common import source_tree_hash,EMA
from crackmeanflow.factory import build_training_components

DEFAULTS={
 'Conference':'configs/conference/crackmeanflow_unet.yaml',
 'A2B_BASELINE':'configs/journal/a2b_original_mismatch_control.yaml',
 'A5_BASELINE':'configs/journal/a5_geocrack_imf_baseline.yaml',
 'A2B_ENDPOINT':'configs/journal/a2b_hybrid_imf_mask_capacity_matched.yaml',
 'A5_ENDPOINT':'configs/journal/a5_geocrack_imf_endpoint_candidate.yaml',
}

def _stress_sample_offset(cfg):
    """Choose a deterministic sample offset that exercises optional expensive branches.

    For GeoCrack configs with GIC>0, select an offset where at least one sample in
    the configured microbatch activates GIC, so VRAM preflight cannot accidentally
    measure only the cheaper no-GIC path.
    """
    loss=cfg.get('loss') or {}; b=int(cfg.get('train',{}).get('batch_size',1))
    if cfg.get('backbone')=='geocrack_imf' and float(loss.get('gic_weight',0))>0 and float(loss.get('gic_probability',0))>0:
        from crackmeanflow.journal.flow.improved_meanflow import _independent_stratified_mask,_stratified_mask
        p=float(loss['gic_probability']); mode=str(loss.get('gic_sampling','stratified'))
        for off in range(2000):
            if mode=='stratified_independent': m=_independent_stratified_mask(b,p,off,torch.device('cpu'),'gic')
            elif mode=='stratified': m=_stratified_mask(b,p,off,torch.device('cpu'))
            else: continue
            if bool(m.any()): return off
        raise RuntimeError('could not find deterministic GIC-active offset for GPU preflight')
    return 0

def _run(name,cfg,device,total_vram,max_reserved_fraction):
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(device);seed_all(int(cfg['train'].get('seed',42)),cfg['train'].get('deterministic',False),cfg['train'].get('deterministic_warn_only',False))
    sample_offset=_stress_sample_offset(cfg);result={'model':name,'backbone':cfg['backbone'],'image_size':int(cfg['model']['img_size']),'microbatch_size':int(cfg['train']['batch_size']),'stress_sample_offset':int(sample_offset)}
    try:
        model,rast,lossfn=build_training_components(cfg,device);model.train();b=int(cfg['train']['batch_size']);h=int(cfg['model']['img_size'])
        img=torch.rand(b,3,h,h,device=device);mask=(torch.rand(b,1,h,h,device=device)>.97).float()*2-1
        if cfg['backbone']=='geocrack_imf':
            from crackmeanflow.journal.geometry.targets import mask_to_geometry_state
            geom,rv=mask_to_geometry_state(mask,cfg['model'].get('max_radius',16),cfg['model'].get('representation','centerline_radius'),cfg['model'].get('distance_encoding','linear'));loss,logs=lossfn(model,geom,img,rv,mask_gt=mask,sample_offset=sample_offset)
        elif cfg['backbone'] in {'sit_imf_mask','hybrid_imf_mask'}:loss,logs=lossfn(model,mask,img,sample_offset=sample_offset)
        else:loss,logs=lossfn(model,mask,{'y':img,'sample_offset':sample_offset})
        # Include the persistent training-state footprint (AdamW moments + EMA shadow),
        # not only forward/backward activations. This is closer to real workstation use.
        opt=torch.optim.AdamW(model.parameters(),lr=float(cfg['train']['lr']),weight_decay=float(cfg['train']['weight_decay']));ema=EMA(model,float(cfg['train']['ema_decay']))
        loss.backward();finite=bool(torch.isfinite(loss).item()) and all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters());grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['train']['max_grad_norm'],error_if_nonfinite=True);opt.step();opt.zero_grad(set_to_none=True);ema.update(model)
        alloc=torch.cuda.max_memory_allocated(device);reserved=torch.cuda.max_memory_reserved(device);frac=reserved/max(total_vram,1)
        required_gic=cfg.get('backbone')=='geocrack_imf' and float((cfg.get('loss') or {}).get('gic_weight',0))>0 and float((cfg.get('loss') or {}).get('gic_probability',0))>0
        gic_exercised=(not required_gic) or int(logs.get('gic_active_samples',0))>0
        result.update({'loss':float(loss.detach()),'gic_active_samples':int(logs.get('gic_active_samples',0)),'gic_branch_required':required_gic,'gic_branch_exercised':gic_exercised,'exact_deployment_samples':int(logs.get('exact_deployment_count',logs.get('boundary_count',0))),'grad_finite':finite,'grad_norm':float(grad_norm.detach().cpu()),'optimizer_state_allocated':True,'ema_state_allocated':True,'peak_allocated_gib':alloc/2**30,'peak_reserved_gib':reserved/2**30,'peak_reserved_fraction_of_vram':frac,'memory_headroom_pass':frac<=max_reserved_fraction,'status':'PASS' if finite and frac<=max_reserved_fraction and h==256 and gic_exercised else 'FAIL'})
        del model,loss,img,mask
    except torch.cuda.OutOfMemoryError as exc:
        result.update({'grad_finite':False,'memory_headroom_pass':False,'status':'FAIL_OOM','error':str(exc)})
    except Exception as exc:
        result.update({'grad_finite':False,'memory_headroom_pass':False,'status':'FAIL_EXCEPTION','error':repr(exc),'traceback':traceback.format_exc(limit=5)})
    finally:
        gc.collect();torch.cuda.empty_cache()
    return result

def main():
    ap=argparse.ArgumentParser(description='Canonical 256x256 CUDA forward/backward + deterministic + peak-VRAM preflight. No dataset required.')
    ap.add_argument('--out',default='reports/GPU_PREFLIGHT.json');ap.add_argument('--configs',nargs='*',default=None);ap.add_argument('--max-reserved-fraction',type=float,default=.90);ap.add_argument('--require-device-substring',default=None);a=ap.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required for GPU preflight')
    if not 0<a.max_reserved_fraction<1:raise ValueError('--max-reserved-fraction must lie in (0,1)')
    root=Path(__file__).resolve().parents[1];device=torch.device('cuda:0');device_name=torch.cuda.get_device_name(device);total=torch.cuda.get_device_properties(device).total_memory
    if a.require_device_substring and a.require_device_substring.lower() not in device_name.lower():raise RuntimeError(f'GPU mismatch: required substring={a.require_device_substring!r}, actual={device_name!r}')
    cps=[root/x for x in DEFAULTS.values()] if not a.configs else [Path(x) for x in a.configs];names=list(DEFAULTS) if not a.configs else [Path(x).stem for x in a.configs]
    rows=[_run(name,yaml.safe_load(open(cp)),device,total,a.max_reserved_fraction) for name,cp in zip(names,cps)]
    report={'source_tree_sha256':source_tree_hash(),'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'device':device_name,'compute_capability':list(torch.cuda.get_device_capability(device)),'total_vram_gib':total/2**30,'max_reserved_fraction_allowed':a.max_reserved_fraction,'deterministic_algorithms_enabled':torch.are_deterministic_algorithms_enabled(),'results':rows,'pass':all(r.get('status')=='PASS' for r in rows)}
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    if not report['pass']:raise SystemExit(2)
if __name__=='__main__':main()
