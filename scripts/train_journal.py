from __future__ import annotations
import argparse,json,os,random,re,sys,time,warnings
import numpy as np, torch, yaml
from torch.utils.data import DataLoader
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crackmeanflow.common import PairedCrackDataset,discover_required_splits,manifest_hash,audit_group_integrity,calibrate_threshold_on_validation,make_warmup_cosine_scheduler,optimizer_steps_per_epoch,save_checkpoint_atomic,load_checkpoint,restore_rng_state,config_hash,EMA,discover_normal_images,append_normal_negatives,source_balancing_weights,EpochWeightedRandomSampler
from crackmeanflow.common.data import write_split_manifest
from crackmeanflow.adapter import CrackMeanFlowModel
from crackmeanflow.sit import build_sit
from crackmeanflow.sampler import crack_meanflow_sampler
from crackmeanflow.conference.losses.meanflow_loss import ConferenceMeanFlowLoss
from crackmeanflow.journal import build_geocrack_imf,GeometryRasterizer,ImprovedMeanFlowGeometryLoss,calibrate_geometry_threshold_on_validation
from crackmeanflow.journal.flow.improved_meanflow import ImprovedMeanFlowStateLoss
from crackmeanflow.journal.models.sit_mask_baseline import MaskIMFSiTModel,HybridMaskIMFModel
from crackmeanflow.journal.engine.dataset import GeometryDataset

def seed_all(s,deterministic=False):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    if deterministic:
        torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
        try: torch.use_deterministic_algorithms(True,warn_only=True)
        except Exception as exc: warnings.warn(f'could not enable deterministic algorithms: {exc}')

def _group_fn(pattern):
    rx=re.compile(pattern)
    def fn(stem):
        m=rx.search(stem)
        if not m: raise RuntimeError(f'group regex {pattern!r} did not match stem {stem!r}')
        return m.group(1) if m.groups() else m.group(0)
    return fn

def _with_optional_normals(data_root,splits,cfg):
    nc=cfg['train'].get('normal_negatives') or {}; out={k:list(v) for k,v in splits.items()}
    for split in ('train','val','test'):
        if bool(nc.get(split,False)):
            normals=discover_normal_images(data_root,split); out[split]=append_normal_negatives(out[split],normals); print(f'[normal-negatives] split={split} added={len(normals)}')
    return out

def _train_loader(ds,pairs,cfg,seed):
    sb=cfg['train'].get('source_balance') or {}; kw=dict(batch_size=cfg['train']['batch_size'],drop_last=cfg['train'].get('drop_last',True),num_workers=cfg['train'].get('num_workers',0))
    if sb.get('enabled',False):
        weights,counts=source_balancing_weights(pairs,pattern=sb.get('regex',r'^([^_]+)'),power=sb.get('power',0.5),cap_ratio=sb.get('cap_ratio',4.0)); sampler=EpochWeightedRandomSampler(weights,num_samples=len(ds),replacement=True,seed=seed); print('[source-balance]',json.dumps(counts,sort_keys=True)); return DataLoader(ds,sampler=sampler,**kw)
    return DataLoader(ds,shuffle=True,**kw)

def build_track(cfg,device):
    if cfg['backbone']=='geocrack_imf':
        model=build_geocrack_imf(cfg['model']).to(device); rast=GeometryRasterizer(cfg['model'].get('max_radius',16),cfg['model'].get('radius_bins',8),representation=cfg['model'].get('representation','centerline_radius'),distance_encoding=cfg['model'].get('distance_encoding','linear')).to(device); return model,rast,ImprovedMeanFlowGeometryLoss(**cfg['loss'],max_radius=cfg['model'].get('max_radius',16),rasterizer=rast)
    if cfg['backbone']=='sit_imf_mask': return MaskIMFSiTModel(cfg['model']['img_size'],cfg['model']['patch'],cfg['model']['size'],cfg['model'].get('background_init',-0.95)).to(device),None,ImprovedMeanFlowStateLoss(**cfg['loss'])
    if cfg['backbone']=='hybrid_imf_mask': return HybridMaskIMFModel(cfg['model']['img_size'],cfg['model']['patch'],cfg['model'].get('size','S'),cfg['model'].get('background_init',-0.95)).to(device),None,ImprovedMeanFlowStateLoss(**cfg['loss'])
    if cfg['backbone']=='sit_mf':
        sit=build_sit(cfg['model']['img_size'],cfg['model']['patch'],cfg['model']['size'],cfg['model'].get('in_ch',1),cfg['model'].get('cond_ch',3)); return CrackMeanFlowModel(sit,T=500).to(device),None,ConferenceMeanFlowLoss(**cfg['loss'])
    raise RuntimeError(f"unsupported journal ablation backbone={cfg['backbone']!r}")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--data',required=True); ap.add_argument('--out',default=None); ap.add_argument('--resume',default=None); ap.add_argument('--allow-config-change',action='store_true'); ap.add_argument('--group-regex',default=None); a=ap.parse_args(); cfg=yaml.safe_load(open(a.config))
    if cfg.get('track') not in {'journal','journal_ablation'}: raise RuntimeError(f"invalid Journal track={cfg.get('track')!r}")
    seed=cfg['train'].get('seed',42); seed_all(seed,cfg['train'].get('deterministic',False)); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); base_sp=discover_required_splits(a.data)
    if a.group_regex: audit_group_integrity(base_sp,_group_fn(a.group_regex))
    sp=_with_optional_normals(a.data,base_sp,cfg); hashes={k:manifest_hash(v) for k,v in sp.items()}; out=a.out or os.path.join('outputs',cfg['experiment']); os.makedirs(out,exist_ok=True); write_split_manifest(sp,os.path.join(out,'split_manifest.json'))
    base=lambda s,aug:PairedCrackDataset(sp[s],cfg['model']['img_size'],aug,aug and cfg['train'].get('photometric_augment',False),cfg['train'].get('mask_resize_mode','nearest'))
    if cfg['backbone']=='geocrack_imf': train_ds=GeometryDataset(base('train',True),cfg['model'].get('max_radius',16),cfg['model'].get('representation','centerline_radius'),cfg['model'].get('distance_encoding','linear')); val_ds=GeometryDataset(base('val',False),cfg['model'].get('max_radius',16),cfg['model'].get('representation','centerline_radius'),cfg['model'].get('distance_encoding','linear'))
    else: train_ds=base('train',True); val_ds=base('val',False)
    tr=_train_loader(train_ds,sp['train'],cfg,seed); va=DataLoader(val_ds,batch_size=1,shuffle=False,num_workers=cfg['train'].get('num_workers',0)); model,rast,lossfn=build_track(cfg,device); opt=torch.optim.AdamW(model.parameters(),lr=cfg['train']['lr'],weight_decay=cfg['train']['weight_decay']); steps=optimizer_steps_per_epoch(len(tr),cfg['train']['grad_accum_steps']); sched=make_warmup_cosine_scheduler(opt,cfg['train']['epochs'],steps,cfg['train'].get('warmup_epochs',0)); ema=EMA(model,cfg['train']['ema_decay'])
    best=-1.; best_th=None; gs=0; hist=[]; start_epoch=0; global_sample_offset=0
    if a.resume:
        ck=load_checkpoint(a.resume,model,opt,sched,map_location='cpu'); ema=EMA(model,cfg['train']['ema_decay'],ck.get('ema')); best=float(ck.get('best_val_metric',-1)); best_th=float(ck.get('best_val_threshold',.5)); gs=int(ck.get('global_optimizer_step',0)); start_epoch=int(ck['epoch'])+1; global_sample_offset=int((ck.get('extra_state') or {}).get('global_sample_offset',0)); restore_rng_state(ck.get('rng_state'))
    for ep in range(start_epoch,cfg['train']['epochs']):
        if hasattr(lossfn,'set_epoch'): lossfn.set_epoch(ep)
        if hasattr(tr.sampler,'set_epoch'): tr.sampler.set_epoch(ep)
        model.train(); opt.zero_grad(set_to_none=True); ls=[]; fm_samples=0; gic_samples=0; seen_samples=0
        for i,b in enumerate(tr):
            img=b['crack'].to(device); bs=int(img.shape[0]); seen_samples+=bs
            if cfg['backbone']=='geocrack_imf': loss,logs=lossfn(model,b['geometry'].to(device),img,b['radius_valid'].to(device),mask_gt=b['mask'].to(device),sample_offset=global_sample_offset)
            elif cfg['backbone'] in {'sit_imf_mask','hybrid_imf_mask'}: loss,logs=lossfn(model,b['mask'].to(device),img,sample_offset=global_sample_offset)
            else: loss,logs=lossfn(model,b['mask'].to(device),{'y':img,'sample_offset':global_sample_offset})
            global_sample_offset+=bs; fm_samples+=int(logs.get('fm_count',0)); gic_samples+=int(logs.get('gic_count',0)); (loss/cfg['train']['grad_accum_steps']).backward(); ls.append(logs['total_loss'])
            if (i+1)%cfg['train']['grad_accum_steps']==0 or i+1==len(tr):
                torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['train']['max_grad_norm']); opt.step(); sched.step(); opt.zero_grad(set_to_none=True); ema.update(model); gs+=1
        ema.apply(model)
        try:
            if cfg['backbone']=='geocrack_imf': sweep,th=calibrate_geometry_threshold_on_validation(model,va,device,rast,cfg['eval']['thresholds'],cfg['eval']['eval_seeds'][0],cfg['model'].get('max_radius',16)); ev=sweep[th]
            else: sweep,th=calibrate_threshold_on_validation(model,va,device,crack_meanflow_sampler,cfg['eval']['thresholds'],1,cfg['eval']['eval_seeds'][0],cfg['eval'].get('cfg_scale',1.0)); ev=sweep[th]
        finally: ema.restore(model)
        score=ev['f1']; improved=score>best
        if improved: best=score; best_th=float(th)
        common=dict(model=model,ema=ema,optimizer=opt,scheduler=sched,epoch=ep,global_optimizer_step=gs,cfg=cfg,best_val_metric=best,best_val_threshold=best_th,split_manifest_hashes=hashes,seed=seed,extra_state={'global_sample_offset':global_sample_offset}); save_checkpoint_atomic(os.path.join(out,'last.pt'),**common)
        if improved: save_checkpoint_atomic(os.path.join(out,'best.pt'),**common)
        hist.append({'epoch':ep,'loss':float(np.mean(ls)),'val_f1':score,'val_threshold':float(th),'optimizer_step':gs,'fm_samples':fm_samples,'gic_samples':gic_samples}); json.dump({'history':hist,'best_val_f1':best,'best_val_threshold':best_th},open(os.path.join(out,'history.json'),'w'),indent=2)
if __name__=='__main__': main()
