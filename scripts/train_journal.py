from __future__ import annotations
import argparse,json,os,random,re,sys,time,warnings
import numpy as np,torch,yaml
from torch.utils.data import DataLoader
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crackmeanflow.common import (
    PairedCrackDataset,discover_required_splits,audit_group_integrity,
    calibrate_threshold_on_validation,make_warmup_cosine_scheduler,optimizer_steps_per_epoch,
    save_checkpoint_atomic,load_checkpoint,restore_rng_state,config_hash,source_tree_hash,protocol_bundle_hash,environment_info,EMA,
    discover_normal_images,append_normal_negatives,source_balancing_weights,EpochRandomSampler,EpochWeightedRandomSampler,
    build_dataset_identity,write_split_manifest,verify_source_dataset_contract,resolve_thresholds,source_splits_for_config,audit_content_split_integrity,
)
from crackmeanflow.adapter import CrackMeanFlowModel
from crackmeanflow.sit import build_sit
from crackmeanflow.sampler import crack_meanflow_sampler
from crackmeanflow.conference.models import build_conference_model
from crackmeanflow.conference.losses import ConferenceMeanFlowLoss
from crackmeanflow.journal import build_geocrack_imf,GeometryRasterizer,ImprovedMeanFlowGeometryLoss,calibrate_geometry_threshold_on_validation
from crackmeanflow.journal.flow.improved_meanflow import ImprovedMeanFlowStateLoss
from crackmeanflow.journal.models.sit_mask_baseline import MaskIMFSiTModel,HybridMaskIMFModel
from crackmeanflow.journal.engine.dataset import GeometryDataset
from crackmeanflow.factory import build_training_components

def seed_all(s,deterministic=False,deterministic_warn_only=False):
    s=int(s); os.environ.setdefault('PYTHONHASHSEED',str(s))
    if deterministic: os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
    if deterministic:
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms(True,warn_only=bool(deterministic_warn_only))

def _seed_worker(worker_id):
    del worker_id
    worker_seed=torch.initial_seed()%2**32
    random.seed(worker_seed);np.random.seed(worker_seed)

def _set_loader_epoch(loader,epoch,seed):
    """Make both sample ordering and DataLoader worker seeding an epoch function.

    This preserves epoch-boundary resume determinism even when num_workers>0.
    """
    if hasattr(loader.sampler,'set_epoch'): loader.sampler.set_epoch(int(epoch))
    gen=getattr(loader,'generator',None)
    if gen is not None: gen.manual_seed(int(seed)+1000003*int(epoch))

def _group_fn(pattern):
    rx=re.compile(pattern)
    def fn(stem):
        m=rx.search(stem)
        if not m:raise RuntimeError(f'group regex {pattern!r} did not match stem {stem!r}')
        return m.group(1) if m.groups() else m.group(0)
    return fn

def _with_optional_normals(data_root,splits,cfg):
    nc=cfg['train'].get('normal_negatives') or {};out={k:list(v) for k,v in splits.items()}
    for split in ('train','val','test'):
        if bool(nc.get(split,False)):
            normals=discover_normal_images(data_root,split);out[split]=append_normal_negatives(out[split],normals);print(f'[normal-negatives] split={split} added={len(normals)}')
    return out

def _train_loader(ds,pairs,cfg,seed):
    trcfg=cfg['train']; balance=trcfg.get('sample_balance')
    if balance is None and trcfg.get('source_balance') is not None:
        warnings.warn('train.source_balance is legacy/ambiguous; use train.sample_balance with explicit unit',RuntimeWarning)
        balance=trcfg.get('source_balance')
    balance=balance or {}
    g=torch.Generator().manual_seed(int(seed))
    kw=dict(batch_size=trcfg['batch_size'],drop_last=trcfg.get('drop_last',True),num_workers=trcfg.get('num_workers',0),worker_init_fn=_seed_worker,generator=g)
    if balance.get('enabled',False):
        unit=balance.get('unit')
        if not unit: raise RuntimeError('sample_balance.enabled=true requires an explicit unit (e.g. parent_group)')
        weights,counts=source_balancing_weights(pairs,pattern=balance.get('regex',r'^([^_]+)'),power=balance.get('power',.5),cap_ratio=balance.get('cap_ratio',4.))
        sampler=EpochWeightedRandomSampler(weights,num_samples=len(ds),replacement=True,seed=seed)
        print('[sample-balance]',json.dumps({'unit':unit,'counts':counts},sort_keys=True))
        return DataLoader(ds,sampler=sampler,**kw)
    sampler=EpochRandomSampler(len(ds),seed=seed)
    return DataLoader(ds,sampler=sampler,**kw)

def build_track(cfg,device):
    return build_training_components(cfg,device)

def _select_checkpoint_metric(cfg,model,loader,device,rasterizer):
    """Select checkpoint on source validation using a preregistered ensemble of inference-noise seeds.

    This reduces checkpoint-selection variance without using any target-domain data.
    The same seed set and threshold policy is applied to Conference/A2B/A5.
    """
    evcfg=cfg.get('eval') or {}
    seeds=evcfg.get('checkpoint_selection_seeds')
    if seeds is None:
        seeds=[int(evcfg.get('checkpoint_selection_seed',(evcfg.get('eval_seeds') or [0])[0]))]
    seeds=[int(x) for x in seeds]
    if not seeds or len(seeds)!=len(set(seeds)): raise RuntimeError('checkpoint_selection_seeds must be non-empty and unique')
    thresholds=resolve_thresholds(evcfg,final=bool(evcfg.get('checkpoint_use_final_threshold_grid',False)))
    sweeps=[]
    for seed in seeds:
        if cfg['backbone']=='geocrack_imf':
            sweep,_=calibrate_geometry_threshold_on_validation(model,loader,device,rasterizer,thresholds,seed,cfg['model'].get('max_radius',16))
        else:
            sweep,_=calibrate_threshold_on_validation(model,loader,device,crack_meanflow_sampler,thresholds,1,seed,evcfg.get('cfg_scale',1.0))
        sweeps.append(sweep)
    mean_f1={float(t):float(np.mean([float(sw[float(t)]['f1']) for sw in sweeps])) for t in thresholds}
    # Stable preregistered tie-break: choose the lowest threshold on an exact mean-F1 tie.
    best_value=max(mean_f1.values()); th=min(t for t in sorted(mean_f1) if mean_f1[t]==best_value)
    return {'f1':mean_f1[th],'per_seed_f1':[float(sw[th]['f1']) for sw in sweeps],'selection_seeds':seeds},float(th)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--data',required=True);ap.add_argument('--dataset-name',default='CFD');ap.add_argument('--dataset-version',required=True);ap.add_argument('--out',default=None);ap.add_argument('--resume',default=None);ap.add_argument('--allow-config-change',action='store_true');ap.add_argument('--group-regex',default=None);ap.add_argument('--max-optimizer-steps',type=int,default=None,help='exact matched-budget stop; recorded into effective config');ap.add_argument('--seed',type=int,default=None,help='override training seed; recorded into EFFECTIVE_CONFIG.yaml');a=ap.parse_args()
    cfg=yaml.safe_load(open(a.config));
    if a.max_optimizer_steps is not None:
        if a.max_optimizer_steps<1: raise ValueError('--max-optimizer-steps must be >=1')
        cfg.setdefault('train',{})['max_optimizer_steps']=int(a.max_optimizer_steps)
    if a.seed is not None:
        cfg.setdefault('train',{})['seed']=int(a.seed)
    if a.group_regex is not None:
        cfg.setdefault('train',{})['parent_group_regex']=str(a.group_regex)
    if not str(a.dataset_name).strip() or not str(a.dataset_version).strip(): raise ValueError('dataset name/version must be non-empty provenance labels')
    track=cfg.get('track')
    if track not in {'conference','journal','journal_ablation'}:raise RuntimeError(f'invalid track={track!r}')
    if int(cfg.get('eval',{}).get('num_steps',1))!=1:raise RuntimeError('headline/research training configs must declare eval.num_steps=1')
    if cfg['train'].get('resize_policy','stretch_square')!='stretch_square': raise RuntimeError('only resize_policy=stretch_square is implemented in the canonical pipeline; use diagnostics for alternatives')
    seed=cfg['train'].get('seed',42);seed_all(seed,cfg['train'].get('deterministic',False),cfg['train'].get('deterministic_warn_only',False));device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_sp=discover_required_splits(a.data);group_status='UNVERIFIED';group_regex=a.group_regex or cfg['train'].get('parent_group_regex')
    if group_regex:audit_group_integrity(base_sp,_group_fn(group_regex));group_status='PASS'
    sp=source_splits_for_config(a.data,cfg);content_leakage_audit=audit_content_split_integrity(sp);identities=build_dataset_identity(sp,include_rows=False);name_hashes={k:v['name_manifest_sha256'] for k,v in identities.items()};content_hashes={k:v['content_manifest_sha256'] for k,v in identities.items()}
    source_provenance={'dataset_name':a.dataset_name,'dataset_version':a.dataset_version,'splits':identities,'parent_group_audit':group_status,'group_regex':group_regex,'content_leakage_audit':content_leakage_audit}
    out=a.out or os.path.join('outputs',cfg['experiment']);os.makedirs(out,exist_ok=True);yaml.safe_dump(cfg,open(os.path.join(out,'EFFECTIVE_CONFIG.yaml'),'w'),sort_keys=False);write_split_manifest(sp,os.path.join(out,'dataset_manifest.json'),include_content_rows=True)
    base=lambda s,aug:PairedCrackDataset(sp[s],cfg['model']['img_size'],aug,aug and cfg['train'].get('photometric_augment',False),cfg['train'].get('mask_resize_mode','nearest'),cfg['train'].get('mask_binarization','auto_binary_safe'))
    if cfg['backbone']=='geocrack_imf':train_ds=GeometryDataset(base('train',True),cfg['model'].get('max_radius',16),cfg['model'].get('representation','centerline_radius'),cfg['model'].get('distance_encoding','linear'));val_ds=GeometryDataset(base('val',False),cfg['model'].get('max_radius',16),cfg['model'].get('representation','centerline_radius'),cfg['model'].get('distance_encoding','linear'))
    else:train_ds=base('train',True);val_ds=base('val',False)
    tr=_train_loader(train_ds,sp['train'],cfg,seed);va=DataLoader(val_ds,batch_size=int(cfg.get('eval',{}).get('batch_size',1)),shuffle=False,num_workers=cfg['train'].get('num_workers',0))
    if len(tr)==0:raise RuntimeError('training loader is empty; reduce batch_size or disable drop_last')
    if bool(cfg['train'].get('drop_incomplete_accumulation',True)) and len(tr)<int(cfg['train']['grad_accum_steps']):raise RuntimeError('training loader has fewer batches than grad_accum_steps under drop_incomplete_accumulation=true')
    model,rast,lossfn=build_track(cfg,device);opt=torch.optim.AdamW(model.parameters(),lr=cfg['train']['lr'],weight_decay=cfg['train']['weight_decay']);drop_incomplete_accum=bool(cfg['train'].get('drop_incomplete_accumulation',True));steps=optimizer_steps_per_epoch(len(tr),cfg['train']['grad_accum_steps'],drop_incomplete_accum);max_steps=cfg['train'].get('max_optimizer_steps');planned_steps=min(int(steps)*int(cfg['train']['epochs']),int(max_steps)) if max_steps is not None else int(steps)*int(cfg['train']['epochs']);sched=make_warmup_cosine_scheduler(opt,cfg['train']['epochs'],steps,cfg['train'].get('warmup_epochs',0),total_optimizer_steps=planned_steps);ema=EMA(model,cfg['train']['ema_decay'])
    nominal_b=int(cfg['train']['batch_size']);ga=int(cfg['train']['grad_accum_steps']);loader_samples=min(len(train_ds),len(tr)*nominal_b) if cfg['train'].get('drop_last',True) else len(train_ds);usable_samples=int(steps)*nominal_b*ga if drop_incomplete_accum else loader_samples
    fairness={'samples_train':len(train_ds),'batch_size':nominal_b,'grad_accum_steps':ga,'effective_batch_size_nominal':nominal_b*ga,'optimizer_steps_per_epoch':int(steps),'planned_epochs':int(cfg['train']['epochs']),'planned_optimizer_steps':planned_steps,'max_optimizer_steps':int(max_steps) if max_steps is not None else None,'budget_protocol':'MATCHED_OPTIMIZER_STEPS' if max_steps is not None else 'BEST_ACHIEVABLE_EPOCHS','nfe':1,'drop_incomplete_accumulation':drop_incomplete_accum,'dropped_microbatches_per_epoch':(len(tr)%ga) if drop_incomplete_accum else 0,'loader_samples_per_epoch':int(loader_samples),'usable_samples_per_epoch':int(usable_samples),'total_omitted_samples_per_epoch':int(max(0,len(train_ds)-usable_samples)),'sample_coverage_fraction_per_epoch':float(usable_samples/max(len(train_ds),1)),'effective_samples_per_full_optimizer_step':nominal_b*ga,'optimizer_recipe':{'lr':float(cfg['train']['lr']),'weight_decay':float(cfg['train']['weight_decay']),'warmup_epochs':int(cfg['train'].get('warmup_epochs',0)),'warmup_optimizer_steps':int(cfg['train'].get('warmup_epochs',0))*int(steps)},'sampling_protocol':cfg['train'].get('sample_balance') or cfg['train'].get('source_balance') or {'enabled':False},'deterministic_requested':bool(cfg['train'].get('deterministic',False)),'checkpoint_selection_seeds':[int(x) for x in cfg.get('eval',{}).get('checkpoint_selection_seeds',[])],'checkpoint_validation_interval_epochs':int(cfg.get('eval',{}).get('checkpoint_validation_interval_epochs',1)),'eval_batch_size':int(cfg.get('eval',{}).get('batch_size',1))}
    best=-1.;best_th=None;gs=0;hist=[];start_epoch=0;global_sample_offset=0;resume_config_mismatch=False
    if a.resume:
        resume_meta=torch.load(a.resume,map_location='cpu',weights_only=False)
        verify_source_dataset_contract(resume_meta,sp,a.dataset_name,a.dataset_version,keys=('train','val','test'))
        saved_tree=resume_meta.get('source_tree_sha256')
        current_tree=source_tree_hash()
        if not saved_tree: raise RuntimeError('resume checkpoint has no source_tree_sha256; provenance is insufficient for scientific resume')
        if saved_tree!=current_tree: raise RuntimeError(f'resume source-tree mismatch: checkpoint={saved_tree} current={current_tree}; start a new run instead of resuming across code changes')
        saved_protocol=resume_meta.get('protocol_bundle_sha256');current_protocol=protocol_bundle_hash()
        if not saved_protocol: raise RuntimeError('resume checkpoint has no protocol_bundle_sha256; scientific protocol provenance is insufficient')
        if saved_protocol!=current_protocol: raise RuntimeError(f'resume protocol-bundle mismatch: checkpoint={saved_protocol} current={current_protocol}; start a new run under the new protocol')
        ck=load_checkpoint(a.resume,model,opt,sched,map_location='cpu')
        resume_config_mismatch=bool(ck.get('config_hash') and ck['config_hash']!=config_hash(cfg))
        if resume_config_mismatch and not a.allow_config_change:raise RuntimeError('resume config differs from checkpoint')
        extra=ck.get('extra_state') or {}
        if extra.get('epoch_complete') is False:
            raise RuntimeError('scientific resume from a partial-epoch budget checkpoint is prohibited; start a new run or choose a budget aligned to full optimizer-accumulation epochs')
        ema=EMA(model,cfg['train']['ema_decay'],ck.get('ema'));best=float(ck.get('best_val_metric',-1));_saved_best_th=ck.get('best_val_threshold');best_th=None if _saved_best_th is None else float(_saved_best_th);gs=int(ck.get('global_optimizer_step',0));start_epoch=int(ck['epoch'])+1;global_sample_offset=int(extra.get('global_sample_offset',0));restore_rng_state(ck.get('rng_state'));hp=os.path.join(out,'history.json')
        if os.path.isfile(hp):hist=json.load(open(hp)).get('history',[])
    json.dump({'source_provenance':source_provenance,'fairness':fairness,'config_hash':config_hash(cfg),'resume_requested':bool(a.resume),'resume_config_mismatch':resume_config_mismatch,'source_tree_sha256':source_tree_hash(),'protocol_bundle_sha256':protocol_bundle_hash(),'environment':environment_info(),'scientific_validity':'TAINTED_RESUME_CONFIG_CHANGE' if resume_config_mismatch else 'VALID_CONFIG'},open(os.path.join(out,'RUN_IDENTITY.json'),'w'),indent=2)
    json.dump(environment_info(),open(os.path.join(out,'ENVIRONMENT.json'),'w'),indent=2)
    for ep in range(start_epoch,cfg['train']['epochs']):
        if gs>=planned_steps: break
        if hasattr(lossfn,'set_epoch'):lossfn.set_epoch(ep)
        _set_loader_epoch(tr,ep,seed)
        model.train();opt.zero_grad(set_to_none=True);ls=[];t0=time.time();stop_budget=False;last_batch_index=-1;fm_samples=gic_active_samples=gic_active_batches=seen_samples=seen_batches=near_deploy=exact_deploy=0
        for i,b in enumerate(tr):
            if drop_incomplete_accum and i >= steps*int(cfg['train']['grad_accum_steps']): break
            last_batch_index=i;img=b['crack'].to(device);bs=int(img.shape[0]);seen_samples+=bs;seen_batches+=1
            if cfg['backbone']=='geocrack_imf':loss,logs=lossfn(model,b['geometry'].to(device),img,b['radius_valid'].to(device),mask_gt=b['mask'].to(device),sample_offset=global_sample_offset)
            elif cfg['backbone'] in {'sit_imf_mask','hybrid_imf_mask'}:loss,logs=lossfn(model,b['mask'].to(device),img,sample_offset=global_sample_offset)
            else:loss,logs=lossfn(model,b['mask'].to(device),{'y':img,'sample_offset':global_sample_offset})
            if not bool(torch.isfinite(loss).all()): raise FloatingPointError(f'non-finite training loss at epoch={ep} batch={i} optimizer_step={gs}')
            global_sample_offset+=bs;fm_samples+=int(logs.get('fm_count',0));gic_active_samples+=int(logs.get('gic_active_samples',0));gic_active_batches+=int(logs.get('gic_active_batches',0));near_deploy+=int(logs.get('near_deployment_count',logs.get('boundary_count',0)));exact_deploy+=int(logs.get('exact_deployment_count',logs.get('boundary_count',0)));ga=int(cfg['train']['grad_accum_steps']);group_start=(i//ga)*ga;group_size=ga if drop_incomplete_accum else min(ga,len(tr)-group_start);(loss/group_size).backward();ls.append(float(logs['total_loss']))
            if (i+1)%cfg['train']['grad_accum_steps']==0 or i+1==len(tr):
                grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['train']['max_grad_norm'],error_if_nonfinite=True);opt.step();sched.step();opt.zero_grad(set_to_none=True);ema.update(model);gs+=1
                if gs>=planned_steps:
                    stop_budget=True;break
        val_interval=max(1,int(cfg.get('eval',{}).get('checkpoint_validation_interval_epochs',1)))
        should_validate=((ep+1)%val_interval==0) or stop_budget or (ep+1==int(cfg['train']['epochs']))
        score=None;th=None;ev=None;improved=False
        if should_validate:
            ema.apply(model)
            try:
                ev,th=_select_checkpoint_metric(cfg,model,va,device,rast)
            finally:ema.restore(model)
            score=float(ev['f1'])
            if not np.isfinite(score): raise FloatingPointError(f'non-finite source-validation F1 at epoch={ep}: {score}')
            improved=score>best
            if improved:best=score;best_th=float(th)
        row={'epoch':ep,'loss':float(np.mean(ls)),'val_f1':score,'val_threshold':None if th is None else float(th),'val_checkpoint_selection_seeds':None if ev is None else ev['selection_seeds'],'val_f1_per_inference_seed':None if ev is None else ev['per_seed_f1'],'validation_performed':bool(should_validate),'validation_interval_epochs':val_interval,'optimizer_step':gs,'lr':opt.param_groups[0]['lr'],'seconds':time.time()-t0,'seen_samples':seen_samples,'seen_batches':seen_batches,'fm_samples':fm_samples,'fm_fraction':fm_samples/max(seen_samples,1),'gic_active_samples':gic_active_samples,'gic_active_batches':gic_active_batches,'gic_sample_fraction':gic_active_samples/max(seen_samples,1),'gic_batch_fraction':gic_active_batches/max(seen_batches,1),'near_deployment_samples':near_deploy,'near_deployment_fraction':near_deploy/max(seen_samples,1),'exact_deployment_samples':exact_deploy,'exact_deployment_fraction':exact_deploy/max(seen_samples,1),'nfe':1,'resume_config_mismatch':resume_config_mismatch};hist.append(row);print(row)
        usable_batches=steps*int(cfg['train']['grad_accum_steps']) if drop_incomplete_accum else len(tr)
        epoch_complete=(last_batch_index+1)>=usable_batches
        common=dict(model=model,ema=ema,optimizer=opt,scheduler=sched,epoch=ep,global_optimizer_step=gs,cfg=cfg,best_val_metric=best,best_val_threshold=best_th,split_manifest_hashes=name_hashes,split_manifest_content_hashes=content_hashes,source_provenance=source_provenance,seed=seed,extra_state={'global_sample_offset':global_sample_offset,'fairness':fairness,'resume_config_mismatch':resume_config_mismatch,'epoch_complete':bool(epoch_complete),'processed_batches_this_epoch':int(last_batch_index+1),'usable_batches_this_epoch':int(usable_batches)})
        save_checkpoint_atomic(os.path.join(out,'last.pt'),**common)
        if improved:save_checkpoint_atomic(os.path.join(out,'best.pt'),**common)
        json.dump({'history':hist,'best_val_f1':best,'best_val_threshold':best_th,'fairness':fairness},open(os.path.join(out,'history.json'),'w'),indent=2)
        if stop_budget: break
if __name__=='__main__':main()
