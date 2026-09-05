from __future__ import annotations
import argparse,copy,json,os,statistics,sys,torch,yaml
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crackmeanflow.common import (
    PairedCrackDataset,discover_required_splits,discover_evaluation_pairs,evaluate_test_with_frozen_threshold,
    verify_source_dataset_contract,target_dataset_identity,config_hash,source_tree_hash,protocol_bundle_hash,audit_group_integrity,audit_cross_dataset_image_content_overlap,
    load_and_verify_target_lock,load_and_verify_threshold_lock,source_splits_for_config,
)
from crackmeanflow.sampler import crack_meanflow_sampler
from crackmeanflow.sit import build_sit
from crackmeanflow.adapter import CrackMeanFlowModel
from crackmeanflow.conference.models import build_conference_model
from crackmeanflow.journal.models.sit_mask_baseline import MaskIMFSiTModel,HybridMaskIMFModel
from crackmeanflow.journal import evaluate_geometry_with_frozen_threshold
from crackmeanflow.factory import build_model_and_rasterizer

CORE_KEYS=['f1','iou','precision','recall','f1_macro_image','iou_macro_image','precision_macro_image','recall_macro_image','f1_macro_positive_image','iou_macro_positive_image','cldice','boundary_f1','gt_foreground_ratio','pred_foreground_ratio','empty_gt_false_positive_rate']

def _ema(model,ck):
    if 'ema' not in ck or not isinstance(ck['ema'],dict): raise RuntimeError('checkpoint has no EMA state')
    current=model.state_dict();ema=ck['ema'];expected={k for k,v in current.items() if torch.is_tensor(v) and v.dtype.is_floating_point};got=set(ema)
    missing=sorted(expected-got);unexpected=sorted(got-set(current))
    if missing or unexpected: raise RuntimeError(f'EMA state mismatch: missing={missing[:10]} ({len(missing)}), unexpected={unexpected[:10]} ({len(unexpected)})')
    merged=dict(current);merged.update(ema);model.load_state_dict(merged,strict=True)

def _summary(rows,keys):
    return {k:{'mean':statistics.mean([float(r[k]) for r in rows]),'std_over_inference_noise':statistics.pstdev([float(r[k]) for r in rows]),'values':[float(r[k]) for r in rows]} for k in keys if all(k in r for r in rows)}

def _method_config_hash(cfg):
    c=copy.deepcopy(cfg)
    if isinstance(c.get('train'),dict): c['train'].pop('seed',None)
    return config_hash(c)

class ForwardCounter(nn.Module):
    def __init__(self,model):super().__init__();self.model=model;self.forward_calls=0;self.flow_output_calls=0
    def forward(self,*a,**kw):self.forward_calls+=1;return self.model(*a,**kw)
    def flow_outputs(self,*a,**kw):self.flow_output_calls+=1;return self.model.flow_outputs(*a,**kw)
    def get_seg_logits(self):return self.model.get_seg_logits() if hasattr(self.model,'get_seg_logits') else None
    def __getattr__(self,name):
        if name in {'model','forward_calls','flow_output_calls'}:return super().__getattr__(name)
        try:return super().__getattr__(name)
        except AttributeError:return getattr(self.model,name)

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser(description='Strict NFE=1 evaluation with source, threshold and target provenance locks.')
    ap.add_argument('--config',required=True);ap.add_argument('--ckpt',required=True);ap.add_argument('--source-data',required=True);ap.add_argument('--data',required=True)
    ap.add_argument('--dataset-name',required=True);ap.add_argument('--dataset-version',required=True);ap.add_argument('--target-lock',default=None);ap.add_argument('--threshold-lock',default=None);ap.add_argument('--out',default=None)
    ap.add_argument('--allow-config-mismatch',action='store_true');ap.add_argument('--per-image-out',default=None);ap.add_argument('--diagnostic-unlocked-target',action='store_true');ap.add_argument('--diagnostic-checkpoint-threshold',action='store_true');ap.add_argument('--include-target-normal-negatives',action='store_true',help='diagnostic only when no target lock is supplied')
    a=ap.parse_args()
    if not str(a.dataset_name).strip() or not str(a.dataset_version).strip(): raise ValueError('target dataset name/version must be non-empty')
    if not a.target_lock and not a.diagnostic_unlocked_target: raise RuntimeError('paper/headline evaluation requires --target-lock; use --diagnostic-unlocked-target only for non-headline diagnostics')
    if not a.threshold_lock and not a.diagnostic_checkpoint_threshold: raise RuntimeError('paper/headline evaluation requires --threshold-lock frozen on source validation; use --diagnostic-checkpoint-threshold only for diagnostics')
    cfg=yaml.safe_load(open(a.config));device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');ck=torch.load(a.ckpt,map_location='cpu',weights_only=False)
    if int(cfg.get('eval',{}).get('num_steps',1))!=1:raise RuntimeError('headline evaluation requires eval.num_steps=1')
    cfg_hash=config_hash(cfg);ck_cfg_hash=ck.get('config_hash');config_match=bool(ck_cfg_hash) and ck_cfg_hash==cfg_hash
    if not config_match and not a.allow_config_mismatch:raise RuntimeError('evaluation config does not match checkpoint config')
    checkpoint_source_tree=ck.get('source_tree_sha256');current_source_tree=source_tree_hash()
    if not checkpoint_source_tree: raise RuntimeError('checkpoint has no source_tree_sha256; strict scientific evaluation refuses unverifiable code provenance')
    code_match=checkpoint_source_tree==current_source_tree
    if not code_match:raise RuntimeError('evaluation source tree differs from checkpoint source_tree_sha256')
    checkpoint_protocol_bundle=ck.get('protocol_bundle_sha256');current_protocol_bundle=protocol_bundle_hash()
    if not checkpoint_protocol_bundle: raise RuntimeError('checkpoint has no protocol_bundle_sha256; strict headline evaluation refuses unverifiable paper-protocol provenance')
    protocol_bundle_match=checkpoint_protocol_bundle==current_protocol_bundle
    if not protocol_bundle_match: raise RuntimeError('evaluation paper/fairness protocol bundle differs from checkpoint protocol_bundle_sha256')
    source_splits=source_splits_for_config(a.source_data,cfg);saved_source=ck.get('source_provenance') or {};source_name=saved_source.get('dataset_name');source_version=saved_source.get('dataset_version')
    if not source_name or source_version is None: raise RuntimeError('checkpoint source dataset name/version provenance is incomplete')
    source_verification=verify_source_dataset_contract(ck,source_splits,source_name,source_version,keys=('train','val','test'))
    saved_group=saved_source.get('group_regex')
    if saved_group:
        import re
        rx=re.compile(saved_group)
        def _group(stem):
            m=rx.search(stem)
            if not m: raise RuntimeError(f'saved group regex does not match source stem={stem!r}')
            return m.group(1) if m.groups() else m.group(0)
        audit_group_integrity(source_splits,_group);source_verification['parent_group_audit']='PASS'
    per_image_payload=[]
    target_lock=None
    if a.target_lock:
        preview=json.loads(Path(a.target_lock).read_text()); include_normals=bool(preview.get('include_normal_negatives',False))
        target_pairs=discover_evaluation_pairs(a.data,'test',include_normals);target_lock=load_and_verify_target_lock(a.target_lock,target_pairs,a.dataset_name,a.dataset_version)
    else:
        include_normals=bool(a.include_target_normal_negatives);target_pairs=discover_evaluation_pairs(a.data,'test',include_normals)
    if not target_pairs:raise RuntimeError('target test split is empty')
    source_target_contamination=audit_cross_dataset_image_content_overlap(source_splits,target_pairs)
    target_splits={'test':target_pairs};target_identity=target_dataset_identity(target_splits,a.dataset_name,a.dataset_version,keys=('test',));target_identity['include_normal_negatives']=include_normals
    threshold_lock=None
    if a.threshold_lock: threshold_lock,th=load_and_verify_threshold_lock(a.threshold_lock,a.ckpt,ck,source_splits['val'],source_name,source_version);threshold_source='CFD_validation_multi_seed_threshold_lock'
    else: th=float(ck['best_val_threshold']);threshold_source='DIAGNOSTIC_checkpoint_single_seed_validation_threshold'
    ds=PairedCrackDataset(target_pairs,cfg['model']['img_size'],False,False,cfg['train'].get('mask_resize_mode','nearest'),cfg['train'].get('mask_binarization','auto_binary_safe'));ld=DataLoader(ds,batch_size=int(cfg.get('eval',{}).get('batch_size',1)),shuffle=False);seeds=[int(x) for x in cfg['eval'].get('eval_seeds',[0])];bb=cfg['backbone'];forward_counts=[]
    if len(set(seeds))!=len(seeds) or not seeds: raise RuntimeError('eval_seeds must be non-empty and unique')
    if bb=='unet':
        base,_=build_model_and_rasterizer(cfg,device);_ema(base,ck);model=ForwardCounter(base);rows=[]
        for s in seeds:
            before=model.forward_calls;row=evaluate_test_with_frozen_threshold(model,ld,device,crack_meanflow_sampler,th,1,s,cfg['eval'].get('cfg_scale',1.0),collect_per_image=bool(a.per_image_out));forward_counts.append(model.forward_calls-before);
            if a.per_image_out: per_image_payload.append({'inference_noise_seed':s,'rows':row.pop('per_image')});
            rows.append(row)
        out=_summary(rows,CORE_KEYS);out.update({'method':'Conference_CrackMeanFlow_UNet'})
    elif bb=='sit_mf':
        base,_=build_model_and_rasterizer(cfg,device);_ema(base,ck);model=ForwardCounter(base);rows=[]
        for s in seeds:
            before=model.forward_calls;row=evaluate_test_with_frozen_threshold(model,ld,device,crack_meanflow_sampler,th,1,s,cfg['eval'].get('cfg_scale',1.0),collect_per_image=bool(a.per_image_out));forward_counts.append(model.forward_calls-before);
            if a.per_image_out: per_image_payload.append({'inference_noise_seed':s,'rows':row.pop('per_image')});
            rows.append(row)
        out=_summary(rows,CORE_KEYS);out.update({'ablation':'A1'})
    elif bb in {'sit_imf_mask','hybrid_imf_mask'}:
        base,_=build_model_and_rasterizer(cfg,device);_ema(base,ck);model=ForwardCounter(base);rows=[]
        for s in seeds:
            before=model.forward_calls;row=evaluate_test_with_frozen_threshold(model,ld,device,crack_meanflow_sampler,th,1,s,1.0,collect_per_image=bool(a.per_image_out));forward_counts.append(model.forward_calls-before);
            if a.per_image_out: per_image_payload.append({'inference_noise_seed':s,'rows':row.pop('per_image')});
            rows.append(row)
        out=_summary(rows,CORE_KEYS);out.update({'ablation':'A2' if bb=='sit_imf_mask' else 'A2B_CAPACITY_MATCHED'})
    elif bb=='geocrack_imf':
        base,rast=build_model_and_rasterizer(cfg,device);_ema(base,ck);model=ForwardCounter(base);rows=[]
        for s in seeds:
            before=model.flow_output_calls;row=evaluate_geometry_with_frozen_threshold(model,ld,device,rast,th,s,cfg['model'].get('max_radius',16),collect_per_image=bool(a.per_image_out));forward_counts.append(model.flow_output_calls-before);
            if a.per_image_out: per_image_payload.append({'inference_noise_seed':s,'rows':row.pop('per_image')});
            rows.append(row)
        out=_summary(rows,CORE_KEYS+['centerline_assd_px','edt_radius_mae_px','skeleton_length_rel_error']);out.update({'mask_source':'geometry_rasterizer','radius_metric_semantics':'EDT-radius proxy; not calibrated physical crack width','representation':cfg['model'].get('representation','centerline_radius'),'distance_encoding':cfg['model'].get('distance_encoding','linear'),'centerline_direct_mask_causality':'auxiliary/shared-representation only for centerline_edt; final rasterized mask is decoded from dense EDT'})
    else:raise RuntimeError(f'unsupported backbone={bb!r}; no silent fallback')
    expected=len(ld);nfe_ok=all(c==expected for c in forward_counts);resume_tainted=bool((ck.get('extra_state') or {}).get('resume_config_mismatch',False))
    strict_locks=bool(a.target_lock and a.threshold_lock);valid=config_match and code_match and strict_locks and not resume_tainted
    out.update({'requested_nfe':1,'actual_nfe':1 if nfe_ok else None,'forward_calls_per_inference_noise_seed':forward_counts,'expected_forward_calls_per_inference_noise_seed':expected,'nfe_contract_pass':nfe_ok,'threshold':float(th),'threshold_source':threshold_source,'threshold_lock':threshold_lock,'training_seed':ck.get('seed'),'inference_noise_seeds':seeds,'inference_noise_rows':rows,'statistical_warning':'inference-noise seeds from one checkpoint are NOT independent training seeds','source_provenance_verification':source_verification,'source_identity':saved_source,'target_identity':target_identity,'target_lock':target_lock,'config_hash':cfg_hash,'method_config_hash':_method_config_hash(cfg),'experiment':cfg.get('experiment'),'track':cfg.get('track'),'backbone':cfg.get('backbone'),'checkpoint_config_hash':ck_cfg_hash,'config_match':config_match,'checkpoint_git_commit':ck.get('git_commit'),'source_tree_sha256':current_source_tree,'checkpoint_source_tree_sha256':checkpoint_source_tree,'source_tree_match':code_match,'checkpoint_protocol_bundle_sha256':checkpoint_protocol_bundle,'protocol_bundle_sha256':current_protocol_bundle,'protocol_bundle_match':protocol_bundle_match,'source_target_contamination_audit':source_target_contamination,'checkpoint_optimizer_step':ck.get('global_optimizer_step'),'training_budget':(ck.get('extra_state') or {}).get('fairness'),'resize_policy':cfg['train'].get('resize_policy','stretch_square'),'mask_resize_mode':cfg['train'].get('mask_resize_mode','nearest'),'mask_binarization':cfg['train'].get('mask_binarization','auto_binary_safe'),'scientific_validity':'VALID_HEADLINE_PROTOCOL' if valid else ('DIAGNOSTIC_CONFIG_MISMATCH' if not config_match else ('TAINTED_RESUME_CONFIG_CHANGE' if resume_tainted else 'DIAGNOSTIC_UNLOCKED_PROTOCOL'))})
    if not nfe_ok:raise RuntimeError(f'NFE/forward-call contract failed: counts={forward_counts} expected={expected}')
    if a.per_image_out:
        pp={'dataset_name':a.dataset_name,'dataset_version':a.dataset_version,'training_seed':ck.get('seed'),'target_identity':target_identity,'method_config_hash':_method_config_hash(cfg),'checkpoint_source_tree_sha256':checkpoint_source_tree,'threshold':float(th),'inference_noise':per_image_payload}
        Path(a.per_image_out).parent.mkdir(parents=True,exist_ok=True);Path(a.per_image_out).write_text(json.dumps(pp,indent=2))
    path=a.out or os.path.join(os.path.dirname(a.ckpt),'TEST_REPORT_'+a.dataset_name+'.json');json.dump(out,open(path,'w'),indent=2);print(json.dumps(out,indent=2))
if __name__=='__main__':main()
