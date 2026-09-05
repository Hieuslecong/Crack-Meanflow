from __future__ import annotations
import argparse, json, os, statistics, sys
from pathlib import Path
import torch, yaml
from torch.utils.data import DataLoader

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from crackmeanflow.common import (
    PairedCrackDataset, discover_required_splits, verify_source_dataset_contract,
    config_hash, source_tree_hash, protocol_bundle_hash, file_sha256, split_identity, resolve_thresholds,
    THRESHOLD_LOCK_TYPE, evaluate_with_threshold, calibrate_threshold_on_validation, source_splits_for_config,
)
from crackmeanflow.sampler import crack_meanflow_sampler
from crackmeanflow.conference.models import build_conference_model
from crackmeanflow.adapter import CrackMeanFlowModel
from crackmeanflow.sit import build_sit
from crackmeanflow.journal.models.sit_mask_baseline import MaskIMFSiTModel, HybridMaskIMFModel
from crackmeanflow.journal import calibrate_geometry_threshold_on_validation, evaluate_geometry_with_frozen_threshold
from crackmeanflow.factory import build_model_and_rasterizer


def _ema(model,ck):
    ema=ck.get('ema')
    if not isinstance(ema,dict): raise RuntimeError('checkpoint has no EMA state')
    current=model.state_dict(); expected={k for k,v in current.items() if torch.is_tensor(v) and v.dtype.is_floating_point}; got=set(ema)
    missing=sorted(expected-got); unexpected=sorted(got-set(current))
    if missing or unexpected: raise RuntimeError(f'EMA state mismatch: missing={missing[:10]} unexpected={unexpected[:10]}')
    merged=dict(current); merged.update(ema); model.load_state_dict(merged,strict=True)


def _build(cfg,device):
    return build_model_and_rasterizer(cfg,device)

def _mean_rows(rows_by_seed,keys):
    out={}
    for k in keys:
        vals=[float(r[k]) for r in rows_by_seed if k in r]
        if vals: out[k]={'mean':statistics.mean(vals),'std_over_inference_noise':statistics.pstdev(vals),'values':vals}
    return out


def main():
    ap=argparse.ArgumentParser(description='Freeze a paper-grade source-validation threshold across multiple inference-noise seeds.')
    ap.add_argument('--config',required=True); ap.add_argument('--ckpt',required=True); ap.add_argument('--source-data',required=True)
    ap.add_argument('--dataset-name',default='CFD'); ap.add_argument('--dataset-version',required=True); ap.add_argument('--out',required=True); a=ap.parse_args()
    if not str(a.dataset_name).strip() or not str(a.dataset_version).strip(): raise ValueError('source dataset name/version must be non-empty')
    cfg=yaml.safe_load(open(a.config)); ck=torch.load(a.ckpt,map_location='cpu',weights_only=False)
    if ck.get('config_hash')!=config_hash(cfg): raise RuntimeError('threshold-freeze config does not exactly match checkpoint config')
    if not ck.get('source_tree_sha256') or ck['source_tree_sha256']!=source_tree_hash(): raise RuntimeError('threshold-freeze source tree does not match checkpoint')
    if not ck.get('protocol_bundle_sha256') or ck['protocol_bundle_sha256']!=protocol_bundle_hash(): raise RuntimeError('threshold-freeze paper/fairness protocol bundle does not match checkpoint')
    splits=source_splits_for_config(a.source_data,cfg); verify_source_dataset_contract(ck,splits,a.dataset_name,a.dataset_version,keys=('train','val','test'))
    evcfg=cfg.get('eval') or {}; seeds=[int(x) for x in evcfg.get('final_threshold_calibration_seeds',evcfg.get('eval_seeds',[0]))]
    if len(set(seeds))!=len(seeds) or not seeds: raise RuntimeError('final threshold calibration seeds must be non-empty and unique')
    thresholds=resolve_thresholds(evcfg,final=True); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds=PairedCrackDataset(splits['val'],cfg['model']['img_size'],False,False,cfg['train'].get('mask_resize_mode','nearest'),cfg['train'].get('mask_binarization','auto_binary_safe')); ld=DataLoader(ds,batch_size=int(cfg.get('eval',{}).get('batch_size',1)),shuffle=False)
    model,rast=_build(cfg,device); _ema(model,ck); per_seed_sweeps=[]
    for seed in seeds:
        if cfg['backbone']=='geocrack_imf': sweep,_=calibrate_geometry_threshold_on_validation(model,ld,device,rast,thresholds,seed,cfg['model'].get('max_radius',16))
        else: sweep,_=calibrate_threshold_on_validation(model,ld,device,crack_meanflow_sampler,thresholds,1,seed,cfg['eval'].get('cfg_scale',1.0) if cfg['backbone'] in {'unet','sit_mf'} else 1.0)
        per_seed_sweeps.append(sweep)
    mean_f1={float(t):statistics.mean(float(sw[float(t)]['f1']) for sw in per_seed_sweeps) for t in thresholds}
    best_value=max(mean_f1.values()); selected=min(t for t in thresholds if mean_f1[float(t)]==best_value)
    final_rows=[]
    for seed in seeds:
        if cfg['backbone']=='geocrack_imf': row=evaluate_geometry_with_frozen_threshold(model,ld,device,rast,selected,seed,cfg['model'].get('max_radius',16))
        else: row=evaluate_with_threshold(model,ld,device,crack_meanflow_sampler,selected,1,seed,cfg['eval'].get('cfg_scale',1.0) if cfg['backbone'] in {'unet','sit_mf'} else 1.0)
        final_rows.append(row)
    lock={
        'lock_type':THRESHOLD_LOCK_TYPE,'checkpoint_path_basename':Path(a.ckpt).name,'checkpoint_sha256':file_sha256(a.ckpt),'checkpoint_config_hash':ck.get('config_hash'),
        'checkpoint_source_tree_sha256':ck.get('source_tree_sha256'),'checkpoint_protocol_bundle_sha256':ck.get('protocol_bundle_sha256'),'source_dataset_name':a.dataset_name,'source_dataset_version':a.dataset_version,
        'source_val_identity':split_identity(splits['val'],include_rows=False),'calibration_seeds':seeds,'threshold_candidates':thresholds,
        'selection_metric':'mean pixel-micro F1 across fixed source-validation inference-noise seeds','tie_break':'lowest threshold on exact mean-F1 tie',
        'mean_f1_by_threshold':{str(t):mean_f1[float(t)] for t in thresholds},'selected_threshold':float(selected),
        'selected_threshold_metrics':_mean_rows(final_rows,['f1','iou','precision','recall','f1_macro_image','iou_macro_image','cldice','boundary_f1']),
        'target_data_used':False,'status':'PASS'
    }
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(lock,indent=2));print(json.dumps(lock,indent=2))

if __name__=='__main__': main()
