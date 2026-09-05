from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crackmeanflow.common import source_splits_for_config, optimizer_steps_per_epoch

DEFAULT_CONFIGS = [
    'configs/conference/crackmeanflow_unet.yaml',
    'configs/journal/a2b_original_mismatch_control.yaml',
    'configs/journal/a5_geocrack_imf_baseline.yaml',
    'configs/journal/a2b_hybrid_imf_mask_capacity_matched.yaml',
    'configs/journal/a5_geocrack_imf_endpoint_candidate.yaml',
]

def _steps(num_samples:int, cfg:dict):
    tr=cfg['train']; b=int(tr['batch_size']); ga=int(tr['grad_accum_steps']); drop=bool(tr.get('drop_last',True))
    batches=(num_samples//b) if drop else math.ceil(num_samples/b)
    if batches < 1: raise RuntimeError(f'empty loader under batch_size={b}, drop_last={drop}')
    drop_accum=bool(tr.get('drop_incomplete_accumulation',True)); spe=optimizer_steps_per_epoch(batches,ga,drop_accum)
    total=spe*int(tr['epochs']); loader_samples=min(num_samples,batches*b) if drop else num_samples; usable=spe*b*ga if drop_accum else loader_samples
    dropped_loader=max(0,num_samples-loader_samples);dropped_accum=max(0,loader_samples-usable);omitted=max(0,num_samples-usable)
    bal=tr.get('sample_balance') or {'enabled':False}
    return {
        'batch_size':b,'grad_accum_steps':ga,'effective_batch_size_nominal':b*ga,'drop_last':drop,
        'drop_incomplete_accumulation':drop_accum,'dropped_microbatches_per_epoch':(batches%ga) if drop_accum else 0,
        'batches_per_epoch':batches,'optimizer_steps_per_epoch':spe,'epochs':int(tr['epochs']),
        'best_achievable_optimizer_steps':total,'loader_samples_per_epoch':loader_samples,'usable_samples_per_epoch':usable,'dropped_samples_by_dataloader_per_epoch':dropped_loader,'dropped_samples_by_incomplete_accumulation_per_epoch':dropped_accum,'total_omitted_samples_per_epoch':omitted,'sample_coverage_fraction_per_epoch':usable/max(num_samples,1),
        'optimizer_recipe':{'lr':float(tr['lr']),'weight_decay':float(tr['weight_decay']),'warmup_epochs':int(tr.get('warmup_epochs',0)),'warmup_steps':int(tr.get('warmup_epochs',0))*spe},
        'sample_balance':bal,'endpoint_probability':float((cfg.get('loss') or {}).get('endpoint_probability',(cfg.get('loss') or {}).get('boundary_prob',0.0))),
        'endpoint_sampling':(cfg.get('loss') or {}).get('endpoint_sampling',(cfg.get('loss') or {}).get('boundary_sampling')),
        'nfe':int((cfg.get('eval') or {}).get('num_steps',-1)),
        'resolution':int((cfg.get('model') or {}).get('img_size',-1)),
    }

def main():
    root=Path(__file__).resolve().parents[1]
    ap=argparse.ArgumentParser(); ap.add_argument('--data',required=True); ap.add_argument('--configs',nargs='*',default=[str(root/x) for x in DEFAULT_CONFIGS]); ap.add_argument('--out',default=str(root/'reports/FAIR_BUDGET_AUDIT.json')); a=ap.parse_args()
    rows=[]
    for cp in a.configs:
        cfg=yaml.safe_load(open(cp)); sp=source_splits_for_config(a.data,cfg); n=len(sp['train']); r=_steps(n,cfg)
        r.update({'train_samples':n,'config':str(Path(cp).resolve()),'experiment':cfg.get('experiment'),'track':cfg.get('track'),'backbone':cfg.get('backbone'),'protocol_role':cfg.get('protocol_role')}); rows.append(r)
    if any(r['nfe']!=1 for r in rows): raise RuntimeError('all paper models must declare NFE=1')
    if len({r['resolution'] for r in rows})!=1: raise RuntimeError('model input resolutions differ in fairness audit')
    matched=min(r['best_achievable_optimizer_steps'] for r in rows)
    recipes={json.dumps(r['optimizer_recipe'],sort_keys=True) for r in rows}
    out={'models':rows,'recommended_matched_optimizer_steps':matched,'matched_budget_label':'MATCHED_OPTIMIZER_UPDATES','fully_matched_optimization_recipe':len(recipes)==1,
         'optimizer_recipe_confounded':len(recipes)!=1,
         'fairness_rule':'Primary fair-budget table uses identical exact optimizer-step budget. Because LR/weight-decay/warmup remain method-specific, call this matched-optimizer-budget, not fully matched optimization. Run configs/fairness/common_recipe_matrix.yaml as a preregistered sensitivity analysis.'}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(out,indent=2))
    md=Path(a.out).with_suffix('.md'); lines=['# Fair-budget audit','',f'- Recommended matched budget: **{matched} optimizer steps**',f'- Fully matched optimizer recipe: **{out["fully_matched_optimization_recipe"]}**',f'- Recipe confound requiring sensitivity matrix: **{out["optimizer_recipe_confounded"]}**','', '| Experiment | Steps/epoch | Best steps | LR | WD | Warmup | Endpoint | Balance unit |','|---|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:
        rec=r['optimizer_recipe'];bal=r['sample_balance'];lines.append(f"| {r['experiment']} | {r['optimizer_steps_per_epoch']} | {r['best_achievable_optimizer_steps']} | {rec['lr']} | {rec['weight_decay']} | {rec['warmup_epochs']} | {r['endpoint_probability']} | {bal.get('unit','none')} |")
    md.write_text('\n'.join(lines)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
