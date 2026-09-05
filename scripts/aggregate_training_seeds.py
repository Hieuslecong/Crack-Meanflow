from __future__ import annotations
import argparse, json, math, statistics
from scipy.stats import t as student_t
from pathlib import Path

METRICS=('f1','iou','precision','recall','f1_macro_image','iou_macro_image','f1_macro_positive_image','iou_macro_positive_image','cldice','boundary_f1','gt_foreground_ratio','pred_foreground_ratio')

def _metric(report,key):
    v=report.get(key)
    if isinstance(v,dict) and 'mean' in v: return float(v['mean'])
    if isinstance(v,(int,float)): return float(v)
    return None

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('reports',nargs='+'); ap.add_argument('--out',required=True); ap.add_argument('--min-seeds',type=int,default=3); a=ap.parse_args()
    rs=[json.load(open(p)) for p in a.reports]
    if a.min_seeds<2: raise ValueError('--min-seeds must be >=2');
    if len(rs)<a.min_seeds: raise RuntimeError(f'need at least {a.min_seeds} independent training-seed reports; got {len(rs)}')
    seeds=[r.get('training_seed') for r in rs]
    if any(s is None for s in seeds): raise RuntimeError('every report must record training_seed')
    if len(set(seeds))!=len(seeds): raise RuntimeError(f'training seeds must be unique, got {seeds}')
    for r in rs:
        if r.get('scientific_validity')!='VALID_HEADLINE_PROTOCOL': raise RuntimeError('all reports must be VALID_HEADLINE_PROTOCOL')
        if r.get('requested_nfe')!=1 or r.get('actual_nfe')!=1 or not r.get('nfe_contract_pass',False): raise RuntimeError('all reports must pass NFE=1 contract')
    identities=[r.get('target_identity') for r in rs]
    canon=json.dumps(identities[0],sort_keys=True)
    if any(json.dumps(x,sort_keys=True)!=canon for x in identities[1:]): raise RuntimeError('target dataset identities differ across training seeds')
    method_hashes=[r.get('method_config_hash') for r in rs]
    if any(not x for x in method_hashes): raise RuntimeError('every report must record method_config_hash')
    if len(set(method_hashes))!=1: raise RuntimeError(f'method/config identity differs across training seeds: {method_hashes}')
    source_ids=[r.get('source_identity') for r in rs]
    if any(not x for x in source_ids): raise RuntimeError('every report must record source_identity')
    source_canon=json.dumps(source_ids[0],sort_keys=True)
    if any(json.dumps(x,sort_keys=True)!=source_canon for x in source_ids[1:]): raise RuntimeError('source dataset identities differ across training seeds')
    code_hashes=[r.get('checkpoint_source_tree_sha256') for r in rs]
    if any(not x for x in code_hashes) or len(set(code_hashes))!=1: raise RuntimeError('checkpoint source-code identity differs or is missing across training seeds')
    protocol_hashes=[r.get('checkpoint_protocol_bundle_sha256') for r in rs]
    if any(not x for x in protocol_hashes) or len(set(protocol_hashes))!=1: raise RuntimeError('paper/fairness protocol bundle differs or is missing across training seeds')
    budgets=[r.get('training_budget') for r in rs]
    budget_sig=lambda b: json.dumps({k:(b or {}).get(k) for k in ('budget_protocol','planned_optimizer_steps','max_optimizer_steps','effective_samples_per_full_optimizer_step','drop_incomplete_accumulation')},sort_keys=True)
    if len({budget_sig(b) for b in budgets})!=1: raise RuntimeError('training budgets differ across training seeds')
    out={'n_training_seeds':len(rs),'training_seeds':seeds,'target_identity':identities[0],'method_config_hash':method_hashes[0],'source_identity':source_ids[0],'checkpoint_source_tree_sha256':code_hashes[0],'checkpoint_protocol_bundle_sha256':protocol_hashes[0],'experiment':rs[0].get('experiment'),'track':rs[0].get('track'),'backbone':rs[0].get('backbone'),'training_budget':budgets[0],'statistics_level':'independent training seeds','metrics':{}}
    for k in METRICS:
        vals=[_metric(r,k) for r in rs]
        if any(v is None for v in vals): continue
        mean=statistics.mean(vals); std=statistics.stdev(vals) if len(vals)>1 else 0.; crit=float(student_t.ppf(0.975,df=len(vals)-1)) if len(vals)>1 else float('nan'); ci=crit*std/math.sqrt(len(vals)) if len(vals)>1 else float('nan')
        out['metrics'][k]={'mean':mean,'std_sample':std,'ci95_student_t':[mean-ci,mean+ci] if len(vals)>1 else None,'t_critical_95':crit if len(vals)>1 else None,'values':vals}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
