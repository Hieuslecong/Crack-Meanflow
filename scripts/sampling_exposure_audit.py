from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from crackmeanflow.common import source_splits_for_config,EpochRandomSampler,optimizer_steps_per_epoch

DEFAULTS={
 'Conference':'configs/conference/crackmeanflow_unet.yaml',
 'A2B_BASELINE':'configs/journal/a2b_original_mismatch_control.yaml',
 'A5_BASELINE':'configs/journal/a5_geocrack_imf_baseline.yaml',
 'A2B_ENDPOINT':'configs/journal/a2b_hybrid_imf_mask_capacity_matched.yaml',
 'A5_ENDPOINT':'configs/journal/a5_geocrack_imf_endpoint_candidate.yaml',
}

def _usable_indices(n,cfg,epoch):
    tr=cfg['train'];b=int(tr['batch_size']);ga=int(tr['grad_accum_steps']);drop_last=bool(tr.get('drop_last',True));drop_acc=bool(tr.get('drop_incomplete_accumulation',True))
    sampler=EpochRandomSampler(n,seed=int(tr.get('seed',42)));sampler.set_epoch(epoch);order=list(sampler)
    batches=(n//b) if drop_last else (n+b-1)//b;steps=optimizer_steps_per_epoch(batches,ga,drop_acc)
    usable=steps*b*ga if drop_acc else n
    return order[:usable],order[usable:]

def main():
    ap=argparse.ArgumentParser(description='Verify identical no-replacement CFD exposure across canonical models.')
    ap.add_argument('--data',required=True);ap.add_argument('--epochs',type=int,default=30);ap.add_argument('--out',default='reports/SAMPLING_EXPOSURE_AUDIT.json');a=ap.parse_args();root=Path(__file__).resolve().parents[1]
    cfgs={k:yaml.safe_load((root/v).read_text()) for k,v in DEFAULTS.items()};splits={k:source_splits_for_config(a.data,c) for k,c in cfgs.items()};n={k:len(v['train']) for k,v in splits.items()}
    if len(set(n.values()))!=1:raise RuntimeError(f'train counts differ across models: {n}')
    count=next(iter(n.values()));rows=[];ever=set();issues=[]
    for ep in range(a.epochs):
        exposures={k:_usable_indices(count,c,ep) for k,c in cfgs.items()};used0=next(iter(exposures.values()))[0];drop0=next(iter(exposures.values()))[1]
        if any(v[0]!=used0 or v[1]!=drop0 for v in exposures.values()):issues.append(f'epoch {ep}: exposure identity differs across models')
        if len(used0)!=len(set(used0)):issues.append(f'epoch {ep}: replacement/duplicate index detected')
        ever.update(used0);rows.append({'epoch':ep,'used_count':len(used0),'omitted_indices':drop0})
    report={'train_count':count,'models':list(cfgs),'epochs_checked':a.epochs,'same_exposure_across_models':not issues,'no_replacement':not issues,'unique_samples_seen_across_checked_epochs':len(ever),'all_samples_seen_over_window':len(ever)==count,'rows':rows,'issues':issues,'status':'PASS' if not issues and len(ever)==count else 'FAIL'}
    out=root/a.out if not Path(a.out).is_absolute() else Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    if report['status']!='PASS':raise SystemExit(2)
if __name__=='__main__':main()
