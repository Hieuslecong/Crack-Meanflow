from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def _as_ints(xs): return [int(x) for x in (xs or [])]
def _same_dict(a,b): return dict(a or {})==dict(b or {})

def main():
    ap=argparse.ArgumentParser(description='Validate executable configs against the preregistered paper protocol. No training is performed.')
    ap.add_argument('--protocol',default=None);ap.add_argument('--out',default='reports/PROTOCOL_PREFLIGHT.json');a=ap.parse_args()
    root=Path(__file__).resolve().parents[1];pp=Path(a.protocol) if a.protocol else root/'configs/protocol/paper_protocol.yaml';proto=yaml.safe_load(pp.read_text());issues=[];rows={}
    source=proto['source_dataset'];ck=proto['checkpoint_selection'];ft=proto['final_threshold_lock']
    required_ck=_as_ints(ck['inference_noise_seeds']);required_final=_as_ints(ft['inference_noise_seeds']);required_interval=int(ck['validation_interval_epochs']);required_eval_batch=int(ck['eval_batch_size'])
    declared=set(proto.get('models') or {})
    for ladder,seq in (proto.get('ablation_ladder') or {}).items():
        for model_name in seq or []:
            if model_name not in declared: issues.append(f'ablation_ladder.{ladder}: undefined model reference {model_name!r}')
    for name,spec in proto['models'].items():
        p=root/spec['config'];cfg=yaml.safe_load(p.read_text());tr=cfg['train'];ev=cfg['eval'];model=cfg.get('model') or {}
        row={'config':spec['config'],'track':cfg.get('track'),'backbone':cfg.get('backbone'),'protocol_role':cfg.get('protocol_role'),'img_size':model.get('img_size'),'nfe':ev.get('num_steps'),'checkpoint_selection_seeds':ev.get('checkpoint_selection_seeds'),'final_threshold_calibration_seeds':ev.get('final_threshold_calibration_seeds'),'sample_balance':tr.get('sample_balance'),'mask_binarization':tr.get('mask_binarization'),'deterministic':tr.get('deterministic'),'normal_negatives':tr.get('normal_negatives')}
        if cfg.get('protocol_role')!=spec.get('role'):issues.append(f'{name}: protocol role mismatch config={cfg.get("protocol_role")!r} spec={spec.get("role")!r}')
        if int(ev.get('num_steps',-1))!=int(spec['nfe']):issues.append(f'{name}: NFE mismatch')
        if _as_ints(ev.get('checkpoint_selection_seeds'))!=required_ck:issues.append(f'{name}: checkpoint-selection seeds mismatch')
        if _as_ints(ev.get('final_threshold_calibration_seeds'))!=required_final:issues.append(f'{name}: final threshold seeds mismatch')
        if _as_ints(ev.get('eval_seeds'))!=required_final:issues.append(f'{name}: headline eval seeds must match final threshold calibration seeds')
        if int(ev.get('checkpoint_validation_interval_epochs',-1))!=required_interval:issues.append(f'{name}: checkpoint validation interval mismatch')
        if int(ev.get('batch_size',-1))!=required_eval_batch:issues.append(f'{name}: eval batch size mismatch')
        if ev.get('checkpoint_use_final_threshold_grid') is not True:issues.append(f'{name}: checkpoint must use final threshold grid')
        grid=ev.get('final_threshold_grid') or {}
        try:
            start,stop,step=float(grid['start']),float(grid['stop']),float(grid['step'])
            if not (step>0 and stop>=start):issues.append(f'{name}: invalid final threshold grid')
        except Exception: issues.append(f'{name}: missing/invalid final threshold grid')
        sb=tr.get('sample_balance') or {}
        if sb.get('enabled') is not False or sb.get('unit')!=source['canonical_sampling']:issues.append(f'{name}: canonical sampling mismatch')
        checks={
          'parent-group regex': tr.get('parent_group_regex')==source['parent_group_regex'],
          'mask-resize': tr.get('mask_resize_mode')==source['mask_resize_mode'],
          'resize-policy': tr.get('resize_policy')==source['resize_policy'],
          'mask-binarization': tr.get('mask_binarization')==source['mask_binarization'],
          'augmentation': tr.get('augment') is source['augment'],
          'photometric augmentation': tr.get('photometric_augment') is source['photometric_augment'],
          'deterministic': tr.get('deterministic') is source['deterministic'],
          'deterministic warn-only': tr.get('deterministic_warn_only') is source['deterministic_warn_only'],
          'drop-incomplete-accumulation': tr.get('drop_incomplete_accumulation') is source['drop_incomplete_accumulation'],
          'normal-negatives': _same_dict(tr.get('normal_negatives'),source['normal_negatives']),
          'num_workers': int(tr.get('num_workers',-1))==int(source['num_workers']),
          'img_size': int(model.get('img_size',-1))==int(source['img_size']),
        }
        for label,ok in checks.items():
            if not ok:issues.append(f'{name}: {label} mismatch')
        rows[name]=row
    report={'protocol_version':proto['protocol_version'],'protocol_file':str(pp),'models':rows,'issues':issues,'status':'PASS' if not issues else 'FAIL'}
    out=root/a.out if not Path(a.out).is_absolute() else Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    if issues:raise SystemExit(2)
if __name__=='__main__':main()
