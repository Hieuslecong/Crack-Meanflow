from __future__ import annotations
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from crackmeanflow.common.option_c_provenance import BASE_V4_COMMIT,PROTOCOL_ID,DESIGN_LOCK_REL,validate_option_c_config,validate_option_c_ladder,semantic_sha256,option_c_design_lock_sha256

FILES={
    'J0':'j0_direct_mask.yaml','J1':'j1_centerline_edt_noncausal.yaml','J2':'j2_centerline_radius_causal.yaml',
    'J3':'j3_centerline_radius_gic.yaml','J4':'j4_centerline_radius_gic_endpoint.yaml'}
CANONICAL_STEPS=[4125,8250,12375,16500,21000]

def sha256(path):
    h=hashlib.sha256();h.update(Path(path).read_bytes());return h.hexdigest()

def _git(*args):
    try:return subprocess.check_output(['git',*args],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
    except Exception:return None

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default=str(ROOT/'reports'/'OPTION_C_V1_ENGINEERING_GATE.json'));a=ap.parse_args()
    protocol_path=ROOT/'configs'/'protocol'/'option_c_protocol_v1.yaml';design_path=ROOT/DESIGN_LOCK_REL
    protocol=yaml.safe_load(protocol_path.read_text());design=yaml.safe_load(design_path.read_text());errors=[]
    if protocol.get('protocol_id')!=PROTOCOL_ID:errors.append('protocol id mismatch')
    if protocol.get('base_v4_commit')!=BASE_V4_COMMIT:errors.append('base V4 mismatch')
    if protocol.get('design_lock')!=DESIGN_LOCK_REL:errors.append('protocol design-lock path mismatch')
    if design.get('protocol_id')!=PROTOCOL_ID or design.get('base_v4_commit')!=BASE_V4_COMMIT:errors.append('design lock identity mismatch')
    if list(protocol.get('training',{}).get('source_checkpoints',[]))!=CANONICAL_STEPS:errors.append('protocol source checkpoint mismatch')
    if list(design.get('milestone_steps',[]))!=CANONICAL_STEPS:errors.append('design-lock milestone mismatch')
    variants={};configs={}
    for variant,name in FILES.items():
        path=ROOT/'configs'/'option_c_v1'/name;cfg=yaml.safe_load(path.read_text());configs[variant]=cfg
        try:validated=validate_option_c_config(cfg,variant)
        except Exception as exc:errors.append(f'{variant}: {exc}');validated={'status':'FAIL','error':str(exc)}
        variants[variant]={'path':str(path.relative_to(ROOT)),'file_sha256':sha256(path),'semantic_sha256':semantic_sha256(cfg),**validated}
    try:ladder=validate_option_c_ladder(configs)
    except Exception as exc:errors.append(f'ladder: {exc}');ladder={'status':'FAIL','error':str(exc)}
    branch=_git('branch','--show-current');head=_git('rev-parse','HEAD')
    if branch is not None and branch!='journal/option-c-v1':errors.append(f'wrong branch: {branch}')
    report={'schema':'CRACKMEANFLOW_OPTION_C_PREFLIGHT_V2','status':'PASS' if not errors else 'FAIL','protocol_id':PROTOCOL_ID,'base_v4_commit':BASE_V4_COMMIT,'branch':branch,'head':head,'protocol_path':str(protocol_path.relative_to(ROOT)),'protocol_sha256':sha256(protocol_path),'design_lock_path':DESIGN_LOCK_REL,'design_lock_sha256':option_c_design_lock_sha256(ROOT),'variants':variants,'ladder':ladder,'errors':errors}
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(report,indent=2,sort_keys=True));raise SystemExit(0 if not errors else 2)
if __name__=='__main__':main()
