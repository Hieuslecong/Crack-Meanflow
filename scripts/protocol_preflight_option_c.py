from __future__ import annotations
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from crackmeanflow.common.option_c_provenance import BASE_V4_COMMIT,PROTOCOL_ID,validate_option_c_config,semantic_sha256

FILES={
    'J0':'j0_direct_mask.yaml','J1':'j1_centerline_edt_noncausal.yaml','J2':'j2_centerline_radius_causal.yaml',
    'J3':'j3_centerline_radius_gic.yaml','J4':'j4_centerline_radius_gic_endpoint.yaml'}

def sha256(path):
    h=hashlib.sha256();h.update(Path(path).read_bytes());return h.hexdigest()

def _git(*args):
    try:return subprocess.check_output(['git',*args],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
    except Exception:return None

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default=str(ROOT/'reports'/'OPTION_C_V1_ENGINEERING_GATE.json'));a=ap.parse_args()
    protocol_path=ROOT/'configs'/'protocol'/'option_c_protocol_v1.yaml';design_path=ROOT/'reports'/'OPTION_C_V1_DESIGN_LOCK.json'
    protocol=yaml.safe_load(protocol_path.read_text());design=json.loads(design_path.read_text());errors=[]
    if protocol.get('protocol_id')!=PROTOCOL_ID:errors.append('protocol id mismatch')
    if protocol.get('base_v4_commit')!=BASE_V4_COMMIT:errors.append('base V4 mismatch')
    if design.get('protocol_id')!=PROTOCOL_ID or design.get('base_v4_commit')!=BASE_V4_COMMIT:errors.append('design lock identity mismatch')
    variants={}
    for variant,name in FILES.items():
        path=ROOT/'configs'/'option_c_v1'/name;cfg=yaml.safe_load(path.read_text())
        try:validated=validate_option_c_config(cfg,variant)
        except Exception as exc:errors.append(f'{variant}: {exc}');validated={'status':'FAIL','error':str(exc)}
        variants[variant]={'path':str(path.relative_to(ROOT)),'file_sha256':sha256(path),'semantic_sha256':semantic_sha256(cfg),**validated}
    branch=_git('branch','--show-current');head=_git('rev-parse','HEAD')
    if branch is not None and branch!='journal/option-c-v1':errors.append(f'wrong branch: {branch}')
    report={'schema':'CRACKMEANFLOW_OPTION_C_PREFLIGHT_V1','status':'PASS' if not errors else 'FAIL','protocol_id':PROTOCOL_ID,'base_v4_commit':BASE_V4_COMMIT,'branch':branch,'head':head,'protocol_path':str(protocol_path.relative_to(ROOT)),'protocol_sha256':sha256(protocol_path),'design_lock_sha256':sha256(design_path),'variants':variants,'errors':errors}
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(report,indent=2,sort_keys=True));raise SystemExit(0 if not errors else 2)
if __name__=='__main__':main()
