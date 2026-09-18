from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from crackmeanflow.common.option_c_provenance import (
    BASE_V4_COMMIT,
    DESIGN_LOCK_REL,
    FACTORIAL_ADDENDUM_REL,
    PROTOCOL_ID,
    option_c_design_lock_sha256,
    option_c_factorial_addendum_sha256,
    semantic_sha256,
    validate_option_c_config,
    validate_option_c_ladder,
)

FROZEN_RELEASE_COMMIT='fb2cf43d719941b41024fbb078c777c485b1a490'
ORIGINAL_BRANCH='journal/option-c-v1'
ADDENDUM_BRANCH='journal/option-c-v1-factorial-addendum'
FILES={
    'J0':'j0_direct_mask.yaml',
    'J1':'j1_centerline_edt_noncausal.yaml',
    'J2':'j2_centerline_radius_causal.yaml',
    'J2E':'j2e_centerline_radius_endpoint_only.yaml',
    'J3':'j3_centerline_radius_gic.yaml',
    'J4':'j4_centerline_radius_gic_endpoint.yaml',
}
CANONICAL_STEPS=[4125,8250,12375,16500,21000]


def sha256(path):
    h=hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def _git(*args):
    try:
        return subprocess.check_output(['git',*args],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _git_success(*args):
    return subprocess.run(['git',*args],cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0


def _validate_factorial_addendum(path):
    errors=[]
    if not path.is_file():
        return ['factorial addendum missing'],None
    addendum=yaml.safe_load(path.read_text()) or {}
    if addendum.get('schema')!='CRACKMEANFLOW_OPTION_C_FACTORIAL_ADDENDUM_V1':
        errors.append('factorial addendum schema mismatch')
    if addendum.get('base_protocol_id')!=PROTOCOL_ID:
        errors.append('factorial addendum base protocol mismatch')
    if addendum.get('base_release_commit')!=FROZEN_RELEASE_COMMIT:
        errors.append('factorial addendum base release mismatch')
    if addendum.get('base_v4_commit')!=BASE_V4_COMMIT:
        errors.append('factorial addendum base V4 mismatch')
    if addendum.get('source_branch')!=ADDENDUM_BRANCH:
        errors.append('factorial addendum source branch mismatch')
    target=addendum.get('target_access_state') or {}
    expected_target={
        'target_metrics_seen_before_addendum':False,
        'target_tuning_allowed':False,
        'GAPS384':'closed',
        'OmniCrack30k_repartitioned_v013_holdout':'closed',
        'final_untouched_external_dataset':'closed',
    }
    for key,expected in expected_target.items():
        if target.get(key)!=expected:
            errors.append(f'target access state mismatch: {key}')
    factorial=addendum.get('factorial') or {}
    if factorial != {
        'gic_off_endpoint_off':'J2',
        'gic_on_endpoint_off':'J3',
        'gic_off_endpoint_on':'J2E',
        'gic_on_endpoint_on':'J4',
    }:
        errors.append('factorial 2x2 mapping mismatch')
    return errors,addendum


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default=str(ROOT/'reports'/'OPTION_C_V1_ENGINEERING_GATE.json'))
    a=ap.parse_args()

    protocol_path=ROOT/'configs'/'protocol'/'option_c_protocol_v1.yaml'
    design_path=ROOT/DESIGN_LOCK_REL
    factorial_path=ROOT/FACTORIAL_ADDENDUM_REL
    protocol=yaml.safe_load(protocol_path.read_text())
    design=yaml.safe_load(design_path.read_text())
    errors=[]
    factorial_errors,factorial_addendum=_validate_factorial_addendum(factorial_path)
    errors.extend(factorial_errors)

    if protocol.get('protocol_id')!=PROTOCOL_ID:errors.append('protocol id mismatch')
    if protocol.get('base_v4_commit')!=BASE_V4_COMMIT:errors.append('base V4 mismatch')
    if protocol.get('design_lock')!=DESIGN_LOCK_REL:errors.append('protocol design-lock path mismatch')
    if protocol.get('source_branch')!=ORIGINAL_BRANCH:errors.append('original protocol source branch changed')
    if design.get('protocol_id')!=PROTOCOL_ID or design.get('base_v4_commit')!=BASE_V4_COMMIT:errors.append('design lock identity mismatch')
    if list(protocol.get('training',{}).get('source_checkpoints',[]))!=CANONICAL_STEPS:errors.append('protocol source checkpoint mismatch')
    if list(design.get('milestone_steps',[]))!=CANONICAL_STEPS:errors.append('design-lock milestone mismatch')

    variants={}
    configs={}
    for variant,name in FILES.items():
        path=ROOT/'configs'/'option_c_v1'/name
        try:
            cfg=yaml.safe_load(path.read_text())
        except Exception as exc:
            errors.append(f'{variant}: config read failed: {exc}')
            cfg={}
        configs[variant]=cfg
        try:
            validated=validate_option_c_config(cfg,variant)
        except Exception as exc:
            errors.append(f'{variant}: {exc}')
            validated={'status':'FAIL','error':str(exc)}
        variants[variant]={
            'path':str(path.relative_to(ROOT)),
            'file_sha256':sha256(path),
            'semantic_sha256':semantic_sha256(cfg),
            **validated,
        }

    try:
        ladder=validate_option_c_ladder(configs)
    except Exception as exc:
        errors.append(f'ladder: {exc}')
        ladder={'status':'FAIL','error':str(exc)}

    branch=_git('branch','--show-current')
    head=_git('rev-parse','HEAD')
    if branch not in {ORIGINAL_BRANCH,ADDENDUM_BRANCH}:
        errors.append(f'wrong branch: {branch}')
    if branch==ORIGINAL_BRANCH and head!=FROZEN_RELEASE_COMMIT:
        errors.append('original branch is not at the frozen release commit')
    if branch==ADDENDUM_BRANCH and not _git_success('merge-base','--is-ancestor',FROZEN_RELEASE_COMMIT,'HEAD'):
        errors.append('addendum branch is not based on the frozen release commit')

    try:
        addendum_sha=option_c_factorial_addendum_sha256(ROOT)
    except Exception as exc:
        addendum_sha=None
        errors.append(str(exc))

    report={
        'schema':'CRACKMEANFLOW_OPTION_C_PREFLIGHT_V3',
        'status':'PASS' if not errors else 'FAIL',
        'protocol_id':PROTOCOL_ID,
        'base_v4_commit':BASE_V4_COMMIT,
        'base_release_commit':FROZEN_RELEASE_COMMIT,
        'branch':branch,
        'head':head,
        'protocol_path':str(protocol_path.relative_to(ROOT)),
        'protocol_sha256':sha256(protocol_path),
        'design_lock_path':DESIGN_LOCK_REL,
        'design_lock_sha256':option_c_design_lock_sha256(ROOT),
        'factorial_addendum_path':str(factorial_path.relative_to(ROOT)),
        'factorial_addendum_sha256':addendum_sha,
        'factorial_addendum':factorial_addendum,
        'factorial_status':'PASS' if not factorial_errors else 'FAIL',
        'variants':variants,
        'ladder':ladder,
        'errors':errors,
    }
    out=Path(a.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(report,indent=2,sort_keys=True))
    raise SystemExit(0 if not errors else 2)


if __name__=='__main__':
    main()
