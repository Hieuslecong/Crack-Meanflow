from __future__ import annotations
import json
from decimal import Decimal
from pathlib import Path

from .data import split_identity
from .checkpointing import file_sha256, config_hash

TARGET_LOCK_TYPE='CRACKMEANFLOW_TARGET_DATASET_LOCK_V1'
THRESHOLD_LOCK_TYPE='CRACKMEANFLOW_SOURCE_THRESHOLD_LOCK_V1'

def _decimal_grid(start,stop,step):
    a=Decimal(str(start)); b=Decimal(str(stop)); d=Decimal(str(step))
    if d<=0: raise ValueError('threshold grid step must be >0')
    if b<a: raise ValueError('threshold grid stop must be >= start')
    vals=[]; x=a
    while x<=b+Decimal('1e-18'):
        vals.append(float(x)); x+=d
    return vals

def resolve_thresholds(eval_cfg,final=False):
    if final and eval_cfg.get('final_threshold_grid'):
        g=eval_cfg['final_threshold_grid']; return _decimal_grid(g['start'],g['stop'],g['step'])
    vals=eval_cfg.get('thresholds')
    if not vals: raise RuntimeError('evaluation threshold candidates are missing')
    vals=[float(x) for x in vals]
    if len(vals)!=len(set(vals)): raise RuntimeError('evaluation threshold candidates contain duplicates')
    if vals!=sorted(vals): raise RuntimeError('evaluation threshold candidates must be sorted')
    return vals

def load_and_verify_target_lock(path,pairs,dataset_name,dataset_version):
    lock=json.loads(Path(path).read_text())
    if lock.get('lock_type')!=TARGET_LOCK_TYPE: raise RuntimeError(f'invalid target lock type: {lock.get("lock_type")!r}')
    if lock.get('status')!='PASS': raise RuntimeError('target lock status is not PASS')
    if not str(lock.get('benchmark_scope','')).strip(): raise RuntimeError('target lock is missing explicit benchmark_scope')
    if lock.get('dataset_name')!=dataset_name or lock.get('dataset_version')!=dataset_version:
        raise RuntimeError('target lock dataset name/version mismatch')
    cur=split_identity(pairs,include_rows=False); saved=lock.get('test_identity') or {}
    for field in ('count','name_manifest_sha256','content_manifest_sha256'):
        if cur.get(field)!=saved.get(field): raise RuntimeError(f'target lock mismatch field={field}: lock={saved.get(field)} current={cur.get(field)}')
    return lock

def load_and_verify_threshold_lock(path,checkpoint_path,checkpoint,source_val_pairs,source_dataset_name=None,source_dataset_version=None):
    lock=json.loads(Path(path).read_text())
    if lock.get('lock_type')!=THRESHOLD_LOCK_TYPE: raise RuntimeError(f'invalid threshold lock type: {lock.get("lock_type")!r}')
    if lock.get('status')!='PASS' or lock.get('target_data_used') is not False: raise RuntimeError('threshold lock is not a clean source-only PASS lock')
    actual_ck=file_sha256(checkpoint_path)
    if lock.get('checkpoint_sha256')!=actual_ck: raise RuntimeError('threshold lock checkpoint SHA256 mismatch')
    if lock.get('checkpoint_config_hash')!=checkpoint.get('config_hash'): raise RuntimeError('threshold lock checkpoint config hash mismatch')
    if lock.get('checkpoint_source_tree_sha256')!=checkpoint.get('source_tree_sha256'): raise RuntimeError('threshold lock checkpoint source-tree mismatch')
    if lock.get('checkpoint_protocol_bundle_sha256')!=checkpoint.get('protocol_bundle_sha256'): raise RuntimeError('threshold lock checkpoint protocol-bundle mismatch')
    if source_dataset_name is not None and lock.get('source_dataset_name')!=source_dataset_name: raise RuntimeError('threshold lock source dataset name mismatch')
    if source_dataset_version is not None and lock.get('source_dataset_version')!=source_dataset_version: raise RuntimeError('threshold lock source dataset version mismatch')
    cur=split_identity(source_val_pairs,include_rows=False); saved=lock.get('source_val_identity') or {}
    for field in ('count','name_manifest_sha256','content_manifest_sha256'):
        if cur.get(field)!=saved.get(field): raise RuntimeError(f'threshold lock source-val mismatch field={field}')
    seeds=lock.get('calibration_seeds') or []
    if not seeds or len(set(seeds))!=len(seeds): raise RuntimeError('threshold lock calibration seeds are missing or duplicated')
    threshold=float(lock['selected_threshold']);candidates=[float(x) for x in (lock.get('threshold_candidates') or [])]
    if threshold not in candidates: raise RuntimeError('threshold lock selected threshold is not in the frozen candidate grid')
    return lock,threshold
