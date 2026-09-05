from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from crackmeanflow.common.data import split_identity,verify_source_provenance,target_dataset_identity

def _pairs(tmp_path):
    out=[]
    for i in range(2):
        ip=tmp_path/f'i{i}.png';mp=tmp_path/f'm{i}.png';Image.fromarray(np.full((4,4,3),i*20,dtype=np.uint8)).save(ip);Image.fromarray(np.eye(4,dtype=np.uint8)*255).save(mp);out.append((f's{i}',str(ip),str(mp)))
    return out

def test_content_provenance_detects_byte_mutation(tmp_path):
    pairs=_pairs(tmp_path);ident=split_identity(pairs);ck={'source_provenance':{'splits':{'train':ident,'val':ident}}};assert verify_source_provenance(ck,{'train':pairs,'val':pairs})['verified']=='content'
    Image.fromarray(np.full((4,4,3),99,dtype=np.uint8)).save(pairs[0][1])
    with pytest.raises(RuntimeError,match='content_manifest_sha256'):verify_source_provenance(ck,{'train':pairs,'val':pairs})

def test_target_identity_is_separate_from_source(tmp_path):
    pairs=_pairs(tmp_path);target=target_dataset_identity({'test':pairs},'GAPS384','v1');assert target['dataset_name']=='GAPS384';assert target['splits']['test']['count']==2;assert 'content_manifest_sha256' in target['splits']['test']

def test_strict_ema_contract_rejects_missing_weight():
    import importlib.util
    import torch
    from torch import nn
    script=Path(__file__).resolve().parents[1]/'scripts/evaluate_journal.py'
    spec=importlib.util.spec_from_file_location('evaluate_journal',script)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    model=nn.Sequential(nn.Linear(3,4),nn.Linear(4,1))
    ema={k:v.detach().clone() for k,v in model.state_dict().items() if v.dtype.is_floating_point}
    ema.pop(next(iter(ema)))
    try:
        mod._ema(model,{'ema':ema})
    except RuntimeError as exc:
        assert 'EMA state mismatch' in str(exc)
    else:
        raise AssertionError('missing EMA weight must be rejected')
