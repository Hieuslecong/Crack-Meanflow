from pathlib import Path
import json
import numpy as np
from PIL import Image
import pytest
import torch

from crackmeanflow.common.checkpointing import file_sha256
from crackmeanflow.common.data import split_identity
from crackmeanflow.common.protocol import load_and_verify_threshold_lock, THRESHOLD_LOCK_TYPE


def _pair(tmp_path):
    ip=tmp_path/'i.png';mp=tmp_path/'m.png'
    Image.fromarray(np.zeros((4,4,3),np.uint8)).save(ip)
    m=np.zeros((4,4),np.uint8);m[1:3,1:3]=255;Image.fromarray(m).save(mp)
    return [('v',str(ip),str(mp))]


def _setup(tmp_path):
    ckpath=tmp_path/'ck.pt';torch.save({'config_hash':'cfg','source_tree_sha256':'tree'},ckpath)
    ck=torch.load(ckpath,weights_only=False)
    pairs=_pair(tmp_path);ident=split_identity(pairs,False)
    lock={
        'lock_type':THRESHOLD_LOCK_TYPE,'status':'PASS','target_data_used':False,
        'checkpoint_sha256':file_sha256(ckpath),'checkpoint_config_hash':'cfg','checkpoint_source_tree_sha256':'tree',
        'source_dataset_name':'CFD','source_dataset_version':'v1','source_val_identity':ident,
        'calibration_seeds':[0,1,2],'threshold_candidates':[0.1,0.2,0.3],'selected_threshold':0.2,
    }
    lp=tmp_path/'lock.json';lp.write_text(json.dumps(lock));return ckpath,ck,pairs,lp,lock


def test_threshold_lock_strict_contract_passes(tmp_path):
    ckpath,ck,pairs,lp,_=_setup(tmp_path)
    _,th=load_and_verify_threshold_lock(lp,ckpath,ck,pairs,'CFD','v1')
    assert th==0.2

@pytest.mark.parametrize('field,value,match',[
    ('source_dataset_version','v2','version mismatch'),
    ('checkpoint_source_tree_sha256','other','source-tree mismatch'),
    ('target_data_used',True,'clean source-only'),
    ('selected_threshold',0.25,'not in the frozen candidate'),
])
def test_threshold_lock_rejects_tampering(tmp_path,field,value,match):
    ckpath,ck,pairs,lp,lock=_setup(tmp_path);lock[field]=value;lp.write_text(json.dumps(lock))
    with pytest.raises(RuntimeError,match=match):
        load_and_verify_threshold_lock(lp,ckpath,ck,pairs,'CFD','v1')
