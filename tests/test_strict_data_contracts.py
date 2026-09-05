from pathlib import Path
import json
import numpy as np
from PIL import Image
import pytest
import torch

from crackmeanflow.common.data import (
    PairedCrackDataset, audit_content_split_integrity, source_splits_for_config,
    split_identity, discover_evaluation_pairs,
)
from crackmeanflow.common.protocol import load_and_verify_target_lock, TARGET_LOCK_TYPE


def _img(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8)).save(path)


def _pair(root, split, stem, mask_value=255, image_value=100):
    ip=root/split/'cracked'/'images'/f'{stem}.png'
    mp=root/split/'cracked'/'masks'/f'{stem}.png'
    _img(ip, np.full((8,8,3), image_value, np.uint8))
    m=np.zeros((8,8),np.uint8);m[2:6,2:6]=mask_value
    _img(mp,m)
    return (stem,str(ip),str(mp))


def test_auto_binary_safe_preserves_zero_one_masks(tmp_path):
    p=_pair(tmp_path,'train','001_a',mask_value=1)
    ds=PairedCrackDataset([p],image_size=8,mask_binarization='auto_binary_safe')
    mask=ds[0]['mask']
    assert (mask>0).sum().item()==16
    assert set(torch.unique(mask).tolist())=={-1.0,1.0}


def test_exact_image_content_leakage_fails_even_with_renamed_stems(tmp_path):
    a=_pair(tmp_path,'train','001_a',image_value=77)
    b=_pair(tmp_path,'val','002_b',image_value=77)
    c=_pair(tmp_path,'test','003_c',image_value=88)
    with pytest.raises(RuntimeError,match='image-content leakage'):
        audit_content_split_integrity({'train':[a],'val':[b],'test':[c]})


def test_source_splits_reconstruct_required_normal_negatives(tmp_path):
    for split,stem,val in [('train','001_a',10),('val','002_a',20),('test','003_a',30)]:
        _pair(tmp_path,split,stem,image_value=val)
    normal=tmp_path/'val'/'normal'/'images'/'negative.png';_img(normal,np.zeros((8,8,3),np.uint8))
    cfg={'train':{'normal_negatives':{'train':False,'val':True,'test':False}}}
    sp=source_splits_for_config(str(tmp_path),cfg)
    assert any(n.startswith('NORMAL::') and m is None for n,_,m in sp['val'])


def test_target_lock_rejects_post_lock_content_mutation(tmp_path):
    p=_pair(tmp_path,'test','x',image_value=11)
    pairs=discover_evaluation_pairs(str(tmp_path),'test',False)
    ident=split_identity(pairs,include_rows=False)
    lock={'lock_type':TARGET_LOCK_TYPE,'status':'PASS','benchmark_scope':'unit-test full test split','dataset_name':'T','dataset_version':'v1','include_normal_negatives':False,'test_identity':ident}
    lp=tmp_path/'lock.json';lp.write_text(json.dumps(lock))
    load_and_verify_target_lock(lp,pairs,'T','v1')
    # mutate image bytes without changing filename
    _img(Path(p[1]),np.full((8,8,3),99,np.uint8))
    mutated=discover_evaluation_pairs(str(tmp_path),'test',False)
    with pytest.raises(RuntimeError,match='target lock mismatch'):
        load_and_verify_target_lock(lp,mutated,'T','v1')
