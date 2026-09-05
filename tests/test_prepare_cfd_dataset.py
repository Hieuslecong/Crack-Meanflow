from pathlib import Path
import subprocess,sys
from PIL import Image
import numpy as np
from crackmeanflow.common import discover_required_splits,audit_group_integrity

ROOT=Path(__file__).resolve().parents[1]

def _group(stem): return stem.split('_',1)[0]

def test_prepare_flat_cfd_is_parent_disjoint_and_loader_compatible(tmp_path):
    src=tmp_path/'flat';(src/'image').mkdir(parents=True);(src/'label').mkdir()
    for parent in range(1,7):
        for crop in range(3):
            Image.fromarray(np.full((16,16,3),parent*10+crop,dtype=np.uint8)).save(src/'image'/f'{parent:03d}_img_{crop}.jpg')
            m=np.zeros((16,16),dtype=np.uint8);m[crop:crop+2,:]=255;Image.fromarray(m).save(src/'label'/f'{parent:03d}_msk_{crop}.png')
    out=tmp_path/'prepared'
    subprocess.check_call([sys.executable,str(ROOT/'scripts/prepare_cfd_dataset.py'),'--input',str(src),'--out',str(out),'--seed','1'])
    splits=discover_required_splits(str(out));audit_group_integrity(splits,_group)
    assert sum(map(len,splits.values()))==18
    assert all('_img_' not in p[0] and '_msk_' not in p[0] for rows in splits.values() for p in rows)

def test_prepare_rejects_exact_duplicate_images_even_with_distinct_names(tmp_path):
    src=tmp_path/'flat';(src/'image').mkdir(parents=True);(src/'label').mkdir()
    # Six parents are needed for a valid 3-way parent split. Two source images are byte-identical.
    base=np.full((16,16,3),77,dtype=np.uint8)
    for parent in range(1,7):
        arr=base if parent in (1,2) else np.full((16,16,3),parent*20,dtype=np.uint8)
        Image.fromarray(arr).save(src/'image'/f'{parent:03d}_img_0.png')
        m=np.zeros((16,16),dtype=np.uint8);m[parent%16,:]=255;Image.fromarray(m).save(src/'label'/f'{parent:03d}_msk_0.png')
    out=tmp_path/'prepared'
    r=subprocess.run([sys.executable,str(ROOT/'scripts/prepare_cfd_dataset.py'),'--input',str(src),'--out',str(out),'--seed','1'],capture_output=True,text=True)
    assert r.returncode!=0
    assert 'exact duplicate image content' in (r.stdout+r.stderr)
