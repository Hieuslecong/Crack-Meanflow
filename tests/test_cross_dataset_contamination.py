from pathlib import Path
import pytest
from crackmeanflow.common import audit_cross_dataset_image_content_overlap

def test_cross_dataset_exact_image_overlap_fails(tmp_path):
    a=tmp_path/'a.bin';a.write_bytes(b'same')
    m=tmp_path/'m.bin';m.write_bytes(b'mask')
    src={'train':[('src',str(a),str(m))],'val':[],'test':[]}
    with pytest.raises(RuntimeError,match='source-target image-content contamination'):
        audit_cross_dataset_image_content_overlap(src,[('tgt',str(a),str(m))])

def test_cross_dataset_disjoint_passes(tmp_path):
    a=tmp_path/'a.bin';a.write_bytes(b'a');b=tmp_path/'b.bin';b.write_bytes(b'b');m=tmp_path/'m.bin';m.write_bytes(b'm')
    r=audit_cross_dataset_image_content_overlap({'train':[('src',str(a),str(m))]},[('tgt',str(b),str(m))])
    assert r['exact_source_target_image_content_overlap_count']==0
