from pathlib import Path
import pytest
from crackmeanflow.common.data import audit_within_split_duplicate_images

def test_within_split_exact_duplicate_image_content_is_rejected(tmp_path):
    a=tmp_path/'a.png';b=tmp_path/'b.png';m1=tmp_path/'m1.png';m2=tmp_path/'m2.png'
    a.write_bytes(b'same');b.write_bytes(b'same');m1.write_bytes(b'm1');m2.write_bytes(b'm2')
    with pytest.raises(RuntimeError,match='duplicate image-content'):
        audit_within_split_duplicate_images({'train':[('a',str(a),str(m1)),('b',str(b),str(m2))],'val':[],'test':[]})
