from pathlib import Path
from crackmeanflow.common.checkpointing import source_tree_hash

def test_source_tree_hash_changes_with_source_content(tmp_path):
    (tmp_path/'crackmeanflow').mkdir(); (tmp_path/'scripts').mkdir(); (tmp_path/'configs').mkdir()
    f=tmp_path/'crackmeanflow'/'x.py'; f.write_text('x=1\n')
    h1=source_tree_hash(tmp_path); f.write_text('x=2\n'); h2=source_tree_hash(tmp_path)
    assert h1!=h2
