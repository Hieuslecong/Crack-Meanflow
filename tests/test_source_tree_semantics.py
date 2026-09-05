from pathlib import Path
import shutil
from crackmeanflow.common.checkpointing import source_tree_hash

ROOT=Path(__file__).resolve().parents[1]

def test_source_tree_hash_ignores_meta_config_but_tracks_execution_code(tmp_path):
    # copy minimum relevant tree into isolated root
    shutil.copytree(ROOT/'crackmeanflow',tmp_path/'crackmeanflow')
    (tmp_path/'scripts').mkdir()
    for name in ('train_journal.py','evaluate_journal.py','freeze_source_threshold.py'):
        shutil.copy2(ROOT/'scripts'/name,tmp_path/'scripts'/name)
    shutil.copy2(ROOT/'requirements.txt',tmp_path/'requirements.txt')
    shutil.copy2(ROOT/'pytest.ini',tmp_path/'pytest.ini')
    h0=source_tree_hash(tmp_path)
    (tmp_path/'configs').mkdir();(tmp_path/'configs'/'new_meta.yaml').write_text('x: 1\n')
    assert source_tree_hash(tmp_path)==h0
    p=tmp_path/'scripts'/'evaluate_journal.py';p.write_text(p.read_text()+'\n# semantic hash test\n')
    assert source_tree_hash(tmp_path)!=h0
