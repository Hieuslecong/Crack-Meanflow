import subprocess, sys, yaml
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def _run(proto,tmp_path):
    p=tmp_path/'proto.yaml';p.write_text(yaml.safe_dump(proto,sort_keys=False))
    return subprocess.run([sys.executable,str(ROOT/'scripts/protocol_preflight.py'),'--protocol',str(p),'--out',str(tmp_path/'out.json')],cwd=ROOT,capture_output=True,text=True)

def test_protocol_locks_training_surface_fields(tmp_path):
    proto=yaml.safe_load((ROOT/'configs/protocol/paper_protocol.yaml').read_text())
    # Mutate the protocol expectation rather than a canonical config: every model should now fail parity.
    proto['source_dataset']['mask_binarization']='unsafe_fake_mode'
    r=_run(proto,tmp_path)
    assert r.returncode!=0
    assert 'mask-binarization mismatch' in (r.stdout+r.stderr)

def test_protocol_uses_precise_parent_reweighting_name():
    proto=yaml.safe_load((ROOT/'configs/protocol/paper_protocol.yaml').read_text())
    fair=proto['fairness']
    assert fair['parent_sqrt_reweighted_replacement_is_sensitivity_only'] is True
    assert 'parent_balanced_replacement_is_sensitivity_only' not in fair
