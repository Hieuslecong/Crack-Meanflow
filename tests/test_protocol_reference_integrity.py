import subprocess, sys, yaml
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def test_protocol_preflight_rejects_dangling_ablation_reference(tmp_path):
    proto=yaml.safe_load((ROOT/'configs/protocol/paper_protocol.yaml').read_text())
    proto['ablation_ladder']['bad_test']=['DOES_NOT_EXIST']
    p=tmp_path/'bad.yaml';p.write_text(yaml.safe_dump(proto,sort_keys=False))
    r=subprocess.run([sys.executable,str(ROOT/'scripts/protocol_preflight.py'),'--protocol',str(p),'--out',str(tmp_path/'out.json')],cwd=ROOT,capture_output=True,text=True)
    assert r.returncode!=0
    assert 'undefined model reference' in (r.stdout+r.stderr)

def test_protocol_declares_every_ablation_reference():
    proto=yaml.safe_load((ROOT/'configs/protocol/paper_protocol.yaml').read_text())
    declared=set(proto['models'])
    refs={m for seq in proto['ablation_ladder'].values() for m in seq}
    assert refs <= declared
