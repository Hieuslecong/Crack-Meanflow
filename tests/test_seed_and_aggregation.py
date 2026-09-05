import json, subprocess, sys
from pathlib import Path

def test_training_seed_aggregator_rejects_duplicate_seed(tmp_path):
    base={'scientific_validity':'VALID_HEADLINE_PROTOCOL','requested_nfe':1,'actual_nfe':1,'nfe_contract_pass':True,'target_identity':{'dataset_name':'X','test':{'content_manifest_sha256':'abc'}},'f1':{'mean':.5}}
    ps=[]
    for i in range(2):
        d=dict(base); d['training_seed']=0; p=tmp_path/f'{i}.json'; p.write_text(json.dumps(d)); ps.append(str(p))
    root=Path(__file__).resolve().parents[1]
    r=subprocess.run([sys.executable,str(root/'scripts/aggregate_training_seeds.py'),*ps,'--out',str(tmp_path/'o.json'),'--min-seeds','2'],capture_output=True,text=True)
    assert r.returncode != 0
    assert 'unique' in (r.stderr+r.stdout)

def test_train_script_exposes_seed_override():
    root=Path(__file__).resolve().parents[1]
    text=(root/'scripts/train_journal.py').read_text()
    assert "--seed" in text
    assert "cfg.setdefault('train',{})['seed']=int(a.seed)" in text


def test_training_seed_aggregator_rejects_mixed_methods(tmp_path):
    base={'scientific_validity':'VALID_HEADLINE_PROTOCOL','requested_nfe':1,'actual_nfe':1,'nfe_contract_pass':True,'target_identity':{'dataset_name':'X','splits':{'test':{'content_manifest_sha256':'abc'}}},'training_budget':{'budget_protocol':'MATCHED_OPTIMIZER_STEPS','planned_optimizer_steps':100,'max_optimizer_steps':100,'effective_samples_per_full_optimizer_step':8,'drop_incomplete_accumulation':True},'f1':{'mean':.5}}
    ps=[]
    for i in range(3):
        d=dict(base); d['training_seed']=i; d['method_config_hash']='A' if i<2 else 'B'; p=tmp_path/f'm{i}.json'; p.write_text(json.dumps(d)); ps.append(str(p))
    root=Path(__file__).resolve().parents[1]
    r=subprocess.run([sys.executable,str(root/'scripts/aggregate_training_seeds.py'),*ps,'--out',str(tmp_path/'o.json')],capture_output=True,text=True)
    assert r.returncode != 0 and 'identity differs' in (r.stderr+r.stdout)

def test_training_seed_aggregator_requires_three_by_default(tmp_path):
    base={'scientific_validity':'VALID_HEADLINE_PROTOCOL','requested_nfe':1,'actual_nfe':1,'nfe_contract_pass':True,'target_identity':{'dataset_name':'X'},'method_config_hash':'A','training_budget':{'budget_protocol':'MATCHED_OPTIMIZER_STEPS','planned_optimizer_steps':100},'f1':{'mean':.5}}
    ps=[]
    for i in range(2):
        d=dict(base); d['training_seed']=i; p=tmp_path/f's{i}.json'; p.write_text(json.dumps(d)); ps.append(str(p))
    root=Path(__file__).resolve().parents[1]
    r=subprocess.run([sys.executable,str(root/'scripts/aggregate_training_seeds.py'),*ps,'--out',str(tmp_path/'o.json')],capture_output=True,text=True)
    assert r.returncode != 0 and 'at least 3' in (r.stderr+r.stdout)
