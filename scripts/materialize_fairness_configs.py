from __future__ import annotations
import argparse,copy,sys
from pathlib import Path
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    ap=argparse.ArgumentParser(description='Materialize preregistered common-optimizer-recipe sensitivity configs without altering model/loss/data protocol.')
    ap.add_argument('--matrix',default=None);ap.add_argument('--out-dir',required=True);a=ap.parse_args();root=Path(__file__).resolve().parents[1];mp=Path(a.matrix) if a.matrix else root/'configs/fairness/common_recipe_matrix.yaml';matrix=yaml.safe_load(mp.read_text());out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    for rname,recipe in matrix['recipes'].items():
        for mname,rel in matrix['models'].items():
            base=yaml.safe_load((root/rel).read_text());cfg=copy.deepcopy(base);tr=cfg['train'];tr['lr']=float(recipe['lr']);tr['weight_decay']=float(recipe['weight_decay']);tr['warmup_epochs']=int(recipe['warmup_epochs']);tr['max_optimizer_steps']=int(matrix['matched_optimizer_steps']);cfg['experiment']=f"{base['experiment']}__{rname}";cfg['protocol_role']=f"optimizer_recipe_sensitivity_{rname}"
            p=out/f'{mname.lower()}__{rname.lower()}.yaml';p.write_text(yaml.safe_dump(cfg,sort_keys=False))
    print(str(out))
if __name__=='__main__':main()
