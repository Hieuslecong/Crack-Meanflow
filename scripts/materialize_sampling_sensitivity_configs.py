from __future__ import annotations
import argparse,copy,sys
from pathlib import Path
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    ap=argparse.ArgumentParser(description='Materialize preregistered CFD sampling sensitivity configs.')
    ap.add_argument('--matrix',default=None);ap.add_argument('--out-dir',required=True);a=ap.parse_args()
    root=Path(__file__).resolve().parents[1];mp=Path(a.matrix) if a.matrix else root/'configs/fairness/data_sampling_matrix.yaml'
    matrix=yaml.safe_load(mp.read_text());out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    for vname,variant in matrix['variants'].items():
        for mname,rel in matrix['models'].items():
            base=yaml.safe_load((root/rel).read_text());cfg=copy.deepcopy(base)
            sb=cfg['train'].setdefault('sample_balance',{});sb.clear();sb.update(copy.deepcopy(variant))
            if not sb.get('enabled',False):
                sb['rationale']='Canonical CFD-only no-replacement sampling: every crop appears once per epoch.'
            cfg['experiment']=f"{base['experiment']}__{vname}";cfg['protocol_role']=f"sampling_sensitivity_{vname.lower()}"
            p=out/f'{mname.lower()}__{vname.lower()}.yaml';p.write_text(yaml.safe_dump(cfg,sort_keys=False))
    print(str(out))
if __name__=='__main__':main()
