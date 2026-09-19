from __future__ import annotations
import json,os,sys
from pathlib import Path
import yaml
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crackmeanflow.common.sit_v1_provenance import validate_sit_v1_pair, sit_v1_protocol_sha256

ROOT=Path(__file__).resolve().parents[1]
FILES={'S0':'s0_sit_mf.yaml','S1':'s1_sit_imf.yaml'}

def main():
    configs={k:yaml.safe_load((ROOT/'configs'/'sit_v1'/v).read_text()) for k,v in FILES.items()}
    pair=validate_sit_v1_pair(configs)
    out={
        'status':'PASS',
        'protocol_id':'CRACKMEANFLOW_SIT_V1',
        'variants':sorted(configs),
        'pairwise_differences':pair['differences'],
        'protocol_sha256':sit_v1_protocol_sha256(ROOT),
        'development_ood_metrics_previously_seen':True,
        'development_ood_role':'model_selection_only',
        'final_external_accessed':False,
    }
    print(json.dumps(out,indent=2,sort_keys=True))
if __name__=='__main__':main()
