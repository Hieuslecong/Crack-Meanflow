from __future__ import annotations
import argparse,json,sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from crackmeanflow.common import discover_required_splits,discover_evaluation_pairs,load_and_verify_target_lock,file_sha256

def main():
    ap=argparse.ArgumentParser(description='Fail-fast exact-content contamination audit between source CFD and a locked target benchmark.')
    ap.add_argument('--source-data',required=True);ap.add_argument('--target-data',required=True);ap.add_argument('--dataset-name',required=True);ap.add_argument('--dataset-version',required=True);ap.add_argument('--target-lock',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    source=discover_required_splits(a.source_data);preview=json.loads(Path(a.target_lock).read_text());target=discover_evaluation_pairs(a.target_data,'test',bool(preview.get('include_normal_negatives',False)));load_and_verify_target_lock(a.target_lock,target,a.dataset_name,a.dataset_version)
    sh=defaultdict(list)
    for split,pairs in source.items():
        for name,ip,_ in pairs:sh[file_sha256(ip)].append((split,name))
    overlap=[]
    for name,ip,_ in target:
        h=file_sha256(ip)
        if h in sh:overlap.append({'target_name':name,'image_sha256':h,'source_matches':sh[h]})
    report={'source_splits':{k:len(v) for k,v in source.items()},'target_dataset':a.dataset_name,'target_version':a.dataset_version,'target_count':len(target),'exact_image_content_overlap_count':len(overlap),'examples':overlap[:50],'status':'PASS' if not overlap else 'FAIL_CONTAMINATION'}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    if overlap:raise SystemExit(2)
if __name__=='__main__':main()
