from __future__ import annotations
import argparse,json,re
from collections import defaultdict
from pathlib import Path
import numpy as np

def _f1_iou(tp,fp,fn):
    if tp+fp+fn==0:return 1.,1.
    pr=tp/max(tp+fp,1e-12);re=tp/max(tp+fn,1e-12);f1=2*pr*re/max(pr+re,1e-12);iou=tp/max(tp+fp+fn,1e-12);return f1,iou

def main():
    ap=argparse.ArgumentParser(description='Parent-group bootstrap for crop datasets; resamples independent parent groups, not crops.')
    ap.add_argument('--per-image-report',required=True);ap.add_argument('--group-regex',default=r'^([^_]+)');ap.add_argument('--bootstrap',type=int,default=10000);ap.add_argument('--seed',type=int,default=12345);ap.add_argument('--out',required=True);a=ap.parse_args()
    d=json.loads(Path(a.per_image_report).read_text());rx=re.compile(a.group_regex);by_name=defaultdict(list)
    for seed_block in d.get('inference_noise',[]):
        for row in seed_block['rows']:by_name[row['name']].append(row)
    if not by_name:raise RuntimeError('per-image report has no rows')
    averaged={}
    for name,rows in by_name.items():
        averaged[name]={k:float(np.mean([float(r[k]) for r in rows])) for k in ('tp','fp','fn','f1','iou')}
    groups=defaultdict(list)
    for name,row in averaged.items():
        m=rx.search(name)
        if not m:raise RuntimeError(f'group regex did not match {name!r}')
        g=m.group(1) if m.groups() else m.group(0);groups[g].append(row)
    gids=sorted(groups)
    if len(gids)<3:raise RuntimeError(f'parent bootstrap requires >=3 groups; got {len(gids)}')
    def stat(selected):
        rows=[r for g in selected for r in groups[g]];tp=sum(r['tp'] for r in rows);fp=sum(r['fp'] for r in rows);fn=sum(r['fn'] for r in rows);f1,iou=_f1_iou(tp,fp,fn);return f1,iou,float(np.mean([r['f1'] for r in rows])),float(np.mean([r['iou'] for r in rows]))
    point=stat(gids);rng=np.random.default_rng(a.seed);boots=np.empty((a.bootstrap,4),dtype=np.float64)
    for i in range(a.bootstrap):boots[i]=stat(rng.choice(gids,size=len(gids),replace=True).tolist())
    names=('pixel_micro_f1','pixel_micro_iou','macro_image_f1','macro_image_iou');metrics={}
    for j,n in enumerate(names):metrics[n]={'point':point[j],'bootstrap_mean':float(boots[:,j].mean()),'ci95_percentile':[float(np.quantile(boots[:,j],.025)),float(np.quantile(boots[:,j],.975))]}
    out={'dataset_name':d.get('dataset_name'),'dataset_version':d.get('dataset_version'),'n_images':len(averaged),'n_parent_groups':len(gids),'group_regex':a.group_regex,'inference_noise_seeds_averaged':[x.get('inference_noise_seed') for x in d.get('inference_noise',[])],'bootstrap_replicates':a.bootstrap,'bootstrap_seed':a.seed,'unit_of_resampling':'parent_group','metrics':metrics}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
