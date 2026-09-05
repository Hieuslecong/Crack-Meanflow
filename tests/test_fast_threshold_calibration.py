import torch
from crackmeanflow.common.evaluation import _micro_threshold_sweep_from_score_gt, _aggregate

def test_bucketized_threshold_sweep_matches_slow_reference_including_equalities():
    score=torch.tensor([[[[-1.0,-0.5,0.0],[0.5,1.0,1.5]]]])
    gt=torch.tensor([[[[0.,1.,0.],[1.,1.,0.]]]])
    thresholds=[-0.5,0.0,0.5,1.0]
    fast=_micro_threshold_sweep_from_score_gt([(score,gt)],thresholds)
    coll=[(score,gt)]
    for t in thresholds:
        slow=_aggregate(coll,t,include_structural=False)
        for k in ('tp','fp','fn','tn','f1','iou','precision','recall'):
            assert abs(float(fast[t][k])-float(slow[k]))<1e-9, (t,k,fast[t][k],slow[k])

def test_bucketized_threshold_sweep_empty_gt_semantics():
    score=torch.tensor([[[[-2.0,-1.0]]]])
    gt=torch.zeros_like(score)
    rows=_micro_threshold_sweep_from_score_gt([(score,gt)],[-0.5,0.0])
    assert rows[-0.5]['f1']==1.0
    assert rows[0.0]['iou']==1.0
