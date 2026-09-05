import torch
from crackmeanflow.common.evaluation import _aggregate

def test_micro_and_macro_metrics_are_reported_separately():
    # image 1 perfect positive pixel; image 2 predicts all foreground against one positive
    gt1=torch.zeros(1,1,2,2);gt1[0,0,0,0]=1
    p1=gt1*2-1
    gt2=torch.zeros(1,1,2,2);gt2[0,0,0,0]=1
    p2=torch.ones_like(gt2) # threshold 0 => all foreground
    out=_aggregate([(p1,gt1),(p2,gt2)],0.0)
    assert 'f1_macro_image' in out and 'iou_macro_image' in out
    assert 'f1_macro_positive_image' in out and 'empty_gt_false_positive_rate' in out
    assert out['metric_aggregation'].startswith('f1/iou')
    assert 0 <= out['gt_foreground_ratio'] <= 1
    assert 0 <= out['pred_foreground_ratio'] <= 1
