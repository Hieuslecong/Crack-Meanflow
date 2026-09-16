import math
import torch
from crackmeanflow.common.metrics import binary_average_precision,geometry_centerline_metrics
from crackmeanflow.common.evaluation import _aggregate

def test_perfect_ranking_has_unit_auprc():
    s=torch.tensor([.99,.9,.2,.1]);g=torch.tensor([1,1,0,0]);assert math.isclose(binary_average_precision(s,g),1.0,rel_tol=0,abs_tol=1e-12)

def test_reversed_ranking_has_lower_auprc():
    good=binary_average_precision(torch.tensor([.9,.8,.2,.1]),torch.tensor([1,1,0,0]));bad=binary_average_precision(torch.tensor([.1,.2,.8,.9]),torch.tensor([1,1,0,0]));assert bad < good

def test_constant_scores_are_finite_and_equal_prevalence():
    s=torch.ones(10);g=torch.tensor([1,1,1,0,0,0,0,0,0,0]);ap=binary_average_precision(s,g);assert math.isfinite(ap) and math.isclose(ap,.3,rel_tol=0,abs_tol=1e-12)

def test_checkpoint_selection_aggregate_can_skip_exact_auprc():
    coll=[(torch.tensor([[[[.9,.1],[.8,.2]]]]),torch.tensor([[[[1.,0.],[1.,0.]]]]))]
    out=_aggregate(coll,.5,include_structural=False,compute_auprc=False)
    assert 'auprc' not in out and out['auprc_skipped_for_checkpoint_selection'] is True

def test_radius_metric_exposes_generic_name_and_backward_compatible_alias():
    c=torch.zeros(1,1,5,5);c[0,0,2,2]=1;r=torch.ones_like(c)
    out=geometry_centerline_metrics(c,c,r,r,max_radius=4.)
    assert out['radius_mae_px']==out['edt_radius_mae_px']==0.0
