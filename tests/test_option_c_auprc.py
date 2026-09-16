import math
import torch
from crackmeanflow.common.metrics import binary_average_precision

def test_perfect_ranking_has_unit_auprc():
    s=torch.tensor([.99,.9,.2,.1]);g=torch.tensor([1,1,0,0]);assert math.isclose(binary_average_precision(s,g),1.0,rel_tol=0,abs_tol=1e-12)

def test_reversed_ranking_has_lower_auprc():
    good=binary_average_precision(torch.tensor([.9,.8,.2,.1]),torch.tensor([1,1,0,0]));bad=binary_average_precision(torch.tensor([.1,.2,.8,.9]),torch.tensor([1,1,0,0]));assert bad < good

def test_constant_scores_are_finite_and_equal_prevalence():
    s=torch.ones(10);g=torch.tensor([1,1,1,0,0,0,0,0,0,0]);ap=binary_average_precision(s,g);assert math.isfinite(ap) and math.isclose(ap,.3,rel_tol=0,abs_tol=1e-12)
