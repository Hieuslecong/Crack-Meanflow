import torch
from crackmeanflow.journal.geometry.rasterizer import GeometryRasterizer
from crackmeanflow.journal.losses.geometry_interval_consistency import geometry_consistency_components


def _state(cpos=(8,8),radius=2.,size=17,max_radius=4):
    s=torch.full((1,2,size,size),-1.);y,x=cpos;s[0,0,y,x]=1.;s[:,1]=2*(radius/max_radius)-1;return s

def test_identical_states_have_near_zero_gic():
    rast=GeometryRasterizer(max_radius=4,bins=5,representation='centerline_radius');a=_state();loss,p=geometry_consistency_components(a,a.clone(),rast);assert float(loss)<1e-5;assert all(float(v)<1e-5 for v in p.values())

def test_center_perturbation_activates_center_and_mask_terms():
    rast=GeometryRasterizer(max_radius=4,bins=5,representation='centerline_radius');a=_state((8,8));b=_state((8,10));_,p=geometry_consistency_components(a,b,rast);assert float(p['gic_center'])>0 and float(p['gic_mask'])>0

def test_radius_perturbation_activates_radius_and_rendered_mask_terms():
    rast=GeometryRasterizer(max_radius=4,bins=5,temperature=.25,representation='centerline_radius');a=_state(radius=1.);b=_state(radius=3.);_,p=geometry_consistency_components(a,b,rast);assert float(p['gic_radius'])>0 and float(p['gic_mask'])>0

def test_gic_is_finite_and_differentiable():
    rast=GeometryRasterizer(max_radius=4,bins=5,representation='centerline_radius');a=torch.nn.Parameter(_state());b=_state((9,8),3.);loss,p=geometry_consistency_components(a,b,rast);loss.backward();assert torch.isfinite(loss) and all(torch.isfinite(v) for v in p.values());assert a.grad is not None and torch.isfinite(a.grad).all() and float(a.grad.abs().sum())>0
