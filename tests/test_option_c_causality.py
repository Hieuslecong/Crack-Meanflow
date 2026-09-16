import torch
import torch.nn.functional as F
from crackmeanflow.journal.geometry.rasterizer import GeometryRasterizer


def _state(center, radius, max_radius=4.0):
    c=(center.clamp(0,1)*2-1);r=((radius/max_radius).clamp(0,1)*2-1);return torch.cat([c,r],1)

def _point(h=25,w=25,y=12,x=12):
    c=torch.zeros(1,1,h,w);c[0,0,y,x]=1;return c

def test_zero_centerline_collapses_mask():
    rast=GeometryRasterizer(max_radius=4,bins=5,representation='centerline_radius');c=torch.zeros(1,1,17,17);r=torch.full_like(c,3.)
    assert float(rast(_state(c,r,4)).max()) == 0.0

def test_centerline_translation_translates_rendered_support():
    rast=GeometryRasterizer(max_radius=3,bins=4,representation='centerline_radius');c1=_point(25,25,10,10);c2=_point(25,25,13,14);r=torch.full_like(c1,2.)
    m1=rast(_state(c1,r,3));m2=rast(_state(c2,r,3));shifted=torch.roll(m1,shifts=(3,4),dims=(-2,-1));assert torch.allclose(m2,shifted,atol=1e-6)

def test_radius_monotonically_increases_area():
    rast=GeometryRasterizer(max_radius=4,bins=5,temperature=.25,representation='centerline_radius');c=_point();r1=torch.full_like(c,1.);r2=torch.full_like(c,3.)
    assert float(rast(_state(c,r2,4)).sum()) > float(rast(_state(c,r1,4)).sum())

def test_mask_loss_has_finite_nonzero_center_and_radius_gradients():
    rast=GeometryRasterizer(max_radius=4,bins=5,temperature=.5,representation='centerline_radius');center_logits=torch.nn.Parameter(torch.full((1,1,17,17),-6.));radius_logits=torch.nn.Parameter(torch.zeros(1,1,17,17))
    with torch.no_grad(): center_logits[0,0,8,8]=4.
    center=torch.sigmoid(center_logits);radius=4*torch.sigmoid(radius_logits);pred=rast.forward_fields(center,radius);target=torch.zeros_like(pred);target[:,:,6:11,6:11]=1
    loss=F.binary_cross_entropy(pred.clamp(1e-6,1-1e-6),target);loss.backward()
    assert center_logits.grad is not None and torch.isfinite(center_logits.grad).all() and float(center_logits.grad.abs().sum())>0
    assert radius_logits.grad is not None and torch.isfinite(radius_logits.grad).all() and float(radius_logits.grad.abs().sum())>0

def test_empty_geometry_is_stable_and_finite():
    rast=GeometryRasterizer(max_radius=4,bins=5,representation='centerline_radius');c=torch.zeros(2,1,19,19);r=torch.zeros_like(c);m=rast(_state(c,r,4));assert torch.isfinite(m).all() and float(m.abs().sum())==0.0

def _check_junction(points):
    rast=GeometryRasterizer(max_radius=2,bins=3,temperature=.25,representation='centerline_radius');c=torch.zeros(1,1,31,31)
    for y,x in points:c[0,0,y,x]=1
    r=torch.full_like(c,1.5);m=rast(_state(c,r,2));vals=torch.stack([m[0,0,y,x] for y,x in points]);assert float(vals.min())>.5

def test_single_line_reconstruction():_check_junction([(15,x) for x in range(8,23)])
def test_y_junction_reconstruction():
    pts=[(y,15) for y in range(8,16)]+[(15+i,15-i) for i in range(0,8)]+[(15+i,15+i) for i in range(0,8)];_check_junction(pts)
def test_x_junction_reconstruction():
    pts=[(8+i,8+i) for i in range(15)]+[(8+i,22-i) for i in range(15)];_check_junction(pts)
