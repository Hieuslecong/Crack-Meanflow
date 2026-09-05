import torch
from crackmeanflow.journal.geometry.rasterizer import GeometryRasterizer

def test_centerline_edt_rasterizer_is_directly_independent_of_center_channel():
    rast=GeometryRasterizer(max_radius=16,representation='centerline_edt',distance_encoding='sqrt');state=torch.rand(2,2,16,16)*2-1;changed=state.clone();changed[:,0]=torch.rand_like(changed[:,0])*2-1
    a=rast(state);b=rast(changed);assert torch.equal(a,b)

def test_centerline_radius_rasterizer_depends_on_center_channel():
    rast=GeometryRasterizer(max_radius=4,bins=4,representation='centerline_radius');state=torch.zeros(1,2,9,9)-1;state[:,0,4,4]=1;state[:,1,4,4]=0;changed=state.clone();changed[:,0]=-1
    assert not torch.equal(rast(state),rast(changed))
