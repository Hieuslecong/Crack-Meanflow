import torch
from torch import nn
from crackmeanflow.journal.flow.improved_meanflow import _stratified_mask,ImprovedMeanFlowGeometryLoss
from crackmeanflow.journal.geometry.rasterizer import GeometryRasterizer

def test_stratified_gic_fraction_is_sample_based_for_multiple_batch_sizes():
    for bs in [1,2,4,8,16,20]:
        total=active=off=0
        while total<1000:
            b=min(bs,1000-total);m=_stratified_mask(b,.25,off,'cpu');active+=int(m.sum());total+=b;off+=b
        assert active/total==.25

def test_gic_logs_active_samples_not_only_batches():
    class Dummy(nn.Module):
        def __init__(self):super().__init__();self.scale=nn.Parameter(torch.tensor(.1))
        def flow_outputs(self,z,t,r,image):
            im=image.mean(1,keepdim=True).repeat(1,z.shape[1],1,1);clean=torch.tanh(self.scale*z+.05*im);tc=t.clamp_min(.05).reshape(-1,1,1,1);u=(z-clean)/tc;return {'u':u,'v':u*.9,'clean_u':clean,'clean_v':clean}
    b=8;g=torch.rand(b,2,4,4)*2-1;image=torch.rand(b,3,4,4);mask=(torch.rand(b,1,4,4)>.7).float()*2-1;rast=GeometryRasterizer(max_radius=4,representation='centerline_edt');lossfn=ImprovedMeanFlowGeometryLoss(gic_probability=.25,gic_weight=.1,rasterizer=rast,max_radius=4)
    loss,logs=lossfn(Dummy(),g,image,mask_gt=mask,sample_offset=0);assert torch.isfinite(loss);assert logs['gic_active_samples']==2.;assert logs['gic_active_batches']==1.;assert logs['batch_size']==8.;assert abs(logs['gic_active_fraction']-.25)<1e-9

def test_endpoint_quota_is_exact_at_sample_level_across_batch_sizes():
    rast=GeometryRasterizer(max_radius=4,representation='centerline_edt')
    lossfn=ImprovedMeanFlowGeometryLoss(endpoint_probability=.15,endpoint_sampling='stratified_disjoint',gic_weight=0,rasterizer=rast,max_radius=4)
    for bs in [1,2,4,8,16,20]:
        active=total=off=0
        while total<2000:
            b=min(bs,2000-total);_,_,_,ep=lossfn.sample_tr(b,'cpu',off);active+=int(ep.sum());total+=b;off+=b
        assert active/total==.15

def test_independent_stratified_streams_remove_fm_gic_aliasing():
    from crackmeanflow.journal.flow.improved_meanflow import _independent_stratified_mask
    n=100000
    fm=_stratified_mask(n,.5,0,'cpu')
    gic=_independent_stratified_mask(n,.25,0,'cpu','gic')
    endpoint=_independent_stratified_mask(n,.15,0,'cpu','endpoint')
    assert abs(float(gic.float().double().mean())-.25)<1e-7
    assert abs(float(endpoint.float().double().mean())-.15)<1e-7
    assert abs(float((fm & gic).float().mean())-.125)<.002
    assert abs(float((fm & endpoint).float().mean())-.075)<.002
    assert abs(float((gic & endpoint).float().mean())-.0375)<.003


def test_endpoint_schedule_preserves_fm_fraction():
    rast=GeometryRasterizer(max_radius=4,representation='centerline_edt')
    lossfn=ImprovedMeanFlowGeometryLoss(data_proportion=.5,endpoint_probability=.15,endpoint_sampling='stratified_disjoint',gic_weight=0,rasterizer=rast,max_radius=4)
    t,r,fm,ep=lossfn.sample_tr(100000,'cpu',0)
    assert abs(float(fm.double().mean())-.50)<1e-12
    assert abs(float(ep.double().mean())-.15)<1e-12
    assert not bool((fm&ep).any())
