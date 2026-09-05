import torch
from crackmeanflow.conference.losses import ConferenceMeanFlowLoss

def test_conference_reports_realized_fm_count_after_endpoint_override():
    loss=ConferenceMeanFlowLoss(ratio_r_not_equal_t=.5,boundary_prob=.15,boundary_sampling='stratified',time_curriculum={'enabled':False})
    r,t,stage=loss._sample_r_t(1000,'cpu',return_stage=True,sample_offset=0)
    assert stage['fm_count']==int((r==t).sum())
    assert abs(stage['realized_fm_fraction']-stage['fm_count']/1000)<1e-7
    assert stage['boundary_count']==150
