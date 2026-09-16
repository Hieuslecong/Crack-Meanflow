import torch
from crackmeanflow.journal.flow.improved_meanflow import _disjoint_stratified_schedule

def test_j4_endpoint_schedule_exact_full_block_and_disjoint():
    fm,ep=_disjoint_stratified_schedule(1000,.50,.15,0,torch.device('cpu'));assert int(ep.sum())==150;assert int(fm.sum())==500;assert not bool((fm & ep).any());assert int((~fm & ~ep).sum())==350

def test_j4_endpoint_schedule_repeats_exact_rate_across_blocks():
    for offset in (0,1000,2000,7000):
        fm,ep=_disjoint_stratified_schedule(1000,.50,.15,offset,torch.device('cpu'));assert int(ep.sum())==150 and int(fm.sum())==500 and not bool((fm&ep).any())
