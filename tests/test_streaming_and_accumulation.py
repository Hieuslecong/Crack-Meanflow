import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from crackmeanflow.common.scheduler import optimizer_steps_per_epoch
from crackmeanflow.common.evaluation import evaluate_with_threshold

class TinyDS(Dataset):
    def __len__(self): return 3
    def __getitem__(self, i):
        m=torch.zeros(1,4,4); m[:,i%4,:2]=1
        return {'crack':torch.zeros(3,4,4), 'mask':m*2-1}

class ZeroModel(nn.Module):
    def forward(self,*args,**kwargs): return args[0]*0

def sampler(model,z,img,num_steps=1,cfg_scale=1.0,clamp=False):
    # deterministic perfect target is not available; just return background-like field
    return torch.full_like(z,-1), None

def test_drop_incomplete_accumulation_uses_full_groups_only():
    assert optimizer_steps_per_epoch(1682,8,True)==210
    assert optimizer_steps_per_epoch(1682,8,False)==211
    assert optimizer_steps_per_epoch(420,2,True)==210

def test_fixed_threshold_evaluation_reports_streaming_mode():
    out=evaluate_with_threshold(ZeroModel(),DataLoader(TinyDS(),batch_size=1),torch.device('cpu'),sampler,0.0,num_steps=1,seed=0)
    assert out['evaluation_memory_mode']=='streaming'
    assert out['empty_gt_images']==0
