import torch
from crackmeanflow.common.scheduler import make_warmup_cosine_scheduler

def test_scheduler_honors_exact_optimizer_budget():
    p=torch.nn.Parameter(torch.tensor(1.));opt=torch.optim.AdamW([p],lr=1e-3);s=make_warmup_cosine_scheduler(opt,epochs=100,optimizer_steps_epoch=20,warmup_epochs=10,total_optimizer_steps=333)
    assert s._cmf_total_steps==333 and s._cmf_warmup_steps==200
