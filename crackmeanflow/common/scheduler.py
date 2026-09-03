import math
import torch

def optimizer_steps_per_epoch(num_batches:int, grad_accum:int)->int:
    if grad_accum < 1: raise ValueError('grad_accum must be >=1')
    return math.ceil(num_batches/grad_accum)

def make_warmup_cosine_scheduler(opt, epochs:int, optimizer_steps_epoch:int, warmup_epochs:int=0):
    total=max(1,int(epochs)*int(optimizer_steps_epoch)); warm=max(0,int(warmup_epochs)*int(optimizer_steps_epoch))
    def f(step):
        if warm and step < warm: return max(step,1)/warm
        progress=(step-warm)/max(1,total-warm); progress=min(max(progress,0.),1.); return .5*(1+math.cos(math.pi*progress))
    sched=torch.optim.lr_scheduler.LambdaLR(opt,f); sched._cmf_total_steps=total; sched._cmf_warmup_steps=warm; return sched
