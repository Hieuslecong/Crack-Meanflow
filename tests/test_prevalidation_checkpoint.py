import torch
from crackmeanflow.common.checkpointing import save_checkpoint_atomic
from crackmeanflow.common.ema import EMA
from crackmeanflow.common.scheduler import make_warmup_cosine_scheduler


def test_last_checkpoint_can_be_saved_before_first_validation(tmp_path):
    model=torch.nn.Conv2d(1,1,1)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
    sched=make_warmup_cosine_scheduler(opt,epochs=5,optimizer_steps_epoch=1,warmup_epochs=0)
    ema=EMA(model,0.99)
    path=tmp_path/'last.pt'
    save_checkpoint_atomic(
        str(path),model=model,ema=ema,optimizer=opt,scheduler=sched,epoch=0,
        global_optimizer_step=1,cfg={'track':'conference'},best_val_metric=-1.0,
        best_val_threshold=None,seed=0,extra_state={'epoch_complete':True},
    )
    ck=torch.load(path,map_location='cpu',weights_only=False)
    assert ck['best_val_threshold'] is None
    assert ck['best_val_metric']==-1.0
