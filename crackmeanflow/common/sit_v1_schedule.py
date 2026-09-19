from __future__ import annotations

SCREEN_STEPS=(4125,8250,12375,16500)
MILESTONE_STEPS=SCREEN_STEPS+(21000,)

def resolve_sit_v1_budget(train_cfg, stop_after_step=None, research_total_steps=None):
    configured=int(train_cfg.get('research_total_steps',0) or 0)
    total=int(research_total_steps) if research_total_steps is not None else configured
    if total!=21000: raise ValueError('CRACKMEANFLOW_SIT_V1 research_total_steps must equal 21000')
    configured_stop=train_cfg.get('stop_after_step')
    stop=int(stop_after_step) if stop_after_step is not None else (int(configured_stop) if configured_stop is not None else total)
    if stop<1 or stop>total: raise ValueError('stop_after_step must lie in [1, research_total_steps]')
    return total,stop

def milestone_steps_from_config(train_cfg):
    steps=tuple(int(x) for x in train_cfg.get('milestone_steps',MILESTONE_STEPS))
    if steps!=MILESTONE_STEPS: raise ValueError(f'CRACKMEANFLOW_SIT_V1 milestone_steps must equal {MILESTONE_STEPS}')
    return steps

def resolve_sit_v1_run_class(run_class, stop_after_step, research_total_steps):
    if run_class is None: raise ValueError('CRACKMEANFLOW_SIT_V1 requires explicit --run-class')
    stop,total=int(stop_after_step),int(research_total_steps)
    if run_class=='headline' and stop!=total: raise ValueError('headline SiT-V1 run must complete 21000 steps')
    if run_class=='screen' and stop not in SCREEN_STEPS: raise ValueError(f'screen SiT-V1 run must stop at one of {SCREEN_STEPS}')
    if run_class=='diagnostic' and (stop in SCREEN_STEPS or stop==total): raise ValueError('diagnostic cannot label preregistered screen/headline milestones')
    if run_class not in {'headline','screen','diagnostic'}: raise ValueError(f'unknown run_class={run_class!r}')
    return run_class

def milestone_checkpoint_name(step):
    step=int(step)
    if step not in MILESTONE_STEPS: raise ValueError(f'not a SiT-V1 milestone: {step}')
    return f'step_{step:06d}.pt'
