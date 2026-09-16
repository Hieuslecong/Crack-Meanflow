from __future__ import annotations

SOURCE_SCREEN_STEPS=(4125,8250,12375,16500)
CANONICAL_MILESTONE_STEPS=SOURCE_SCREEN_STEPS+(21000,)


def resolve_option_c_budget(train_cfg, stop_after_step=None, research_total_steps=None):
    """Resolve canonical scheduler horizon separately from the current run stop."""
    configured_total=int(train_cfg.get('research_total_steps',0) or 0)
    total=int(research_total_steps) if research_total_steps is not None else configured_total
    if total < 1: raise ValueError('research_total_steps must be >=1')
    configured_stop=train_cfg.get('stop_after_step')
    stop=int(stop_after_step) if stop_after_step is not None else (int(configured_stop) if configured_stop is not None else total)
    if stop < 1: raise ValueError('stop_after_step must be >=1')
    if stop > total: raise ValueError('stop_after_step cannot exceed research_total_steps')
    return total,stop


def milestone_steps_from_config(train_cfg):
    steps=tuple(int(x) for x in train_cfg.get('milestone_steps',CANONICAL_MILESTONE_STEPS))
    if steps != tuple(sorted(set(steps))): raise ValueError('milestone_steps must be strictly increasing and unique')
    if steps != CANONICAL_MILESTONE_STEPS: raise ValueError(f'Option-C milestone_steps must equal {CANONICAL_MILESTONE_STEPS}')
    total=int(train_cfg.get('research_total_steps',0) or 0)
    if total != CANONICAL_MILESTONE_STEPS[-1]: raise ValueError('canonical Option-C research horizon must end at 21000')
    return steps


def resolve_option_c_run_class(run_class, stop_after_step, research_total_steps, screen_steps=SOURCE_SCREEN_STEPS):
    """Fail closed on run-class semantics so scientific screens cannot be mislabeled."""
    if run_class is None: raise ValueError('OPTION_C_V1 requires explicit --run-class')
    stop=int(stop_after_step); total=int(research_total_steps); run_class=str(run_class)
    screens=tuple(int(x) for x in screen_steps)
    if run_class=='headline':
        if stop != total: raise ValueError('headline Option-C runs must complete the full research scheduler horizon')
    elif run_class=='screen':
        if stop not in screens: raise ValueError(f'screen Option-C runs must stop at one of {screens}')
    elif run_class=='diagnostic':
        if stop in screens or stop==total: raise ValueError('diagnostic run_class cannot be used for preregistered screen/headline milestones')
    else:
        raise ValueError(f'unknown run_class={run_class!r}')
    return run_class


def milestone_checkpoint_name(step):
    step=int(step)
    if step not in CANONICAL_MILESTONE_STEPS: raise ValueError(f'not a canonical Option-C milestone: {step}')
    return f'step_{step:06d}.pt'
