from __future__ import annotations

def resolve_option_c_budget(train_cfg, stop_after_step=None, research_total_steps=None):
    """Resolve canonical scheduler horizon separately from the current run stop.

    Option-C diagnostics must follow the exact prefix of the canonical 21k LR
    trajectory. ``research_total_steps`` controls the scheduler; ``run_stop``
    controls how many optimizer updates are executed in this invocation.
    """
    configured_total=int(train_cfg.get('research_total_steps',train_cfg.get('max_optimizer_steps',0) or 0))
    total=int(research_total_steps) if research_total_steps is not None else configured_total
    if total < 1: raise ValueError('research_total_steps must be >=1')
    configured_stop=train_cfg.get('stop_after_step')
    stop=int(stop_after_step) if stop_after_step is not None else (int(configured_stop) if configured_stop is not None else total)
    if stop < 1: raise ValueError('stop_after_step must be >=1')
    if stop > total: raise ValueError('stop_after_step cannot exceed research_total_steps')
    return total,stop
