import torch


def _get_model_attr(model, name):
    return getattr(model.module, name) if hasattr(model, "module") else getattr(model, name)


@torch.no_grad()
def crack_meanflow_sampler(model, z, crack_image, num_steps=1, cfg_scale=1.0, clamp=True):
    batch_size = z.shape[0]
    device = z.device
    do_cfg = cfg_scale > 1.0
    sampled = z

    def _forward(sample, r, t, y):
        return model(sample, r, t, y=y)

    if num_steps == 1:
        r = torch.zeros(batch_size, device=device)
        t = torch.ones(batch_size, device=device)
        if do_cfg:
            z2 = torch.cat([sampled, sampled], dim=0)
            r2 = torch.cat([r, r], dim=0)
            t2 = torch.cat([t, t], dim=0)
            y2 = torch.cat([crack_image, torch.zeros_like(crack_image)], dim=0)
            u2 = _forward(z2, r2, t2, y2)
            u_cond, u_uncond = torch.chunk(u2, 2, dim=0)
            u = u_uncond + cfg_scale * (u_cond - u_uncond)
        else:
            u = _forward(sampled, r, t, crack_image)
        sampled = sampled - u
    else:
        grid = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
        for idx in range(num_steps):
            t_cur = torch.full((batch_size,), float(grid[idx].item()), device=device)
            t_next = torch.full((batch_size,), float(grid[idx + 1].item()), device=device)
            if do_cfg:
                z2 = torch.cat([sampled, sampled], dim=0)
                r2 = torch.cat([t_next, t_next], dim=0)
                t2 = torch.cat([t_cur, t_cur], dim=0)
                y2 = torch.cat([crack_image, torch.zeros_like(crack_image)], dim=0)
                u2 = _forward(z2, r2, t2, y2)
                u_cond, u_uncond = torch.chunk(u2, 2, dim=0)
                u = u_uncond + cfg_scale * (u_cond - u_uncond)
            else:
                u = _forward(sampled, t_next, t_cur, crack_image)
            sampled = sampled + (t_next[:, None, None, None] - t_cur[:, None, None, None]) * u

    if clamp:
        sampled = sampled.clamp(-1.0, 1.0)
    seg_logits = _get_model_attr(model, "get_seg_logits")()
    return sampled, seg_logits
