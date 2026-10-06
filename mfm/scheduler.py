from timm.scheduler.cosine_lr import CosineLRScheduler


def build_scheduler(cfg, optimizer, n_iter_per_epoch):
    scheduler_name = cfg.training.lr_scheduler.name.lower()

    if scheduler_name != "cosine":
        raise ValueError(
            f"Unsupported scheduler: {scheduler_name}"
        )

    num_steps = int(
        cfg.training.epochs * n_iter_per_epoch
    )
    warmup_steps = int(
        cfg.training.warmup_epochs * n_iter_per_epoch
    )

    return CosineLRScheduler(
        optimizer,
        t_initial=num_steps,
        cycle_mul=1.0,
        lr_min=cfg.training.min_lr,
        warmup_lr_init=cfg.training.warmup_lr,
        warmup_t=warmup_steps,
        cycle_limit=1,
        t_in_epochs=False,
    )