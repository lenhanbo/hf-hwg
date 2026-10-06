from torch import optim


def check_keywords_in_name(name, keywords=()):
    return any(keyword in name for keyword in keywords)


def get_pretrain_param_groups(model,skip_list=(),skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        should_skip_decay = (
            len(param.shape) == 1
            or name.endswith(".bias")
            or name in skip_list
            or check_keywords_in_name(
                name,
                skip_keywords,
            )
        )

        if should_skip_decay:
            no_decay.append(param)
        else:
            has_decay.append(param)

    return [
        {"params": has_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_optimizer(cfg, model):
    skip = set()
    skip_keywords = set()

    if hasattr(model, "no_weight_decay"):
        skip = model.no_weight_decay()

    if hasattr(model, "no_weight_decay_keywords"):
        skip_keywords = model.no_weight_decay_keywords()

    parameters = get_pretrain_param_groups(
        model,
        skip_list=skip,
        skip_keywords=skip_keywords,
    )

    optimizer_name = cfg.training.optimizer.name.lower()

    if optimizer_name != "adamw":
        raise ValueError(
            f"Unsupported optimizer: {optimizer_name}"
        )

    return optim.AdamW(
        parameters,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        eps=cfg.training.optimizer.eps,
        betas=tuple(cfg.training.optimizer.betas),
    )
