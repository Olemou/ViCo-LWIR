import torch
import numpy as np
from utils.weight_param import set_lr_para

def get_optimizer(model):
    """
    Create an AdamW optimizer with separate parameter groups for head and backbone.
    """
    # Head parameters
    head_params = list(model.head.parameters())
    head_params_set = {id(p) for p in head_params}

    # Backbone parameters
    backbone_params = [p for p in model.parameters() if id(p) not in head_params_set]

    # Optimizer hyperparameters
    optimizer_params = set_lr_para()

    optimizer = torch.optim.AdamW([
        {"params": head_params, "lr": optimizer_params["base_lr_head"], "weight_decay": optimizer_params["weight_decay_head"]},
        {"params": backbone_params, "lr": optimizer_params["base_lr_backbone"], "weight_decay": optimizer_params["weight_decay_backbone"]}
    ])

    return optimizer

def cosine_schedule(
    epoch: int = 0,
    optimizer: torch.optim.Optimizer = None,
    warmup_epochs: int = 10,
    max_epochs: int = 100,
    min_lr: float = 1e-6
):
    """
    Update optimizer learning rates for two parameter groups (head and backbone)
    using quadratic warmup + cosine decay.

    Args:
        epoch (int): current epoch
        optimizer (torch.optim.Optimizer): optimizer with param_groups
        warmup_epochs (int): number of warmup epochs
        max_epochs (int): total number of epochs
        min_lr (float): minimum learning rate
    """
    for _, param_group in enumerate(optimizer.param_groups):
        base_lr = param_group.get('initial_lr', param_group['lr'])
        if epoch < warmup_epochs:
            # Quadratic warmup: min_lr → base_lr
            lr = min_lr + (base_lr - min_lr) * (epoch / warmup_epochs) ** 2
        else:
            # Cosine decay: base_lr → min_lr
            progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
            lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * progress))
        param_group['lr'] = lr

