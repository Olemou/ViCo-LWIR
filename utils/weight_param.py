def set_lr_para():
    """
     parameter groups for head and backbone.

    Args:
        
        base_lr_head (learning rate for the head)
        base_lr_backbone (learning rate for the backbone)
        weight_decay_head ( weight decay for the head) 
        weight_decay_backbone (weight decay for the backbone) :
    """
    OPTIMIZER_PARAMS = {
        "base_lr_head": 9e-5,
        "base_lr_backbone": 7e-5,
        "weight_decay_head": 0.05,
        "weight_decay_backbone": 0.1
    }
    return OPTIMIZER_PARAMS
