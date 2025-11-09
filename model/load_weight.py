import torch
import torch.nn as nn
import torch.nn.init as init
import timm


def load_pretrained_vit_weights(custom_model, device, model_size='base', patch_size=16, img_size=224):
    """
    Load pretrained weights from a TIMM Vision Transformer (ViT) into a custom VisionTransformer model.

    Args:
        custom_model (nn.Module): Your VisionTransformer instance.
        model_size (str): One of {'tiny', 'small', 'base', 'large', 'huge'}.
        patch_size (int): Patch size used (typically 8, 14, or 16).
        img_size (int): Image resolution used (e.g., 224, 384).
        device (torch.device): CUDA or CPU device.

    Returns:
        nn.Module: The custom model with pretrained weights loaded.
    """
    if device is None :
        raise ValueError("Device must be specified (e.g., torch.device('cuda') or torch.device('cpu'))")

    # Construct TIMM model name dynamically
    timm_model_name = f"vit_{model_size}_patch{patch_size}_{img_size}"

    print(f"\n🔍 Loading pretrained weights from '{timm_model_name}'...")

    # Create pretrained model from timm
    try:
        pretrained_model = timm.create_model(timm_model_name, pretrained=True).to(device)
    except Exception as e:
        raise ValueError(f"❌ Unsupported ViT model '{timm_model_name}'. "
                         f"Ensure it's a valid timm model (error: {e})")

    pretrained_state_dict = pretrained_model.state_dict()
    custom_state_dict = custom_model.state_dict()

    # Filter only compatible parameters (matching names + shapes)
    filtered_state_dict = {
        k: v for k, v in pretrained_state_dict.items()
        if k in custom_state_dict and v.shape == custom_state_dict[k].shape
    }

    # Identify mismatched or missing keys
    missing_keys = [k for k in custom_state_dict if k not in filtered_state_dict]
    unexpected_keys = [k for k in pretrained_state_dict if k not in custom_state_dict]

    print(f"✅ Pretrained weights summary:")
    print(f"   ├─ Matched keys:     {len(filtered_state_dict)}")
    print(f"   ├─ Missing keys:     {len(missing_keys)}")
    print(f"   └─ Unexpected keys:  {len(unexpected_keys)}\n")

    # Load matching weights
    custom_state_dict.update({k: v.to(device) for k, v in filtered_state_dict.items()})
    custom_model.load_state_dict(custom_state_dict)

    # === Initialize missing parameters ===
    for key in missing_keys:
        parts = key.split('.')
        module_name, attr = '.'.join(parts[:-1]), parts[-1]
        module = custom_model
        for name in module_name.split('.') if module_name else []:
            module = getattr(module, name, None)
            if module is None:
                break
        if module is None:
            continue

        param = getattr(module, attr, None)
        if param is None:
            continue

        if isinstance(param, nn.Parameter):
            data = param.data
            if 'weight' in attr:
                if data.ndim >= 2:
                    init.xavier_uniform_(data)
                else:
                    init.normal_(data, mean=0, std=0.02)
            elif 'bias' in attr:
                init.zeros_(data)
            elif any(t in attr for t in ['mask_token', 'cls_token', 'pos_embed']):
                init.normal_(data, mean=0, std=0.02)

        elif isinstance(module, nn.BatchNorm2d):
            if attr == 'running_mean':
                init.zeros_(module.running_mean)
            elif attr == 'running_var':
                init.ones_(module.running_var)
            elif attr == 'num_batches_tracked':
                module.num_batches_tracked.zero_()
            elif attr == 'weight':
                init.ones_(module.weight)
            elif attr == 'bias':
                init.zeros_(module.bias)

    print(f"🎯 Model '{timm_model_name}' successfully adapted and loaded.\n")
    return custom_model
