import torch
from torch.utils.data import Dataset, random_split
from torchvision import datasets, transforms
from PIL import Image
import requests
import numpy as np
from io import BytesIO
from torchvision import io as tv_io
from thermal import occlusion
import kornia.augmentation as K
from utils.data_aug_config import dataloadConfig, ThermalAugConfig, RgbAugConfig


# ------------------------------
# RGB image loader (local or URL)
# ------------------------------
'''def rgb_loader(path_or_url):
    if path_or_url.startswith("http"):
        response = requests.get(path_or_url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        img = Image.open(path_or_url).convert("RGB")
    return img'''

def rgb_loader(path_or_url):
    """
    Load an RGB image from a local path or URL as a torch.Tensor.
    Output: float32 tensor [C,H,W] in [0,1]
    """
    if path_or_url.startswith("http"):
        # Load from URL
        response = requests.get(path_or_url, timeout=10)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img = torch.tensor(np.array(img), dtype=torch.float32).permute(2,0,1) / 255.0
    else:
        # Load from local path using torchvision (faster)
        img = tv_io.read_image(path_or_url).float() / 255.0  # [C,H,W], float32 [0,1]
        if img.shape[0] == 1:  # grayscale fallback
            img = img.repeat(3,1,1)
    return img

# ------------------------------
#  Contrastive Dataset
# ------------------------------
class ConDataset(Dataset):
    def __init__(self, base_dataset,device, transform=None):
        self.base_dataset = base_dataset
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        img = img.to(self.device, non_blocking=True)  # [C,H,W], float32
        label = torch.tensor(label, device=self.device, dtype=torch.long)
        idx = torch.tensor(idx, device=self.device, dtype=torch.long)
        xi = self.transform(img) 
        xj = self.transform(img) 
        return (xi, xj), label, idx


# ------------------------------
# Create datasets and dataloaders
# ------------------------------
def get_datasets_and_loaders(
    root=None,
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
    dataset_class=None,
    transform=None,
    train_ratio=dataloadConfig().train_ratio,
    val_ratio=dataloadConfig().val_ratio,
    seed=dataloadConfig().seed,
):
    # Load base dataset
    if dataset_class is not None:
        base_dataset = dataset_class(
            root=root, train=True, download=True, transform=None
        )
    elif root is not None:
        base_dataset = datasets.ImageFolder(
            root=root, transform=None, loader=rgb_loader
        )
    else:
        raise ValueError("Provide either root path or dataset_class")

    # Split into train/val/test
    total_len = len(base_dataset)
    train_len = int(total_len * train_ratio)
    val_len = int(total_len * val_ratio)
    test_len = total_len - train_len - val_len

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        base_dataset, [train_len, val_len, test_len], generator=generator
    )

    # Wrap in SupConDataset
    train_dataset = ConDataset(train_dataset, transform=transform,device = device)
    val_dataset = ConDataset(val_dataset, transform=transform,device = device)
    test_dataset = ConDataset(test_dataset, transform=transform,device = device)

    return train_dataset, val_dataset, test_dataset


# -----------------------------------------------------------------------------------------------
#  Thermal modality
# -----------------------------------------------------------------------------------------------
'''class ThermalAugmentation:
    def __init__(self, cfg: ThermalAugConfig):
        self.cfg = cfg
        self.transform = self.build_transform()

    def build_transform(self):
        return transforms.Compose(
            [
                transforms.Resize((self.cfg.image_size, self.cfg.image_size)),
                # --- Combined geometric & photometric transforms ---
                transforms.RandomOrder(
                    [
                        transforms.RandomApply(
                            [
                                transforms.RandomAffine(
                                    self.cfg.degrees,
                                    self.cfg.translate,
                                    self.cfg.scale,
                                    self.cfg.shear,
                                )
                            ],
                            p=self.cfg.random_affine_prob,
                        ),
                        transforms.RandomApply(
                            [
                                transforms.RandomRotation(
                                    degrees=self.cfg.random_rotation_degrees
                                ),
                            ],
                            p=self.cfg.random_rotation_degrees_prob,
                        ),
                        transforms.RandomApply(
                            [
                                transforms.RandomResizedCrop(
                                    self.cfg.image_size,
                                    scale=self.cfg.resized_crop_scale,
                                    ratio=self.cfg.resized_crop_ratio,
                                )
                            ],
                            p=self.cfg.random_crop_prob,
                        ),
                    ]
                ),
                # --- Thermal-specific augmentations ---
                transforms.RandomApply(
                    [
                        transforms.Lambda(
                            lambda img: occlusion(
                                img,
                                mask_width_ratio=self.cfg.mask_width_ratio,
                                mask_height_ratio=self.cfg.mask_height_ratio,
                                max_attempts=self.cfg.max_attempts,
                            )
                        )
                    ],
                    p=self.cfg.occlusion_prob,
                ),
                # --- Flipping ---
                transforms.RandomHorizontalFlip(p=self.cfg.horizontal_flip_prob),
                transforms.RandomVerticalFlip(p=self.cfg.vertical_flip_prob),
                # --- Tensor and normalization ---
                transforms.ToTensor(),
                transforms.Normalize(mean=self.cfg.mean, std=self.cfg.std),
            ]
        )
        
        

    def __call__(self, img):
        return self.transform(img)
'''
class ThermalAugmentation:
    def __init__(self, cfg):
        self.cfg = cfg
        self.transform = K.AugmentationSequential(
            # Resize
            K.Resize((cfg.image_size, cfg.image_size)),

            # Geometric transforms (random order)
            K.AugmentationSequential(
                K.RandomAffine(
                    degrees=cfg.degrees,
                    translate=cfg.translate,
                    scale=cfg.scale,
                    shear=cfg.shear,
                    p=cfg.random_affine_prob,
                ),
                K.RandomRotation(
                    degrees=cfg.random_rotation_degrees,
                    p=cfg.random_rotation_degrees_prob,
                ),
                K.RandomResizedCrop(
                    size=(cfg.image_size, cfg.image_size),
                    scale=cfg.resized_crop_scale,
                    ratio=cfg.resized_crop_ratio,
                    p=cfg.random_crop_prob,
                ),
                random_apply=True,
            ),

            # Thermal occlusion
            K.RandomApply(
                K.Lambda(
                    lambda img: occlusion(
                        img,
                        mask_width_ratio=cfg.mask_width_ratio,
                        mask_height_ratio=cfg.mask_height_ratio,
                        max_attempts=cfg.max_attempts,
                    )
                ),
                p=cfg.occlusion_prob,
            ),  

            # Flips
            K.RandomHorizontalFlip(p=cfg.horizontal_flip_prob),
            K.RandomVerticalFlip(p=cfg.vertical_flip_prob),

            # Normalize
            K.Normalize(mean=cfg.mean, std=cfg.std),
        )

    def __call__(self, img):
        out = self.transform(img)
        return out


# ------------------------------------------------------------------------------------------
#  RGB modality
# -------------------------------------------------------------------------------------------


class RgbAugmentation:
    def __init__(self, cfg: RgbAugConfig):
        self.cfg = cfg
        self.transform = self.build_transform()

    def build_transform(self):
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self.cfg.image_size, scale=self.cfg.random_resized_crop_scale
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.cfg.mean, std=self.cfg.std),
            ]
        )

    def __call__(self, img):
        return self.transform(img)
