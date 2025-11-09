# config.py
from dataclasses import dataclass


@dataclass
class ThermalAugConfig:
    # Image size
    image_size: int = 224

    # Thermal erase hyperparameters
    mask_width_ratio: float = 0.6
    mask_height_ratio: float = 0.2
    max_attempts: int = 5
    erase_prob: float = 0.9

    # Geometric / photometric augmentation
    random_affine_degrees: int = 10
    random_rotation_degrees: int = 5
    resized_crop_scale: tuple = (0.85, 1.0)
    resized_crop_ratio: tuple = (0.75, 1.25)
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.3
    degrees = 10
    translate = (0.1, 0.1)
    scale = (0.85, 1.15)
    shear = 10

    # probabilities
    random_affine_prob: float = 0.3
    random_rotation_degrees_prob: float = 0.5
    random_crop_prob: float = 0.5
    occlusion_prob: float = 0.4

    # Normalization
    mean: tuple = (0.24, 0.24, 0.24)
    std: tuple = (0.07, 0.07, 0.07)


@dataclass
class RgbAugConfig:
    image_size: int = 224
    mean: tuple = (0.5071, 0.4867, 0.4408)
    std: tuple = (0.2675, 0.2565, 0.2761)
    random_resized_crop_scale: tuple = (0.2, 1.0)


@dataclass
class dataloadConfig:
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    seed: int = 42
