"""
spatialcl.utils

Provides default PyTorch imports, device configuration,
and commonly used helper functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
from typing import Optional, Tuple, Union, Any
from scipy.ndimage import gaussian_filter, map_coordinates
from dataclasses import dataclass
from torch import Tensor
import random

# Automatically detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


__all__ = [
    "DataLoader",
    "Image",
    "Optional",
    "Tuple",
    "Union",
    "Any",
    "map_coordinates" ,
    "gaussian_filter",
    "dataclass",
    "F",
    "np",
    "Tensor",
    "torch",
    "random",
]
