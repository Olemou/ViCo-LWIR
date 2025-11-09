from clr_utils import *
from spatialcl._config.config import load_config
from typing import Dict
from dataclasses import field


@dataclass
class ImageLabelDTO:
    """
    Data Transfer Object for image identifiers and labels.

    Attributes:
        img_id (torch.Tensor): Tensor of image identifiers.
        label (torch.Tensor): Tensor of corresponding class labels.
    """

    img_id: torch.Tensor
    label: torch.Tensor
    eps = 1e-8
    device: Optional[torch.device] = None
    _B: int = field(default=64, init=False)
    _V: int = field(default=2, init=False)
    _D: int = field(default=128, init=False)
    _N: int = field(default=512, init=False)
    # Internal config storage (not in __init__)
    _config: Dict[str, Any] = field(init=False, default_factory=dict)

    def __post_init__(self):
        """Load configuration after object initialization"""
        self._config = load_config()

    def to(self, device: Optional[torch.device] = None) -> "ImageLabelDTO":
        """
        Move tensors to the specified device.

        Args:
            device (torch.device, optional): Target device ("cuda" or "cpu").

        Returns:
            ImageLabelDTO: A new instance with tensors on the target device.
        """
        if device is None:
            return self
        return ImageLabelDTO(img_id=self.img_id.to(device), label=self.label.to(device))

    def compute_dimensions(self, z: Tensor):
        """Compute and set dimensions based on input tensor shape"""
        if (
            z.dim()
            == self._config["learning_mechanism"]["init_args"]["minimum_dim_output"]
        ):
            self._B, self._D = z.shape
            self._V = self._config["learning_mechanism"]["init_args"][
                "number_patch_or_single_view"
            ]
            # Single view per sample
            self._N = self._B
        else:  # dim == 3
            self._B, self.V, self._D = z.shape
            self._N = self._B * self._V

        return self

    def flatten_inputs(self, z: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # Déterminer le device : self.device ou CPU
        device = self.device if self.device is not None else torch.device("cpu")

        # Labels tensor
        if isinstance(self.label, torch.Tensor):
            labels_tensor = self.label.to(device)
            if labels_tensor.dtype != torch.long:
                labels_tensor = labels_tensor.long()
        else:  # list ou tuple
            labels_tensor = torch.tensor(self.label, dtype=torch.long, device=device)

        # Img_ids tensor
        if isinstance(self.img_id, torch.Tensor):
            img_ids_tensor = self.img_id.to(device)
            if img_ids_tensor.dtype != torch.long:
                img_ids_tensor = img_ids_tensor.long()
        else:
            img_ids_tensor = torch.tensor(self.img_id, dtype=torch.long, device=device)

        # Calculer dimensions
        self.compute_dimensions(z)

        # Flatten et normalisation
        if z.dim() == 2:
            z_flat = F.normalize(z.to(device), dim=1, eps=self.eps)
            labels_flat = labels_tensor
            img_ids_flat = img_ids_tensor
        else:
            B, V, D = self._B, self._V, self._D
            z_flat = F.normalize(z.view(B * V, D).to(device), dim=1, eps=self.eps)
            labels_flat = labels_tensor.repeat_interleave(V)
            img_ids_flat = img_ids_tensor.repeat_interleave(V)

        return z_flat, labels_flat, img_ids_flat
