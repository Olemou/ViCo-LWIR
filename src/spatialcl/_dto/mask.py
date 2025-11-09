from clr_utils import *


@dataclass
class Maskdto:
    """
    Data Transfer Object for contrastive learning masks with efficient
    device management and validation.

    Attributes:
        strong_pos_mask: Mask for strong positive pairs
        weak_pos_mask: Mask for weak positive pairs
        pos_mask: Mask for all positive pairs
        neg_mask: Mask for negative pairs
        eye_mask: Identity mask to exclude self-comparisons
    """

    strong_pos_mask: torch.Tensor
    weak_pos_mask: torch.Tensor
    pos_mask: torch.Tensor
    neg_mask: torch.Tensor
    eye_mask: torch.Tensor

    def __post_init__(self):
        """Validate mask shapes and types after initialization."""
        self._validate_masks()

    def _validate_masks(self) -> None:
        """Ensure all masks have consistent shapes and are boolean tensors."""
        shapes = {mask.shape for mask in self.all_masks}
        if len(shapes) != 1:
            raise ValueError(f"Inconsistent mask shapes: {shapes}")

        for name, mask in self.mask_items:
            if not isinstance(mask, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor, got {type(mask)}")
            if mask.dtype != torch.bool:
                raise ValueError(
                    f"{name} must be boolean tensor, got dtype {mask.dtype}"
                )

    @property
    def mask_items(self):
        """Iterator over mask names and tensors."""
        return (
            ("strong_pos_mask", self.strong_pos_mask),
            ("weak_pos_mask", self.weak_pos_mask),
            ("pos_mask", self.pos_mask),
            ("neg_mask", self.neg_mask),
            ("eye_mask", self.eye_mask),
        )

    @property
    def all_masks(self):
        """List of all mask tensors."""
        return [
            self.strong_pos_mask,
            self.weak_pos_mask,
            self.pos_mask,
            self.neg_mask,
            self.eye_mask,
        ]

    @property
    def device(self) -> torch.device:
        """Get device of the first mask (all should be on same device)."""
        return self.strong_pos_mask.device

    @property
    def shape(self) -> torch.Size:
        """Get common shape of all masks."""
        return self.strong_pos_mask.shape

    # ----------- Device Management -----------
    def to(self, device: torch.device) -> "Maskdto":
        """Move all masks to specified device."""
        return Maskdto(*[mask.to(device) for mask in self.all_masks])

    def cpu(self) -> "Maskdto":
        """Move all masks to CPU."""
        return self.to(torch.device("cpu"))

    def cuda(self, device: Optional[int] = None) -> "Maskdto":
        """Move all masks to CUDA device."""
        return self.to(torch.device(f"cuda:{device}" if device is not None else "cuda"))

    # ----------- Factory Methods -----------
    @classmethod
    def from_dict(cls, mask_dict: dict) -> "Maskdto":
        """Create MaskDTO from dictionary."""
        return cls(
            strong_pos_mask=mask_dict["strong_pos_mask"],
            weak_pos_mask=mask_dict["weak_pos_mask"],
            pos_mask=mask_dict["pos_mask"],
            neg_mask=mask_dict["neg_mask"],
            eye_mask=mask_dict["eye_mask"],
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "strong_pos_mask": self.strong_pos_mask,
            "weak_pos_mask": self.weak_pos_mask,
            "pos_mask": self.pos_mask,
            "neg_mask": self.neg_mask,
            "eye_mask": self.eye_mask,
        }

    # ----------- Utility Methods -----------
    def apply_eye_mask(
        self, tensor: torch.Tensor, fill_value: float = 0.0
    ) -> torch.Tensor:
        """Apply eye mask to a tensor, setting diagonal to fill_value."""
        return tensor.masked_fill(self.eye_mask, fill_value)

    def count_positives(self) -> dict:
        """Count number of positive pairs for each mask type."""
        return {
            "strong_pos": self.strong_pos_mask.sum().item(),
            "weak_pos": self.weak_pos_mask.sum().item(),
            "pos": self.pos_mask.sum().item(),
            "neg": self.neg_mask.sum().item(),
        }

    def __repr__(self) -> str:
        """Enhanced representation with shape and device info."""
        counts = self.count_positives()
        return (
            f"MaskDTO(shape={self.shape}, device={self.device}, "
            f"strong_pos={counts['strong_pos']}, weak_pos={counts['weak_pos']}, "
            f"pos={counts['pos']}, neg={counts['neg']})"
        )
