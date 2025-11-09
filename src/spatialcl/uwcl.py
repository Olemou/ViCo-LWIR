from typing import Literal
from clr_utils import *
from spatialcl.clearn import compute_weights_from_uncertainty
from spatialcl._dto.mask import Maskdto
from spatialcl._dto.config import ConfigDto
from spatialcl._dto.cldto import clDto
from spatialcl._dto.image_label_dto import ImageLabelDTO
from spatialcl.uncertainty import co_cluster_uncertainty
from dataclasses import field


class DenseContrastiveLoss:
    """implementation of dense contrastive loss with uncertainty weighting."""

    z_flat: Tensor = field(default=None, init=False)
    label_flat: Tensor = field(default=None, init=False)
    img_id_flat: Tensor = field(default=None, init=False)

    def __init__(
        self,
        config_dto: ConfigDto,
        image_label_dto: ImageLabelDTO,
    ):
        # Load configuration from YAML
        self.image_label_dto = image_label_dto
        self.config_dto = config_dto

    def __call__(
        self,
        z: Tensor,
        epoch: int,
    ) -> Tensor:
        return self.forward(z, epoch)

    def forward(
        self,
        z: Tensor,
        epoch: int,
    ) -> Tensor:
        """
        Compute dense contrastive loss with uncertainty weighting.

        Args:
            z: Feature tensor of shape [B, D] or [B, V, D]
            labels: Class labels of shape [B]
            img_ids: Image identifiers of shape [B]
            epoch: Current training epoch
            u_batch: Optional pre-computed uncertainty matrix. If None,
                    uncertainty_fn will be used to compute it.

        Returns:
            Scalar loss tensor
        """
        self.__validate_inputs(z)
        # Flatten and normalize features

        u_batch = co_cluster_uncertainty(
            z=z, labels=self.image_label_dto.label, img_ids=self.image_label_dto.img_id
        )

        loss = self.__compute_loss_per_batch(
            self.z_flat, self.label_flat, self.img_id_flat, u_batch, epoch
        )

        return loss

    def __validate_inputs(
        self,
        z: Tensor,
    ) -> None:
        """Validate input tensor shapes and properties."""
        self.z_flat, self.label_flat, self.img_id_flat = (
            self.image_label_dto.flatten_inputs(z)
        )

        assert self.z_flat.dim() in [2, 3], f"z must be 2D or 3D tensor, got {z.dim()}D"
        assert (
            self.label_flat.dim() == 1
        ), f"labels must be 1D tensor, got {self.label_flat.dim()}D"
        assert (
            self.img_id_flat.dim() == 1
        ), f"img_ids must be 1D tensor, got {self.img_id_flat.dim()}D"

        B = self.z_flat.size(0)
        assert (
            self.label_flat.size(0) == B
        ), f"Labels batch size mismatch: z={B}, labels={self.label_flat.size(0)}"
        assert (
            self.img_id_flat.size(0) == B
        ), f"Image IDs batch size mismatch: z={B}, img_ids={self.img_id_flat.size(0)}"

    def __compute_similarity_matrix(self, z_flat: Tensor) -> Tensor:
        """Compute pairwise cosine similarity matrix."""
        return torch.matmul(z_flat, z_flat.T) / self.config_dto.temperature

    def __compute_loss_per_batch(
        self,
        z_flat: torch.Tensor,
        labels_flat: torch.Tensor,
        img_ids_flat: torch.Tensor,
        u_batch: torch.Tensor,
        epoch: int,
    ) -> torch.Tensor:
        device = labels_flat.device

        # Identity and masks
        eye_mask = torch.eye(labels_flat.size(0), dtype=torch.bool, device=device)
        same_class = labels_flat.unsqueeze(0) == labels_flat.unsqueeze(1)
        same_image = img_ids_flat.unsqueeze(0) == img_ids_flat.unsqueeze(1)
        same_class[eye_mask] = False

        strong_pos_mask = same_class & same_image
        weak_pos_mask = same_class & ~same_image
        pos_mask = strong_pos_mask | weak_pos_mask
        neg_mask = ~same_class

        # Get uncertainty-based weights
        diff_img_weight = compute_weights_from_uncertainty(
            uncertainty=u_batch,
            epoch=epoch,
            T=self.config_dto.T,
            device=self.config_dto.device,
            eps=self.config_dto.eps,
        )

        pos_weights = strong_pos_mask.float() + diff_img_weight.to(strong_pos_mask.device) * weak_pos_mask.float()
        pos_weights[~pos_mask] = 0.0

        # Compute similarity
        sim_matrix = self.__compute_similarity_matrix(z_flat)

        # Center similarities to avoid very large exp()
        sim_matrix = sim_matrix - sim_matrix.max(dim=1, keepdim=True)[0].detach()

        # Apply exp safely
        exp_sim = torch.exp(sim_matrix).clamp_min(self.config_dto.eps)

        numerator = exp_sim * pos_weights

        exp_neg = exp_sim * neg_mask.float()
        exp_neg_sum = exp_neg.sum(dim=1, keepdim=True).clamp_min(self.config_dto.eps)
        neg_weights = exp_neg / exp_neg_sum
        neg_term = neg_weights * exp_sim * neg_mask.float()

        # Denominator with stabilization
        denominator = numerator + neg_term.sum(dim=1, keepdim=True)
        denominator = denominator.clamp_min(self.config_dto.eps)

        # Log-prob
        log_prob = torch.log((numerator / denominator).clamp_min(self.config_dto.eps))
        loss_matrix = -log_prob * pos_mask.float()

        num_positives = pos_mask.sum(dim=1)
        valid = num_positives > 0
        if valid.sum() == 0:
            print("Warning: no positive pairs found.")
            return torch.tensor(0.0, device=device)

        loss = loss_matrix.sum(dim=1)[valid] / (
            num_positives[valid] + self.config_dto.eps
        )

        return loss.mean()


# Example usage with co_cluster_uncertainty function
def build_uwcl(
    z: Tensor,
    img_ids: Tensor,
    labels: Tensor,
    epoch: int,
    device:torch.device,
    temperature: float = 0.1,
    T: int = 100,
    eps: float = 1e-12,
) -> DenseContrastiveLoss:
    """
    Factory function to create DenseContrastiveLoss with co-cluster uncertainty.

    Args:
        temperature: Temperature scaling parameter
        same_img_weight: Weight for same-image positives
        T: total number of epochs
        eps: Numerical stability constant
        device: Device for computation

    Returns:
        Configured DenseContrastiveLoss instance
    """

    dense_contrastive = DenseContrastiveLoss(
        ConfigDto(temperature=temperature, T=T, eps=eps, device=device),
        ImageLabelDTO(img_id=img_ids, label=labels),
    )
    return dense_contrastive(z, epoch)


__all__ = ["build_uwcl"]  # Only this function is public
