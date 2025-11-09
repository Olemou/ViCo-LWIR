from clr_utils import *
from spatialcl._dto.image_label_dto import ImageLabelDTO


class CoClusterUncertainty:
    """Compute masked uncertainty between embeddings using Subjective Logic."""

    def __init__(self, image_label_dto: ImageLabelDTO, prior_weight: int = 2):

        self.prior_weight = prior_weight
        self.image_label_dto = image_label_dto

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        return self.compute(z)

    def compute(self, z: torch.Tensor) -> torch.Tensor:
        """Compute uncertainty matrix with label masking."""
        # Déterminer le device à utiliser
        device = (
            self.image_label_dto.device
            if self.image_label_dto.device is not None
            else z.device
        )

        # Flatten inputs
        Z_flat, labels_flat, _ = self.image_label_dto.flatten_inputs(z)

        # Déplacer tensors sur le device correct
        Z_flat = Z_flat.to(device)
        labels_flat = labels_flat.to(device)

        # Compute similarity
        sim = torch.matmul(Z_flat, Z_flat.T)
        sim.fill_diagonal_(0.0)

        # Compute uncertainty
        uncertainty = self._similarity_to_evidence(sim)

        # Mask: same labels only
        mask = labels_flat.unsqueeze(1) == labels_flat.unsqueeze(0)

        # Apply label mask
        uncertainty[~mask] = 0.0

        return uncertainty

    def _similarity_to_evidence(self, sim: torch.Tensor) -> torch.Tensor:
        """Convert similarity to uncertainty using subjective logic."""
        g_sim = sim
        g_dsim = 1 - sim

        e_pos = torch.exp(F.softmax(g_sim, dim=1))
        e_neg = torch.exp(F.softmax(g_dsim, dim=1))

        total_mass = e_pos + e_neg + self.prior_weight
        return self.prior_weight / total_mass


# Module-level functions
def co_cluster_uncertainty(
    z: torch.Tensor, labels: Tensor, img_ids: Tensor, prior_weight: int = 2
) -> torch.Tensor:
    """
    Direct function interface for co-cluster uncertainty.

    Args:
        Z (torch.Tensor): Embedding feature tensor of shape (B, D) or (B,V,D),
            where B is the batch size, D is the embedding dimension and V le number of views.
        image_label_dto (ImageLabelDTO): Data Transfer Object containing
            sample metadata, including:
            - `labels` (torch.Tensor): Class labels for each sample.
            - `image_ids` (torch.Tensor): Unique identifiers for images or views.
        prior_weight (int): Prior weighting factor indicating how many times
            an instance is assumed to be observed under a prior assumption.
    Returns:
        torch.Tensor: Co-cluster uncertainty matrix representing pairwise
            uncertainty relationships between samples.
    """
    image_label_dto = ImageLabelDTO(img_id=img_ids, label=labels)
    computer = CoClusterUncertainty(
        image_label_dto=image_label_dto, prior_weight=prior_weight
    )
    return computer(z)


__all__ = ["co_cluster_uncertainty"]  # Only this function is public
