import torch
from torch import Tensor
from spatialcl._dto.config import ConfigDto
from clr_utils import *


class UncertaintyWeightComputer:
    """
    Compute uncertainty-based weights with rank modulation.

    This class provides optimized methods to compute Λ(t, u_ij) where rank
    is used to modulate uncertainty influence over training epochs.

    Args:
    is in encapsulate in configDto with para
        T (int): Total number of epochs for scheduling.
        method (str): Weight computation method ("exp" for exponential,
            "tanh" for hyperbolic tangent).
        eps (float): Numerical stability constant.
        device (torch.device): Target device for computations.
    """

    def __init__(self, config_dto: ConfigDto):
        self.config_dto = config_dto
        self._validate_parameters()

    def __call__(self, uncertainty: Tensor, epoch: int) -> Tensor:
        """Compute weights for given uncertainty matrix and epoch."""
        return self.compute(uncertainty, epoch)

    def compute(self, uncertainty: Tensor, epoch: int) -> Tensor:
        """
        Compute uncertainty-based weights with rank modulation.

        Args:
            uncertainty (torch.Tensor): Uncertainty matrix of shape [N, M].
            epoch (int): Current training epoch.

        Returns:
            torch.Tensor: Weight matrix of shape [N, M].
        """

        self._validate_inputs(uncertainty, epoch)
        return self._compute_lambda(uncertainty, epoch)

    def _compute_lambda(self, uncertainty: Tensor, epoch: int):
        """
        Compute Lambda(t, u_ij) = tanh( (t/T) * (rank(u_ij)/n) ) + 1

        Args:
            u: Tensor of shape [N, M] — uncertainty matrix
            t: int — current epoch
            T: int — total number of epochs

        Returns:
            Lambda: Tensor of shape [N, M]
        """
        self._validate_inputs(uncertainty, epoch)

        N, M = uncertainty.shape
        delta_t = epoch / self.config_dto.T  # scalar in [0, 1]

        # Get descending rank index of u_ij along dim=1 for each row
        # u_desc_indices[i] gives the order of uncertainties in row i, high to low
        u_desc_indices = torch.argsort(uncertainty, dim=1, descending=True)

        # Initialize rank tensor where rank[i][j] is the rank of u[i][j]
        ranks = torch.zeros_like(uncertainty, dtype=torch.float32)
        for i in range(N):
            ranks[i, u_desc_indices[i]] = torch.arange(
                M, dtype=torch.float32, device=self.config_dto.device
            )

        # Normalize ranks: phi(rho(u_ij)) = rank / n
        phi_rho = ranks / M

        # Lambda computation
        Lambda = 1 + torch.exp(-delta_t * phi_rho)

        return Lambda

    def _validate_parameters(self) -> None:
        """Validate initialization parameters."""
        assert self.config_dto.T > 0, f"T must be positive, got {self.config_dto.T}"
        assert (
            self.config_dto.eps > 0
        ), f"Epsilon must be positive, got {self.config_dto.eps}"

    def _validate_inputs(self, uncertainty: Tensor, epoch: int) -> None:
        """Validate input tensors and parameters."""
        assert (
            uncertainty.dim() == 2
        ), f"Uncertainty must be 2D tensor, got {uncertainty.dim()}D"
        assert epoch >= 0, f"Epoch must be non-negative, got {epoch}"
        assert (
            epoch <= self.config_dto.T
        ), f"Epoch {epoch} exceeds total epochs {self.T}"

        _, M = uncertainty.shape
        assert M > 1, f"Uncertainty matrix must have at least 2 columns, got {M}"


def compute_weights_from_uncertainty(
    uncertainty: torch.Tensor,
    epoch: int,
    T: int = 100,
    eps: Optional[float] = 1e-8,
    device: torch.device = 'cpu',
) -> torch.Tensor:
    """
    Compute uncertainty-based weights using the specified method.

    Args:
        uncertainty (torch.Tensor): Uncertainty matrix of shape [N, M].
        epoch (int): Current training epoch.
        T (number of epochs):
        eps (Optional[float]): Small value for numerical stability.
        device (torch.device): Target device ("cuda" or "cpu").
    Returns:
        torch.Tensor: Weight matrix of shape [N, M].
    """
    config_dto = ConfigDto(
        T=T,
        device=device,
        eps=eps,
    )
    compute_weight = UncertaintyWeightComputer(config_dto)
    return compute_weight(uncertainty, epoch)


__all__ = ["compute_weights_from_uncertainty"]  # Only this function is public
