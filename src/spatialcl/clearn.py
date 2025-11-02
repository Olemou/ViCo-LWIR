
import torch
from torch import Tensor  
from .dto.config import ConfigDto

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
    
    def __init__(
        self,
        config_dto: ConfigDto
    ):
       self.config_dto = config_dto
       self._validate_parameters()
    
    def __call__(
        self, 
        uncertainty: Tensor, 
        epoch: int
    ) -> Tensor:
        """Compute weights for given uncertainty matrix and epoch."""
        return self.compute(uncertainty, epoch)
    
    def compute(
        self, 
        uncertainty: Tensor, 
        epoch: int
    ) -> Tensor:
        """
        Compute uncertainty-based weights with rank modulation.

        Args:
            uncertainty (torch.Tensor): Uncertainty matrix of shape [N, M].
            epoch (int): Current training epoch.

        Returns:
            torch.Tensor: Weight matrix of shape [N, M].
        """
        
        self._validate_inputs(uncertainty, epoch)
        return self._compute_vectorized(uncertainty, epoch)
        
    
    def _compute_vectorized(self, u: Tensor, epoch: int) -> Tensor:
        """Fully vectorized implementation using torch's built-in ranking."""
        delta_t = epoch/self.config_dto.T
        ranks = self._compute_ranks_vectorized(u, descending=True)
        phi_rho = ranks / (u.size(1) - 1 + self.eps)
        
        return self._apply_lambda_formula(phi_rho, delta_t)
    
    
    def _compute_ranks_vectorized(self, x: Tensor, descending: bool = True) -> Tensor:
        """
        Compute ranks of tensor elements along the last dimension.

        Args:
            x (torch.Tensor): Input tensor.
            descending (bool): If True, higher values receive lower rank numbers.

        Returns:
            torch.Tensor: Rank tensor with the same shape as `x`. """
        # Use negative values for descending sort to maintain stability
        sort_target = -x if descending else x
        
        # Double argsort technique for rank computation
        argsort_indices = torch.argsort(sort_target, dim=-1)
        ranks = torch.argsort(argsort_indices, dim=-1)
        
        return ranks.float()
    
    def _apply_lambda_formula(self, phi_rho: Tensor, delta_t: float) -> Tensor:
        """Apply the selected lambda computation formula."""
        if self.config_dto.method == "exp":
            return 1.0 + torch.exp(-delta_t * phi_rho)
        elif self.config_dto.method == "tanh":
            return torch.tanh(delta_t * phi_rho) + 1.0
        else:
            raise ValueError(f"Unknown method: {self.config_dto.method}")
    
    def _validate_parameters(self) -> None:
        """Validate initialization parameters."""
        assert self.config_dto.T > 0, f"T must be positive, got {self.config_dto.T}"
        assert self.config_dto.method in ["exp", "tanh"], f"Method must be 'exp' or 'tanh', got { self.config_dto.method}"
        assert  self.config_dto.eps > 0, f"Epsilon must be positive, got {self.config_dto.eps}"
    
    def _validate_inputs(self, uncertainty: Tensor, epoch: int) -> None:
        """Validate input tensors and parameters."""
        assert uncertainty.dim() == 2, f"Uncertainty must be 2D tensor, got {uncertainty.dim()}D"
        assert epoch >= 0, f"Epoch must be non-negative, got {epoch}"
        assert epoch <= self.T, f"Epoch {epoch} exceeds total epochs {self.T}"
        
        _, M = uncertainty.shape
        assert M > 1, f"Uncertainty matrix must have at least 2 columns, got {M}"

def compute_weights_from_uncertainty(
    uncertainty: torch.Tensor,
    epoch: int,
    config_dto : ConfigDto,
) -> torch.Tensor:
    """
    Compute uncertainty-based weights using the specified method.

    Args:
        uncertainty (torch.Tensor): Uncertainty matrix of shape [N, M].
        epoch (int): Current training epoch.
        config_dto (ConfigDto): Hyperparameters including T, device, same_img_weight, and method.
        device (torch.device, optional): Target device ("cuda" or "cpu"). 
            Overrides the device in config_dto if provided.

    Returns:
        torch.Tensor: Weight matrix of shape [N, M].
    """
    compute_weight = UncertaintyWeightComputer(
        config_dto
    )
    return compute_weight(uncertainty, epoch)
     
