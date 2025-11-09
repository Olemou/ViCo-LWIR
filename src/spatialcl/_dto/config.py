from clr_utils import *


class ConfigDto:
    """
    Data Transfer Object for configuration parameters and tensor dimensions.

    Attributes:
        temperature (Optional[float]): Temperature for scaling.
        same_img_weight (Optional[float]): Weight assigned to same-image pairs.
        T (Optional[int]): Total number of epochs.
        eps (Optional[float]): Small value for numerical stability.
        device (Optional[torch.device]): Target device ("cuda" or "cpu").

    """

    def __init__(
        self,
        temperature: Optional[float] = None,
        same_img_weight: Optional[float] = 1,
        T: int = 100,
        eps: Optional[float] = 1e-8,
        device: Optional[torch.device] = None,
    ):
        self.temperature = temperature
        self.same_img_weight = same_img_weight
        self.T = T
        self.eps = eps
        self.device = device

    def to(self, device: Optional[torch.device] = None) -> "ConfigDto":
        """
        Set the target device for this configuration.

        Args:
            device (torch.device, optional): Target device ("cuda" or "cpu").

        Returns:
            ConfigDto: A new instance with the same values and updated device.
        """
        target_device = device or self.device
        return ConfigDto(
            temperature=self.temperature,
            same_img_weight=self.same_img_weight,
            T=self.T,
            eps=self.eps,
            device=target_device,
        )
