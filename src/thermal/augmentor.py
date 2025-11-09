
from clr_utils import *
class ThermalAugmentor:
    def __init__(
        self,
        image_size: int = 224,
    ):
        self.image_size = image_size

    def thermal_erase(
        self,
        img,
        mask_width_ratio: float = 0.6,
        mask_height_ratio: float = 0.2,
        max_attempts: int = 5,
    ):

        x = np.array(img)
        if x.ndim == 2:
            h, w = x.shape
            x = np.expand_dims(x, axis=-1)
        elif x.ndim == 3 and x.shape[2] in [1, 3]:
            h, w, _ = x.shape
        else:
            raise ValueError(f"Unexpected image shape: {x.shape}")

        # --- Validate hyperparameters ---
        if not (0 < mask_width_ratio <= 1):
            raise ValueError(
                f"mask_width_ratio must be in (0,1], got {mask_width_ratio}"
            )
        if not (0 < mask_height_ratio <= 1):
            raise ValueError(
                f"mask_height_ratio must be in (0,1], got {mask_height_ratio}"
            )
        if mask_width_ratio + mask_height_ratio > 1:
            raise ValueError(
                f"Sum of mask_width_ratio and mask_height_ratio should not exceed 1. "
                f"Got mask_width_ratio={mask_width_ratio}, mask_height_ratio={mask_height_ratio}"
            )

        if max_attempts < 1:
            raise ValueError(f"max_attempts must be >=1, got {max_attempts}")

        mask_w = int(w * mask_width_ratio)
        mask_h = int(h * mask_height_ratio)

        center_x1, center_y1 = int(w * 0.3), int(h * 0.3)
        center_x2, center_y2 = int(w * 0.7), int(h * 0.7)

        for _ in range(max_attempts):
            x1 = np.random.randint(0, w - mask_w)
            y1 = np.random.randint(0, h - mask_h)
            if not (
                x1 + mask_w > center_x1
                and x1 < center_x2
                and y1 + mask_h > center_y1
                and y1 < center_y2
            ):
                x[y1 : y1 + mask_h, x1 : x1 + mask_w, :] = 0
                break
        x = np.clip(x, 0, 255).astype(np.uint8)
        return (
            Image.fromarray(x[:, :, 0], mode="L")
            if x.shape[2] == 1
            else Image.fromarray(x)
        )

    def brightness_contrast(
        self,
        img: Image.Image,
        brightness: Optional[float] = None,
        contrast: Optional[float] = None,
    ) -> Image.Image:

        # Convert image to NumPy array
        arr = np.array(img).astype(np.float32)
        # Validate or randomize parameters
        if brightness is None:
            brightness = np.random.uniform(0.8, 1.4)
        if contrast is None:
            contrast = np.random.uniform(0.2, 1.2)

        if brightness <= 0 or contrast <= 0:
            raise ValueError(
                f"Brightness and contrast must be positive, got brightness={brightness}, contrast={contrast}"
            )
        # Apply brightness and contrast adjustment
        mean = np.mean(arr)
        arr = brightness * arr + (contrast - 1.0) * (arr - mean) + mean
        arr = np.clip(arr, 0, 255).astype(np.uint8)

        return Image.fromarray(arr)

    # contrast augmentation
    def thermal_contrast(
        self, img: Image.Image, alpha: float | None = None
    ) -> Image.Image:

        if alpha is not None and alpha > 1:
            raise ValueError(f"alpha must be <= 1 to increase contrast, got {alpha}")
        # Determine the contrast scaling factor
        factor = (
            np.random.uniform(0.5, 1.0)
            if alpha is None
            else np.random.uniform(alpha, 1 + alpha)
        )
        # Apply the contrast adjustment
        x_arr = np.asarray(img, dtype=np.float32)
        x_contrasted = np.clip(x_arr * factor, 0, 255).astype(np.uint8)

        return Image.fromarray(x_contrasted)

    def elastic_transform(
        self,
        img: Image.Image,
        alpha: Optional[float] = None,
        sigma: Optional[float] = None,
        random_state: Optional[np.random.RandomState] = None,
    ) -> Image.Image:

        arr = np.array(img)
        alpha = alpha if alpha is not None else self.image_size * 0.08
        sigma = sigma if sigma is not None else self.image_size * 0.08
        random_state = random_state or np.random.RandomState(None)

        # Handle grayscale or RGB
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=-1)

        if arr.ndim != 3:
            raise ValueError(f"Unexpected image shape: {arr.shape}")

        h, w, c = arr.shape
        dx = (
            gaussian_filter(
                (random_state.rand(h, w) * 2 - 1), sigma, mode="constant", cval=0
            )
            * alpha
        )
        dy = (
            gaussian_filter(
                (random_state.rand(h, w) * 2 - 1), sigma, mode="constant", cval=0
            )
            * alpha
        )

        x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
        indices = np.vstack([(y_coords + dy).ravel(), (x_coords + dx).ravel()])

        # Apply deformation per channel
        distorted_channels = [
            map_coordinates(arr[:, :, i], indices, order=1, mode="reflect").reshape(
                h, w
            )
            for i in range(c)
        ]
        distorted = np.stack(distorted_channels, axis=-1)

        return Image.fromarray(np.clip(distorted, 0, 255).astype(np.uint8))


# --- Module-level convenience functions ---
_default_augmentor = ThermalAugmentor()


def occlusion(img: Image.Image, **kwargs) -> Image.Image:
    """
    Directly apply thermal erase on an image without instantiating the class.

    Args:
        img (PIL.Image.Image): Input image.
        mask_width_ratio (float, optional): Width of the erase mask relative to image width. Defaults to 0.6.
        mask_height_ratio (float, optional): Height of the erase mask relative to image height. Defaults to 0.2.
        max_attempts (int, optional): Maximum attempts to place the mask outside center. Defaults to 5.
        erase_prob (float, optional): Probability of applying thermal erase. Defaults to 0.9.

    Returns:
        PIL.Image.Image: Augmented image.

    Example:
        >>> from augmentor import thermal_erase
        >>> aug_img = thermal_erase(img, mask_width_ratio=0.5)
    """
    return _default_augmentor.thermal_erase(img, **kwargs)


def contrast(img: Image.Image, **kwargs) -> Image.Image:
    """
    Directly apply thermal contrast adjustment on an image without instantiating the class.

    Args:
        img (PIL.Image.Image): Input image.
        alpha (float, optional): Contrast adjustment factor.
                                 If None, a random factor in [0.5, 1] is used.
                                 If provided, must be <=1 to increase contrast.
    Returns:
        PIL.Image.Image: Augmented image."""

    return _default_augmentor.thermal_contrast(img, **kwargs)


def brightness_contrast(img: Image.Image, **kwargs) -> Image.Image:
    """
    Adjust the brightness and contrast of a thermal image.

    This function modifies the pixel intensity values of a thermal image
    by applying brightness and contrast scaling. When no values are provided,
    random factors are drawn for each augmentation.

    Args:
        img (PIL.Image.Image): Input thermal image.
        brightness (float, optional):
            Brightness adjustment factor.
            - Values < 1.0 darken the image.
            - Values > 1.0 brighten the image.
            - If None, a random factor in [0.8, 1.4] is used.
        contrast (float, optional):
            Contrast adjustment factor.
            - Values < 1.0 decrease contrast.
            - Values > 1.0 increase contrast.
            - If None, a random factor in [0.2, 1.2] is used.

    Returns:
        PIL.Image.Image: Augmented thermal image.

    Raises:
        ValueError: If brightness or contrast are non-positive.

    Example:
        >>> img = Image.open("thermal.png").convert("RGB")
        >>> aug = ThermalAugmentor()
        >>> img_aug = aug.brightness_contrast(img, brightness=1.2, contrast=0.8)
    """
    return _default_augmentor.brightness_contrast(img, **kwargs)


def elastic(img: Image.Image, **kwargs) -> Image.Image:
    """
    Directly apply elastic transformation on an image without instantiating the class.

    Args:
        img (PIL.Image.Image): Input image.
        alpha (float, optional): Scaling factor for displacement intensity.
            Overrides the instance default if provided.
        sigma (float, optional): Standard deviation of Gaussian filter.
            Controls smoothness of displacement fields.
            Overrides the instance default if provided.

    Returns:
        PIL.Image.Image: Deformed image with elastic transformation.

    Example:
        >>> from augmentor import elastic_transform
        >>> img_elastic = elastic_transform(img, alpha=34, sigma=4)
    """
    return _default_augmentor.elastic_transform(img, **kwargs)
