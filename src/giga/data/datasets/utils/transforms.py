import torch.nn.functional as F
import torchvision.ops as tvops
import torchvision.transforms.functional as TF
from jaxtyping import Float
from torch import Tensor


class BBoxCropResize:
    def __init__(self, target_size: tuple[int, int]):
        """
        Args:
            target_size: (width, height) of the output tensor
        """
        self.target_width, self.target_height = target_size

    def __call__(self, tensor: Float[Tensor, "C H W"]) -> tuple[Float[Tensor, "C H W"], dict]:
        assert tensor.shape[0] == 4, "Tensor must have 4 channels for RGBA images"
        image, mask = tensor[:3, :, :], tensor[3:, :, :]
        bboxes = tvops.masks_to_boxes(mask).squeeze(0).long().tolist()

        top, left = bboxes[1], bboxes[0]
        height, width = bboxes[3] - top, bboxes[2] - left

        image_crop = TF.crop(image, top, left, height, width)
        image_resized = TF.resize(image_crop, (self.target_height, self.target_width), antialias=True)

        return image_resized, {"original_shape": tensor.shape[1:], "target_shape": (self.target_height, self.target_width)}


class AdaptiveResize:
    """Resize tensor with optional aspect ratio preservation."""

    def __init__(
        self, target_size: tuple[int, int], preserve_aspect_ratio: bool = False, fill: float = 0.0, padding_mode: str = "constant"
    ):
        """
        Args:
            target_size: (width, height) of the output tensor
            preserve_aspect_ratio: If True, maintain aspect ratio and pad; if False, stretch to fit
            fill: Fill value for padding (only used when preserve_aspect_ratio=True)
            padding_mode: Type of padding (only used when preserve_aspect_ratio=True)
        """
        self.target_width, self.target_height = target_size
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, tensor: Float[Tensor, "C H W"]) -> tuple[Float[Tensor, "C H W"], dict]:
        """Apply resize with optional aspect ratio preservation.

        Returns:
            tuple: (resized_tensor, transform_metadata)
        """
        # Get tensor dimensions (C, H, W)
        _, original_height, original_width = tensor.shape

        if not self.preserve_aspect_ratio:
            # Standard resize (stretch to fit)
            tensor_resized = TF.resize(tensor, (self.target_height, self.target_width), antialias=True)

            metadata = {
                "original_shape": (original_height, original_width),  # (H, W)
                "target_shape": (self.target_height, self.target_width),  # (H, W)
                "offset_x": 0,
                "offset_y": 0,
                "content_width": self.target_width,
                "content_height": self.target_height,
            }
            return tensor_resized, metadata

        # Aspect-ratio-preserving resize with padding
        scale_height = self.target_height / original_height
        scale_width = self.target_width / original_width
        scale = min(scale_height, scale_width)

        # Calculate new size after scaling
        scaled_height = int(original_height * scale)
        scaled_width = int(original_width * scale)

        # Resize tensor
        tensor_resized = TF.resize(tensor, (scaled_height, scaled_width), antialias=True)

        # Calculate padding needed
        pad_height = self.target_height - scaled_height
        pad_width = self.target_width - scaled_width

        # Padding: (left, top, right, bottom)
        pad_left = pad_width // 2
        pad_top = pad_height // 2
        pad_right = pad_width - pad_left
        pad_bottom = pad_height - pad_top

        metadata = {
            "original_shape": (original_height, original_width),  # (H, W)
            "target_shape": (self.target_height, self.target_width),  # (H, W)
            "offset_x": pad_left,
            "offset_y": pad_top,
            "content_width": scaled_width,
            "content_height": scaled_height,
        }

        # Apply padding - torchvision expects (left, right, top, bottom)
        tensor_padded = TF.pad(tensor_resized, (pad_left, pad_right, pad_top, pad_bottom), fill=self.fill, padding_mode=self.padding_mode)

        return tensor_padded, metadata

    def inverse(self, tensor, metadata) -> Float[Tensor, "B C H W"]:
        """Inverse transform to restore original shape.

        Args:
            tensor: Tensor with shape (B, C, H, W)
            metadata: Complete transformation metadata from forward pass

        Returns:
            Tensor with original shape
        """

        original_shape = metadata["original_shape"]  # (H, W)
        offset_x = metadata["offset_x"]
        offset_y = metadata["offset_y"]
        content_width = metadata["content_width"]
        content_height = metadata["content_height"]

        # Crop the tensor to remove padding - extract only the content region
        cropped = tensor[:, :, offset_y : offset_y + content_height, offset_x : offset_x + content_width]

        # Resize back to original shape
        restored = F.interpolate(cropped, size=original_shape, mode="bilinear", align_corners=False)

        return restored
