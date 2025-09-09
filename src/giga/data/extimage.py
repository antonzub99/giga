from dataclasses import dataclass
from typing import Any, Optional

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data._utils.collate import default_collate
from torchvision.transforms import v2 as tvt


@dataclass
class ExtendedImage:
    image: Tensor
    """The image tensor in the format C x H x W, can be 1/2/3/4 channel image."""
    intrinsics: Float[Tensor, "3 3"]
    """Camera intrinsics in OpenCV format."""
    extrinsics: Float[Tensor, "4 4"]
    """Camera intrinsics and extrinsics in OpenCV cam-to-world format."""
    distortion: Optional[Float[Tensor, "5"]] = None
    """Camera distortion coefficients in OpenCV 5-parameter format."""

    @property
    def channels(self) -> int:
        """Number of channels in the image."""
        return self.image.shape[0]

    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape of the image tensor."""
        return self.image.shape

    @property
    def resolution(self) -> tuple[int, int]:
        """Resolution of the image."""
        return self.image.shape[1:]

    def crop(self, bbox: Int[Tensor, "4"]) -> "ExtendedImage":
        """
        Crops the image to the given bounding box and updates camera intrinsics respectively.
        Args:
            bbox: Bounding box in the format (top, left, height, width) for each image in the batch.
        Returns:
            Cropped ExtendedImage with updated image and intrinsics.
        """

        cropped_image = tvt.functional.crop(self.image, *bbox.tolist())
        cropped_intrinsics = self.intrinsics.clone()
        cropped_intrinsics[0, 2] -= bbox[1].float()  # left
        cropped_intrinsics[1, 2] -= bbox[0].float()

        return ExtendedImage(
            image=cropped_image,
            intrinsics=cropped_intrinsics,
            extrinsics=self.extrinsics,
            distortion=self.distortion,
        )

    def resize(self, size: tuple[int, int]) -> "ExtendedImage":
        """
        Resizes the image and updates camera intrinsics accordingly.
        Args:
            size: New size for the image in the format (height, width).
        Returns:
            Resized ExtendedImage with updated image and intrinsics.
        """
        resized_image = tvt.functional.resize(self.image, size)  # bilinear and antialiased
        scale_factor = torch.tensor([size[1] / self.resolution[1], size[0] / self.resolution[0]]).view(2, 1)
        resized_intrinsics = self.intrinsics.clone()
        resized_intrinsics[:2, :3] *= scale_factor

        return ExtendedImage(
            image=resized_image,
            intrinsics=resized_intrinsics,
            extrinsics=self.extrinsics,
            distortion=self.distortion,
        )

    def to(self, device: torch.device | str) -> "ExtendedImage":
        """
        Move the image to a specified device.
        Args:
            device: The target device (e.g., 'cuda' or 'cpu').
        Returns:
            ExtendedImage on the specified device.
        """
        return ExtendedImage(
            image=self.image.to(device),
            intrinsics=self.intrinsics.to(device),
            extrinsics=self.extrinsics.to(device),
            distortion=self.distortion.to(device) if self.distortion is not None else None,
        )

    def scale_camera_translations(self, scale: float) -> "ExtendedImage":
        """Scale camera translation vectors.

        Args:
            scale: Scale factor to apply to camera translations

        Returns:
            ExtendedImage with scaled camera translations
        """
        scaled_extrinsics = self.extrinsics.clone()
        scaled_extrinsics[:3, 3] *= scale  # Scale only translation component

        return ExtendedImage(image=self.image, intrinsics=self.intrinsics, extrinsics=scaled_extrinsics, distortion=self.distortion)


@dataclass
class ExtendedImageBatch:
    """A batch of ExtendedImage instances with batch operations."""

    image: Float[Tensor, "b c h w"]
    """Batch of images in the format B x C x H x W."""
    intrinsics: Float[Tensor, "b 3 3"]
    """Batch of camera intrinsics in OpenCV format."""
    extrinsics: Float[Tensor, "b 4 4"]
    """Batch of camera extrinsics in camera-to-world OpenCV format."""
    distortion: Optional[Float[Tensor, "b 5"]] = None
    """Batch of camera distortion coefficients in OpenCV 5-parameter format, optional."""

    @property
    def channels(self) -> int:
        """Number of channels in the image."""
        return self.image.shape[1]

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """Shape of the image tensor."""
        return self.image.shape

    @property
    def resolution(self) -> tuple[int, int]:
        """Resolution of the image, (height, width)."""
        return self.image.shape[2:]

    @classmethod
    def stack(cls, images: list["ExtendedImage"]) -> "ExtendedImageBatch":
        """Creates a batch from a list of ExtendedImage instances."""
        return cls(
            image=torch.stack([img.image for img in images]),
            intrinsics=torch.stack([img.intrinsics for img in images]),
            extrinsics=torch.stack([img.extrinsics for img in images]),
            distortion=torch.stack([img.distortion for img in images]) if images[0].distortion is not None else None,
        )

    def __len__(self) -> int:
        """Get batch size."""
        return len(self.image)

    def __getitem__(self, idx: int) -> ExtendedImage:
        """Get single ExtendedImage from batch."""
        return ExtendedImage(
            image=self.image[idx],
            intrinsics=self.intrinsics[idx],
            extrinsics=self.extrinsics[idx],
            distortion=self.distortion[idx] if self.distortion is not None else None,
        )

    def map(self, fn: callable) -> "ExtendedImageBatch":
        """Apply a function to each ExtendedImage in batch."""
        results = [fn(self[i]) for i in range(len(self.image))]
        return self.stack(results)

    def resize(self, size: tuple[int, int]) -> "ExtendedImageBatch":
        """
        Resizes all images in batch to the same size and updates camera intrinsics.
        Args:
            size: New size for all images in format (height, width).
        Returns:
            Resized ExtendedImageBatch with updated images and intrinsics.
        """
        resized_images = tvt.functional.resize(self.image, size)

        old_h, old_w = self.image.shape[-2:]
        scale_x = size[1] / old_w
        scale_y = size[0] / old_h
        scale_factor = torch.tensor([scale_x, scale_y]).view(1, 2, 1)

        resized_intrinsics = self.intrinsics.clone()
        resized_intrinsics[:, :2, :3] *= scale_factor

        return ExtendedImageBatch(
            image=resized_images,
            intrinsics=resized_intrinsics,
            extrinsics=self.extrinsics,
            distortion=self.distortion,
        )

    def crop_all(self, bbox: Int[Tensor, "4"]) -> "ExtendedImageBatch":
        """
        Efficiently crops all images in batch using the same bounding box.
        Args:
            bbox: Single bounding box to apply to all images,
                 in format (top, left, height, width).
        Returns:
            Cropped ExtendedImageBatch with updated images and intrinsics.
        """
        cropped_images = tvt.functional.crop(self.image, *bbox.tolist())

        cropped_intrinsics = self.intrinsics.clone()
        cropped_intrinsics[:, 0, 2] -= bbox[1].float()  # left offset for all images
        cropped_intrinsics[:, 1, 2] -= bbox[0].float()  # top offset for all images

        return ExtendedImageBatch(
            image=cropped_images,
            intrinsics=cropped_intrinsics,
            extrinsics=self.extrinsics,
            distortion=self.distortion,
        )

    def crop(self, bboxes: Int[Tensor, "... 4"]) -> "ExtendedImageBatch":
        """
        Crops images in batch. If single bbox provided, applies it to all images.
        Args:
            bboxes: Either single bbox or tensor of bounding boxes (one per image),
                   in format (top, left, height, width).
        Returns:
            Cropped ExtendedImageBatch with updated images and intrinsics.
        """
        # If single bbox provided, use efficient batch cropping
        if bboxes.ndim == 1:
            return self.crop_all(bboxes)

        # Otherwise process each image separately
        if len(bboxes) != len(self.image):
            raise ValueError(f"Expected {len(self.image)} bounding boxes, got {len(bboxes)}")

        return self.stack([self[i].crop(bboxes[i]) for i in range(len(self.image))])

    def to(self, device: torch.device | str) -> "ExtendedImageBatch":
        """
        Move the entire batch to a specified device.
        Args:
            device: The target device (e.g., 'cuda' or 'cpu').
        Returns:
            ExtendedImageBatch on the specified device.
        """
        return ExtendedImageBatch(
            image=self.image.to(device),
            intrinsics=self.intrinsics.to(device),
            extrinsics=self.extrinsics.to(device),
            distortion=self.distortion.to(device) if self.distortion is not None else None,
        )

    def scale_camera_translations(self, scale: float | Float[Tensor, " b "]) -> "ExtendedImageBatch":
        """Scale camera translation vectors for entire batch.

        Args:
            scale: Scale factor to apply to camera translations.
                  Can be single float or per-batch scale factors.

        Returns:
            ExtendedImageBatch with scaled camera translations
        """
        scaled_extrinsics = self.extrinsics.clone()

        if isinstance(scale, float):
            scaled_extrinsics[..., :3, 3] *= scale
        else:
            # Handle per-batch scaling
            scale = scale.view(-1, 1)  # [B, 1]
            scaled_extrinsics[..., :3, 3] *= scale

        return ExtendedImageBatch(image=self.image, intrinsics=self.intrinsics, extrinsics=scaled_extrinsics, distortion=self.distortion)


def extimage_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Custom collate function that handles ExtendedImage, ExtendedImageBatch and other tensor types.
    Dictionaries will also be collated into a list of dictionaries.

    Args:
        batch: List of dictionaries containing mixed types of data.

    Returns:
        Collated dictionary with properly batched data.

    Example:
        ```python
        dataset = YourDataset(...)  # returns dict with ExtendedImageBatch
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            collate_fn=extimage_collate_fn
        )
        ```
    """
    if not batch:
        return {}

    result = {}
    elem = batch[0]  # all the batches have the same set of dictionary keys

    for key in elem:
        values = [d[key] for d in batch]

        if isinstance(values[0], ExtendedImage):
            result[key] = [ExtendedImageBatch.stack(values)]  # wrap into 1-item list
        elif isinstance(values[0], (ExtendedImageBatch, dict)):
            result[key] = values  # keep as a list
        else:
            result[key] = default_collate(values)

    return result
