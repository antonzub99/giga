# Custom Datasets

To train/evaluate GIGA on custom datasets, you need to create a dataset class that follows the expected interface.

## Requirements

### Dataset Structure
Your dataset class should:
1. Inherit from `torch.utils.data.Dataset`
2. Be registered using `@register_dataset("your_dataset_name")`
3. Have a corresponding config class inheriting from `DatasetConfig`
4. Implement `select_characters()` class method for character filtering
5. Return the expected data format from `__getitem__()`

### Required Returns from `__getitem__()`

Your dataset's `__getitem__()` method must return a dictionary with these keys:

```python
{
    "name": str,                    # Character/sample name
    "input_views": ExtendedImageBatch,  # Source view images with cameras
    "target_views": ExtendedImageBatch, # Target view images with cameras  
    "motion": dict,                 # SMPL-X parameters (see below)
    "texture": Tensor,              # Static RGB texture (3, H, W) in [0,1]
    "patch_data": dict,             # Patch sampling info to evaluate perceptual loss during training (see below)
}
```

### Motion Parameters Format

The `motion` dict must contain SMPL-X parameters:

```python
motion = {
    "gender": str,                      # "neutral", "male", or "female"
    "flat_hand": bool,                  # Use flat hand pose
    "hand_pose_dim": int,               # Hand pose dimensionality (might be any number, typically 12 for PCA version or 45 for full hand pose)
    "pelvis_rotation": Tensor,          # (1, 3) pelvis rotation (global_orient in terms of EasyMocap; first 3 channel of body_pose in SMPLX)
    "pelvis_translation": Tensor,       # (1, 3) pelvis translation (interchangable with global translation)
    "shape": Tensor,                    # (1, 10) shape parameters (betas)
    "expression": Tensor,               # (1, 10) expression parameters
    "body_pose": Tensor,                # (1, 63) body pose (21 joints × 3)
    "hand_pose": Tensor,                # (1, 90) hand pose (concatenated left and right hand poses, padded to 90)
    "head_pose": Tensor,                # (1, 12) head pose (jaw, left_eye, right_eye)
    "global_rotation": Tensor,          # (1, 3) global rotation
    "global_translation": Tensor,       # (1, 3) global translation
}
```

### ExtendedImageBatch Format

Images should be provided as `ExtendedImageBatch` (see [`extimage.py`](./src/giga/data/extimage.py)) objects containing:
- RGBA images (with alpha channel for masks)
- Camera intrinsics and extrinsics

### Patch Data Format

```python
patch_data = {
    "center": Tensor,       # (N, 2) patch centers in image coordinates
    "radius": Tensor,       # (N, 2) patch radii
    "size": int,            # Patch size in pixels
}
```

## Example Implementation

See [`src/giga/data/datasets/mvh.py`](./src/giga/data/datasets/mvh.py) for a complete reference implementation, particularly:
- Camera calibration loading
- SMPL-X annotation parsing
- Image and mask loading with proper preprocessing
- Character selection logic

## Registration

Register your dataset in the config system:

```python
@dataclass
class YourDatasetConfig(DatasetConfig):
    _target_: str = "your.module.path.YourDataset"
    # Add your specific config fields

@register_dataset("your_dataset")
class YourDataset(Dataset):
    __config_class__ = YourDatasetConfig
    # Implementation
```