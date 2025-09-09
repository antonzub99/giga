import importlib

import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from giga.data import ExtendedImage, ExtendedImageBatch


def instantiate_from_config(config):
    if "_target_" not in config:
        raise ValueError("config must have a target field")
    return get_obj_from_str(config["_target_"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def to_device(data, device=None):
    if device is None:
        device = torch.device("cuda:0")

    if isinstance(data, (torch.Tensor, torch.nn.Module)):
        return data.to(device)
    elif isinstance(data, (ExtendedImage, ExtendedImageBatch)):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_device(v, device) for v in data]
    elif isinstance(data, tuple):
        return tuple(to_device(v, device) for v in data)
    else:
        return data


def tensor_to_4d(x: Float[Tensor, "... c h w"]) -> Float[Tensor, "b c h w"]:
    if x.ndim == 3:
        x = x.unsqueeze(0)
    elif x.ndim == 5:
        x = rearrange(x, "b n c h w -> (b n) c h w")
    return x


def tensor_as_gaussians(x: Float[Tensor, "b c ..."], spec: dict[str, int] | None = None) -> dict[str, Tensor]:
    """
    Splits a tensor into 3D Gaussian parameters along the channel dimension based on the provided specification.
    Channel is assumed to be the first dimension.
    """

    if spec is None:
        spec = {
            "colors": 3,
            "means": 3,
            "quats": 4,
            "scales": 3,
            "opacities": 1,
        }
    assert x.shape[1] == sum(spec.values()), (
        f"The input tensor has {x.shape[1]} channels, but specification provides {sum(spec.values())} channels."
    )

    parts: tuple[Tensor, ...] = x.split(list(spec.values()), dim=1)
    gaussian_params: dict[str, Tensor] = dict(zip(spec.keys(), parts))
    return gaussian_params


def batch_dict(dict_list: list[dict]) -> dict:
    """Combine a list of dictionaries into a single dictionary with batched values.

    Args:
        dict_list: List of dictionaries with the same keys

    Returns:
        Dictionary with batched values (tensors or lists)

    Example:
        ```python
        items = [
            {"gender": "male", "shape": tensor[10]},
            {"gender": "female", "shape": tensor[10]}
        ]
        batch = batch_dict(items)  # {"gender": ["male", "female"], "shape": tensor[2, 10]}
        ```
    """
    if not dict_list:
        return {}

    # Get all keys from first dict (assuming all dicts have same keys)
    keys = dict_list[0].keys()
    if not all(d.keys() == keys for d in dict_list):
        raise ValueError("All dictionaries must have the same keys")

    result = {}
    for key in keys:
        values = [d[key] for d in dict_list]

        if isinstance(values[0], Tensor):
            result[key] = torch.stack(values)
        elif isinstance(values[0], np.ndarray):
            result[key] = np.stack(values)
        elif isinstance(values[0], (str, int, float, bool)):
            result[key] = values
        else:
            raise ValueError(f"Unsupported value type: {type(values[0])} for key {key}")

    return result


def unbatch_dict(batch_dict: dict) -> list[dict]:
    """Split a dictionary with batched values into a list of individual dictionaries.

    Args:
        batch_dict: Dictionary with batched values (tensors or lists)

    Returns:
        List of dictionaries, where each dictionary contains single items from the batch

    Example:
        ```python
        batch = {
            "gender": ["male", "female"],
            "shape": torch.randn(2, 10),
            "body_pose": torch.randn(2, 24, 3)
        }
        items = unbatch_dict(batch)  # Returns list of 2 dicts
        ```
    """
    # Get batch size from first tensor/list in dict
    first_value = next(iter(batch_dict.values()))
    if isinstance(first_value, torch.Tensor):
        batch_size = first_value.shape[0]
    else:
        batch_size = len(first_value)

    result = []
    for idx in range(batch_size):
        item_dict = {}
        for key, value in batch_dict.items():
            if isinstance(value, Tensor) or isinstance(value, np.ndarray):
                item_dict[key] = value[idx]
            elif isinstance(value, list):
                item_dict[key] = value[idx]
            result.append(item_dict)

    return result
