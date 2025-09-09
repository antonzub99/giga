import numpy as np
import torch
from torch import Tensor
from torch.nn import Module


def zero_module(module: Module) -> Module:
    """Initialize module parameters to zero and mark it to preserve during weight initialization."""
    for p in module.parameters():
        p.data.zero_()
    # Mark the module to preserve zero initialization during general weight init
    module._is_zero_module = True
    return module


def dim_norm(x: Tensor, dim: int | None = None, eps=1e-5) -> Tensor:
    dim = dim if dim is not None else list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.type_as(x)


def mp_cat(a: Tensor, b: Tensor, dim: int = 1, weight: float = 0.5) -> Tensor:
    na = a.shape[dim]
    nb = b.shape[dim]
    c = np.sqrt((na + nb) / (1 - weight) ** 2 + weight**2)
    wa = c / np.sqrt(na) * (1 - weight)
    wb = c / np.sqrt(nb) * weight
    return torch.cat([wa * a, wb * b], dim=dim)


def timestep_embedding(timesteps: Tensor, dim: int, max_period: int = 10_000, time_factor: float = 1000.0) -> Tensor:
    timesteps = time_factor * timesteps
    half_dim = dim // 2
    freqs = torch.exp(-np.log(max_period) * torch.arange(0, half_dim, dtype=torch.float32) / half_dim).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    padding = torch.zeros(emb.shape[0], dim % 2, device=emb.device, dtype=emb.dtype)
    emb = torch.cat([emb, padding], dim=-1)
    return emb
