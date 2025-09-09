# rope and apply_rope function borrowed from
# https://github.com/black-forest-labs/flux/blob/main/src/flux/math.py

from enum import Enum
from functools import partial

import torch
from einops import rearrange
from jaxtyping import Float
from loguru import logger
from torch import Tensor


class AttentionBackend(Enum):
    XFORMERS = "xformers"
    FLASH = "flash"
    EFFICIENT = "efficient"
    CUDNN = "cudnn"
    MATH = "math"
    AUTO = "auto"


def _test_attention_params():
    """Create test parameters for attention capability checking."""
    batch_size, num_heads, seq_len, head_dim = 2, 8, 128, 64
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    value = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)

    return query, key, value


def _check_backend_availability(backend: AttentionBackend) -> tuple[bool, str]:
    """Check if a specific backend is available with detailed reasoning."""

    q, k, v = _test_attention_params()
    params = torch.backends.cuda.SDPAParams(q, k, v, None, 0.0, False, False)

    if backend == AttentionBackend.XFORMERS:
        try:
            from xformers.ops import memory_efficient_attention

            # xFormers expects different tensor layout
            q_xf = rearrange(q, "b h l d -> b l h d")
            k_xf = rearrange(k, "b h l d -> b l h d")
            v_xf = rearrange(v, "b h l d -> b l h d")
            _ = memory_efficient_attention(q_xf, k_xf, v_xf)
            return True, "xFormers available and working"
        except ImportError:
            return False, "xFormers not installed"
        except Exception as e:
            return False, f"xFormers test failed: {str(e)[:500]}"

    elif backend == AttentionBackend.FLASH:
        if not torch.cuda.is_available():
            return False, "Flash Attention requires CUDA"
        try:
            can_use = torch.backends.cuda.can_use_flash_attention(params, debug=False)
            return can_use, "Flash Attention capability check passed" if can_use else "Flash Attention capability check failed"
        except Exception as e:
            return False, f"Flash Attention check error: {str(e)[:500]}"

    elif backend == AttentionBackend.EFFICIENT:
        if not torch.cuda.is_available():
            return False, "Efficient Attention requires CUDA"
        try:
            can_use = torch.backends.cuda.can_use_efficient_attention(params, debug=False)
            return can_use, "Efficient Attention capability check passed" if can_use else "Efficient Attention capability check failed"
        except Exception as e:
            return False, f"Efficient Attention check error: {str(e)[:500]}"

    elif backend == AttentionBackend.CUDNN:
        if not torch.cuda.is_available():
            return False, "cuDNN Attention requires CUDA"
        try:
            can_use = torch.backends.cuda.can_use_cudnn_attention(params, debug=False)
            return can_use, "cuDNN Attention capability check passed" if can_use else "cuDNN Attention capability check failed"
        except Exception as e:
            return False, f"cuDNN Attention check error: {str(e)[:500]}"

    elif backend == AttentionBackend.MATH:
        return True, "Math backend always available"

    elif backend == AttentionBackend.AUTO:
        return True, "Auto selection always available"

    return False, f"Unknown backend: {backend}"


def _select_auto_backend() -> AttentionBackend:
    """Auto-select the best available backend."""
    # Priority order: Flash -> xFormers -> Efficient -> cuDNN -> Math
    priority_order = [
        AttentionBackend.FLASH,
        AttentionBackend.XFORMERS,
        AttentionBackend.EFFICIENT,
        AttentionBackend.CUDNN,
        AttentionBackend.MATH,
    ]

    for backend in priority_order:
        available, reason = _check_backend_availability(backend)
        if available:
            return backend

    # Fallback to math (should never reach here)
    logger.warning("No backends available, falling back to math")
    return AttentionBackend.MATH


def setup_attention_backend(preferred_backend: AttentionBackend = AttentionBackend.AUTO):
    """Setup attention backend with proper capability checking."""
    global ATTN_CTX, attn_op, CURRENT_BACKEND

    # Determine actual backend to use
    if preferred_backend == AttentionBackend.AUTO:
        actual_backend = _select_auto_backend()
    else:
        available, reason = _check_backend_availability(preferred_backend)
        if available:
            actual_backend = preferred_backend
        else:
            logger.warning(f"Preferred {preferred_backend.value} backend not available: {reason}")
            actual_backend = _select_auto_backend()

    # Configure the selected backend
    if actual_backend == AttentionBackend.XFORMERS:
        from contextlib import nullcontext

        from xformers.ops import memory_efficient_attention

        ATTN_CTX = nullcontext

        def attn_op(q, k, v):
            """xFormers expects (batch, seq_len, heads, head_dim) format."""
            q = rearrange(q, "b hq l d -> b l hq d")
            k = rearrange(k, "b h l d -> b l h d")
            v = rearrange(v, "b h l d -> b l h d")
            x = memory_efficient_attention(q, k, v)
            x = rearrange(x, "b l hq d -> b hq l d")
            return x

    else:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from torch.nn.functional import scaled_dot_product_attention as sdpa

        backend_map = {
            AttentionBackend.FLASH: SDPBackend.FLASH_ATTENTION,
            AttentionBackend.EFFICIENT: SDPBackend.EFFICIENT_ATTENTION,
            AttentionBackend.CUDNN: SDPBackend.CUDNN_ATTENTION,
            AttentionBackend.MATH: SDPBackend.MATH,
        }

        ATTN_CTX = partial(sdpa_kernel, backends=[backend_map[actual_backend]])
        attn_op = sdpa

    CURRENT_BACKEND = actual_backend


def get_available_backends() -> dict[str, tuple[bool, str]]:
    """Get status of all attention backends."""
    backends_status = {}
    for backend in AttentionBackend:
        if backend != AttentionBackend.AUTO:
            available, reason = _check_backend_availability(backend)
            backends_status[backend.value] = (available, reason)
    return backends_status


def log_backend_status():
    """Log the availability status of all backends."""
    logger.info("Attention backend availability:")
    for backend_name, (available, reason) in get_available_backends().items():
        status = "✓" if available else "✗"
        logger.info(f"  {status} {backend_name:12}: {reason}")


# Initialize with AUTO by default and log status
setup_attention_backend(AttentionBackend.AUTO)


def attention(
    q: Float[Tensor, "b hq l d"], k: Float[Tensor, "b h l d"], v: Float[Tensor, "b h l d"], pe: Float[Tensor, "b 1 l d2 2 2"]
) -> Float[Tensor, "b l (hq d)"]:
    """Apply attention with RoPE positional encoding."""
    q, k = apply_rope(q, k, pe)

    with ATTN_CTX():
        x = attn_op(q, k, v)
    x = rearrange(x, "b hq l d -> b l (hq d)")

    return x


# Keep existing rope and apply_rope functions unchanged
def rope(pos: Float[Tensor, "... n"], dim: int, theta: int) -> Float[Tensor, "b n d x y"]:
    scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(
    xq: Float[Tensor, "b hq l d"], xk: Float[Tensor, "b h l d"], freq_cis: Float[Tensor, "... 2"]
) -> tuple[Float[Tensor, "b hq l d"], Float[Tensor, "b h l d"]]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freq_cis[..., : xq.shape[-2], :, :, 0] * xq_[..., 0] + freq_cis[..., : xq.shape[-2], :, :, 1] * xq_[..., 1]
    xk_out = freq_cis[..., : xk.shape[-2], :, :, 0] * xk_[..., 0] + freq_cis[..., : xk.shape[-2], :, :, 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)
