from abc import abstractmethod
from typing import Iterable, Literal

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import Module

from .attention import attention, rope
from .operations import zero_module


class IdentityAny(nn.Identity):
    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        return super().forward(x)


class RopeEmbedding(Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat([rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)], dim=-3)
        return emb.unsqueeze(1)


class RMSNorm(Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        rrms = torch.rsqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype) * self.scale


class QKNorm(Module):
    def __init__(self, dim: int):
        super().__init__()
        self.q_norm = RMSNorm(dim)
        self.k_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.q_norm(q)
        k = self.k_norm(k)
        return q.to(v), k.to(v)


class GEGLU(Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.projection = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: Tensor) -> Tensor:
        x, gate = self.projection(x).chunk(2, dim=-1)
        return x * F.gelu(gate, approximate="tanh")


class FeedForward(Module):
    def __init__(self, dim: int, dim_out: int = None, mult: int = 4, glu: bool = False, dropout: float = 0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        project_in = nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU(approximate="tanh")) if not glu else GEGLU(dim, inner_dim)
        self.net = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class LayerNorm32(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).to(x.dtype)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x: Tensor):
        return super().forward(x.float()).to(x.dtype)


def Normalize(in_channels: int) -> Module:
    return GroupNorm32(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class SelfAttention(Module):
    """Multi-head self-attention with query-key normalization.
    Processes sequences using scaled dot-product attention."""

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.projection = nn.Linear(dim, dim)

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = self.projection(x)
        return x


class CrossAttention(Module):
    """Multi-head cross-attention between query and context sequences.
    Allows attention between different feature spaces. Default to self-attention if no context is provided."""

    def __init__(self, q_dim: int, ctx_dim: int = None, head_dim: int = 64, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()

        inner_dim = head_dim * num_heads
        ctx_dim = ctx_dim if ctx_dim is not None else q_dim
        self.num_heads = num_heads

        self.q = nn.Linear(q_dim, inner_dim, bias=qkv_bias)
        self.kv = nn.Linear(ctx_dim, inner_dim * 2, bias=qkv_bias)
        self.norm = QKNorm(head_dim)

        self.projection = nn.Linear(inner_dim, q_dim)

    def forward(self, x: Tensor, pe: Tensor, context: Tensor | None = None) -> Tensor:
        q = self.q(x)
        q = rearrange(q, "B L (H D) -> B H L D", H=self.num_heads)
        context = context if context is not None else x
        kv = self.kv(context)
        k, v = rearrange(kv, "B L (K H D) -> K B H L D", K=2, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = self.projection(x)
        return x


class TransformerBlock(Module):
    """Basic transformer block with cross attention and feed-forward.
    Implements the standard transformer architecture with normalizations."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        ctx_dim: int | None = None,
        gated_ff: bool = True,
        disable_self_attn: bool = False,
    ):
        super().__init__()
        effective_ctx_dim = ctx_dim if ctx_dim is not None else dim

        self.attn1 = CrossAttention(
            q_dim=dim,
            ctx_dim=effective_ctx_dim if disable_self_attn else None,
            head_dim=head_dim,
            num_heads=num_heads,
        )
        self.disable_self_attn = disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(
            q_dim=dim,
            ctx_dim=effective_ctx_dim if disable_self_attn else None,
            head_dim=head_dim,
            num_heads=num_heads,
        )

        # self.lnorm1 = nn.LayerNorm(dim)
        # self.lnorm2 = nn.LayerNorm(dim)
        # self.lnorm3 = nn.LayerNorm(dim)

        self.lnorm1 = LayerNorm32(dim)
        self.lnorm2 = LayerNorm32(dim)
        self.lnorm3 = LayerNorm32(dim)

    def forward(self, x: Tensor, pe: Tensor, context: Tensor | None = None) -> Tensor:
        x = self.attn1(self.lnorm1(x), pe, context=context if self.disable_self_attn else None) + x
        x = self.attn2(self.lnorm2(x), pe, context=context if self.disable_self_attn else None) + x
        x = self.ff(self.lnorm3(x)) + x
        return x


class SpatialTransformer(Module):
    """Transformer that operates on spatial (image-like) inputs.
    Converts spatial data to sequence form for attention processing."""

    def __init__(
        self,
        in_channels: int,
        num_heads: int,
        head_dim: int,
        depth: int = 1,
        dropout: float = 0.0,
        ctx_dim: int | None = None,
        use_linear: bool = False,
        disable_self_attn: bool = False,
        rope_theta: int = 10000,
    ):
        super().__init__()

        inner_dim = head_dim * num_heads
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)

        self.rope = RopeEmbedding(
            dim=head_dim,
            theta=rope_theta,
            axes_dim=[head_dim // 2, head_dim // 2],  # Assuming 2D spatial coordinates
        )

        if use_linear:
            self.projection_in = nn.Sequential(
                nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0),
                Rearrange("b c h w -> b (h w) c"),
            )
        else:
            self.projection_in = nn.Sequential(
                Rearrange("b c h w -> b (h w) c"),
                nn.Linear(in_channels, inner_dim),
            )

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    inner_dim,
                    num_heads,
                    head_dim,
                    dropout,
                    ctx_dim,
                    gated_ff=True,
                    disable_self_attn=disable_self_attn,
                )
                for _ in range(depth)
            ]
        )

        # a bit hacky, but helps to avoid conditional branching in the forward pass
        if use_linear:
            self.projection_out_1 = zero_module(nn.Linear(inner_dim, in_channels))
            self.projection_out_2 = nn.Identity()
        else:
            self.projection_out_1 = nn.Identity()
            self.projection_out_2 = zero_module(nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0))
        self.use_linear = use_linear

    def forward(self, x: Float[Tensor, "b c h w"], condition: Float[Tensor, "b l d"] | Float[Tensor, "b f h w"] | None = None) -> Tensor:
        b, c, h, w = x.shape

        if condition is not None and condition.ndim == 3:
            # since we treat each element in H x W grid as a token, sequence length should be l = h * w
            assert condition.shape[1] == 1, (
                f"Condition shape must have sequence length of 1 for spatial transformer, got {condition.shape[1]}"
            )
            seq_len = h * w
            context = repeat(condition, "b l d -> b (m l) d", m=seq_len)
        elif condition is not None and condition.ndim == 4:
            assert condition.shape[2:] == (h, w), f"Condition shape must match spatial dimensions of input x, got {condition.shape[2:]}"
            context = rearrange(condition, "b f h w -> b (h w) f")
        else:
            context = None

        pos_ids = self._create_positional_ids(h, w).to(x.device)
        pe = self.rope(pos_ids)
        h = self.norm(x)
        h = self.projection_in(h).contiguous()

        for block in self.transformer_blocks:
            h = block(h, pe, context=context)

        # unfortunately, we need to know input shapes for rearranging here, so using rearrange as a layer is not possible...
        h = self.projection_out_1(h)
        h = rearrange(h, "b (h w) c -> b c h w", h=int(x.shape[2]), w=int(x.shape[3])).contiguous()
        h = self.projection_out_2(h)
        return x + h

    def _create_positional_ids(self, height: int, width: int) -> Tensor:
        """Create 2D positional IDs for the spatial dimensions."""
        y_pos = torch.linspace(-1.0, 1.0, height)
        x_pos = torch.linspace(-1.0, 1.0, width)

        grid_y, grid_x = torch.meshgrid(y_pos, x_pos, indexing="ij")
        pos = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=-1)

        # Shape: [1, h*w, 2] - batch dimension of 1
        return pos.unsqueeze(0)


class ParallelTransformerBlock(Module):
    """Parallel Transformer block with self-attention and without context conditioning."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        mlp_ratio: float = 3.0,
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_hidden_dim = mlp_hidden_dim

        # qkv projection + mlp_in
        self.linear1 = nn.Linear(dim, dim * 3 + mlp_hidden_dim)
        # projection and mlp_out
        self.linear2 = nn.Linear(dim + mlp_hidden_dim, dim)

        self.norm = QKNorm(head_dim)
        self.pre_norm = LayerNorm32(dim, elementwise_affine=False, eps=1e-6)
        self.activation = nn.GELU(approximate="tanh")

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        qkv_mlp = self.linear1(self.pre_norm(x))
        qkv, mlp = torch.split(qkv_mlp, [3 * self.dim, self.mlp_hidden_dim], dim=-1)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        attn = attention(q, k, v, pe=pe)
        mlp_act = self.activation(mlp)
        output = self.linear2(torch.cat([attn, mlp_act], dim=2))
        return x + output


class ParallelCrossTransformerBlock(Module):
    """Parallel Transformer block with cross-attention and MLP branch.
    Projects input as queries and condition as keys/values."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        context_dim: int,
        mlp_ratio: float = 3.0,
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.context_dim = context_dim
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_hidden_dim = mlp_hidden_dim

        # query projection + mlp_in combined
        self.q_mlp = nn.Linear(dim, dim + mlp_hidden_dim)
        # key-value projection for context
        self.kv = nn.Linear(context_dim, dim * 2)
        # projection and mlp_out combined
        self.linear2 = nn.Linear(dim + mlp_hidden_dim, dim)

        self.norm = QKNorm(head_dim)
        self.pre_norm = LayerNorm32(dim, elementwise_affine=False, eps=1e-6)
        self.context_norm = LayerNorm32(context_dim, elementwise_affine=False, eps=1e-6)
        self.activation = nn.GELU(approximate="tanh")

    def forward(self, x: Tensor, context: Tensor, pe: Tensor | None = None) -> Tensor:
        """
        Args:
            x: Input tensor of shape [B, L, D]
            context: Conditioning tensor of shape [B, S, D']
            pe: Optional positional encoding
        Returns:
            Output tensor of shape [B, L, D]
        """
        q_mlp = self.q_mlp(self.pre_norm(x))
        q, mlp = torch.split(q_mlp, [self.dim, self.mlp_hidden_dim], dim=-1)
        q = rearrange(q, "b l (h d) -> b h l d", h=self.num_heads)

        kv = self.kv(self.context_norm(context))
        k, v = rearrange(kv, "b s (t h d) -> t b h s d", t=2, h=self.num_heads)

        q, k = self.norm(q, k, v)
        attn = attention(q, k, v, pe=pe)

        mlp_act = self.activation(mlp)
        output = self.linear2(torch.cat([attn, mlp_act], dim=2))

        return x + output


class Upsample(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        use_conv: bool = False,
        kernel_size: int = 3,
        padding: int = 1,
        scale_factor: int = 2,
    ):
        super().__init__()

        self.out_channels = out_channels if out_channels is not None else in_channels

        self.scale_factor = scale_factor
        if use_conv:
            self.op = nn.Conv2d(in_channels, self.out_channels, kernel_size, padding=padding)
        else:
            self.op = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        x = self.op(x)
        return x


class Downsample(Module):
    def __init__(self, in_channels: int, out_channels: int | None = None, use_conv: bool = False, padding: int = 1):
        super().__init__()
        self.out_channels = out_channels if out_channels is not None else in_channels
        if use_conv:
            self.op = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, stride=2, padding=padding)
        else:
            self.op = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.op(x)
        return x


class TimestepBlock(Module):
    """Base class for blocks that accept timestep embeddings.
    Defines interface for conditional processing."""

    @abstractmethod
    def forward(self, x: Tensor, embedding: Tensor | None = None) -> Tensor:
        """
        Apply the module to `x` given `embedding` as condition.
        """


class TimestepSequential(nn.Sequential, TimestepBlock):
    """Sequential container that handles timestep conditions.
    Routes conditions to appropriate layers in the sequence."""

    def forward(self, x: Tensor, condition: Tensor | None = None, other: Tensor | None = None) -> Tensor:
        for module in self:
            layer = module
            if isinstance(layer, TimestepBlock):
                x = module(x, other)
            elif isinstance(layer, SpatialTransformer):
                x = module(x, condition)
            else:
                x = module(x)
        return x


class ZeroCondition(Module):
    def forward(self, x: Tensor, other: Tensor | None = None) -> Tensor:
        return torch.zeros_like(x)


class ConditionLayer(Module):
    """Processes conditioning information for residual blocks.
    Projects conditions to match feature dimensions."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.op = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_channels, out_channels),
        )

    def forward(self, x: Tensor, other: Tensor) -> Tensor:
        cond = self.op(other).type_as(x)
        extra_dims = x.ndim - cond.ndim  # assuming it is always non-negative
        cond = cond.view(*cond.shape, *([1] * extra_dims)).contiguous()  # insert singleton spatial dimensions
        return cond


class ResBlock(TimestepBlock):
    """Residual block with optional up/downsampling and conditioning.
    Core building block for UNet architectures."""

    def __init__(
        self,
        in_channels: int,
        condition_channels: int | None = None,
        out_channels: int | None = None,
        use_conv: bool = False,
        dropout: float = 0.0,
        mode: Literal["up", "down", "none"] = "none",
        kernel_size: int | Iterable = 3,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        if isinstance(kernel_size, Iterable):
            padding = [k // 2 for k in kernel_size]
        else:
            padding = kernel_size // 2

        self.activate = nn.Sequential(
            GroupNorm32(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True),
            nn.SiLU(),
        )

        if mode == "up":
            self.h_update = Upsample(out_channels, use_conv=use_conv)
            self.x_update = Upsample(out_channels, use_conv=use_conv)
        elif mode == "down":
            self.h_update = Downsample(out_channels, use_conv=use_conv)
            self.x_update = Downsample(out_channels, use_conv=use_conv)
        else:
            self.h_update = self.x_update = nn.Identity()

        self.in_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

        if condition_channels is None:
            self.condition_layer = ZeroCondition()
        else:
            # maybe put back use_scale_shift_norm ?
            self.condition_layer = ConditionLayer(condition_channels, out_channels)

        self.out_layers = nn.Sequential(
            GroupNorm32(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True),
            nn.SiLU(),
            nn.Dropout(dropout),
            zero_module(nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)),
        )

        if out_channels == in_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: Tensor, condition: Tensor | None = None) -> Tensor:
        """
        Args:
            x: input tensor of shape [B, C, H, W]
            condition: an optional embedding for timestep of shape [B, C']
        """
        h = self.activate(x)
        h = self.h_update(h)
        h = self.in_conv(h)

        x = self.x_update(x)
        embedding = self.condition_layer(h, condition)
        h = h + embedding
        h = self.out_layers(h)
        return self.skip_connection(x) + h
