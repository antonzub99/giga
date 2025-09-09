import torch
from einops import rearrange
from jaxtyping import Float
from loguru import logger
from timm.models.vision_transformer import PatchEmbed
from torch import Tensor, nn

from ..attention import rope
from ..components import IdentityAny, LayerNorm32, ParallelCrossTransformerBlock, ParallelTransformerBlock


class ViT(nn.Module):
    """DiT-like Vision Transformer."""

    def __init__(
        self,
        input_size: int,
        patch_size: int,
        in_channels: int,
        out_channels: int | None = None,
        context_dim: int | None = None,
        model_channels: int = 1024,
        depth: int = 14,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        out_channels = out_channels or in_channels
        self.out_channels = patch_size * patch_size * out_channels
        self.model_channels = model_channels
        self.input_size = input_size
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.head_dim = model_channels // num_heads
        self.patchifier = PatchEmbed(
            img_size=input_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=model_channels,
            bias=True,
        )  # cut image into patches and embed them
        self.num_patches = self.patchifier.num_patches
        # table for rope frequencies
        self.rope_embedding = nn.Parameter(torch.zeros(1, 1, self.num_patches, self.head_dim // 2, 2, 2), requires_grad=False)

        if depth % 2 != 0:
            logger.warning("Depth is odd, assuming bottleneck layer was included")
            depth = depth - 1
        self.encoder = nn.ModuleList(
            [
                ParallelTransformerBlock(
                    dim=model_channels,
                    num_heads=num_heads,
                    head_dim=self.head_dim,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(depth // 2)
            ]
        )
        if context_dim is not None:
            self.bottleneck = ParallelCrossTransformerBlock(
                dim=model_channels,
                num_heads=num_heads,
                head_dim=self.head_dim,
                mlp_ratio=mlp_ratio,
                context_dim=context_dim,
            )
        else:
            self.bottleneck = IdentityAny()  # do nothing

        self.decoder = nn.ModuleList(
            [
                ParallelTransformerBlock(
                    dim=model_channels,
                    num_heads=num_heads,
                    head_dim=self.head_dim,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(depth // 2)
            ]
        )

        self.output_layer = nn.Sequential(
            LayerNorm32(model_channels, elementwise_affine=False, eps=1e-6),
            nn.Linear(model_channels, self.out_channels, bias=True),
        )

        self.init_weights()

    def init_weights(self):
        y_pos = torch.linspace(-1.0, 1.0, self.input_size // self.patch_size)
        x_pos = torch.linspace(-1.0, 1.0, self.input_size // self.patch_size)

        grid_y, grid_x = torch.meshgrid(y_pos, x_pos, indexing="ij")
        pos = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=-1).unsqueeze(0)  # [1, h*w, 2]
        rope_freqcs = torch.cat([rope(pos[..., i], self.head_dim // 2, 10000) for i in range(2)], dim=-3).unsqueeze(
            1
        )  # [1, 1, h*w, d, 2, 2]
        self.rope_embedding.data.copy_(rope_freqcs.data)

        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.patchifier.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.patchifier.proj.bias, 0)

    def unpatchify(self, x: Float[Tensor, "b l c2"]) -> Float[Tensor, "b c h w"]:
        out_channels = self.out_channels // self.patch_size**2
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape((x.shape[0], h, w, self.patch_size, self.patch_size, out_channels))
        x = rearrange(x, "b h w p1 p2 c -> b c (h p1) (w p2)")
        return x

    def forward(self, x: Float[Tensor, "b c1 h w"], condition: Float[Tensor, "b l f"] | None = None) -> Float[Tensor, "b c2 h w"]:
        x = self.patchifier(x)
        for block in self.encoder:
            x = block(x, self.rope_embedding)
        x = self.bottleneck(x, condition, self.rope_embedding)
        for block in self.decoder:
            x = block(x, self.rope_embedding)
        x = self.output_layer(x)
        x = self.unpatchify(x)
        return x
