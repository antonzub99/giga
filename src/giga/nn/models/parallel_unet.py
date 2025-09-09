import torch
from torch import Tensor, nn

from .unet import UNet


class ParallelUNet(nn.Module):
    def __init__(
        self,
        in_channels: list[int],
        out_channels: list[int],
        model_channels: int = 64,
        channel_multipliers: list[int] | tuple[int, ...] = (1, 2, 4, 8),
        resblocks_per_layer: int | list[int] = 2,
        transformer_depth: int = 1,
        num_heads: int = 8,
        head_dim: int | None = None,
        transformer_in_layers: bool = False,
        transformer_in_bottleneck: bool = True,
        context_dim: int | None = None,
        use_self_attention: bool = True,
        resblock_resize: bool = True,
        dropout: float = 0.0,
        initialize_weights: bool = False,
    ):
        super().__init__()
        if len(in_channels) != len(out_channels):
            raise ValueError("in_channels and out_channels must have the same length.")

        self.num_models = len(in_channels)
        self.channel_multipliers = channel_multipliers
        self.unets = nn.ModuleList(
            [
                UNet(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    model_channels=model_channels,
                    channel_multipliers=channel_multipliers,
                    resblocks_per_layer=resblocks_per_layer,
                    transformer_depth=transformer_depth,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    transformer_in_layers=transformer_in_layers,
                    transformer_in_bottleneck=transformer_in_bottleneck,
                    context_dim=context_dim,
                    use_self_attention=use_self_attention,
                    resblock_resize=resblock_resize,
                    dropout=dropout,
                    initialize_weights=initialize_weights,
                )
                for in_ch, out_ch in zip(in_channels, out_channels)
            ]
        )

    def forward(self, x: Tensor, context: Tensor | None = None) -> list[Tensor]:
        outputs = []
        for i in range(self.num_models):
            model_output = self.unets[i](x, context)
            outputs.append(model_output)

        return torch.cat(outputs, dim=1)
