import torch
from torch import Tensor, nn

from ..components import Downsample, GroupNorm32, ResBlock, SpatialTransformer, TimestepSequential, Upsample


class UNet(nn.Module):
    """
    Customizable UNet architecture with optional spatial transformers.

    Features:
    - Configurable number of layers in encoder/decoder
    - Configurable number of ResBlocks per layer
    - Optional SpatialTransformer at the end of each layer and in bottleneck
    - Conditional input support for transformer blocks
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
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
        """
        Initialize the UNet.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            model_channels: Base channel count for the model
            channel_multipliers: Channel multipliers for each resolution level
            resblocks_per_layer: Number of ResBlocks per layer, can be a single int or list per layer
            transformer_depth: Depth of transformer blocks
            num_heads: Number of attention heads in transformers
            head_dim: Dimension per attention head, if None, will be model_channels // num_heads
            transformer_in_layers: Whether to use transformers at the end of each layer
            transformer_in_bottleneck: Whether to use transformer in bottleneck
            context_dim: Dimension of context/condition for transformers
            use_self_attention: Whether to default to self-attention when no context is provided
            resblock_resize: Whether to use ResBlock when resizing
            dropout: Dropout probability
        """
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.context_dim = context_dim
        self.use_self_attention = use_self_attention
        self.channel_multipliers = channel_multipliers

        # Input projection
        self.input_projection = TimestepSequential(nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1))

        num_layers = len(channel_multipliers)
        if isinstance(transformer_depth, int):
            transformer_depth = num_layers * [transformer_depth]

        if isinstance(resblocks_per_layer, int):
            resblocks_per_layer = num_layers * [resblocks_per_layer]

        # Encoder (downsampling path)
        self.encoder = nn.ModuleList()
        encoder_feature_channels = [model_channels]
        current_ch = model_channels

        for level, ch_mult in enumerate(channel_multipliers):
            encoder_block = []
            for _ in range(resblocks_per_layer[level] - 1):
                encoder_block.append(
                    ResBlock(
                        in_channels=current_ch,
                        out_channels=model_channels * ch_mult,
                        use_conv=True,
                        dropout=dropout,
                        mode="none",  # does not change spatial size
                    )
                )
                current_ch = model_channels * ch_mult

            # put transformer after same-size ResBlocks
            if transformer_in_layers:
                head_dim = current_ch // num_heads
                encoder_block.append(
                    SpatialTransformer(
                        in_channels=current_ch,
                        depth=transformer_depth[level],
                        num_heads=num_heads,
                        head_dim=head_dim,
                        ctx_dim=context_dim,
                        disable_self_attn=not use_self_attention,
                        dropout=dropout,
                    )
                )
            # downsampling block at the end of each layer
            encoder_block.append(
                ResBlock(in_channels=current_ch, out_channels=current_ch, use_conv=True, dropout=dropout, mode="down")
                if resblock_resize
                else Downsample(in_channels=current_ch, out_channels=current_ch, use_conv=True)
            )
            encoder_feature_channels.append(current_ch)
            self.encoder.append(TimestepSequential(*encoder_block))

        self.bottleneck_feature_channels = current_ch

        self.bottleneck = TimestepSequential(
            ResBlock(
                in_channels=current_ch,
                out_channels=current_ch,
                use_conv=True,
                dropout=dropout,
                mode="none",  # does not change spatial size
            ),
            SpatialTransformer(
                in_channels=current_ch,
                depth=transformer_depth[-1],
                num_heads=num_heads,
                head_dim=current_ch // num_heads,
                ctx_dim=context_dim,
                disable_self_attn=not use_self_attention,
                dropout=dropout,
            )
            if transformer_in_bottleneck
            else torch.nn.Identity(),
            ResBlock(
                in_channels=current_ch,
                out_channels=current_ch,
                use_conv=True,
                dropout=dropout,
                mode="none",  # does not change spatial size
            ),
        )

        self.decoder = nn.ModuleList()

        for level, ch_mult in enumerate(reversed(channel_multipliers)):
            decoder_block = []
            skip_channels = encoder_feature_channels.pop()
            # skip connection from encoder to decoder
            decoder_block.append(
                ResBlock(
                    in_channels=current_ch + skip_channels,
                    out_channels=model_channels * ch_mult,
                    use_conv=True,
                    dropout=dropout,
                    mode="none",  # does not change spatial size
                )
            )
            current_ch = model_channels * ch_mult
            for _ in range(resblocks_per_layer[level] - 2):
                decoder_block.append(
                    ResBlock(
                        in_channels=current_ch,
                        out_channels=current_ch,
                        use_conv=True,
                        dropout=dropout,
                        mode="none",  # does not change spatial size
                    )
                )
                current_ch = model_channels * ch_mult
            if transformer_in_layers:
                head_dim = current_ch // num_heads
                decoder_block.append(
                    SpatialTransformer(
                        in_channels=current_ch,
                        depth=transformer_depth[level],
                        num_heads=num_heads,
                        head_dim=head_dim,
                        ctx_dim=context_dim,
                        disable_self_attn=not use_self_attention,
                        dropout=dropout,
                    )
                )
            # upsampling block at the end of each layer
            decoder_block.append(
                ResBlock(in_channels=current_ch, out_channels=current_ch, use_conv=True, dropout=dropout, mode="up")
                if resblock_resize
                else Upsample(in_channels=current_ch, out_channels=current_ch, use_conv=True)
            )
            self.decoder.append(TimestepSequential(*decoder_block))

        # Output projection
        self.output_blocks = nn.Sequential(
            GroupNorm32(32, model_channels), nn.SiLU(), nn.Conv2d(model_channels, out_channels, kernel_size=1, padding=0)
        )

        if initialize_weights:
            self._initialize_weights()
            self._re_zero_modules()

    def _re_zero_modules(self):
        """Re-zero modules that were marked as zero-initialized."""
        for m in self.modules():
            if hasattr(m, "_is_zero_module") and m._is_zero_module:
                for p in m.parameters():
                    p.data.zero_()

    def _initialize_weights(self):
        """
        Applies initial weights to certain layers while preserving zero-initialized modules.
        """
        for m in self.modules():
            # Skip modules that are marked as zero-initialized
            if hasattr(m, "_is_zero_module") and m._is_zero_module:
                continue

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor, condition: Tensor | None = None, other: Tensor | None = None) -> Tensor:
        """
        Forward pass through the UNet.

        Args:
            x: Input tensor of shape [batch, channels, height, width]
            condition: Optional conditioning tensor for transformers
                       of shape [batch, sequence_length, dim]
            other: Optional tensor for injecting embedding into ResBlocks (not used now)

        Returns:
            Output tensor of shape [batch, out_channels, height, width]
        """
        h = self.input_projection(x)
        # Store encoder outputs for skip connections
        feature_stack = []

        # Encoder (downsampling path)
        for encoder_block in self.encoder:
            h = encoder_block(h, condition, other)
            feature_stack.append(h)

        h = self.bottleneck(h, condition, other)
        for decoder_block in self.decoder:
            h = torch.cat([h, feature_stack.pop()], dim=1)
            h = decoder_block(h, condition, other)

        h = h.type_as(x)
        output = self.output_blocks(h)

        return output
