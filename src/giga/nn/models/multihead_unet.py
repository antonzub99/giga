import torch
from torch import Tensor, nn

from ..components import Downsample, GroupNorm32, ResBlock, SpatialTransformer, TimestepSequential, Upsample


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        model_channels: int = 64,
        channel_multipliers: list[int] | tuple[int, ...] = (1, 2, 4),
        resblocks_per_layer: int | list[int] = 2,
        transformer_depth: int = 1,
        num_heads: int = 8,
        head_dim: int | None = None,
        transformer_in_layers: bool = False,
        context_dim: int | None = None,
        use_self_attention: bool = True,
        resblock_resize: bool = True,
        dropout: float = 0.0,
    ):
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

        self.encoder_feature_channels = encoder_feature_channels
        self.encoder_final_channels = current_ch

    def forward(self, x: Tensor, context: Tensor | None = None) -> tuple[Tensor, list[Tensor]]:
        x = self.input_projection(x)

        features = []
        for block in self.encoder:
            x = block(x, context)
            features.append(x)

        return x, features


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        encoder_feature_channels: list[int],
        model_channels: int = 64,
        channel_multipliers: list[int] | tuple[int, ...] = (1, 2, 4),
        resblocks_per_layer: int | list[int] = 2,
        transformer_depth: int = 1,
        num_heads: int = 8,
        head_dim: int | None = None,
        transformer_in_layers: bool = False,
        context_dim: int | None = None,
        use_self_attention: bool = True,
        resblock_resize: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.context_dim = context_dim
        self.use_self_attention = use_self_attention

        self.decoder = nn.ModuleList()

        current_ch = self.in_channels

        num_layers = len(channel_multipliers)
        if isinstance(transformer_depth, int):
            transformer_depth = num_layers * [transformer_depth]

        if isinstance(resblocks_per_layer, int):
            resblocks_per_layer = num_layers * [resblocks_per_layer]
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
            GroupNorm32(32, model_channels), nn.SiLU(), nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x: Tensor, features: list[Tensor], context: Tensor | None = None) -> Tensor:
        num_stacks = len(features)
        h = x
        for idx in range(num_stacks):
            feature = features[num_stacks - idx - 1]
            h = torch.cat([h, feature], dim=1)
            h = self.decoder[idx](h, context)
        output = self.output_blocks(h)
        return output


class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        context_dim: int | None = None,
        transformer_depth: int = 1,
        num_heads: int = 8,
        head_dim: int | None = None,
        use_self_attention: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.bottleneck = TimestepSequential(
            ResBlock(
                in_channels=in_channels,
                out_channels=in_channels,
                use_conv=True,
                dropout=dropout,
                mode="none",  # does not change spatial size
            ),
            SpatialTransformer(
                in_channels=in_channels,
                depth=transformer_depth,
                num_heads=num_heads,
                head_dim=in_channels // num_heads,
                ctx_dim=context_dim,
                disable_self_attn=not use_self_attention,
                dropout=dropout,
            ),
            ResBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                use_conv=True,
                dropout=dropout,
                mode="none",  # does not change spatial size
            ),
        )

    def forward(self, x: Tensor, context: Tensor | None = None) -> Tensor:
        return self.bottleneck(x, context)


class MultiHeadUnet(nn.Module):
    def __init__(
        self,
        in_channels: list[int] | tuple[int, ...],
        out_channels: list[int] | tuple[int, ...],
        model_channels: int = 64,
        channel_multipliers: list[int] | tuple[int, ...] = (1, 2, 4),
        resblocks_per_layer: int | list[int] = 2,
        transformer_in_layers: bool = False,
        transformer_depth: int = 1,
        num_heads: int = 8,
        context_dim: int | None = None,
        use_self_attention: bool = True,
        resblock_resize: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.in_channels = [_ for _ in in_channels]
        self.out_channels = [_ for _ in out_channels]
        self.in_heads = len(in_channels)
        self.out_heads = len(out_channels)
        self.channel_multipliers = channel_multipliers

        self.encoder_blocks = nn.ModuleList()
        self.encoder_feature_channels = []

        self.bottleneck_feature_channels = model_channels * channel_multipliers[-1] * self.in_heads
        for _in_channels in in_channels:
            encoder_block = EncoderBlock(
                in_channels=_in_channels,
                out_channels=model_channels,
                model_channels=model_channels,
                channel_multipliers=channel_multipliers,
                resblocks_per_layer=resblocks_per_layer,
                transformer_depth=transformer_depth,
                num_heads=num_heads,
                transformer_in_layers=transformer_in_layers,
                context_dim=context_dim,
                use_self_attention=use_self_attention,
                resblock_resize=resblock_resize,
                dropout=dropout,
            )
            # Store feature channel dims for each downsampling layer for each encoder block
            self.encoder_feature_channels.append(encoder_block.encoder_feature_channels)
            self.encoder_blocks.append(
                TimestepSequential(
                    encoder_block,
                )
            )
            self.bottleneck = Bottleneck(
                in_channels=model_channels * channel_multipliers[-1] * self.in_heads,
                out_channels=model_channels * channel_multipliers[-1],
                context_dim=context_dim,
                transformer_depth=transformer_depth,
                num_heads=num_heads,
                use_self_attention=use_self_attention,
                dropout=dropout,
            )

        self.decoder_blocks = nn.ModuleList()
        num_stacks = len(self.encoder_feature_channels[0])  # how many downsampling layers in each encoder
        stacked_encoder_feature_channels = [
            [sum([self.encoder_feature_channels[i][j] for i in range(self.in_heads)]) for j in range(num_stacks)]
            for odx in range(self.out_heads)
        ]  # sum feature channels from all encoders for each downsampling layer, repeat for each decoder head
        for _out_channels, stacked_feature_channels in zip(out_channels, stacked_encoder_feature_channels):
            decoder_block = DecoderBlock(
                in_channels=model_channels * channel_multipliers[-1],  # base dimension of the decoder input
                out_channels=_out_channels,
                encoder_feature_channels=stacked_feature_channels,  # stacked downsampled features from all encoders
                model_channels=model_channels,
                channel_multipliers=channel_multipliers,
                resblocks_per_layer=resblocks_per_layer,
                transformer_depth=transformer_depth,
                num_heads=num_heads,
                transformer_in_layers=False,
                context_dim=None,
                use_self_attention=use_self_attention,
                resblock_resize=resblock_resize,
                dropout=dropout,
            )
            self.decoder_blocks.append(decoder_block)

    def forward(self, x: Tensor, context: Tensor | None = None) -> Tensor:
        x = torch.split(x, self.in_channels, dim=1)  # split input tensor into multiple heads based on in_channels
        assert len(x) == self.in_heads, f"Expected {self.in_heads} input heads, got {len(x)}"
        main_features = []
        stacked_feature_pyramids = []

        for x_input, encoder_block in zip(x, self.encoder_blocks):
            feature, pyramid = encoder_block(x_input, context)
            main_features.append(feature)
            stacked_feature_pyramids.append(pyramid)

        latent = torch.cat(main_features, dim=1)
        latent = self.bottleneck(latent, context)

        stacked_pyramid = [
            torch.cat([pyramid[i] for pyramid in stacked_feature_pyramids], dim=1) for i in range(len(stacked_feature_pyramids[0]))
        ]

        del main_features

        outputs: list[Tensor] = []
        for decoder_block in self.decoder_blocks:
            output = decoder_block(latent, stacked_pyramid)
            outputs.append(output)

        del stacked_pyramid
        outputs = torch.cat(outputs, dim=1)  # stack outputs from all decoder heads
        return outputs
