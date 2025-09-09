from jaxtyping import Float
from torch import Tensor, nn

from ..components import FeedForward


class FFMLP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 3,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks

        project_in = nn.Linear(in_channels, out_channels, bias=False)
        blocks = [project_in]
        for _ in range(num_blocks):
            blocks.append(
                FeedForward(
                    dim=out_channels,
                    dim_out=out_channels,
                    mult=1,
                )
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: Float[Tensor, "b ... in_channels"]) -> Float[Tensor, "b ... out_channels"]:
        """Forward pass of the MLP."""
        x = self.blocks(x)
        return x
