import torch.nn as nn
from torch.nn.utils import spectral_norm

from . import Swish


class SimpleDecoderBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int,
        upsample: bool = True,
    ):
        super().__init__()
        if upsample:
            self.main = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                spectral_norm(
                    nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(negative_slope=0.2)
            )
        else:
            self.main = nn.Sequential(
                spectral_norm(
                    nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(negative_slope=0.2)
            )

    def forward(self, x):
        return self.main(x)


class SLE(nn.Module):
    def __init__(
        self,
        from_channels,
        to_channels,
        *,
        max_pool=False,
        swish=False,
    ):
        super().__init__()
        self.f = nn.Sequential(
            # 適応的プーリング -> (from_channels, 4, 4)
            nn.AdaptiveMaxPool2d(4) if max_pool else nn.AdaptiveAvgPool2d(4),
            # 畳み込み -> (to_channels, 1, 1)
            spectral_norm(nn.Conv2d(from_channels, to_channels, 4)),
            # 活性化
            Swish() if swish else nn.LeakyReLU(0.1),
            # 畳み込み
            spectral_norm(nn.Conv2d(to_channels, to_channels, 1)),
            # 活性化
            nn.Sigmoid()
        )

    def forward(self, x_from, x_to):
        return x_to * self.f(x_from)
