from logging import getLogger
from typing import Tuple, Union

import torch
from torch import nn
from torch.nn.utils import spectral_norm
from ..modules import init_xavier_uniform
from ..modules.lightweight import SimpleDecoderBlock

logger = getLogger(__name__)
logger.debug('スクリプトを読み込みました。')


class SimpleDecoder(nn.Module):
    def __init__(self, in_channels: int, nc: int):
        super().__init__()
        self.decode = nn.Sequential(
            # 1 -> 2
            SimpleDecoderBlock(in_channels, 64),
            # 2 -> 4
            SimpleDecoderBlock(64, 32),
            # 4 -> 8
            SimpleDecoderBlock(32, 16),
            # 8 -> 8
            spectral_norm(
                nn.Conv2d(16, nc, 1, stride=1, padding=0, bias=False)),
            nn.Sigmoid())

    def forward(self, x):
        return self.decode(x)


class DBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        num_classes: int = 0, prob_dropout: float = 0.3
    ):
        super().__init__()
        self.main = nn.Sequential(
            spectral_norm(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size, stride, padding, bias=True)),
            nn.LeakyReLU(0.2),
        )
        self.main.apply(init_xavier_uniform)  # 重みの初期化

    def forward(self, inputs):
        return self.main(inputs)


class Discriminator(nn.Module):
    def __init__(
        self, nc: int,
        num_classes: int = 0
    ):
        super().__init__()
        logger.debug('Discriminatorのインスタンスを作成します。')

        self.blocks = nn.Sequential(
            # 32 -> 16
            DBlock(nc, 32, 3, stride=2, padding=1),
            # 16 -> 8
            DBlock(32, 64, 3, stride=2, padding=1),
            # 8 -> 4
            DBlock(64, 128, 3, stride=2, padding=1),
            # 4 -> 1
            DBlock(128, 256, 4, stride=1, padding=0),
        )

        # ブロックの出力から8×8画像再構築
        self.recons = SimpleDecoder(256, nc)

        self.real_fake = nn.Linear(256, 1)
        if num_classes > 0:
            self.sn_embedding = spectral_norm(
                nn.Embedding(num_classes, 256))
            self.sn_embedding.apply(init_xavier_uniform)
        else:
            self.sn_embedding = None

    def forward(
        self, x, classes=None,
        detach: bool = False, reconstruct: bool = False,
    ):
        if detach:
            x = x.detach()
        x = self.blocks(x)
        # 再構築
        recons = self.recons(x) if reconstruct else None
        h = x.view(-1, x.size(1))
        real_fake = self.real_fake(h)
        # cGANs with Projection Discriminator
        if classes is not None:
            real_fake += torch.sum(
                h * self.sn_embedding(classes),
                1, keepdim=True)
        return real_fake, recons
