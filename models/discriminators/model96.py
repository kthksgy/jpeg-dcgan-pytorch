from logging import getLogger

import torch
from torch import nn
from torch.nn.utils import spectral_norm
from .model32 import DBlock
from ..modules import init_xavier_uniform
from ..modules.lightweight import SimpleDecoderBlock

logger = getLogger(__name__)
logger.debug('スクリプトを読み込みました。')


class SimpleDecoder(nn.Module):
    def __init__(self, in_channels: int, nc: int):
        super().__init__()
        self.decode = nn.Sequential(
            # 1 -> 3
            nn.Upsample(size=3, mode='nearest'),
            spectral_norm(
                nn.Conv2d(in_channels, 128, 3, 1, 1, bias=False)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2),
            # 3 -> 6
            SimpleDecoderBlock(128, 64),
            # 6 -> 12
            SimpleDecoderBlock(64, 32),
            # 12 -> 24
            SimpleDecoderBlock(32, 16),
            # 24 -> 24
            spectral_norm(
                nn.Conv2d(16, nc, 1, stride=1, padding=0, bias=False)),
            nn.Sigmoid())

    def forward(self, x):
        return self.decode(x)


class Discriminator(nn.Module):
    def __init__(
        self, nc: int,
        num_classes: int = 0
    ):
        super().__init__()
        logger.debug('Discriminatorのインスタンスを作成します。')

        self.blocks = nn.Sequential(
            # 96 -> 48
            DBlock(nc, 32, 3, stride=2, padding=1),
            # 48 -> 24
            DBlock(32, 64, 3, stride=2, padding=1),
            # 24 -> 12
            DBlock(64, 128, 3, stride=2, padding=1),
            # 12 -> 6
            DBlock(128, 256, 3, stride=2, padding=1),
            # 6 -> 1
            DBlock(256, 512, 6, stride=1, padding=0),
        )

        # ブロックの出力から8×8画像再構築
        self.recons = SimpleDecoder(512, nc)

        self.real_fake = nn.Linear(512, 1)
        if num_classes > 0:
            self.sn_embedding = spectral_norm(
                nn.Embedding(num_classes, 512))
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
