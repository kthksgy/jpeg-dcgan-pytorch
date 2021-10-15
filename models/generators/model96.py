from logging import getLogger

import torch
import torch.nn as nn

from .model32 import GBlock

logger = getLogger(__name__)
logger.debug('スクリプトを読み込みました。')


class Generator(nn.Module):
    def __init__(
        self, nz: int, nc: int,
        num_classes: int = 0
    ):
        super().__init__()
        logger.debug('Generatorのインスタンスを作成します。')
        self.blocks = nn.ModuleList([
            # 1 -> 6
            GBlock(
                nz, 256, 6,
                stride=1, padding=0,
                num_classes=num_classes),
            # 6 -> 12
            GBlock(
                256, 128, 4,
                stride=2, padding=1,
                num_classes=num_classes),
            # 12 -> 24
            GBlock(
                128, 64, 4,
                stride=2, padding=1,
                num_classes=num_classes),
            # 24 -> 48
            GBlock(
                64, 32, 4,
                stride=2, padding=1,
                num_classes=num_classes),
            # 48 -> 96
            GBlock(
                32, 16, 4,
                stride=2, padding=1,
                num_classes=num_classes),
        ])

        self.last = nn.Conv2d(16, nc, 1, stride=1, padding=0)

    def forward(
        self, z: torch.Tensor, classes: torch.Tensor = None,
    ):
        x = z.view(-1, z.size(1), 1, 1)
        for block in self.blocks:
            x = block(x, classes)
        return self.last(x)
