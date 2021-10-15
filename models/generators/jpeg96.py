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
        logger.debug('JPEG版Generatorのインスタンスを作成します。')
        self.color = nc == 3
        self.blocks = nn.ModuleList([
            # 1 -> 3
            GBlock(
                nz, 512, 3,
                stride=1, padding=0,
                num_classes=num_classes),
            # 3 -> 6
            GBlock(
                512, 256, 4,
                stride=2, padding=1,
                num_classes=num_classes),
            # 6 -> 12
            GBlock(
                256, 128, 4,
                stride=2, padding=1,
                num_classes=num_classes),
        ])
        # 12 -> 12
        self.last_y = nn.Conv2d(128, 64, 1, stride=1, padding=0)
        if self.color:
            self.last_cbcr = nn.Conv2d(128, 128, 2, stride=2, padding=0)

    def forward(
        self, z: torch.Tensor, classes: torch.Tensor = None,
    ):
        x = z.view(-1, z.size(1), 1, 1)
        for block in self.blocks:
            x = block(x, classes)
        return self.last_y(x), self.last_cbcr(x) if self.color else None
