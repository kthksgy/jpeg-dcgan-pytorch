from logging import getLogger

import torch
from torch import nn
from torch.nn.utils import spectral_norm
from .model32 import DBlock, SimpleDecoder
from ..modules import init_xavier_uniform

logger = getLogger(__name__)
logger.debug('スクリプトを読み込みました。')


class Discriminator(nn.Module):
    def __init__(
        self, nc: int,
        num_classes: int = 0
    ):
        super().__init__()
        logger.debug('JPEG版Discriminatorのインスタンスを作成します。')
        self.color = nc == 3

        # 4 -> 2
        self.b1 = DBlock(64, 128, 3, stride=2, padding=1)
        # 2 -> 1
        if self.color:
            self.b2 = DBlock(128 + 128, 256, 2, stride=1, padding=0)
        else:
            self.b2 = DBlock(128, 256, 2, stride=1, padding=0)

        # b3の出力から8×8画像再構築
        self.recons = SimpleDecoder(256, nc)

        self.real_fake = nn.Linear(256, 1)
        if num_classes > 0:
            self.sn_embedding = spectral_norm(
                nn.Embedding(num_classes, 256))
            self.sn_embedding.apply(init_xavier_uniform)
        else:
            self.sn_embedding = None

    def forward(
        self, y_cbcr, classes=None,
        detach: bool = False, reconstruct: bool = False,
    ):
        if self.color:
            y, cbcr = y_cbcr
            if detach:
                y = y.detach()
                cbcr = cbcr.detach()
            x = self.b1(y)
            x = self.b2(torch.cat([x, cbcr], dim=1))
        else:
            y = y_cbcr[0]
            if detach:
                y = y.detach()
            x = self.b1(y)
            x = self.b2(x)
        # 再構築
        recons = self.recons(x) if reconstruct else None
        h = x.view(-1, x.size(1))
        real_fake = self.real_fake(h)
        # cGANs with Projection Discriminator
        if classes is not None:
            real_fake += torch.sum(
                h * self.sn_embedding(classes),
                1, keepdim=True)
        # Real(1) or Fake(0)を出力する
        return real_fake, recons
