from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.utils as vutils

from models.jpeg_modules import JPEGDecoder


def do(magnification: int = 15, padding: int = 8):
    '''8×8画素の二次元離散コサイン変換の基底画像を出力します。

    Args:
        magnification: 各基底画像の拡大率
            各基底画像は(8 * magnification, 8 * magnification)のサイズになります。
        padding: 各基底画像の間の画素数
    '''
    # デコーダとテンソルを定義
    decoder = JPEGDecoder()
    y_blocks = torch.zeros(64, 64, 1, 1, dtype=torch.float32)
    # 64種類の基底画像の元となるDCT係数のバッチを作成
    for i in range(64):
        y_blocks[i, i, 0, 0] = 1
    # DCT係数から画像へ変換
    images = decoder(y_blocks)
    # 各基底画像を拡大(Nearest Neighbor)
    images = F.interpolate(images, scale_factor=magnification, mode='nearest')
    # 出力ディレクトリの作成
    output_dir = Path('./outputs/')
    output_dir.mkdir(exist_ok=True)
    # 画像の保存
    vutils.save_image(
        images,
        output_dir.joinpath('dct_bases.png'),
        nrow=8, normalize=True, padding=padding)
