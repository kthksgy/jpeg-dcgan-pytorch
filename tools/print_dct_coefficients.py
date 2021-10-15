import cv2
import numpy as np
from utils.common.image import to_uint8
from utils.transforms.jpeg import (
    ToYCbCr,
    ChromaSubsampling,
    BlockwiseDCT,
    JPEGQuantize,
)


def do(filename='test32.png', quality=50):
    np.set_printoptions(precision=1, suppress=True)
    image = cv2.imread(f'assets/{filename}')[:8, :8, [2, 1, 0]]
    to_ycbcr = ToYCbCr()
    chroma_subsampling = ChromaSubsampling(ratio='4:2:0')
    bwdct = BlockwiseDCT(block_size=(8, 8))
    quantize = JPEGQuantize(quality=quality)

    ycbcr = to_ycbcr(image)
    y, _ = chroma_subsampling(ycbcr)
    print('画素(Y成分):')
    print(to_uint8(y).reshape(8, 8))
    y_block = bwdct(y)
    print('DCT係数(Y成分):')
    print(np.trunc(y_block).astype(np.int64).reshape(8, 8))
    y_block, _ = quantize(y_block)
    print(f'量子化済みDCT係数(Y成分, 品質{quality}):')
    print(y_block.astype(np.int64).reshape(8, 8))
    print(f'量子化テーブル(輝度, 品質{quality}):')
    print(quantize.y_table.astype(np.int64).reshape(8, 8))
