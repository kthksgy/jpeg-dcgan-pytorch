import numpy as np

from .transforms.jpeg import (
    PILImageToYCbCr,
    OpenCVImageToYCbCr,
    ToYCbCr,
    ToRGB,
    ChromaSubsampling,
    BlockwiseDCT,
    JPEGQuantize,
)


class JPEGCompression:
    def __init__(
        self, *,
        quality: int = 100,
        chroma_subsampling_ratio: str = '4:2:0',
        image_type: str = None
    ):
        '''
        Args:
            quality: 画質[1, ..., 99] (100で量子化を無効化)
            chroma_subsampling_ratio: 色差間引きの比率
            image_type: ['PIL', 'OpenCV', None]
        '''
        self.quality = quality
        if image_type == 'PIL':
            self.to_ycbcr = PILImageToYCbCr()
        elif image_type == 'OpenCV':
            self.to_ycbcr = OpenCVImageToYCbCr()
        else:
            self.to_ycbcr = ToYCbCr()
        self.to_rgb = ToRGB()
        self.chroma_subsampling = ChromaSubsampling(
            ratio=chroma_subsampling_ratio)
        self.bwdct = BlockwiseDCT()
        if self.quality < 100:
            self.jpeg_quantize = JPEGQuantize(quality=self.quality)

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def encode(self, image):
        image = self.to_ycbcr(image)
        y, cbcr = self.chroma_subsampling(image)
        y_blocks = self.bwdct(y)
        if cbcr is not None:
            cbcr_blocks = np.concatenate(
                [self.bwdct(cbcr[:, :, 0]), self.bwdct(cbcr[:, :, 1])],
                axis=-1)
        else:
            cbcr_blocks = None
        if self.quality < 100:
            y_blocks, cbcr_blocks = self.jpeg_quantize(
                y_blocks, cbcr_blocks, rounding=True)
        return y_blocks, cbcr_blocks

    def decode(self, y_blocks, cbcr_blocks=None):
        if self.quality < 100:
            y_blocks, cbcr_blocks = self.jpeg_quantize.inverse(
                y_blocks, cbcr_blocks)
        y = self.bwdct.inverse(y_blocks)
        if cbcr_blocks is not None:
            cbcr = np.concatenate([
                self.bwdct.inverse(cbcr_blocks[:, :, :64]),
                self.bwdct.inverse(cbcr_blocks[:, :, 64:]),
            ], axis=-1)
            image = self.chroma_subsampling.inverse(y, cbcr)
            image = self.to_rgb(image)
        else:
            image = y
        return image


def zigzag(a: np.ndarray, block_size=8, inverse=False) -> np.ndarray:
    x = 0
    order = np.zeros(a.shape, dtype=np.int32)
    k = 1
    while x < block_size:
        x += 1
        order[k] = x
        k += 1
        for _ in range(x):
            order[k] = order[k-1] + 7
            k += 1
        if x == block_size - 1:
            break
        order[k] = order[k-1] + 8
        k += 1
        for _ in range(x):
            order[k] = order[k-1] - 7
            k += 1
        x += 1
        order[k] = x
        k += 1
    order[block_size**2//2+block_size//2:] = \
        63 - order[block_size**2//2-block_size//2-1::-1]
    ret = np.zeros(a.shape, dtype=a.dtype)
    if not inverse:
        for i in range(ret.shape[0]):
            ret[i] = a[order[i]]
    else:
        for i in range(ret.shape[0]):
            ret[i] = a[np.argwhere(order == i)[0]]
    return ret


def marker_sof(f):
    length = int.from_bytes(f.read(2), 'big')
    frame_header = {}
    frame_header['sample_precision'] = int.from_bytes(f.read(1), 'big')
    frame_header['height'] = int.from_bytes(f.read(2), 'big')
    frame_header['width'] = int.from_bytes(f.read(2), 'big')
    frame_header['num_channels'] = int.from_bytes(f.read(1), 'big')
    k = length - 8
    while k > 0:
        n = int.from_bytes(f.read(1), 'big')
        frame_header[f'channel{n}'] = {}
        tmp = int.from_bytes(f.read(1), 'big')
        hn = tmp >> 4
        vn = tmp - (hn << 4)
        frame_header[f'channel{n}']['horizontal_ratio'] = hn
        frame_header[f'channel{n}']['vertical_ratio'] = vn
        frame_header[f'channel{n}']['target_quantization_table'] = \
            int.from_bytes(f.read(1), 'big')
        k -= 3
    return frame_header


def marker_sof0(f):
    frame_header = marker_sof(f)
    frame_header['method'] = 'baseline'
    return frame_header


def marker_sof2(f):
    frame_header = marker_sof(f)
    frame_header['method'] = 'progressive'
    return frame_header


def marker_dqt(f):
    length = int.from_bytes(f.read(2), 'big')
    k = length - 2
    ret = {}
    while k > 0:
        tmp = int.from_bytes(f.read(1), 'big')
        pqn = tmp >> 4
        tqn = tmp - (pqn << 4)
        k -= 65 if pqn == 0 else 129
        ret[f'quantization_table{tqn}'] = np.array([
                int.from_bytes(f.read(1 if pqn == 0 else 2), 'big')
                for _ in range(64)],
                dtype=np.uint8 if pqn == 0 else np.uint16)
    return ret


MARKERS = {
    b'\xff\xc0': marker_sof0,
    b'\xff\xc2': marker_sof2,
    b'\xff\xdb': marker_dqt,
}


def inspect(path: str):
    f = open(str(path), 'rb')
    # SOI(Start of Image)
    assert f.read(2) == b'\xff\xd8'
    info = {}
    telled = 0
    while telled < f.tell():
        telled = f.tell()
        if f.read(1) != b'\xff':
            continue
        marker = b'\xff' + f.read(1)
        info.update(MARKERS.get(marker, lambda _: {})(f))
    return info


if __name__ == '__main__':
    # DCTの出力
    zzo_2d = np.array([
        [0, 1, 5, 6, 14, 15, 27, 28],
        [2, 4, 7, 13, 16, 26, 29, 42],
        [3, 8, 12, 17, 25, 30, 41, 43],
        [9, 11, 18, 24, 31, 40, 44, 53],
        [10, 19, 23, 32, 39, 45, 52, 54],
        [20, 22, 33, 38, 46, 51, 55, 60],
        [21, 34, 37, 47, 50, 56, 59, 61],
        [35, 36, 48, 49, 57, 58, 62, 63]
    ])
    print(zzo_2d)
    # JPEGファイルのDQT等の形式
    zzo_1d = np.arange(64)
    print(zzo_1d.reshape(8, 8))

    # 配列をジグザグスキャンする
    assert (zigzag(zzo_2d.ravel()) == zzo_1d).all()
    # ジグザグスキャン済みの配列を元に戻す
    assert (zzo_2d.ravel() == zigzag(zzo_1d, inverse=True)).all()
