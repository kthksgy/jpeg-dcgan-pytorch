from typing import Optional, Tuple

import cv2
import numpy as np
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dctn.html#scipy.fft.dctn
from scipy.fft import dctn, idctn


# JPEGの色変換に関する資料
# https://www.w3.org/Graphics/JPEG/jfif3.pdf
# https://github.com/LuaDist/libjpeg/blob/master/jdcolor.c
def rgb_to_ycbcr(rgb):
    if rgb.ndim == 3 and rgb.shape[-1] == 3:
        return np.stack([
            # Y
            0.299 * rgb[:, :, 0]
            + 0.587 * rgb[:, :, 1]
            + 0.114 * rgb[:, :, 2],
            # Cb
            -0.1687 * rgb[:, :, 0]
            - 0.3313 * rgb[:, :, 1]
            + 0.5 * rgb[:, :, 2]
            + 0.5,
            # Cr
            0.5 * rgb[:, :, 0]
            - 0.4187 * rgb[:, :, 1]
            - 0.0813 * rgb[:, :, 2]
            + 0.5
        ], axis=-1)
    else:
        return rgb


def ycbcr_to_rgb(ycbcr):
    if ycbcr.ndim == 3 and ycbcr.shape[-1] == 3:
        return np.stack([
            # R
            ycbcr[:, :, 0]
            + 1.402 * (ycbcr[:, :, 2] - 0.5),
            # G
            ycbcr[:, :, 0]
            - 0.344136286 * (ycbcr[:, :, 1] - 0.5)
            - 0.714136286 * (ycbcr[:, :, 2] - 0.5),
            # B
            ycbcr[:, :, 0]
            + 1.772 * (ycbcr[:, :, 1] - 0.5)
        ], axis=-1)
    else:
        return ycbcr


class PILImageToYCbCr:
    def __call__(self, image):
        if image.mode == 'L':
            return np.asarray(image, dtype=np.float32) / 255.0
        else:
            return rgb_to_ycbcr(
                np.asarray(image.convert('RGB'), dtype=np.float32) / 255.0)


class OpenCVImageToYCbCr:
    def __call__(self, image):
        if image.ndim == 3 and image.shape[-1] == 3:
            return rgb_to_ycbcr(
                image[:, :, [2, 1, 0]].astype(np.float32) / 255.0)
        else:
            return image.astype(np.float32) / 255.0


class ToYCbCr:
    def __call__(self, image):
        return rgb_to_ycbcr(image)


class ToRGB:
    def __call__(self, image):
        return ycbcr_to_rgb(image)


class BlockwiseDCT:
    def __init__(
        self, *, block_size: Tuple[int] = (8, 8)
    ):
        self.block_size = block_size
        self.num_coefficients = np.prod(block_size)

    def __call__(self, image, inplace=False):
        num_vblocks = image.shape[0] // self.block_size[0]
        return dctn(
            self.__split(image, num_vblocks), type=2,
            axes=[1, 2],
            norm='ortho',
            overwrite_x=inplace,
            workers=-1
        ) \
            .reshape(
                num_vblocks,
                -1,
                self.num_coefficients)

    def inverse(self, blocks, inplace=True):
        h, w = (
            blocks.shape[0] * self.block_size[0],
            blocks.shape[1] * self.block_size[1]
        )
        return self.__concatenate(idctn(
            blocks.reshape(-1, self.block_size[0], self.block_size[1]), type=2,
            axes=[1, 2],
            norm='ortho',
            overwrite_x=inplace,
            workers=-1
        ), blocks.shape[0], h, w)

    # https://stackoverflow.com/questions/16873441/form-a-big-2d-array-from-multiple-smaller-2d-arrays/16873755#16873755
    def __split(self, image, num_vblocks):
        return image \
            .reshape(
                num_vblocks,
                self.block_size[0],
                -1,
                self.block_size[1]) \
            .swapaxes(1, 2) \
            .reshape(-1, self.block_size[0], self.block_size[1])

    def __concatenate(self, blocks, num_vblocks, h, w):
        return blocks \
            .reshape(
                num_vblocks,
                -1,
                self.block_size[0],
                self.block_size[1]) \
            .swapaxes(1, 2) \
            .reshape(h, w, 1)


class JPEGQuantize:
    def __init__(
        self, *,
        quality: int, source: str = 'jpeg_standard',
    ):
        self.quality = quality
        self.source = source
        luma_table, chroma_table = \
            self.get_table(quality=self.quality, source=self.source)
        self.y_table = luma_table.ravel()
        self.cbcr_table = np.concatenate((chroma_table.ravel(),) * 2)

    def __call__(
        self, y_blocks, cbcr_blocks=None, *, rounding=True
    ) -> np.ndarray:
        y_blocks = y_blocks / self.y_table
        if cbcr_blocks is not None:
            cbcr_blocks = cbcr_blocks / self.cbcr_table
        if rounding:
            y_blocks = np.round(y_blocks)
            cbcr_blocks = np.round(y_blocks)
        return y_blocks, cbcr_blocks

    def inverse(self, y_blocks, cbcr_blocks=None) -> np.ndarray:
        y_blocks = y_blocks * self.luma
        if cbcr_blocks is not None:
            cbcr_blocks = cbcr_blocks * self.cbcr_table
        return y_blocks, cbcr_blocks

    @classmethod
    def get_table(cls, *, quality: int = 50, source: str = 'jpeg_standard'):
        assert quality > 0, 'Qualityパラメータは1以上の整数で指定してください。'
        # https://www.impulseadventure.com/photo/jpeg-quantization.html
        if source == 'jpeg_standard':
            luma = np.asarray([
                [16, 11, 10, 16,  24,  40,  51,  61],
                [12, 12, 14, 19,  26,  58,  60,  55],
                [14, 13, 16, 24,  40,  57,  69,  56],
                [14, 17, 22, 29,  51,  87,  80,  62],
                [18, 22, 37, 56,  68, 109, 103,  77],
                [24, 35, 55, 64,  81, 104, 113,  92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103,  99]
            ])
            chroma = np.asarray([
                [17, 18, 24, 47, 99, 99, 99, 99],
                [18, 21, 26, 66, 99, 99, 99, 99],
                [24, 26, 56, 99, 99, 99, 99, 99],
                [47, 66, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99]
            ])
        else:
            raise KeyError(f'指定した量子化テーブルのキー〈{source}〉は未実装です。')
        luma, chroma = \
            [
                np.floor((100 - quality) / 50 * table + 0.5).clip(min=1)
                if quality >= 50 else
                np.floor(50 / quality * table + 0.5)
                for table in [luma, chroma]
            ]
        return luma, chroma


class ChromaSubsampling:
    # https://en.wikipedia.org/wiki/Chroma_subsampling
    def __init__(self, *, ratio: str):
        '''
        '''
        self.ratio = ratio
        sampling_factor = self.get_sampling_factor(ratio=ratio)
        self.fy = 1 / sampling_factor[0]
        if self.fy.is_integer():
            self.fy = int(self.fy)
        self.fx = 1 / sampling_factor[1]
        if self.fx.is_integer():
            self.fx = int(self.fx)

    def __call__(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''色差の解像度を減らします。

        Args:
            image: YCbCr画像
        Returns:
            (Y成分, CbCr成分)
        '''
        if image.ndim == 3 and image.shape[-1] == 3:
            if self.fy == 1 and self.fx == 1:
                return (image[:, :, 0:1], image[:, :, 1:])
            else:
                return (
                    image[:, :, 0:1],
                    cv2.resize(image[:, :, 1:], None, fx=self.fx, fy=self.fy)
                )
        else:
            return (image, None)

    def inverse(
        self, y: np.ndarray, cbcr: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        '''逆クロマサブサンプリングを実行します。

        Args:
            y: 形状(H, W, 1)のY成分
            cbcr: 形状(H, W, 2)のCbCr成分
        Returns:
            形状(H, W, 3)のYCbCr画像
        '''
        if self.fy == 1 and self.fx == 1:
            return np.concatenate([y, cbcr], axis=-1)
        else:
            return np.concatenate(
                [y, cv2.resize(cbcr, y.shape[:2])],
                axis=-1
            )

    @classmethod
    def get_sampling_factor(cls, *, ratio: str = '4:4:4'):
        if ratio == '4:4:4':
            sampling_factor = (1, 1)
        elif ratio == '4:2:2':
            sampling_factor = (1, 2)
        elif ratio == '4:2:0':
            sampling_factor = (2, 2)
        elif ratio == '4:4:0':
            sampling_factor = (2, 1)
        elif ratio == '4:1:1':
            sampling_factor = (1, 4)
        else:
            raise KeyError(f'指定した比率〈{ratio}〉は未実装です。')
        return sampling_factor


class LowPassFilter:
    def __init__(
        self, ratio: float, block_size: Tuple[int] = (8, 8),
    ):
        self.block_size = block_size
        self.new_block_size = (
            max(1, int(self.block_size[0] * ratio)),
            max(1, int(self.block_size[1] * ratio))
        )
        self.new_num_coefficients = np.prod(self.new_block_size)
        self.padding = (
            (0, 0), (0, 0),
            (0, self.block_size[0] - self.new_block_size[0]),
            (0, self.block_size[1] - self.new_block_size[1]),
        )

    def __call__(self, blocks):
        return blocks \
            .reshape(blocks.shape[:-1] + self.block_size)[
                :,
                :,
                :self.new_block_size[0],
                :self.new_block_size[1]] \
            .reshape(blocks.shape[:-1] + (-1,))

    def inverse(self, blocks):
        return np.pad(
            blocks.reshape(blocks.shape[:-1] + self.new_block_size),
            self.padding) \
            .reshape(blocks.shape[:-1] + (-1,))

    def get_num_features(self):
        return self.new_num_coefficients
