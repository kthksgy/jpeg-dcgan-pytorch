from copy import deepcopy
from PIL import Image
import unittest

import cv2
import numpy as np
import torch

from models.modules.jpeg import (
    JPEGEncoder, JPEGDecoder
)
from utils import to_uint8
from utils.jpeg import (
    JPEGCompression
)
from utils.transforms.jpeg import (
    OpenCVImageToYCbCr,
    PILImageToYCbCr,
    ToRGB,
    BlockwiseDCT,
)


class TestJPEGMethods(unittest.TestCase):
    def test_ycbcr(self):
        pil_image = Image.open('assets/test512.png').convert('RGB')
        ocv_image = cv2.imread('assets/test512.png')
        np.testing.assert_array_equal(
            np.asarray(pil_image),
            ocv_image[:, :, [2, 1, 0]]
        )

        to_ycbcr_pil = PILImageToYCbCr()
        to_ycbcr_ocv = OpenCVImageToYCbCr()
        to_rgb = ToRGB()

        ycbcr1 = to_ycbcr_pil(pil_image)
        ycbcr2 = to_ycbcr_ocv(ocv_image)

        rgb1 = to_rgb(ycbcr1)
        rgb2 = to_rgb(ycbcr2)

        np.testing.assert_array_equal(rgb1, rgb2)

        np.testing.assert_array_equal(
            np.asarray(pil_image), to_uint8(rgb1 * 255)
        )

        np.testing.assert_array_equal(
            ocv_image[:, :, [2, 1, 0]], to_uint8(rgb2 * 255)
        )

    def test_bwdct(self):
        bwdct = BlockwiseDCT(block_size=(8, 8))
        # (H, W)
        image = cv2.imread('assets/test512.png', 0)
        # (H, W, 1)
        image = image[:, :, np.newaxis]
        # (H // 8, W // 8, 64)
        blocks = bwdct(image)
        # (H, W, 1)
        recons = to_uint8(bwdct.inverse(blocks))
        self.assertTrue((image == recons).all())

    def test_encode_decode(self):
        # OpenCV画像テスト
        jpeg = JPEGCompression(
            quality=100,
            chroma_subsampling_ratio='4:4:4',
            image_type='OpenCV')
        # グレースケール画像
        image = cv2.imread('assets/test512.png', 0)
        y_blocks, cbcr_blocks = jpeg(image)
        self.assertIsNone(cbcr_blocks)
        decoded = jpeg.decode(y_blocks, cbcr_blocks)
        np.testing.assert_array_equal(
            image[:, :, np.newaxis],
            to_uint8(decoded * 255)
        )
        # 色コンポーネント個別
        image = cv2.imread('assets/test512.png')
        for c in image.transpose(2, 0, 1):
            y_blocks, cbcr_blocks = jpeg(c)
            self.assertIsNone(cbcr_blocks)
            decoded = jpeg.decode(y_blocks, cbcr_blocks)
            np.testing.assert_array_equal(
                c[:, :, np.newaxis],
                to_uint8(decoded * 255)
            )
        # カラー画像
        image = cv2.imread('assets/test512.png')
        y_blocks, cbcr_blocks = jpeg(image)
        decoded = jpeg.decode(y_blocks, cbcr_blocks)
        np.testing.assert_array_equal(
            image[:, :, [2, 1, 0]], to_uint8(decoded * 255)
        )

        # PIL画像テスト
        jpeg = JPEGCompression(
            quality=100,
            chroma_subsampling_ratio='4:4:4',
            image_type='PIL')
        image = Image.open('assets/test512.png')
        y_blocks, cbcr_blocks = jpeg(image)
        decoded = jpeg.decode(y_blocks, cbcr_blocks)
        np.testing.assert_array_equal(
            np.asarray(image.convert('RGB')), to_uint8(decoded * 255)
        )

        # 汎用的なRGB画像テスト
        jpeg = JPEGCompression(
            quality=100,
            chroma_subsampling_ratio='4:4:4',
            image_type=None)
        image = cv2.imread('assets/test512.png')[:, :, [2, 1, 0]]
        # RGB画像を入力
        y_blocks, cbcr_blocks = jpeg(image)
        decoded = jpeg.decode(y_blocks, cbcr_blocks)
        np.testing.assert_array_equal(
            image, to_uint8(decoded)
        )
        # 0.0 ~ 1.0の範囲に正規化されたRGB画像を入力
        y_blocks, cbcr_blocks = jpeg(image.astype(np.float32) / 255.0)
        decoded = jpeg.decode(y_blocks, cbcr_blocks)
        np.testing.assert_array_equal(
            image, to_uint8(decoded * 255)
        )

    def test_torch_modules(self):
        jpeg = JPEGCompression(
            quality=100,
            chroma_subsampling_ratio='4:4:4',
            image_type=None)
        encoder = JPEGEncoder(
            interpolation_mode='bilinear', chroma_subsampling=False)
        decoder = JPEGDecoder(
            interpolation_mode='bilinear')
        original_rgb = cv2.imread('assets/test512.png')[:, :, [2, 1, 0]]

        # 元画像のコピーを作成
        image = deepcopy(original_rgb).astype(np.float32) / 255.0
        # バッチテンソル化 (H, W, C) -> (1, C, H, W)
        image = torch.from_numpy(image[np.newaxis].transpose(0, 3, 1, 2))
        y_blocks, cbcr_blocks = encoder(image)
        # NumPy配列に戻す
        y_blocks = np.squeeze(y_blocks.numpy().transpose(0, 2, 3, 1))
        cbcr_blocks = np.squeeze(cbcr_blocks.numpy().transpose(0, 2, 3, 1))
        # 通常実装でデコード
        decoded = jpeg.decode(y_blocks, cbcr_blocks)
        # 同一判定
        np.testing.assert_array_equal(
            original_rgb, to_uint8(decoded * 255)
        )

        # 元画像のコピーを作成
        image = deepcopy(original_rgb).astype(np.float32) / 255.0
        # 通常実装でエンコード
        y_blocks, cbcr_blocks = jpeg(image)
        # バッチテンソル化 (H, W, C) -> (1, C, H, W)
        y_blocks = torch.from_numpy(
            y_blocks[np.newaxis].transpose(0, 3, 1, 2))
        cbcr_blocks = torch.from_numpy(
            cbcr_blocks[np.newaxis].transpose(0, 3, 1, 2))
        # PyTorch実装でデコード
        decoded = decoder([y_blocks, cbcr_blocks])
        # NumPy配列に戻す
        decoded = np.squeeze(decoded.numpy().transpose(0, 2, 3, 1))
        # 同一判定
        np.testing.assert_array_equal(
            original_rgb, to_uint8(decoded * 255)
        )
