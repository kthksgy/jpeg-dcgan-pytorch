import PIL.Image
from typing import Optional

import numpy as np

import torch
import torchvision.transforms as transforms


class Normalize(transforms.Normalize):
    '''逆変換を追加実装したバージョンのtransforms.Normalize

    nチャンネルの平均(mean[1],...,mean[n])と標準偏差(std[1],..,std[n])が与えられた時、
    output[c] = (input[c] - mean[c]) / std[c]を計算します。
    '''
    def inverse(self, tensor: torch.Tensor) -> Optional[torch.Tensor]:
        '''正規化された状態を元に戻します。
        inplaceの場合は入力テンソルを上書きします。

        Args:
            tensor: (C, H, W)形式の画像テンソル
        Returns:
            正規化前の画像テンソル
        Note:
            正規化はoutput[c] = (input[c] - mean[c]) / std[c]を計算します。
            逆正規化はoutput[c] = (input[c] * std[c]) + mean[c]を計算します。
        '''
        if self.inplace:
            for t, m, s in zip(tensor, self.mean, self.std):
                t.mul_(s).add_(m)
        else:
            ret = tensor.clone()
            for i, t, m, s in enumerate(zip(tensor, self.mean, self.std)):
                ret[i] = t.mul(s).add(m)
            return ret

    def batch(self, tensor: torch.Tensor) -> Optional[torch.Tensor]:
        '''.()のバッチ版です。
        inplaceの場合は入力テンソルを上書きします。

        Args:
            tensor: (B, C, H, W)形式の画像テンソルバッチ
        Returns:
            正規化後の画像テンソルバッチ
        '''
        if self.inplace:
            for i in range(tensor.size(1)):
                tensor[:, i].sub_(self.mean[i]).div_(self.std[i])
        else:
            ret = tensor.clone()
            for i in range(tensor.size(1)):
                ret[:, i] = tensor[:, i].sub(self.mean[i]).div(self.std[i])
            return ret

    def batch_inverse(self, tensor: torch.Tensor) -> Optional[torch.Tensor]:
        '''.inverse()のバッチ版です。
        inplaceの場合は入力テンソルを上書きします。

        Args:
            tensor: (B, C, H, W)形式の画像テンソルバッチ
        Returns:
            正規化前の画像テンソルバッチ
        '''
        if self.inplace:
            for i in range(tensor.size(1)):
                tensor[:, i].mul_(self.std[i]).add_(self.mean[i])
        else:
            ret = tensor.clone()
            for i in range(tensor.size(1)):
                ret[:, i] = tensor[:, i].mul(self.std[i]).add(self.mean[i])
            return ret


class ToNumPyArray:
    '''PIL画像をNumPy配列に変換するクラス

    Note:
        PIL画像はRGB形式、OpenCV画像はBGR形式である。
        MatplotlibはRGB形式で表示する。
    '''
    def __call__(self, image: PIL.Image.Image) -> np.ndarray:
        '''PIL画像をNumPy配列に変換する。
        Args:
            image: PIL画像
        Returns:
            NumPy配列
        '''
        return np.asarray(image)

    def inverse(self, image: np.ndarray) -> PIL.Image.Image:
        '''NumPy配列をPIL画像に変換する。
        Args:
            image: NumPy配列
        Returns:
            PIL画像
        '''
        return PIL.Image.fromarray(image[:, :, ::-1])
