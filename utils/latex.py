from typing import Union

import numpy as np
import torch


def to_matrix(a: Union[np.ndarray, torch.Tensor], elem_len: int = 3) -> str:
    if isinstance(a, torch.Tensor):
        a = a.numpy()
    a = a.squeeze()
    if a.ndim > 2:
        raise ValueError('三次元以上の配列には未対応です。')
    elif np.prod(a.shape) == 0:
        raise ValueError('空の配列には未対応です。')
    elif a.ndim == 1:
        a = np.expand_dims(a, 0)

    ret = ''
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            ret += f'{a[i, j]:{elem_len}} & '
        ret = ret[:-2] + '\\\\\n'
    ret = ret[:-4] + '\n'
    return ret
