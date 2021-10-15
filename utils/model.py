from typing import Tuple

import torch.nn as nn


def count_params(model: nn.Module) -> Tuple[int, int]:
    '''モデルのパラメータ数を数えます。

    Args:
        model: パラメータ数をカウントするモデル

    Returns:
        (学習可能パラメータ数, 固定パラメータ数)
    '''
    trainable = 0
    fixed = 0
    for params in model.parameters():
        tmp = params.numel()
        if params.requires_grad:
            trainable += tmp
        else:
            fixed += tmp
    return trainable, fixed
