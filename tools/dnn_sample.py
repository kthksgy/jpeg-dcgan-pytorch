import numpy as np
import torch
import torch.nn as nn

from utils.latex import to_matrix


def do():
    with torch.no_grad():
        a = np.arange(1, 26).reshape(1, 1, 5, 5).astype(np.float32)
        a = torch.from_numpy(a)
        print('畳み込みの入力:')
        print(to_matrix(a.numpy().astype(np.int64)))

        conv = nn.Conv2d(1, 1, 3, stride=2, padding=0, bias=False)
        conv.weight.requires_grad = False
        for i in range(3):
            for j in range(3):
                conv.weight[0, 0, i, j] = i * 3 + j + 1
        print('畳み込みのカーネル:')
        print(to_matrix(conv.weight.numpy().astype(np.int64), elem_len=2))

        b = conv(a)
        print('畳み込みの出力:')
        print(to_matrix(b.numpy().astype(np.int64), elem_len=3))

        ta = np.arange(1, 5).reshape(1, 1, 2, 2).astype(np.float32)
        ta = torch.from_numpy(ta)
        print('転置畳み込みの入力:')
        print(to_matrix(ta.numpy().astype(np.int64), elem_len=2))

        tconv = nn.ConvTranspose2d(1, 1, 3, stride=2, padding=0, bias=False)
        tconv.weight.requires_grad = False
        for i in range(3):
            for j in range(3):
                tconv.weight[0, 0, i, j] = i * 3 + j + 1
        print('転置畳み込みのカーネル:')
        print(to_matrix(conv.weight.numpy().astype(np.int64), elem_len=2))

        tb = tconv(ta)
        print('転置畳み込みの出力:')
        print(to_matrix(tb.numpy().astype(np.int64), elem_len=3))
