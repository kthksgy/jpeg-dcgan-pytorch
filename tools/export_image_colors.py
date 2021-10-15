from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np

from utils import to_uint8
from utils.transforms.jpeg import ToYCbCr, ToRGB


def do(filename='test32.png', scale=32):
    output_dir = Path(f'outputs/{filename}_colors')
    output_dir.mkdir(exist_ok=True)
    image = cv2.imread(f'assets/{filename}')
    image = cv2.resize(
        image, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(output_dir.joinpath('source.png')), image)
    to_ycbcr = ToYCbCr()
    to_rgb = ToRGB()
    rgb = image[:, :, [2, 1, 0]]

    r = np.zeros_like(image, dtype=np.uint8)
    r[:, :, 2] = rgb[:, :, 0]
    cv2.imwrite(str(output_dir.joinpath('r.png')), r)

    g = np.zeros_like(image, dtype=np.uint8)
    g[:, :, 1] = rgb[:, :, 1]
    cv2.imwrite(str(output_dir.joinpath('g.png')), g)

    b = np.zeros_like(image, dtype=np.uint8)
    b[:, :, 0] = rgb[:, :, 2]
    cv2.imwrite(str(output_dir.joinpath('b.png')), b)

    ycbcr = to_uint8(to_ycbcr(rgb))

    y = ycbcr[:, :, 0]
    y = np.stack([y, y, y], axis=-1)
    cv2.imwrite(str(output_dir.joinpath('y.png')), y)

    cb = deepcopy(ycbcr)
    cb[:, :, [0, 2]] = 0
    cb = to_uint8(to_rgb(cb))[:, :, [2, 1, 0]]
    cv2.imwrite(str(output_dir.joinpath('cb.png')), cb)

    cr = deepcopy(ycbcr)
    cr[:, :, [0, 1]] = 0
    cr = to_uint8(to_rgb(cr))[:, :, [2, 1, 0]]
    cv2.imwrite(str(output_dir.joinpath('cr.png')), cr)
