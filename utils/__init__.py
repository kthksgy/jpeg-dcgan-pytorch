import numpy as np

DATASET_NAMES = ['mnist', 'fashion_mnist', 'cifar10', 'stl10', 'imagenet2012']
DATASETS_ROOT = '~/.datasets/vision'


def default(value, default_value):
    try:
        return value if value else default_value
    except Exception:
        return default_value


def to_uint8(a: np.ndarray):
    return a.clip(0, 255).astype(np.uint8)
