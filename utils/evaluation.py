import numpy as np
from scipy.linalg import sqrtm
from torchvision.transforms.functional import normalize


def imagenet2012_normalize(images, inplace: bool = True):
    return normalize(
        images,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        inplace=inplace)


def inception_score(logits: np.ndarray) -> float:
    p_y = np.mean(logits, axis=0)  # p(y)
    e = logits * np.log(logits / p_y)  # p(y|x)log(P(y|x)/P(y))
    e = np.sum(e, axis=1)  # KL(x) = Σ_y p(y|x)log(P(y|x)/P(y))
    e = np.mean(e, axis=0)
    return np.exp(e)  # Inception score


def fid(feats1: np.ndarray, feats2: np.ndarray) -> float:
    assert feats1.shape[0] >= 2048, '最低でも2048個の特徴ベクトルが必要です。'
    assert feats2.shape[0] >= 2048, '最低でも2048個の特徴ベクトルが必要です。'
    mean1 = np.mean(feats1, axis=0)
    cov1 = np.cov(feats1, rowvar=False)
    mean2 = np.mean(feats2, axis=0)
    cov2 = np.cov(feats2, rowvar=False)
    cov_mean, _ = sqrtm(np.dot(cov1, cov2), disp=False)
    return np.sum((mean1 - mean2) ** 2)\
        + np.trace(cov1) + np.trace(cov2)\
        - 2 * np.trace(cov_mean)
