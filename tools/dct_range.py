# 標準モジュール
from pathlib import Path

# 追加モジュール
import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm

from utils.jpeg import JPEGCompression


def do(root='~/.datasets/vision/'):
    pad = transforms.Compose([
        transforms.Pad(2, fill=0, padding_mode='constant'),
        # transforms.ToTensor()
    ])
    grayscale = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        # transforms.ToTensor()
    ])
    datasets = [
        ('mnist', dset.MNIST(
            root=root, train=True, download=True,
            transform=pad)),
        ('fashion_mnist', dset.FashionMNIST(
            root=root, train=True, download=True,
            transform=pad)),
        # ('cifar10', dset.CIFAR10(
        #     root=root, train=True, download=True,
        #     transform=transforms.ToTensor())),
        # ('stl10', dset.STL10(
        #     root=root, split='train', download=True,
        #     transform=transforms.ToTensor())),
        ('cifar10_grayscale', dset.CIFAR10(
            root=root, train=True, download=True,
            transform=grayscale)),
        # ('stl10_grayscale', dset.STL10(
        #     root=root, split='train', download=True,
        #     transform=grayscale))
    ]
    jpeg = JPEGCompression(image_type='PIL')
    np.set_printoptions(precision=3, suppress=True)
    for name, dataset in datasets:
        pbar = tqdm(
            dataset,
            desc='データセットの画像を抽出中... ',
            total=len(dataset),
            leave=False)
        coefs = []
        for (image, label) in pbar:
            y, cbcr = jpeg.encode(image)
            coefs.append(y * 255)
        coefs = np.stack(coefs).transpose(3, 0, 1, 2).reshape(64, -1)
        max_coefs = np.max(coefs, axis=1).reshape(8, 8)
        min_coefs = np.min(coefs, axis=1).reshape(8, 8)
        average_coefs = np.average(coefs, axis=1).reshape(8, 8)
        median_coefs = np.median(coefs, axis=1).reshape(8, 8)
        std_coefs = np.std(coefs, axis=1).reshape(8, 8)
        print(name)
        print(max_coefs)
        print(min_coefs)
        print(average_coefs)
        print(median_coefs)
        print(std_coefs)
        pbar.close()
