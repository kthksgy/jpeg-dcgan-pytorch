# 標準モジュール
from pathlib import Path

# 追加モジュール
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm


def do(
    root='~/.datasets/vision/'
):
    pad = transforms.Compose([
        transforms.Pad(2, fill=0, padding_mode='constant'),
        transforms.ToTensor()
    ])
    grayscale = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    datasets = [
        ('mnist', dset.MNIST(
            root=root, train=True, download=True,
            transform=pad)),
        ('fashion_mnist', dset.FashionMNIST(
            root=root, train=True, download=True,
            transform=pad)),
        ('cifar10', dset.CIFAR10(
            root=root, train=True, download=True,
            transform=transforms.ToTensor())),
        ('stl10', dset.STL10(
            root=root, split='train', download=True,
            transform=transforms.ToTensor())),
        ('cifar10_grayscale', dset.CIFAR10(
            root=root, train=True, download=True,
            transform=grayscale)),
        ('stl10_grayscale', dset.STL10(
            root=root, split='train', download=True,
            transform=grayscale))
    ]
    for name, dataset in datasets:
        num_classes = len(dataset.classes)
        num_samples = num_classes
        images = [[] for _ in range(num_classes)]
        pbar = tqdm(
            dataset,
            desc='データセットの画像を抽出中... ',
            total=len(dataset),
            leave=False)
        for (image, label) in pbar:
            if len(images[label]) < num_samples:
                images[label].append(image)
            filled = 0
            for arr in images:
                if len(arr) >= num_samples:
                    filled += 1
            if filled >= num_classes:
                break
        pbar.close()
        output_dir = Path('outputs')
        output_dir.mkdir(parents=True, exist_ok=True)
        images = torch.cat([torch.stack(arr, dim=0) for arr in images], dim=0)
        # 画像の保存
        vutils.save_image(
            images,
            output_dir.joinpath(f'{name}.png'),
            nrow=num_samples)
