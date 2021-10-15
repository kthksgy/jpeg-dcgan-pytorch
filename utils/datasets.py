from typing import Callable, List, Literal, Optional, Union
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

DATASET_NAMES = ['mnist', 'fashion_mnist', 'cifar10', 'stl10']
DATASETS_ROOT = '~/.datasets/vision'


def load_dataset(
    name: Literal['mnist', 'fashion_mnist', 'cifar10', 'stl10'],
    root: str, train: bool = True,
    transform: Optional[Union[List[Callable], Callable]] = None,
    download: bool = True
) -> torch.utils.data.Dataset:
    '''データセットを読み込みます。

    Args:
        name: データセット名
        root: データセットのルートディレクトリのパス
        train: 訓練スプリットを読み込む
        transform: 画像の前処理
        download: データセットが存在しない場合にダウンロードする

    Returns:
        指定した前処理のデータセットの指定したスプリット
    '''
    if isinstance(transform, (list, tuple)):
        transform = transforms.Compose(transform)
    if name == 'mnist':
        # THE MNIST DATABASE of handwritten digits
        # http://yann.lecun.com/exdb/mnist/
        dataset = dset.MNIST(
            root=root, download=download, train=train,
            transform=transform)
    elif name == 'fashion_mnist':
        # zalandoresearch / fashion-mnist
        # https://github.com/zalandoresearch/fashion-mnist
        dataset = dset.FashionMNIST(
            root=root, download=download, train=train,
            transform=transform)
    elif name == 'cifar10':
        # The CIFAR-10 dataset
        # https://www.cs.toronto.edu/~kriz/cifar.html
        dataset = dset.CIFAR10(
            root=root, download=download, train=train,
            transform=transform)
    elif name == 'stl10':
        # STL-10 dataset
        # https://ai.stanford.edu/~acoates/stl10/
        dataset = dset.STL10(
            root=root, split='train',
            transform=transform, download=download)
    else:
        raise Exception(f'指定したデータセット〈{name}〉は未実装です。')
    return dataset


def get_classes(
    name: Literal['mnist', 'fashion_mnist', 'cifar10', 'stl10']
) -> List[str]:
    '''データセットの日本語のクラス名を取得します。

    Args:
        name: データセット名

    Returns:
        クラス名のリスト

    Note:
        パスに使えない文字は使わないでください。
    '''
    if name == 'mnist':
        # THE MNIST DATABASE of handwritten digits
        # http://yann.lecun.com/exdb/mnist/
        classes = [
            'Zero', 'One', 'Two', 'Three', 'Four',
            'Five', 'Six', 'Seven', 'Eight', 'Nine'
        ]
    elif name == 'fashion_mnist':
        # zalandoresearch / fashion-mnist
        # https://github.com/zalandoresearch/fashion-mnist
        classes = [
            'トップス', 'トラウザー', 'プルオーバー', 'ドレス', 'コート',
            'サンダル', 'シャツ', 'スニーカー', 'バッグ', 'アンクルブーツ'
        ]
    elif name == 'cifar10' or name == 'stl10':
        # The CIFAR-10 dataset
        # https://www.cs.toronto.edu/~kriz/cifar.html
        classes = [
            '飛行機', '鳥', '自動車', '猫', '鹿',
            '犬', '馬', '猿', '船', 'トラック'
        ]
    else:
        raise Exception(f'指定したデータセット〈{name}〉は未実装です。')
    return classes
