# 標準モジュール
import argparse
from datetime import datetime
from logging import (
    getLogger, basicConfig,
    DEBUG, INFO, WARNING
)
from math import ceil
from pathlib import Path
import random
from time import perf_counter
from typing import Tuple

# 追加モジュール
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torchvision.models import inception_v3
import torchvision.transforms as transforms
from tqdm import tqdm

# 自作モジュール
from utils import DATASETS_ROOT
from utils.datasets import load_dataset
from utils.evaluation import imagenet2012_normalize, inception_score, fid
from utils.device import AutoDevice


# =========================================================================== #
# コマンドライン引数の設定
# =========================================================================== #
parser = argparse.ArgumentParser(
    prog='PyTorch Generative Adversarial Network',
    description='PyTorchを用いてGANの画像生成を行います。'
)

# 訓練に関する引数
parser.add_argument(
    'load_generator', help='指定したパスのGeneratorのセーブファイルを読み込みます。',
    type=str, default=None
)

parser.add_argument(
    '-b', '--batch-size', help='バッチサイズを指定します。',
    type=int, default=64, metavar='B'
)

# PyTorchに関するコマンドライン引数
parser.add_argument(
    '--seed', help='乱数生成器のシード値を指定します。',
    type=int, default=42
)
parser.add_argument(
    '--disable-cuda', '--cpu', help='CUDAを無効化しGPU上では計算を行わず全てCPU上で計算します。',
    action='store_true'
)
parser.add_argument(
    '--data-path', help='データセットのパスを指定します。',
    type=str, default=DATASETS_ROOT
)
parser.add_argument(
    '--info', help='ログ表示レベルをINFOに設定し、詳細なログを表示します。',
    action='store_true'
)
parser.add_argument(
    '--debug', help='ログ表示レベルをDEBUGに設定し、より詳細なログを表示します。',
    action='store_true'
)
# コマンドライン引数をパースする
args = parser.parse_args()

# 結果を出力するために起動日時を保持する
launch_datetime = datetime.now()

# ロギングの設定
basicConfig(
    format='%(asctime)s %(name)s %(funcName)s %(levelname)s: %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=DEBUG if args.debug else INFO if args.info else WARNING,
)
# 名前を指定してロガーを取得する
logger = getLogger('main')

generator_path = Path(args.load_generator)
generator_dir = generator_path.parent

# チェックポイントの読み込み
checkpoint = torch.load(generator_path)
batch_size = args.batch_size
nz = checkpoint['nz']
nc = checkpoint['nc']
num_classes = checkpoint['num_classes']
g_jpeg = checkpoint['jpeg']
state_dict = checkpoint['model_state_dict']
h, w = checkpoint['h'], checkpoint['w']

idx = 1
output_path = generator_dir.joinpath(f'evaluation{idx}.txt')
while output_path.exists():
    idx += 1
    output_path = generator_dir.joinpath(f'evaluation{idx}.txt')
output_txt = open(output_path, mode='w', encoding='utf-8')

# =========================================================================== #
# 再現性の設定 https://pytorch.org/docs/stable/notes/randomness.html
# =========================================================================== #
random.seed(args.seed)                     # Pythonの乱数生成器のシード値の設定
np.random.seed(args.seed)                  # NumPyの乱数生成器のシード値の設定
torch.manual_seed(args.seed)               # PyTorchの乱数生成器のシード値の設定
torch.backends.cudnn.deterministic = True  # Pytorchの決定性モードの有効化
torch.backends.cudnn.benchmark = False     # Pytorchのベンチマークモードの無効化
logger.info('乱数生成器のシード値を設定しました。')
output_txt.write(f'乱数生成器のシード値: {args.seed}\n')

# デバイスについての補助クラスをインスタンス化
auto_device = AutoDevice(disable_cuda=args.disable_cuda)
logger.info('デバイスの優先順位を計算しました。')
device = auto_device()
logger.info(f'メインデバイスとして〈{device}〉が選択されました。')
if device != 'cpu':
    prop = torch.cuda.get_device_properties(device)
    output_txt.write(
        f'使用デバイス: {prop.name}\n'
        f'  メモリ容量: {prop.total_memory // 1048576}[MiB]\n'
        f'  Compute Capability: {prop.major}.{prop.minor}\n'
        f'  ストリーミングマルチプロセッサ数: {prop.multi_processor_count}[個]\n'
    )
else:
    output_txt.write('使用デバイス: CPU\n')

# =========================================================================== #
# JPEGエンコーダ／デコーダの定義
# =========================================================================== #
if checkpoint['jpeg']:
    from models.modules.jpeg import JPEGDecoder
    jpeg_decoder = JPEGDecoder()
    jpeg_decoder = jpeg_decoder.to(device)

# =========================================================================== #
# モデルのインポート
# =========================================================================== #
if h == 32:
    if checkpoint['jpeg']:
        from models.generators.jpeg32 import Generator
    else:
        from models.generators.model32 import Generator
elif h == 96:
    if checkpoint['jpeg']:
        from models.generators.jpeg96 import Generator
    else:
        from models.generators.model96 import Generator

# =========================================================================== #
# モデルの読み込み
# =========================================================================== #
g_model = Generator(
    nz=nz, nc=nc, num_classes=num_classes)
g_model = g_model.to(device)
g_model.eval()
g_model.load_state_dict(state_dict)

# ImageNet2012訓練済みInception V3を読み込む
inception = inception_v3(pretrained=True, aux_logits=False)
inception_children = list(inception.children())
# 特徴抽出器(FIDで使用)
feature_extractor = nn.Sequential(*inception_children[:-2]).to(device)
feature_extractor.eval()
# クラス分類器(Inception Scoreで使用)
classifier = nn.Sequential(
    inception_children[-1], nn.Softmax(dim=1)).to(device)
classifier.eval()

# Inception V3の入力(299×299)のための画像拡大モジュール
upsample = nn.UpsamplingBilinear2d((299, 299))


def preprocess(images: torch.Tensor) -> torch.Tensor:
    '''Inception V3用の画像の前処理を行います。

    Args:
        images: 画素値[0.0 ~ 1.0]の画像バッチ(B, C, H, W)

    Returns:
        前処理済みの画像バッチ(B, C, 299, 299)
    '''
    if images.size(1) == 1:
        images = images.repeat(1, 3, 1, 1)
    images = imagenet2012_normalize(images, inplace=True)
    images = upsample(images)
    return images


def inception_forward(images: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    '''FID／Inception Scoreの算出に必要な計算結果を取得する。

    Args:
        前処理済みの画像バッチ(B, C, 299, 299)

    Returns:
        (最終層の特徴量(B, 2048)，各クラスに属する確率(B, 1000))
    '''
    features = feature_extractor(images).view(-1, 2048)
    logits = classifier(features)
    return features.cpu().numpy(), logits.cpu().numpy()


# =========================================================================== #
# キャッシュの読み込み／訓練画像の評価
# =========================================================================== #
if nc == 1 and checkpoint['dataset'] in ['cifar10', 'stl10']:
    suffix = '_grayscale'
else:
    suffix = ''
assets_root = Path('./assets')
assets_dir = assets_root.joinpath(checkpoint['dataset'] + suffix)
features_path = assets_dir.joinpath('features.npz')
logits_path = assets_dir.joinpath('logits.npz')
labels_path = assets_dir.joinpath('labels.npz')
if (
    assets_dir.exists() and assets_dir.is_dir() and
    features_path.exists() and features_path.is_file() and
    logits_path.exists() and logits_path.is_file() and
    labels_path.exists() and labels_path.is_file()
):
    # 既に計算された結果がある場合は読み込み
    train_features = np.load(features_path)['features']
    train_logits = np.load(logits_path)['logits']
    train_labels = np.load(labels_path)['labels']
else:  # 無い場合は計算してその結果を保存
    # ======================================================================= #
    # データ整形
    # ======================================================================= #
    logger.info('画像に適用する変換のリストを定義します。')
    data_transforms = []

    if checkpoint['dataset'] in ['mnist', 'fashion_mnist']:
        # MNIST/Fashion MNISTは28×28画素なのでゼロパディング
        data_transforms.append(
            transforms.Pad(2, fill=0, padding_mode='constant')
        )
        logger.info('変換リストにゼロパディングを追加しました。')

    if nc == 1 and checkpoint['dataset'] in ['cifar10', 'stl10']:
        data_transforms.append(
            transforms.Grayscale(num_output_channels=1)
        )
        logger.info('変換リストにグレースケール化を追加しました。')

    data_transforms.append(transforms.ToTensor())
    logger.info('変換リストにテンソル化を追加しました。')
    # ======================================================================= #
    # データセットの読み込み／データローダーの定義
    # ======================================================================= #
    dataset = load_dataset(
        checkpoint['dataset'], root=args.data_path, transform=data_transforms)
    logger.info(f"データセット〈{checkpoint['dataset']}〉を読み込みました。")

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        shuffle=False, drop_last=False)
    logger.info('データローダを生成しました。')

    features_list = []
    logits_list = []
    labels_list = []
    pbar = tqdm(
        enumerate(dataloader),
        desc='データセット画像から特徴を抽出中...',
        total=ceil(len(dataset) / batch_size),
        leave=False)
    for i, (images, labels) in pbar:
        with torch.no_grad():
            images = preprocess(images)
            images = images.to(device)
            features, logits = inception_forward(images)
            features_list.append(features)
            logits_list.append(logits)
            labels_list.append(labels)
    train_features = np.concatenate(features_list, axis=0)
    train_logits = np.concatenate(logits_list, axis=0)
    train_labels = np.concatenate(labels_list, axis=0)
    assets_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        assets_dir.joinpath('features.npz'), features=train_features)
    np.savez_compressed(
        assets_dir.joinpath('logits.npz'), logits=train_logits)
    np.savez_compressed(
        assets_dir.joinpath('labels.npz'), labels=train_labels)
    with open(
        assets_dir.joinpath('evaluation.txt'), mode='w', encoding='utf-8'
    ) as f:
        f.write(f'画像数: {train_labels.shape[0]}\n')
        f.write(f'Inception Score: {inception_score(train_logits)}\n')
    print(f'{assets_dir}にデータセット画像の評価を保存しました。')

# =========================================================================== #
# 画像生成時間の計測
# =========================================================================== #
pbar = tqdm(range(10000), desc='画像生成時間を計測中...', total=10000, leave=False)
with torch.no_grad():
    begin_time = perf_counter()
    for _ in pbar:
        z = torch.randn(1, nz, device=device)
        classes = torch.randint_like(
            z, high=num_classes, dtype=torch.long, device=device)
        fakes = g_model(z, classes)
    end_time = perf_counter()
s = f'画像生成時間: {(end_time - begin_time) / 10000:.07f}[s/image]'
print(s)
output_txt.write(f'{s}\n')

# =========================================================================== #
# 生成画像の評価
# =========================================================================== #
features_list = []
logits_list = []
with torch.no_grad():
    pbar = tqdm(
        [
            train_labels[
                i * batch_size:
                min((i + 1) * batch_size, train_labels.shape[0])
            ]
            for i in range(ceil(train_labels.shape[0] / batch_size))
        ],
        desc='生成画像から特徴を抽出中...',
        total=ceil(train_labels.shape[0] / batch_size),
        leave=False)
    for labels in pbar:
        labels = torch.from_numpy(labels).to(device)
        z = torch.randn(labels.size(0), nz, device=device)
        images = g_model(z, labels)
        if checkpoint['jpeg']:
            images = jpeg_decoder(images)
        images = preprocess(images)
        features, logits = inception_forward(images)
        features_list.append(features)
        logits_list.append(logits)
features = np.concatenate(features_list, axis=0)
logits = np.concatenate(logits_list, axis=0)
s = f'Inception Score: {inception_score(logits)}'
print(s)
output_txt.write(f'{s}\n')
s = f'FID: {fid(features, train_features)}'
print(s)
output_txt.write(f'{s}\n')
output_txt.close()
