# 標準モジュール
import argparse
from datetime import datetime
from logging import (
    getLogger, basicConfig,
    DEBUG, INFO, WARNING
)
from pathlib import Path
import random

# 追加モジュール
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.utils as vutils
from tqdm import tqdm

# 自作モジュール
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

parser.add_argument(
    '-n', '--num-images', help='サンプル画像を生成する枚数を指定します。',
    type=int, default=10, metavar='B'
)

# PyTorchに関するコマンドライン引数
parser.add_argument(
    '--seed', help='乱数生成器のシード値を指定します。',
    type=int, default=None
)
parser.add_argument(
    '--disable-cuda', '--cpu', help='CUDAを無効化しGPU上では計算を行わず全てCPU上で計算します。',
    action='store_true'
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

# =========================================================================== #
# 再現性の設定 https://pytorch.org/docs/stable/notes/randomness.html
# =========================================================================== #
if args.seed is not None:
    random.seed(args.seed)                     # Pythonの乱数生成器のシード値の設定
    np.random.seed(args.seed)                  # NumPyの乱数生成器のシード値の設定
    torch.manual_seed(args.seed)               # PyTorchの乱数生成器のシード値の設定
    torch.backends.cudnn.deterministic = True  # Pytorchの決定性モードの有効化
    torch.backends.cudnn.benchmark = False     # Pytorchのベンチマークモードの無効化
    logger.info('乱数生成器のシード値を設定しました。')

# デバイスについての補助クラスをインスタンス化
auto_device = AutoDevice(disable_cuda=args.disable_cuda)
logger.info('デバイスの優先順位を計算しました。')
device = auto_device()
logger.info(f'メインデバイスとして〈{device}〉が選択されました。')

# =========================================================================== #
# JPEGエンコーダ／デコーダの定義
# =========================================================================== #
if g_jpeg:
    from models.modules.jpeg import JPEGDecoder
    jpeg_decoder = JPEGDecoder()
    jpeg_decoder = jpeg_decoder.to(device)

# =========================================================================== #
# モデルのインポート
# =========================================================================== #
if h == 32:
    if g_jpeg:
        from models.generators.jpeg32 import Generator
    else:
        from models.generators.model32 import Generator
elif h == 96:
    if g_jpeg:
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

# =========================================================================== #
# 画像の生成
# NOTE: クラス数×クラス数の画像を生成しグリッドで保存するので
#       クラス数が極端に多い場合は注意が必要
# =========================================================================== #
features_list = []
logits_list = []
with torch.no_grad():
    idx = 1
    output_path = generator_dir.joinpath(f'sample{idx}.png')
    for cnt in range(args.num_images):
        while output_path.exists():
            idx += 1
            output_path = generator_dir.joinpath(f'sample{idx}.png')
        overview = []
        pbar = tqdm(
            range(num_classes),
            desc=f'[{cnt + 1}/{args.num_images}] {output_path}を生成中...',
            total=num_classes,
            leave=False)
        for i in range(num_classes):
            classes = torch.ones(
                num_classes, dtype=torch.long, device=device) * i
            z = torch.randn(num_classes, nz, device=device)
            images = g_model(z, classes)
            if g_jpeg:
                images = jpeg_decoder(images)
            images = images.cpu()
            overview.append(images)
        vutils.save_image(
            torch.cat(overview, dim=0),
            output_path,
            nrow=num_classes,
            range=(0.0, 1.0)
        )
        pbar.close()
        print(f'[{cnt + 1}/{args.num_images}] {output_path}を生成完了')
