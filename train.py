# 標準モジュール
import argparse
import csv
from datetime import datetime
import json
from logging import (
    getLogger, basicConfig,
    DEBUG, INFO, WARNING
)
from pathlib import Path
import random
import sys
from time import perf_counter

# 追加モジュール
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm

# 自作モジュール
from utils import default
from utils.datasets import (
    DATASET_NAMES, DATASETS_ROOT,
    load_dataset, get_classes)
from utils.device import AutoDevice
from utils.model import count_params
from utils.path import remove_invalid_char

# =========================================================================== #
# コマンドライン引数の設定
# =========================================================================== #
parser = argparse.ArgumentParser(
    prog='PyTorch Generative Adversarial Network',
    description='PyTorchを用いてGANの画像生成を行います。'
)

# データセット
parser.add_argument(
    'dataset', help='データセットを指定します。',
    type=str, choices=DATASET_NAMES
)
# 訓練パラメータ
parser.add_argument(
    '-b', '--batch-size', help='バッチサイズを指定します。',
    type=int, default=None, metavar='N'
)
parser.add_argument(
    '-e', '--num-epochs', help='学習エポック数を指定します。',
    type=int, default=None, metavar='N'
)
parser.add_argument(
    '-z', '--nz', help='潜在空間のサイズを指定します。',
    type=int, default=None, metavar='N'
)
parser.add_argument(
    '--grayscale', help='カラー画像の場合グレースケール化します。',
    action='store_true'
)
parser.add_argument(
    '--ms-lambda', help='Mode Seeking損失の係数λを指定します。',
    type=float, default=0.0
)
parser.add_argument(
    '--lrs-ss', help='学習率スケジューラのステップサイズを指定します。',
    type=int, default=None
)
parser.add_argument(
    '--lrs-gamma', help='学習率スケジューラの学習率減衰係数γを指定します。',
    type=float, default=None
)
parser.add_argument(
    '--g-jpeg', help='DCT係数を生成するJPEG Generatorを使用します。',
    action='store_true'
)
parser.add_argument(
    '--d-jpeg', help='DCT係数を生成するJPEG Discriminatorを使用します。',
    action='store_true'
)
parser.add_argument(
    '--d-weak', help='JPEG Generatorに釣り合う強さのDiscriminatorを使用します。',
    action='store_true'
)
# 学習率／学習回数
parser.add_argument(
    '--g-lr', help='Generatorの学習率を指定します。',
    type=float, default=None
)
parser.add_argument(
    '--d-lr', help='Discriminatorの学習率を指定します。',
    type=float, default=None
)
parser.add_argument(
    '--d-ur', help='D:Gの更新回数比を整数で指定します。',
    type=int, default=None
)
# 出力に関するコマンドライン引数
parser.add_argument(
    '--dir-name', help='出力ディレクトリの名前を指定します。',
    type=str, default=None,
)
# 画像生成
parser.add_argument(
    '--num-samples', help='結果を見るための1クラス当たりのサンプル数を指定します。',
    type=int, default=100
)
parser.add_argument(
    '--sample-interval', help='生成画像の保存間隔をエポック数で指定します。',
    type=int, default=25,
)
parser.add_argument(
    '--first-sample', help='1エポック学習直後の画像を保存します。',
    action='store_true'
)
# モデルのセーブ／ロード
parser.add_argument(
    '--save', help='訓練したモデルを保存します。',
    action='store_true'
)
parser.add_argument(
    '--save-interval', help='モデルの保存間隔をエポック数で指定します。',
    type=int, default=25,
)
parser.add_argument(
    '-lg', '--load-generator', help='指定したパスのGeneratorのセーブファイルを読み込みます。',
    type=str, default=None
)
parser.add_argument(
    '-ld', '--load-discriminator', help='指定したパスのDiscriminatorのセーブファイルを読み込みます。',
    type=str, default=None
)
# プログラムの動作
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

# =========================================================================== #
# 再現性の設定 https://pytorch.org/docs/stable/notes/randomness.html
# =========================================================================== #
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
# 結果の保存ディレクトリの準備
# =========================================================================== #
if args.dir_name is None:
    args.dir_name = launch_datetime.strftime('%Y%m%d%H%M%S')
suffix = '_grayscale' if args.grayscale else ''
suffix = f'{suffix}_jpeg' if args.g_jpeg else suffix
output_root = Path(
    f'./outputs/{args.dataset}{suffix}/{args.dir_name}')
output_root.mkdir(parents=True)
logger.info(f'結果出力用のディレクトリ({output_root})を作成しました。')
# 結果を出力するためのファイルを開く
output_csv = open(
    output_root.joinpath('outputs.csv'),
    mode='w',
    encoding='utf-8',
)
csv_writer = csv.writer(output_csv, lineterminator='\n')
output_csv.write(' '.join(sys.argv) + '\n')

# =========================================================================== #
# 訓練パラメータの読み込み
# =========================================================================== #
with open(Path(f'./configs/params/{args.dataset}.json')) as f:
    params = json.load(f)
# バッチサイズ／学習エポック数
batch_size = default(args.batch_size, params['batch_size'])
num_epochs = default(args.num_epochs, params['num_epochs'])
nz = default(args.nz, params['nz'])
# 学習率
g_lr = default(args.g_lr, params['g_lr'])
d_lr = default(args.d_lr, params['d_lr'])
d_ur = default(args.d_ur, params['d_ur'])
# 学習率スケジューラのパラメータ
lrs_ss = default(args.lrs_ss, params['lrs_ss'])
lrs_gamma = default(args.lrs_gamma, params['lrs_gamma'])
ms_lambda = args.ms_lambda
csv_writer.writerow([
    'バッチサイズ', batch_size,
    '更新回数比(D:G)', d_ur,
    'Mode Seeking Loss', ms_lambda
])

# =========================================================================== #
# データ整形／Data Augmentationの定義
# =========================================================================== #
logger.info('画像に適用する変換のリストを定義します。')
data_transforms = []

if args.dataset in ['mnist', 'fashion_mnist']:
    # MNIST/Fashion MNISTは28×28画素なのでゼロパディング
    data_transforms.append(
        transforms.Pad(2, fill=0, padding_mode='constant'))
    logger.info('変換リストにゼロパディングを追加しました。')

if args.grayscale and args.dataset in ['cifar10', 'stl10']:
    data_transforms.append(
        transforms.Grayscale(num_output_channels=1)
    )
    logger.info('変換リストにグレースケール化を追加しました。')

# if args.dataset == 'mnist':
#     data_transforms.extend([
#         transforms.RandomRotation(15, fill=0)
#     ])
# elif args.dataset == 'cifar10':
#     data_transforms.extend([
#         transforms.RandomHorizontalFlip(p=0.5),
#         transforms.RandomRotation(15, fill=0),
#         transforms.RandomResizedCrop(
#             32, scale=(0.75, 1.0), ratio=(3. / 4., 4. / 3.))
#     ])

data_transforms.append(transforms.ToTensor())
logger.info('変換リストにテンソル化を追加しました。')

# =========================================================================== #
# データセットの読み込み／データローダーの定義
# =========================================================================== #
dataset = load_dataset(
    args.dataset, root=args.data_path, transform=data_transforms)
# データセットの1番目の画像から色数を取得
nc, h, w = dataset[0][0].size()  # dataset[0][0].size() = (C, H, W)
assert h == w, '正方形以外の画像には未対応です。縦と横の画素数を一致させてください。'
try:
    class_names = get_classes(args.dataset)
except Exception:  # 日本語名が用意されていない場合
    class_names = [remove_invalid_char(name) for name in dataset.classes]
num_classes = len(class_names)
logger.info(f'データセット〈{args.dataset}〉を読み込みました。')

dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, drop_last=True)
logger.info('データローダを生成しました。')

# =========================================================================== #
# JPEGエンコーダ／デコーダの定義
# =========================================================================== #
if args.g_jpeg:
    from models.modules.jpeg import JPEGEncoder, JPEGDecoder
    jpeg_encoder = JPEGEncoder(chroma_subsampling=True)
    jpeg_encoder = jpeg_encoder.to(device)
    jpeg_decoder = JPEGDecoder()
    jpeg_decoder = jpeg_decoder.to(device)

# =========================================================================== #
# モデルのインポート
# =========================================================================== #
if h == 32:
    if args.g_jpeg:
        from models.generators.jpeg32 import Generator
    else:
        from models.generators.model32 import Generator
    if args.d_jpeg:
        from models.discriminators.jpeg32 import Discriminator
    else:
        from models.discriminators.model32 import Discriminator
elif h == 96:
    if args.g_jpeg:
        from models.generators.jpeg96 import Generator
    else:
        from models.generators.model96 import Generator
    if args.d_jpeg:
        from models.discriminators.jpeg96 import Discriminator
    else:
        from models.discriminators.model96 import Discriminator

# =========================================================================== #
# モデルの定義
# =========================================================================== #
g_model = Generator(
    nz=nz, nc=nc,
    num_classes=num_classes
).to(device)
if args.load_generator is not None:
    g_model.load_state_dict(
        torch.load(args.load_generator)['model_state_dict'])  # パラメータのロード
trainable, fixed = count_params(g_model)
csv_writer.writerow([
    'Generator',
    '総パラメータ数', trainable + fixed,
    '学習可能パラメータ数', trainable,
    '固定パラメータ数', fixed
])

d_model = Discriminator(
    nc=nc, num_classes=num_classes
).to(device)
if args.load_discriminator is not None:
    d_model.load_state_dict(
        torch.load(args.load_discriminator)['model_state_dict'])  # パラメータのロード
trainable, fixed = count_params(d_model)
csv_writer.writerow([
    'Discriminator',
    '総パラメータ数', trainable + fixed,
    '学習可能パラメータ数', trainable,
    '固定パラメータ数', fixed
])

# =========================================================================== #
# オプティマイザ／学習率スケジューラの定義
# =========================================================================== #
g_optimizer = torch.optim.Adam(
    g_model.parameters(),
    lr=g_lr,
    betas=[0.5, 0.999])
g_lrs = torch.optim.lr_scheduler.StepLR(
    g_optimizer, step_size=lrs_ss, gamma=lrs_gamma,
)
d_optimizer = torch.optim.Adam(
    d_model.parameters(),
    lr=d_lr,
    betas=[0.5, 0.999])
d_lrs = torch.optim.lr_scheduler.StepLR(
    d_optimizer, step_size=lrs_ss, gamma=lrs_gamma,
)

sample_z = torch.randn(args.num_samples, nz, device=device)

# =========================================================================== #
# 結果ログファイルの準備
# =========================================================================== #
output_csv.write('訓練ログ\n')
result_items = [
    'Epoch',
    'Generator Learning Rate', 'Generator Loss Mean',
    'Discriminator Learning Rate', 'Discriminator Loss Mean',
    'Train Elapsed Time',
]
csv_writer.writerow(result_items)
csv_idx = {item: i for i, item in enumerate(result_items)}

output_csv.flush()  # 書き込みバッファをフラッシュ

# =========================================================================== #
# 訓練
# =========================================================================== #
num_dis_update = 0  # Discriminatorの更新回数カウントを初期化
for epoch in range(num_epochs):
    output_dir = output_root.joinpath(f'epoch{epoch + 1}')
    results = ['' for _ in range(len(csv_idx))]
    results[csv_idx['Epoch']] = f'{epoch + 1}'

    g_loss_list = []
    d_loss_list = []

    # 損失関数: Hinge Loss
    # https://github.com/ajbrock/BigGAN-PyTorch/blob/master/losses.py
    pbar = tqdm(
        enumerate(dataloader),
        desc=f'[{epoch+1}/{num_epochs}] 訓練開始',
        total=len(dataset)//batch_size,
        leave=False)
    g_model.train()
    d_model.train()
    begin_time = perf_counter()  # 時間計測開始
    for i, (reals_orig, labels) in pbar:
        labels = labels.to(device)
        reals_orig = reals_orig.to(device)

        z = torch.randn(batch_size, nz, device=device)
        fakes_orig = g_model(z, labels)

        if args.g_jpeg:
            if args.d_jpeg:
                reals = jpeg_encoder(reals_orig)
                fakes = fakes_orig
            else:
                reals = reals_orig
                fakes = jpeg_decoder(fakes_orig)
        else:
            reals = reals_orig
            fakes = fakes_orig

        # =================================================================== #
        # Discriminatorの訓練
        # =================================================================== #
        d_model.zero_grad()  # 各パラメータの勾配の初期化

        # Real画像についてDを訓練
        output, reconstructed = d_model(reals, labels, reconstruct=True)
        d_loss_real = F.relu(1.0 - output).mean()
        # Reconstruction Loss
        d_loss_real += F.mse_loss(
            reconstructed,
            F.interpolate(reals_orig, reconstructed.size()[2:]))

        # Fake画像についてDを訓練
        output, _ = d_model(fakes, labels, detach=True)
        d_loss_fake = F.relu(1.0 + output).mean()

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()  # 逆伝播
        d_loss_list.append(d_loss.item())
        d_optimizer.step()  # Discriminatorの重みを更新
        num_dis_update += 1
        pbar_d_loss = f'{d_loss.item():.016f}'

        # =================================================================== #
        # Generatorの訓練
        # =================================================================== #
        if num_dis_update >= d_ur:  # 更新回数比が一定以上の場合にGを更新
            g_model.zero_grad()
            output, _ = d_model(fakes, labels)
            g_loss = -output.mean()

            # Mode Seeking Loss
            if ms_lambda > 0.0:
                z2 = torch.randn(batch_size, nz, device=device)
                fakes2 = g_model(z2, labels)
                z_loss = F.l1_loss(z, z2)
                if args.g_jpeg:
                    if args.d_jpeg:
                        fakes = jpeg_decoder(fakes_orig)
                    fakes2 = jpeg_decoder(fakes2)
                g_loss += ms_lambda * z_loss \
                    / F.l1_loss(
                        torch.clip(fakes, 0, 1),
                        torch.clip(fakes2, 0, 1))

            g_loss.backward()
            g_loss_list.append(g_loss.item())
            g_optimizer.step()  # Generatorの重みを更新
            num_dis_update = 0  # Discriminatorの更新回数カウントを初期化
            pbar_g_loss = f'{g_loss.item():.016f}'
        else:
            pbar_g_loss = '損失計算無し'

        pbar.set_description_str(  # プログレスバーの情報を更新
            f'[{epoch+1}/{num_epochs}] 訓練中... '
            f'<損失: (G={pbar_g_loss}, D={pbar_d_loss})>')
    end_time = perf_counter()  # 時間計測終了
    pbar.close()

    g_loss_mean = np.mean(g_loss_list)
    d_loss_mean = np.mean(d_loss_list)
    results[csv_idx['Generator Loss Mean']] = f'{g_loss_mean:.016f}'
    results[csv_idx['Discriminator Loss Mean']] = f'{d_loss_mean:.016f}'

    train_elapsed_time = end_time - begin_time
    results[csv_idx['Train Elapsed Time']] = f'{train_elapsed_time:.07f}'

    results[csv_idx['Generator Learning Rate']] = \
        f'{g_lrs.get_last_lr()[0]}'
    results[csv_idx['Discriminator Learning Rate']] = \
        f'{d_lrs.get_last_lr()[0]}'

    print(
        f'[{epoch+1}/{num_epochs}] 訓練完了. <'
        f'エポック処理時間: {train_elapsed_time:.07f}[s/epoch]'
        f', 平均損失: (G={g_loss_mean:.016f}, D={d_loss_mean:.016f})'
        ', 学習率: '
        f'(G={g_lrs.get_last_lr()[0]},'
        f' D={d_lrs.get_last_lr()[0]})'
        '>')

    g_lrs.step()  # Gの学習率スケジューラを進める
    d_lrs.step()  # Dの学習率スケジューラを進める

    g_model.eval()  # Generatorを評価モードに
    d_model.eval()  # Discriminatorを評価モードに

    if (
        epoch == 0 and args.first_sample
        or (epoch + 1) % args.sample_interval == 0
        or epoch == num_epochs - 1
    ):
        output_dir.mkdir(exist_ok=True)  # このエポック用の出力ディレクトリを作成
        overview = []
        for i in range(num_classes):
            with torch.no_grad():
                sample_classes = torch.ones(
                    sample_z.size(0), dtype=torch.long, device=device) * i
                sample_images = g_model(sample_z, sample_classes)
                if args.g_jpeg:
                    sample_images = jpeg_decoder(sample_images)
                sample_images = sample_images.cpu()
                overview.append(sample_images[:int(np.sqrt(args.num_samples))])
                vutils.save_image(
                    sample_images,
                    output_dir.joinpath(f'class{i}_{class_names[i]}.png'),
                    nrow=int(np.sqrt(args.num_samples)),
                    range=(0.0, 1.0)
                )
        vutils.save_image(
            torch.cat(overview, dim=0),
            output_dir.joinpath('overview.png'),
            nrow=int(np.sqrt(args.num_samples)),
            range=(0.0, 1.0)
        )

    if args.save and (
            (epoch + 1) % args.save_interval == 0
            or epoch == num_epochs - 1):
        output_dir.mkdir(exist_ok=True)  # このエポック用の出力ディレクトリを作成
        torch.save(  # Generatorのセーブ
            {
                'model_state_dict': g_model.state_dict(),
                'optimizer_state_dict': g_optimizer.state_dict(),
                'lrs_state_dict': g_lrs.state_dict(),
                'last_epoch': epoch,
                'batch_size': batch_size,
                'dataset': args.dataset,
                'h': h,
                'w': w,
                'nz': nz,
                'nc': nc,
                'num_classes': num_classes,
                'jpeg': args.g_jpeg
            },
            output_dir.joinpath('generator.pt')
        )
        torch.save(  # Discriminatorのセーブ
            {
                'model_state_dict': d_model.state_dict(),
                'optimizer_state_dict': d_optimizer.state_dict(),
                'lrs_state_dict': d_lrs.state_dict(),
                'last_epoch': epoch,
                'batch_size': batch_size,
                'dataset': args.dataset,
                'nc': nc,
                'num_classes': num_classes,
                'jpeg': args.d_jpeg
            },
            output_dir.joinpath('discriminator.pt')
        )
    csv_writer.writerow(results)
    output_csv.flush()
output_csv.close()
