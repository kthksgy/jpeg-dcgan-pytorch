# 標準モジュール
import argparse
import importlib

# コマンドライン引数を取得するパーサー
parser = argparse.ArgumentParser(
    prog='ツール',
    description='様々なツールを実行するためのポータルです。'
)

# 訓練に関する引数
parser.add_argument(
    'name', help='実行するツール名を指定します。',
    type=str, choices=[
        'dnn_sample',
        'dct_range',
        'export_dataset',
        'export_dct_bases',
        'export_image_colors',
        'print_dct_coefficients',
    ]
)
# コマンドライン引数をパース
args = parser.parse_args()

if __name__ == '__main__':
    # 指定されたツールを動的インポート
    mod = importlib.import_module(f'tools.{args.name}')
    # ツールを実行
    mod.do()
