from typing import Optional

import torch


class AutoDevice:
    '''自動的に最適なデバイスを判断して選択するクラス
    '''
    def __init__(self, disable_cuda: bool = False):
        '''
        Args:
            disable_cuda: CUDAを無効化するかどうか
        Todo:
            * 他の尺度も実装する
              現在は最大メモリ容量を基準に優先順位を決定している。
        '''
        self.use_cuda = not disable_cuda and torch.cuda.is_available()
        if self.use_cuda:
            self.cuda_devices = []
            for i in range(torch.cuda.device_count()):
                prop = torch.cuda.get_device_properties(f'cuda:{i}')
                self.cuda_devices.append({
                    'key': f'cuda:{i}',
                    'name': prop.name,
                    'capability': (prop.major, prop.minor),
                    'memory': prop.total_memory,
                    'num_processors': prop.multi_processor_count,
                })
            self.cuda_devices.sort(key=lambda d: d['memory'], reverse=True)
        else:
            self.cuda_devices = None

    def __call__(self, priority: Optional[int] = None) -> str:
        '''自動計算された優先度順でデバイスを返します。

        Args:
            priority: 特定の優先順位のデバイスを得る場合に指定します。
        Returns:
            デバイスを表す文字列
        '''
        if self.use_cuda:
            if priority is None:
                return 'cuda'
            else:
                return self.cuda_devices[
                    min(priority, len(self.cuda_devices) - 1)
                ]['key']
        else:
            return 'cpu'
