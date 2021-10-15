import numpy as np
import torch
import torch.nn as nn


class BWDCT(nn.Module):
    def __init__(self):
        super().__init__()
        self.bwdct = nn.Conv2d(1, 64, 8, 8, bias=False)
        self.bwdct.weight.requires_grad = False
        for m in range(8):
            for n in range(8):
                for p in range(8):
                    for q in range(8):
                        self.bwdct.weight[p * 8 + q, 0, m, n] \
                            = np.cos(np.pi * (2 * m + 1) * p / 16) \
                            * np.cos(np.pi * (2 * n + 1) * q / 16) \
                            * (np.sqrt(1 / 8) if p == 0 else (1 / 2)) \
                            * (np.sqrt(1 / 8) if q == 0 else (1 / 2))

    def forward(self, x):
        return self.bwdct(x)


class JPEGEncoder(nn.Module):
    def __init__(
        self, chroma_subsampling=True
    ):
        super().__init__()
        self.bwdct = BWDCT()
        self.chroma_subsampling = chroma_subsampling
        if self.chroma_subsampling:
            self.cs = nn.AvgPool2d(2)

    def forward(self, image):
        '''JPEGのDCT係数から正規化済みRGB画素へ変換する。
        Args:
            y_coefs: YのDCT係数
            cr_coefs: CrのDCT係数
            cb_coefs: CbのDCT係数
        Note:
            入力は(B, 64, H, W)の形状で入力する。
            各成分はジグザグスキャンせずそのまま平坦化した状態である。
        '''
        if image.size(1) == 1:
            return self.bwdct(image), None
        else:
            # Y
            y = 0.299 * image[:, 0:1] \
                + 0.587 * image[:, 1:2] \
                + 0.114 * image[:, 2:3]
            cbcr = torch.cat([
                # Cb
                -0.1687 * image[:, 0:1]
                - 0.3313 * image[:, 1:2]
                + 0.5 * image[:, 2:3]
                + 0.5,
                # Cr
                0.5 * image[:, 0:1]
                - 0.4187 * image[:, 1:2]
                - 0.0813 * image[:, 2:3]
                + 0.5
            ], dim=1)

            if self.chroma_subsampling:
                cbcr = self.cs(cbcr)
            y_blocks = self.bwdct(y)
            cbcr_blocks = torch.cat([
                self.bwdct(cbcr[:, 0:1]),
                self.bwdct(cbcr[:, 1:2])
            ], dim=1)
            return y_blocks, cbcr_blocks


class IBWDCT(nn.Module):
    def __init__(self,):
        super().__init__()
        self.ibwdct = nn.ConvTranspose2d(64, 1, 8, 8, bias=False)
        self.ibwdct.weight.requires_grad = False
        for m in range(8):
            for n in range(8):
                for p in range(8):
                    for q in range(8):
                        self.ibwdct.weight[p * 8 + q, 0, m, n] \
                            = np.cos(np.pi * (2 * m + 1) * p / 16) \
                            * np.cos(np.pi * (2 * n + 1) * q / 16) \
                            * (np.sqrt(1 / 8) if p == 0 else (1 / 2)) \
                            * (np.sqrt(1 / 8) if q == 0 else (1 / 2))

    def forward(self, x):
        return self.ibwdct(x)


class JPEGDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.ibwdct = IBWDCT()
        self.ics = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, y_cbcr):
        '''JPEGのDCT係数から正規化済みRGB画素へ変換する。

        Args:
            y_cbcr: DCT係数(Y, CbCr)のタプル ((B, 64, H, W), (B, 128, H, W))
        Note:
            各成分はジグザグスキャンせずそのまま平坦化した状態である。
        '''
        y = self.ibwdct(y_cbcr[0])
        if y_cbcr[1] is not None:
            cbcr = torch.cat([
                self.ibwdct(y_cbcr[1][:, :64]),
                self.ibwdct(y_cbcr[1][:, 64:])
            ], dim=1)
            if cbcr.size()[2:] != y.size()[2:]:
                cbcr = self.ics(cbcr)
            return torch.cat([
                # R
                y
                + 1.402 * (cbcr[:, 1:2] - 0.5),
                # G
                y
                - 0.344136286 * (cbcr[:, 0:1] - 0.5)
                - 0.714136286 * (cbcr[:, 1:2] - 0.5),
                # B
                y
                + 1.772 * (cbcr[:, 0:1] - 0.5)
            ], dim=1)
        else:
            return y
