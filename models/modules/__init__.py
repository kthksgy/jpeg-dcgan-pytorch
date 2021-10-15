import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


def init_xavier_uniform(layer):
    if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
        if hasattr(layer, "weight"):
            torch.nn.init.xavier_uniform_(layer.weight)
        if hasattr(layer, "bias"):
            if hasattr(layer.bias, "data"):
                layer.bias.data.fill_(0)


def swish(x):
    return x * torch.sigmoid(x)


class Swish(nn.Module):
    def forward(self, x):
        return swish(x)


# https://github.com/sxhxliang/BigGAN-pytorch/blob/4cbad24f7b49bf55f2b1b6aa8451b2db495b707c/model_resnet.py
class SelfAttention(nn.Module):
    """Self attention Layer"""
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

        self.query_conv.apply(init_xavier_uniform)
        self.key_conv.apply(init_xavier_uniform)
        self.value_conv.apply(init_xavier_uniform)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        # B X CX(N)
        proj_query = \
            self.query_conv(x) \
                .view(m_batchsize, -1, width*height) \
                .permute(0, 2, 1)
        # B X C x (*W*H)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        # transpose check
        energy = torch.bmm(proj_query, proj_key)
        # BX (N) X (N)
        attention = self.softmax(energy)
        # B X C X N
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma*out + x
        return out


class CondBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        if num_classes > 0:
            self.bn = nn.BatchNorm2d(self.num_features, affine=False)
            self.gamma_embed = nn.Embedding(num_classes, self.num_features)
            nn.init.normal_(self.gamma_embed.weight, 1, 0.02)
            self.beta_embed = nn.Embedding(num_classes, self.num_features)
            nn.init.zeros_(self.beta_embed.weight)
        else:
            self.bn = nn.BatchNorm2d(self.num_features)

    def forward(self, x, classes=None):
        if classes is not None:
            return \
                self.gamma_embed(classes).view(-1, self.num_features, 1, 1) \
                * self.bn(x) \
                + self.beta_embed(classes).view(-1, self.num_features, 1, 1)
        else:
            return self.bn(x)
