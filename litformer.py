'''
LIT-Former
'''
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _triple
import ipdb
import copy
from torch.nn.parameter import Parameter
import numbers
from einops import rearrange
from torch.nn import init
import matplotlib.pyplot as plt

torch.backends.cudnn.enabled = False
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class eMSM_T(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(eMSM_T, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.position_embedding = PositionalEncoding(d_model=dim)

        self.project_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(0.)
        )

    def forward(self, x):
        b, c, t, h, w = x.shape  # (pytorch_ssim_package,pytorch_ssim_package,pytorch_ssim_package,512,512)

        x = F.adaptive_avg_pool3d(x, (t, 1, 1))  # (pytorch_ssim_package,32,pytorch_ssim_package,pytorch_ssim_package,pytorch_ssim_package)

        x = x.squeeze(-1).squeeze(-1).permute(2, 0, 1)  # t,b,c  #(pytorch_ssim_package,pytorch_ssim_package,32)

        x = self.position_embedding(x).permute(1, 0, 2)  # b,t,c   #(pytorch_ssim_package,pytorch_ssim_package,32)

        q = self.to_q(x)  # (pytorch_ssim_package,pytorch_ssim_package,32)
        k = self.to_k(x)  # (pytorch_ssim_package,pytorch_ssim_package,32)
        v = self.to_v(x)  # (pytorch_ssim_package,pytorch_ssim_package,32)
        # ipdb.set_trace()

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=self.num_heads), (q, k, v))  # (pytorch_ssim_package,pytorch_ssim_package,32)

        scale = (c // self.num_heads) ** -0.5
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * scale  # (pytorch_ssim_package,pytorch_ssim_package,pytorch_ssim_package)

        attn = sim.softmax(dim=-1)  # (pytorch_ssim_package,pytorch_ssim_package,pytorch_ssim_package)

        out = torch.einsum('b i j, b j d -> b i d', attn, v)  # (pytorch_ssim_package,pytorch_ssim_package,32)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=self.num_heads)  # (pytorch_ssim_package,pytorch_ssim_package,32)

        out = self.project_out(out).permute(0, 2, 1)  # (pytorch_ssim_package,32,pytorch_ssim_package)

        return out


class eMSM_I(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(eMSM_I, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, t, h, w = x.shape  # (pytorch_ssim_package,32,pytorch_ssim_package,512,512)
        x = F.adaptive_avg_pool3d(x, (1, h, w))  # (pytorch_ssim_package,32,pytorch_ssim_package,512,512)  #没有用
        x = x.permute(0, 1, 3, 4, 2).squeeze(-1)  # (pytorch_ssim_package,32,512,512)

        qkv = self.qkv_dwconv(self.qkv(x))  # (pytorch_ssim_package,96,512,512)
        q, k, v = qkv.chunk(3, dim=1)  # (pytorch_ssim_package,32,512,512)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  # (pytorch_ssim_package,pytorch_ssim_package,32,262144)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  # (pytorch_ssim_package,pytorch_ssim_package,32,262144)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  # (pytorch_ssim_package,pytorch_ssim_package,32,262144)

        q = torch.nn.functional.normalize(q, dim=-1)  # (pytorch_ssim_package,pytorch_ssim_package,32,262144)
        k = torch.nn.functional.normalize(k, dim=-1)  # (pytorch_ssim_package,pytorch_ssim_package,32,262144)

        attn = (q @ k.transpose(-2, -1)) * self.temperature  # (pytorch_ssim_package,pytorch_ssim_package,32,32)
        attn = attn.softmax(dim=-1)  # (pytorch_ssim_package,pytorch_ssim_package,32,32)

        out = (attn @ v)  # (pytorch_ssim_package,pytorch_ssim_package,32,262144)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)  # (pytorch_ssim_package,32,512,512)

        out = self.project_out(out)  # (pytorch_ssim_package,32,512,512)
        return out


class LITFormerBlock(nn.Module):
    def __init__(self, input_channel, output_channel, num_heads_s=8, num_heads_t=2, kernel_size=1,
                 stride=1, padding=0,
                 groups=1, bias=False, res=True, attention_s=False, attention_t=False):
        super().__init__()
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        assert len(kernel_size) == len(stride) == len(padding) == 3
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.bias = bias
        self.res = res
        self.attn_s = attention_s
        self.attn_t = attention_t
        self.num_heads_s = num_heads_s
        self.num_heads_t = num_heads_t
        self.activation = nn.LeakyReLU(inplace=True)

        if attention_s == True:
            self.attention_s = eMSM_I(dim=input_channel, num_heads=num_heads_s, bias=False)
        self.conv_1x3x3 = nn.Conv3d(input_channel, output_channel, kernel_size=(1, kernel_size[1], kernel_size[2]),
                                    stride=(1, stride[1], stride[2]), padding=(0, padding[1], padding[2]),
                                    groups=groups)
        if attention_t == True:
            self.attention_t = eMSM_T(dim=input_channel, num_heads=num_heads_t, bias=False)
        self.conv_3x1x1 = nn.Conv3d(input_channel, output_channel, kernel_size=(kernel_size[0], 1, 1),
                                    stride=(stride[0], 1, 1), padding=(padding[0], 0, 0), groups=groups)

        if self.input_channel != self.output_channel:
            self.shortcut = nn.Conv3d(in_channels=input_channel, out_channels=output_channel, kernel_size=1, padding=0,
                                      stride=1, groups=1, bias=False)

    def forward(self, inputs):

        if self.attn_s == True or self.attn_t == True:
            # （pytorch_ssim_package，32，pytorch_ssim_package，512，512）
            attn_s = self.attention_s(inputs).unsqueeze(2) if self.attn_s == True else 0  # （pytorch_ssim_package，32，pytorch_ssim_package，512，512）
            attn_t = self.attention_t(inputs).unsqueeze(-1).unsqueeze(-1) if self.attn_t == True else 0  # （pytorch_ssim_package，32，pytorch_ssim_package，pytorch_ssim_package，pytorch_ssim_package）

            inputs_attn = inputs + attn_t + attn_s  # （pytorch_ssim_package，32，pytorch_ssim_package，512，512）

            conv_S = self.conv_1x3x3(inputs_attn)  # （pytorch_ssim_package，32，pytorch_ssim_package，512，512）
            conv_T = self.conv_3x1x1(inputs_attn)  # （pytorch_ssim_package，32，pytorch_ssim_package，512，512）

            if self.input_channel == self.output_channel:
                identity_out = inputs_attn
            else:
                identity_out = self.shortcut(inputs_attn)

        else:
            if self.input_channel == self.output_channel:
                identity_out = inputs
            else:
                identity_out = self.shortcut(inputs)  # (pytorch_ssim_package,32,pytorch_ssim_package,512,512)

            conv_S = self.conv_1x3x3(inputs)  # (pytorch_ssim_package,32,pytorch_ssim_package,512,512)
            conv_T = self.conv_3x1x1(inputs)  # (pytorch_ssim_package,32,pytorch_ssim_package,512,512)

        if self.res:
            output = conv_S + conv_T + identity_out  # (pytorch_ssim_package,32,pytorch_ssim_package,512,512)
        elif not self.res:
            output = conv_S + conv_T

        return output


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads_s=8, num_heads_t=2,
                 res=True, attention_s=False, attention_t=False):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            LITFormerBlock(in_channels, in_channels, num_heads_s=num_heads_s, num_heads_t=num_heads_t, res=res,
                           attention_s=attention_s, attention_t=attention_t),
            nn.LeakyReLU(inplace=True),
            LITFormerBlock(in_channels, out_channels, res=res),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):

    def __init__(self, in_channels, out_channels, num_heads_s=8, num_heads_t=2,
                 res=True, attention_s=False, attention_t=False):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool3d((1, 2, 2), (1, 2, 2)),
            DoubleConv(in_channels, out_channels, num_heads_s=num_heads_s, num_heads_t=num_heads_t,
                       res=res, attention_s=attention_s, attention_t=attention_t)
        )

    def forward(self, x):
        return self.encoder(x)


class LastDown(nn.Module):

    def __init__(self, in_channels, out_channels, num_heads_s=8, num_heads_t=2,
                 res=True, attention_s=False, attention_t=False):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.MaxPool3d((1, 2, 2), (1, 2, 2)),
            LITFormerBlock(in_channels, 2 * in_channels, num_heads_s=num_heads_s, num_heads_t=num_heads_t, res=res,
                           attention_s=attention_s, attention_t=attention_t),
            nn.LeakyReLU(inplace=True),
            LITFormerBlock(2 * in_channels, out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.encoder(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, res_unet=True, trilinear=True, num_heads_s=8, num_heads_t=2,
                 res=True, attention_s=False, attention_t=False):
        super().__init__()
        self.res_unet = res_unet
        if trilinear:
            self.up = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels, num_heads_s=num_heads_s, num_heads_t=num_heads_t,
                               res=res, attention_s=attention_s, attention_t=attention_t)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if self.res_unet:
            x = x1 + x2
        else:
            x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels, res=True, activation=False):
        super().__init__()
        self.act = activation
        self.conv = LITFormerBlock(in_channels, out_channels, res=res)
        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)  # (pytorch_ssim_package,32,pytorch_ssim_package,512,512)
        if self.act == True:
            x = self.activation(x)  # (pytorch_ssim_package,32,pytorch_ssim_package,512,512)
        return x


#我加的
class PTPBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        sobel_x = torch.tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]], dtype=torch.float32)
        sobel_y = torch.tensor([[1,  2,  1],
                                [0,  0,  0],
                                [-1, -2, -1]], dtype=torch.float32)

        kernel = torch.stack([sobel_x, sobel_y], dim=0).unsqueeze(1)
        self.register_buffer('grad_kernel', kernel)

        self.proj = nn.Conv3d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        B, C, D, H, W = x.shape

        x_2d = x.view(B * D * C, 1, H, W)
        grad = F.conv2d(x_2d, self.grad_kernel, padding=1)
        grad_mag = torch.sqrt(grad[:, 0]**2 + grad[:, 1]**2 + 1e-6)
        grad_mag = grad_mag.view(B, C, D, H, W)

        x_enhanced = x + x * torch.tanh(grad_mag)
        out = self.proj(x_enhanced)

        return out


class LITFormer(nn.Module):
    def __init__(self, in_channels, out_channels, n_channels, num_heads_s=[1, 2, 4, 8], num_heads_t=[1, 2, 4, 8],
                 res=True, attention_s=False, attention_t=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_channels = n_channels

        self.firstconv = SingleConv(in_channels, n_channels // 2, res=res, activation=True)
        self.enc1 = DoubleConv(n_channels // 2, n_channels, num_heads_s=num_heads_s[0], num_heads_t=num_heads_t[0],
                               res=res, attention_s=attention_s, attention_t=attention_t)

        self.enc2 = Down(n_channels, 2 * n_channels, num_heads_s=num_heads_s[1], num_heads_t=num_heads_t[1],
                         res=res, attention_s=attention_s, attention_t=attention_t)

        self.enc3 = Down(2 * n_channels, 4 * n_channels, num_heads_s=num_heads_s[2], num_heads_t=num_heads_t[2],
                         res=res, attention_s=attention_s, attention_t=attention_t)

        self.ptp_block = PTPBlock(channels=4 * n_channels)#我加的

        self.enc4 = LastDown(4 * n_channels, 4 * n_channels, num_heads_s=num_heads_s[3], num_heads_t=num_heads_t[3],
                             res=res, attention_s=attention_s, attention_t=attention_t)

        self.dec1 = Up(4 * n_channels, 2 * n_channels, num_heads_s=num_heads_s[2], num_heads_t=num_heads_t[2],
                       res=res, attention_s=attention_s, attention_t=attention_t)

        self.dec2 = Up(2 * n_channels, 1 * n_channels, num_heads_s=num_heads_s[1], num_heads_t=num_heads_t[1],
                       res=res, attention_s=attention_s, attention_t=attention_t)

        self.dec3 = Up(1 * n_channels, n_channels // 2, num_heads_s=num_heads_s[0], num_heads_t=num_heads_t[0],
                       res=res, attention_s=attention_s, attention_t=attention_t)
        self.out1 = SingleConv(n_channels // 2, n_channels // 2, res=res, activation=True)
        self.depth_up = nn.Upsample(scale_factor=tuple([2.5, 1, 1]), mode='trilinear')
        self.out2 = SingleConv(n_channels // 2, out_channels, res=res, activation=False)

    def forward(self, x):
        b, c, d, h, w = x.shape  # (pytorch_ssim_package,pytorch_ssim_package,pytorch_ssim_package,512,512)
        x = self.firstconv(x)  # (pytorch_ssim_package,32,pytorch_ssim_package,512,512)
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x3 = self.ptp_block(x3)
        x4 = self.enc4(x3)
        output = self.dec1(x4, x3)
        output = self.dec2(output, x2)
        output = self.dec3(output, x1)
        output = self.out1(output) + x  # （pytorch_ssim_package，32，pytorch_ssim_package，512，512）
        # output = self.depth_up(output)#（pytorch_ssim_package，32，2，512，512）
        output = self.out2(output)
        return output