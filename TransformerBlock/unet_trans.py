import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import numpy as np
from os.path import join
from models.Transformer import TransformerModel
from models.PositionalEncoding import FixedPositionalEncoding


def save_density_as_cmap(density, origin, step, fname='pockets.cmap', mode='w', name='protein'):
    """Save predcited pocket density as .cmap file (which can be opened in
    UCSF Chimera or ChimeraX)
    """
    if len(density) != 1:
        raise ValueError('saving more than one prediction at a time is not'
                         ' supported')
    density = density[0].transpose([3, 2, 1, 0])

    with h5py.File(fname, mode) as cmap:
        g1 = cmap.create_group('Chimera')
        for i, channel_dens in enumerate(density):
            g2 = g1.create_group('image%s' % (i + 1))
            g2.attrs['chimera_map_version'] = 1
            g2.attrs['name'] = name.encode() + b' binding sites'
            g2.attrs['origin'] = origin
            g2.attrs['step'] = step
            g2.create_dataset('data_zyx', data=channel_dens,
                              shape=channel_dens.shape,
                              dtype='float32')


def channelPool(input):
    n, c, w, h, z = input.size()
    input = input.view(n, c, w * h * z).permute(0, 2, 1)
    pooled = nn.functional.max_pool1d(
        input,
        c,
    )
    _, _, c = pooled.size()
    pooled = pooled.permute(0, 2, 1)
    return pooled.view(n, c, w, h, z)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1):
        super().__init__()
        self.block = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding),
                                   nn.BatchNorm3d(out_channels),
                                   nn.ReLU(),
                                   nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding),
                                   nn.BatchNorm3d(out_channels),
                                   nn.ReLU())

    def forward(self, x):
        out = self.block(x)
        return out


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_pad, stride=2):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool3d(kernel_size_pad, stride=stride), DoubleConv(in_channels, out_channels, 3))

    def forward(self, x):
        out = self.block(x)
        return out


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_up,padding=0,stride=2, out_pad=0, upsample=None):
        super().__init__()
        if upsample:
            self.up_s = nn.Upsample(scale_factor=2, mode=upsample, align_corners=True)
        else:
            self.up_s = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size_up, stride=stride, padding=padding,
                                           output_padding=out_pad)

        self.convT = DoubleConv(in_channels, out_channels, 3)

    def forward(self, x1, x2):
        out = self.up_s(x1)
        out = self.convT(torch.cat((x2, out), dim=1))
        return out


class Unet(nn.Module):
    def __init__(self, n_classes=1, upsample=False):
        super().__init__()
        self.n_classes = n_classes
        self.in1 = DoubleConv(14, 32, 3)
        self.down1 = Down(32, 64, 3)
        self.down2 = Down(64, 128, 3)
        self.down3 = Down(128, 256, 3)
        factor = 2 if upsample else 1
        self.down4 = Down(256, 512 // factor, 3)
        self.up1 = Up(512, 256 // factor, 3, upsample=upsample, stride=2, out_pad=0)
        self.up2 = Up(256, 128 // factor, 3, upsample=upsample)
        self.up3 = Up(128, 64 // factor, 3, upsample=upsample, out_pad=1)
        self.up4 = Up(64, 32, 3, upsample=upsample)

        self.in1 = nn.Sequential(nn.Conv3d(14, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU())
        self.in2 = nn.Sequential(nn.Conv3d(32, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU())

        self.poll1_0 = nn.MaxPool3d(3, stride=2)
        self.down1_1 = nn.Sequential(nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU())
        self.down1_2 = nn.Sequential(nn.Conv3d(64, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU())

        self.poll2_0 = nn.MaxPool3d(3, stride=2, padding=1)
        self.down2_1 = nn.Sequential(nn.Conv3d(64, 128, 3, padding=1), nn.BatchNorm3d(128), nn.ReLU())
        self.down2_2 = nn.Sequential(nn.Conv3d(128, 128, 3, padding=1), nn.BatchNorm3d(128), nn.ReLU())

        self.poll3_0 = nn.MaxPool3d(3, stride=2, padding=1)
        self.down3_1 = nn.Sequential(nn.Conv3d(128, 256, 3, padding=1), nn.BatchNorm3d(256), nn.ReLU())
        self.down3_2 = nn.Sequential(nn.Conv3d(256, 256, 3, padding=1), nn.BatchNorm3d(256), nn.ReLU())

        factor = 2 if upsample else 1
        self.poll4_0 = nn.MaxPool3d(3, stride=2, padding=1)
        self.down4_1 = nn.Sequential(nn.Conv3d(256, 512, 3, padding=1), nn.BatchNorm3d(512), nn.ReLU())
        self.down4_2 = nn.Sequential(nn.Conv3d(512, 512, 3, padding=1), nn.BatchNorm3d(512), nn.ReLU())

        self.upsample1_0 = nn.ConvTranspose3d(512, 512//2, 3, stride=2, padding=1, output_padding=1)
        self.up1_1 = nn.Sequential(nn.Conv3d(512, 256, 3, padding=1), nn.BatchNorm3d(256), nn.ReLU())
        self.up1_2 = nn.Sequential(nn.Conv3d(256, 256, 3, padding=1), nn.BatchNorm3d(256), nn.ReLU())

        self.upsample2_0 = nn.ConvTranspose3d(256, 256 // 2, 3, stride=2, padding=1, output_padding=1)
        self.up2_1 = nn.Sequential(nn.Conv3d(256, 128, 3, padding=1), nn.BatchNorm3d(128), nn.ReLU())
        self.up2_2 = nn.Sequential(nn.Conv3d(128, 128, 3, padding=1), nn.BatchNorm3d(128), nn.ReLU())

        self.upsample3_0 = nn.ConvTranspose3d(128, 128 // 2, 3, stride=2, padding=1, output_padding=1)
        self.up3_1 = nn.Sequential(nn.Conv3d(128, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU())
        self.up3_2 = nn.Sequential(nn.Conv3d(64, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU())

        self.upsample4_0 = nn.ConvTranspose3d(64, 64 // 2, 3, stride=2, padding=0, output_padding=0)
        self.up4_1 = nn.Sequential(nn.Conv3d(64, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU())
        self.up4_2 = nn.Sequential(nn.Conv3d(32, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU())

        self.conv = nn.Conv3d(32, self.n_classes, 1)
        self.sigmoid = nn.Sigmoid()

        ''' add '''
        self.embedding_dim = 256
        self.dropout_rate = 0.1
        self.attn_dropout_rate = 0.1
        num_heads = 8
        num_layers = 4
        # hidden_dim = 4096
        hidden_dim = 256
        self.position_encoding = FixedPositionalEncoding(self.embedding_dim)
        self.pe_dropout = nn.Dropout(p=self.dropout_rate)
        self.transformer = TransformerModel(
            self.embedding_dim,
            num_layers,
            num_heads,
            hidden_dim,
            self.dropout_rate,
            self.attn_dropout_rate,
        )
        self.pre_head_ln = nn.LayerNorm(self.embedding_dim)
        print("num_layers = ", num_layers)
        print('original=4')


    def _reshape_output(self, x):
        x = x.view(x.size(0), 8, 8, 8, self.embedding_dim)
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # 1,8,8,8,256 -> 1,256,8,8,8
        return x

    def forward(self, x):
        # x1 = self.in1(x)
        # x2 = self.down1(x1)
        # x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # x11 = self.up1(x5, x4)
        # x22 = self.up2(x11, x3)
        # x33 = self.up3(x22, x2)
        # x44 = self.up4(x33, x1)
        # logits = self.conv(x44)
        # prob = self.sigmoid(logits)

        x1 = self.in1(x)
        x1 = self.in2(x1)
        # print(x1.shape) # C 32
        x1_0 = self.poll1_0(x1)
        x1_1 = self.down1_1(x1_0)
        x2 = self.down1_2(x1_1)
        # print(x2.shape)  #2C

        x2_0 = self.poll2_0(x2)
        x2_1 = self.down2_1(x2_0)
        x3 = self.down2_2(x2_1)
        # print(x3.shape) #4C

        x3_0 = self.poll3_0(x3)
        x3_1 = self.down3_1(x3_0)
        x4 = self.down3_2(x3_1)
        # print(x4.shape) #8C

        x4_0 = self.poll4_0(x4)
        x4_1 = self.down4_1(x4_0)
        x5 = self.down4_2(x4_1)
        # print(x5.shape)

        # print('=============================')
        x11_0 = self.upsample1_0(x5)
        x11_1 = self.up1_1(torch.cat((x11_0, x4), dim=1))
        x11 = self.up1_2(x11_1)
        # print(x11.shape)  # [1, 256, 8, 8, 8]

        ''' transformer '''
        x11 = x11.permute(0, 2, 3, 4, 1).contiguous()  # [1, 8, 8, 8, 256]
        x11 = x11.view(x11.size(0), -1, self.embedding_dim)  # [1, 512, 256]
        x11 = self.position_encoding(x11)  # [1, 512, 256]
        x11 = self.pe_dropout(x11)  # [1, 512, 256]
        # apply transformer
        x11, intmd_x = self.transformer(x11)  # x [1, 512, 256]
        x11 = self.pre_head_ln(x11)
        # print('encoder_output=', x11.shape)  # [1, 512, 256]

        encoder_outputs = {}
        all_keys = []
        for i in [1, 2, 3, 4]:
            val = str(2 * i - 1)  # 1,3,5,7
            _key = 'Z' + str(i)  # Z1, Z2, Z3, Z4
            all_keys.append(_key)
            encoder_outputs[_key] = intmd_x[val]
        all_keys.reverse()  # # Z4, Z3, Z2, Z1
        x11 = encoder_outputs[all_keys[0]]

        # print('enter decoder --------------')
        # print('x11', x11.shape)  # [1, 512, 256]
        x11 = self._reshape_output(x11)  # p[1, 256, 8, 8, 8]
        # print('reshape=', x11.shape)

        x22_0 = self.upsample2_0(x11)
        x22_1 = self.up2_1(torch.cat((x22_0, x3), dim=1))
        x22 = self.up2_2(x22_1)
        # print(x22.shape)

        x33_0 = self.upsample3_0(x22)
        x33_1 = self.up3_1(torch.cat((x33_0, x2), dim=1))
        x33 = self.up3_2(x33_1)
        # print(x33.shape)

        x44_0 = self.upsample4_0(x33)
        x44_1 = self.up4_1(torch.cat((x44_0, x1), dim=1))
        x44 = self.up4_2(x44_1)
        # print(x44.shape)

        logits = self.conv(x44)
        prob = self.sigmoid(logits)
        return prob


if __name__ == '__main__':
    model = Unet()
    input = torch.randn(size=(1, 14, 65, 65, 65))
    output = model(input)
    print('output,shape=', output.shape)


