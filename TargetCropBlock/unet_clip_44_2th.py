# _date_:2022/8/5 21:59
# _date_:2022/8/5 16:22
import torch
import torch.nn as nn
import torch.nn.functional as F
# import h5py
import numpy as np
from os.path import join
import math
# def save_density_as_cmap(density, origin, step, fname='pockets.cmap', mode='w', name='protein'):
#     """Save predcited pocket density as .cmap file (which can be opened in
#     UCSF Chimera or ChimeraX)
#     """
#     if len(density) != 1:
#         raise ValueError('saving more than one prediction at a time is not'
#                          ' supported')
#     density = density[0].transpose([3, 2, 1, 0])
#
#     with h5py.File(fname, mode) as cmap:
#         g1 = cmap.create_group('Chimera')
#         for i, channel_dens in enumerate(density):
#             g2 = g1.create_group('image%s' % (i + 1))
#             g2.attrs['chimera_map_version'] = 1
#             g2.attrs['name'] = name.encode() + b' binding sites'
#             g2.attrs['origin'] = origin
#             g2.attrs['step'] = step
#             g2.create_dataset('data_zyx', data=channel_dens,
#                               shape=channel_dens.shape,
#                               dtype='float32')


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

        print(' ================ unet 44 2-th ===============')

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

        self.upsample1_0 = nn.ConvTranspose3d(512, 512 // 2, 3, stride=2, padding=1, output_padding=0)
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

    def forward(self, x, center):
        # x = self.clip(x, f=1, condidate_center=center, a1a2=(36, 36))
        x1 = self.in1(x)
        x1 = self.in2(x1)
        # print('x1.shape=', x1.shape)
        skip_1 = self.clip(x1, f=1, condidate_center=center, a1a2=(36, 36))

        x1_0 = self.poll1_0(x1)
        x1_1 = self.down1_1(x1_0)
        x2 = self.down1_2(x1_1)
        x2 = self.clip(x2, f=2, condidate_center=center, a1a2=(17, 18))
        # print('x2.shape=', x2.shape)

        x2_0 = self.poll2_0(x2)
        x2_1 = self.down2_1(x2_0)
        x3 = self.down2_2(x2_1)
        # print('x3.shape=', x3.shape)

        x3_0 = self.poll3_0(x3)
        x3_1 = self.down3_1(x3_0)
        x4 = self.down3_2(x3_1)
        # print('x4.shape=', x4.shape)

        x4_0 = self.poll4_0(x4)
        x4_1 = self.down4_1(x4_0)
        x5 = self.down4_2(x4_1)
        # print('x5.shape=', x5.shape)

        # print('=============================')
        x11_0 = self.upsample1_0(x5)
        # print(x11_0.shape)
        x11_1 = self.up1_1(torch.cat((x11_0, x4), dim=1))
        x11 = self.up1_2(x11_1)
        # print('x11.shape=', x11.shape)

        x22_0 = self.upsample2_0(x11)
        x22_1 = self.up2_1(torch.cat((x22_0, x3), dim=1))
        x22 = self.up2_2(x22_1)
        # print('x22.shape=', x22.shape)

        x33_0 = self.upsample3_0(x22)
        x33_1 = self.up3_1(torch.cat((x33_0, x2), dim=1))
        x33 = self.up3_2(x33_1)
        # print('x33.shape=', x33.shape)

        x44_0 = self.upsample4_0(x33)
        x44_1 = self.up4_1(torch.cat((x44_0, skip_1), dim=1))
        x44 = self.up4_2(x44_1)
        # print('x44.shape=', x44.shape)

        logits = self.conv(x44)
        prob = self.sigmoid(logits)
        return prob


    def clip(self, data_x, f, condidate_center, a1a2):
        center = condidate_center / f
        # block_list = []
        a1, a2 = a1a2
        min_center = (center - a1).int()
        max_center = (center + a2).int()
        x1, y1, z1 = min_center[0]
        x2, y2, z2 = max_center[0]
        # print(center)
        # print('a1={} a2={}'.format(a1, a2))
        # print('x1,y1,z1=', x1, y1, z1)
        # print('x2,y2,z2=', x2, y2, z2)
        batch_block = data_x[:, :, x1: x2 + 1, y1: y2 + 1, z1:z2 + 1]
        return batch_block
        # pad_x = F.pad(data_x, (a2, a2, a2, a2, a2, a2), "constant")
        # center = center.int() + a2
        # min_center = (center - a1).int()
        # max_center = (center + a2).int()
        # pad_x = pad(data_x)
        # print(pad_x.shape)
        # for i in range(len(data_x)):
        #     x1, y1, z1 = min_center[i]
        #     x2, y2, z2 = max_center[i]
        #     # data = pad_x[i:i+1, :, x1: x2 + 1, y1: y2 + 1, z1:z2 + 1]
        #     data = data_x[i:i+1, :, x1: x2 + 1, y1: y2 + 1, z1:z2 + 1]
            # if not (data.shape[2] == data.shape[3] and data.shape[3] == data.shape[4]):
            #     print('unet get feature, out of range')
            #     print('data_x.shape=', data_x.shape)
            #     print('feature_clip_block.shape=', data.shape, 'a2={}'.format(a2), 'f={}'.format(f))
            #     print('center:', condidate_center[i], center[i])
            #     print(x1, y1, z1)
            #     print(x2, y2, z2)
            #     print()
            # block_list.append(data)
            # print('block.shape=', data.shape)
        # batch_block = torch.stack(block_list, dim=0)
        # batch_block = torch.cat(block_list, dim=0)
#

if __name__ == '__main__':
    model = Unet()
    # input = torch.randn(size=(1, 14, 65, 65, 65))
    input = torch.randn(size=(1, 14, 129, 129, 129))
    centers = torch.tensor([[80, 80, 80]])
    output = model(input, centers)
    print('output,shape=', output.shape)


