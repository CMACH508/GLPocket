import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Transformer import TransformerModel
from models.PositionalEncoding import FixedPositionalEncoding

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

    def _reshape_output(self, x):
        x = x.view(x.size(0), 9, 9, 9, self.embedding_dim)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x


    def forward(self, x, center):
        x1 = self.in1(x)
        x1 = self.in2(x1)
        # print(x1.shape)
        skip_1 = self.clip(x1, f=1, condidate_center=center, a1a2=(36, 36))

        x1_0 = self.poll1_0(x1)
        x1_1 = self.down1_1(x1_0)
        x2 = self.down1_2(x1_1)
        x2 = self.clip(x2, f=2, condidate_center=center, a1a2=(17, 18))
        # print(x2.shape)

        x2_0 = self.poll2_0(x2)
        x2_1 = self.down2_1(x2_0)
        x3 = self.down2_2(x2_1)
        # print(x3.shape)

        x3_0 = self.poll3_0(x3)
        x3_1 = self.down3_1(x3_0)
        x4 = self.down3_2(x3_1)
        # print(x4.shape)

        x4_0 = self.poll4_0(x4)
        x4_1 = self.down4_1(x4_0)
        x5 = self.down4_2(x4_1)
        # print(x5.shape)

        # print('=============================')
        x11_0 = self.upsample1_0(x5)
        # print(x11_0.shape, x4.shape)
        x11_1 = self.up1_1(torch.cat((x11_0, x4), dim=1))
        x11 = self.up1_2(x11_1)
        # print(x11.shape)

        ''' transformer '''
        x11 = x11.permute(0, 2, 3, 4, 1).contiguous()
        x11 = x11.view(x11.size(0), -1, self.embedding_dim)
        x11 = self.position_encoding(x11)
        x11 = self.pe_dropout(x11)
        # apply transformer
        x11, intmd_x = self.transformer(x11)
        x11 = self.pre_head_ln(x11)
        # print('encoder_output=', x11.shape)

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
        # print('x11', x11.shape)
        x11 = self._reshape_output(x11)
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
        x44_1 = self.up4_1(torch.cat((x44_0, skip_1), dim=1))
        x44 = self.up4_2(x44_1)
        # print(x44.shape)

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
        batch_block = data_x[:, :, x1: x2 + 1, y1: y2 + 1, z1:z2 + 1]
        return batch_block


if __name__ == '__main__':
    model = Unet()
    # input = torch.randn(size=(1, 14, 65, 65, 65))
    input = torch.randn(size=(1, 14, 129, 129, 129))
    centers = torch.tensor([[80, 80, 80]])
    output = model(input, centers)
    print('output,shape=', output.shape)


