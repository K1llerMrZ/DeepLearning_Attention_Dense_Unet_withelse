import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)


class up_conv_1x1(nn.Module):
    def __init__(self, ch_in, ch_out, scale_factor):
        super(up_conv_1x1, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor),
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)


class pool_conv(nn.Module):
    def __init__(self, ch_in, ch_out, layer_num):
        super(pool_conv, self).__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True)
        self.pools = nn.ModuleList()
        for i in range(layer_num):
            self.pools.append(nn.Sequential(
                nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=2, padding=1, bias=True),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True)
            ))

    def forward(self, x):
        x = self.conv(x)
        for n in self.pools:
            x = n(x)
        return x


class AttU_Net_plus_with_identify(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, scale_factor=1):
        super(AttU_Net_plus_with_identify, self).__init__()
        filters = np.array([32, 64, 128, 256, 512])
        filters = filters // scale_factor
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.scale_factor = scale_factor
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.Sigmoid= nn.Sigmoid

        self.Conv1 = conv_block(ch_in=n_channels, ch_out=filters[0])
        self.Conv2 = conv_block(ch_in=filters[0], ch_out=filters[1])
        self.Conv3 = conv_block(ch_in=filters[1], ch_out=filters[2])
        self.Conv4 = conv_block(ch_in=filters[2], ch_out=filters[3])
        self.Conv5 = conv_block(ch_in=filters[3], ch_out=filters[4])

        self.Up5 = up_conv(ch_in=filters[4], ch_out=filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = conv_block(ch_in=filters[4], ch_out=filters[3])

        self.Up4 = up_conv(ch_in=filters[3], ch_out=filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(ch_in=filters[3], ch_out=filters[2])

        self.Up3 = up_conv(ch_in=filters[2], ch_out=filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(ch_in=filters[2], ch_out=filters[1])

        self.Up2 = up_conv(ch_in=filters[1], ch_out=filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=filters[0] // 2)
        self.Up_conv2 = conv_block(ch_in=filters[1], ch_out=filters[0])

        self.Conv_1x1 = nn.Conv2d(filters[0], n_classes, kernel_size=1, stride=1, padding=0)

        self.Conv_Res_1_2 = nn.Conv2d(filters[0], filters[1], kernel_size=1, stride=1, padding=0)
        self.Conv_Res_2_3 = nn.Conv2d(filters[1], filters[2], kernel_size=1, stride=1, padding=0)
        self.Conv_Res_3_4 = nn.Conv2d(filters[2], filters[3], kernel_size=1, stride=1, padding=0)
        self.Conv_Res_4_5 = nn.Conv2d(filters[3], filters[4], kernel_size=1, stride=1, padding=0)

        self.Conv_Res_1_3 = nn.Conv2d(filters[0], filters[2], kernel_size=2, stride=2, padding=0)
        self.Conv_Res_2_4 = nn.Conv2d(filters[1], filters[3], kernel_size=2, stride=2, padding=0)
        self.Conv_Res_3_5 = nn.Conv2d(filters[2], filters[4], kernel_size=2, stride=2, padding=0)

        self.Conv_Res_1_4 = nn.Conv2d(filters[0], filters[3], kernel_size=4, stride=4, padding=0)
        self.Conv_Res_2_5 = nn.Conv2d(filters[1], filters[4], kernel_size=4, stride=4, padding=0)

        self.Conv_Res_1_5 = nn.Conv2d(filters[0], filters[4], kernel_size=8, stride=8, padding=0)

        self.UpConv_Res_3_2 = up_conv_1x1(filters[2], filters[1], scale_factor=2)
        self.UpConv_Res_4_3 = up_conv_1x1(filters[3], filters[2], scale_factor=2)
        self.UpConv_Res_5_4 = up_conv_1x1(filters[4], filters[3], scale_factor=2)

        self.UpConv_Res_4_2 = up_conv_1x1(filters[3], filters[1], scale_factor=4)
        self.UpConv_Res_5_3 = up_conv_1x1(filters[4], filters[2], scale_factor=4)

        self.UpConv_Res_5_2 = up_conv_1x1(filters[4], filters[1], scale_factor=8)

        self.Conv_pooing_1 = nn.Sequential(nn.Conv2d(filters[0], filters[0], kernel_size=2, stride=2, padding=0), nn.ReLU())
        self.Conv_pooing_2 = nn.Sequential(nn.Conv2d(filters[1], filters[1], kernel_size=2, stride=2, padding=0), nn.ReLU())
        self.Conv_pooing_3 = nn.Sequential(nn.Conv2d(filters[2], filters[2], kernel_size=2, stride=2, padding=0), nn.ReLU())
        self.Conv_pooing_4 = nn.Sequential(nn.Conv2d(filters[3], filters[3], kernel_size=2, stride=2, padding=0), nn.ReLU())

        self.Conv_identify_get_4 = pool_conv(filters[3] * 2, filters[3] // 2, layer_num=0)
        self.Conv_identify_get_3 = pool_conv(filters[2] * 2, filters[2] // 2, layer_num=1)
        self.Conv_identify_get_2 = pool_conv(filters[1] * 2, filters[1] // 2, layer_num=2)
        self.Conv_identify_get_1 = pool_conv(filters[0] * 2, filters[0], layer_num=3)

        channels_sum = filters[3] // 2 + filters[2] // 2 + filters[1] // 2 + filters[0]
        self.Conv_identify_4 = conv_block(ch_in=channels_sum, ch_out=filters[3])
        self.Conv_identify_3 = conv_block(ch_in=filters[3], ch_out=filters[2])
        self.Conv_identify_2 = conv_block(ch_in=filters[2], ch_out=filters[1])
        self.Conv_identify_1 = conv_block(ch_in=filters[1], ch_out=filters[0])
        #
        # self.Conv_identify_pooling_4 = nn.Conv2d(filters[4], filters[4], kernel_size=2, stride=2, padding=0)
        # self.Conv_identify_pooling_3 = nn.Conv2d(filters[3], filters[3], kernel_size=2, stride=2, padding=0)
        # self.Conv_identify_pooling_2 = nn.Conv2d(filters[2], filters[2], kernel_size=2, stride=2, padding=0)
        # self.Conv_identify_pooling_1 = nn.Conv2d(filters[1], filters[1], kernel_size=2, stride=2, padding=0)

        self.Linear_identify_1 = nn.Sequential(nn.Flatten(), nn.Linear(filters[0] * 2 * 8 ** 2, 1024), nn.ReLU())
        self.Linear_identify_2 = nn.Sequential(nn.Linear(1024, 128), nn.ReLU())
        self.Linear_feature = nn.Sequential(nn.Linear(128, 16), nn.Sigmoid())
        self.Linear_out = nn.Sequential(nn.Linear(16, 1), nn.Sigmoid())

        self.IdeConv_Res_1_2 = nn.Conv2d(channels_sum, filters[3], kernel_size=1, stride=1, padding=0)
        self.IdeConv_Res_2_3 = nn.Conv2d(filters[3], filters[2], kernel_size=1, stride=1, padding=0)
        self.IdeConv_Res_3_4 = nn.Conv2d(filters[2], filters[1], kernel_size=1, stride=1, padding=0)

        self.IdeConv_Res_1_3 = nn.Conv2d(channels_sum, filters[2], kernel_size=1, stride=1, padding=0)
        self.IdeConv_Res_2_4 = nn.Conv2d(filters[3], filters[1], kernel_size=1, stride=1, padding=0)

        self.IdeConv_Res_1_4 = nn.Conv2d(channels_sum, filters[1], kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2_t = self.Conv_pooing_1(x1)
        x2_o = self.Conv2(x2_t)

        x2 = x2_o + self.Conv_Res_1_2(x2_t)
        x3_t = self.Conv_pooing_2(x2)
        x3_o = self.Conv3(x3_t)

        x3 = x3_o + self.Conv_Res_2_3(x3_t)
        x3 = x3 + self.Conv_Res_1_3(x2_t)
        x4_t = self.Conv_pooing_3(x3)
        x4_o = self.Conv4(x4_t)

        x4 = x4_o + self.Conv_Res_3_4(x4_t)
        x4 = x4 + self.Conv_Res_2_4(x3_t)
        x4 = x4 + self.Conv_Res_1_4(x2_t)
        x5_t = self.Conv_pooing_4(x4)
        x5_o = self.Conv5(x5_t)

        x5 = x5_o + self.Conv_Res_4_5(x5_t)
        x5 = x5 + self.Conv_Res_3_5(x4_t)
        x5 = x5 + self.Conv_Res_2_5(x3_t)
        x5 = x5 + self.Conv_Res_1_5(x2_t)

        # decoding + concat path
        d5 = self.Up5(x5)
        a4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((a4, d5), dim=1)
        d5_t = self.Up_conv5(d5)
        d5 = d5_t + self.UpConv_Res_5_4(x5)

        d4 = self.Up4(d5)
        a3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((a3, d4), dim=1)
        d4_t = self.Up_conv4(d4)
        d4 = d4_t + self.UpConv_Res_4_3(d5)
        d4 = d4 + self.UpConv_Res_5_3(x5)

        d3 = self.Up3(d4)
        a2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((a2, d3), dim=1)
        d3_t = self.Up_conv3(d3)
        d3 = d3_t + self.UpConv_Res_3_2(d4)
        d3 = d3 + self.UpConv_Res_4_2(d5)
        d3 = d3 + self.UpConv_Res_5_2(x5)

        d2 = self.Up2(d3)
        a1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((a1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        i4 = torch.cat((a4, x4), dim=1)
        i3 = torch.cat((a3, x3), dim=1)
        i2 = torch.cat((a2, x2), dim=1)
        i1 = torch.cat((a1, x1), dim=1)

        i4 = self.Conv_identify_get_4(i4)
        i3 = self.Conv_identify_get_3(i3)
        i2 = self.Conv_identify_get_2(i2)
        i1 = self.Conv_identify_get_1(i1)

        i = torch.cat((i1, i2, i3, i4), dim=1)

        o1 = self.Conv_identify_4(i)
        o1 = o1 + self.IdeConv_Res_1_2(i)

        o2 = self.Conv_identify_3(o1)
        # o2 = o2 + self.IdeConv_Res_1_3(i)
        o2 = o2 + self.IdeConv_Res_2_3(o1)

        o3 = self.Conv_identify_2(o2)
        # o3 = o3 + self.IdeConv_Res_1_4(i)
        # o3 = o3 + self.IdeConv_Res_2_4(o1)
        o3 = o3 + self.IdeConv_Res_3_4(o2)

        o4 = self.Linear_identify_1(o3)
        o4 = self.Linear_identify_2(o4)

        o5 = self.Linear_feature(o4)

        o6 = self.Linear_out(o5)
        o6 = o6.squeeze(-1)

        return d1, o6, o5


if __name__ == "__main__":
    net = AttU_Net_plus_with_identify(1, 1, 1)
    print(net)

    inputs = torch.randn(1, 1, 64, 64)
    outputs, res, dis = net(inputs)

    inputs, outputs = inputs.detach().numpy(), outputs.detach().numpy()
    res, dis = res.detach().numpy(), dis.detach().numpy()

    print(outputs.shape, res.shape, dis.shape)
    print(res, "\n", dis)
    plt.subplot(121)
    plt.imshow(inputs[0][0])
    plt.subplot(122)
    plt.imshow(outputs[0][0])
    plt.show()
