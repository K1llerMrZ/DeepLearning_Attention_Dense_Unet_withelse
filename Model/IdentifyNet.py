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


class pool_conv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=2, stride=2):
        super(pool_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


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


class linear_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(linear_block, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(ch_in, ch_out, bias=True),
            nn.BatchNorm1d(ch_out),
            nn.ReLU(inplace=True),
            nn.Linear(ch_out, ch_out, bias=True),
            nn.BatchNorm1d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.linear(x)





class IDentifyNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, scale_factor=1):
        super(IDentifyNet, self).__init__()

        filters = np.array([16, 32, 64, 128, 256, 512])
        filters = filters // scale_factor
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.scale_factor = scale_factor

        self.Conv1 = conv_block(ch_in=n_channels, ch_out=filters[0])
        self.Conv2 = conv_block(ch_in=filters[0] * 2, ch_out=filters[1])
        self.Conv3 = conv_block(ch_in=filters[1] * 2, ch_out=filters[2])
        self.Conv4 = conv_block(ch_in=filters[2] * 2, ch_out=filters[3])
        self.Conv5 = conv_block(ch_in=filters[3], ch_out=filters[4])
        self.Conv6 = conv_block(ch_in=filters[4], ch_out=filters[5])

        self.Conv_pooing_1 = pool_conv(filters[0], filters[0], kernel_size=2, stride=2)
        self.Conv_pooing_2 = pool_conv(filters[1], filters[1], kernel_size=2, stride=2)
        self.Conv_pooing_3 = pool_conv(filters[2], filters[2], kernel_size=2, stride=2)
        self.Conv_pooing_4 = pool_conv(filters[3], filters[3], kernel_size=2, stride=2)
        self.Conv_pooing_5 = pool_conv(filters[4], filters[4], kernel_size=2, stride=2)

        self.Conv1_mask = conv_block(ch_in=n_channels, ch_out=filters[0])
        self.Conv2_mask = conv_block(ch_in=filters[0], ch_out=filters[1])
        self.Conv3_mask = conv_block(ch_in=filters[1], ch_out=filters[2])

        self.Att1 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=filters[1])
        self.Att2 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[2])
        self.Att3 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[3])

        self.Flatten = nn.Flatten()
        self.Linear1 = linear_block(filters[5] * 2 ** 2, 1024)
        self.Linear2 = linear_block(1024, 16)
        # self.Linear3 = linear_block(128, 16)

        self.Linear_out = linear_block(16, 5)
        self.Softmax = nn.Softmax(dim=1)

        self.Conv_Res_1_2 = nn.Conv2d(filters[0], filters[1], kernel_size=2, stride=2, padding=0)
        self.Conv_Res_2_3 = nn.Conv2d(filters[1], filters[2], kernel_size=2, stride=2, padding=0)
        self.Conv_Res_3_4 = nn.Conv2d(filters[2], filters[3], kernel_size=2, stride=2, padding=0)
        self.Conv_Res_4_5 = nn.Conv2d(filters[3], filters[4], kernel_size=2, stride=2, padding=0)
        self.Conv_Res_5_6 = nn.Conv2d(filters[4], filters[5], kernel_size=2, stride=2, padding=0)

        self.Conv_Res_1_3 = nn.Conv2d(filters[0], filters[2], kernel_size=4, stride=4, padding=0)
        self.Conv_Res_2_4 = nn.Conv2d(filters[1], filters[3], kernel_size=4, stride=4, padding=0)
        self.Conv_Res_3_5 = nn.Conv2d(filters[2], filters[4], kernel_size=4, stride=4, padding=0)
        self.Conv_Res_4_6 = nn.Conv2d(filters[3], filters[5], kernel_size=4, stride=4, padding=0)

        self.Conv_Res_1_4 = nn.Conv2d(filters[0], filters[3], kernel_size=8, stride=8, padding=0)
        self.Conv_Res_2_5 = nn.Conv2d(filters[1], filters[4], kernel_size=8, stride=8, padding=0)
        self.Conv_Res_3_6 = nn.Conv2d(filters[2], filters[5], kernel_size=8, stride=8, padding=0)

        self.Conv_Res_1_5 = nn.Conv2d(filters[0], filters[4], kernel_size=16, stride=16, padding=0)
        self.Conv_Res_2_6 = nn.Conv2d(filters[1], filters[5], kernel_size=16, stride=16, padding=0)

        self.Conv_Res_1_6 = nn.Conv2d(filters[0], filters[5], kernel_size=32, stride=32, padding=0)


    def forward(self, x, m):
        x1_o = self.Conv1(x)
        x1 = self.Conv_pooing_1(x1_o)
        m1 = self.Conv1_mask(m)
        m1 = self.Conv_pooing_1(m1)

        a1 = self.Att1(x1, m1)
        x2 = torch.cat((x1, a1), dim=1)
        x2_o = self.Conv2(x2)
        m2 = self.Conv2_mask(m1)
        x2 = x2_o + self.Conv_Res_1_2(x1_o)
        x2 = self.Conv_pooing_2(x2)
        m2 = self.Conv_pooing_2(m2)

        a2 = self.Att2(x2, m2)
        x3 = torch.cat((x2, a2), dim=1)
        x3_o = self.Conv3(x3)
        m3 = self.Conv3_mask(m2)
        x3 = x3_o + self.Conv_Res_2_3(x2_o)
        x3 = x3 + self.Conv_Res_1_3(x1_o)
        x3 = self.Conv_pooing_3(x3)
        m3 = self.Conv_pooing_3(m3)

        a3 = self.Att3(x3, m3)
        x3 = torch.cat((x3, a3), dim=1)
        x4_o = self.Conv4(x3)
        x4 = x4_o + self.Conv_Res_3_4(x3_o)
        x4 = x4 + self.Conv_Res_2_4(x2_o)
        x4 = x4 + self.Conv_Res_1_4(x1_o)
        x4 = self.Conv_pooing_4(x4)

        x5_o = self.Conv5(x4)
        x5 = x5_o + self.Conv_Res_4_5(x4_o)
        x5 = x5 + self.Conv_Res_3_5(x3_o)
        x5 = x5 + self.Conv_Res_2_5(x2_o)
        x5 = x5 + self.Conv_Res_1_5(x1_o)
        x5 = self.Conv_pooing_5(x5)

        x6_o = self.Conv6(x5)
        x6 = x6_o + self.Conv_Res_5_6(x5_o)
        x6 = x6 + self.Conv_Res_4_6(x4_o)
        x6 = x6 + self.Conv_Res_3_6(x3_o)
        x6 = x6 + self.Conv_Res_2_6(x2_o)
        x6 = x6 + self.Conv_Res_1_6(x1_o)

        x7 = self.Flatten(x6)
        x7 = self.Linear1(x7)

        x8 = self.Linear2(x7)

        out = self.Linear_out(x8)

        return out, x8

if __name__ == "__main__":
    net = IDentifyNet(1, 1, 1)
    print(net)

    inputs = torch.randn(1, 1, 64, 64)
    tag = torch.randn(1, 1, 64, 64)

    output, o = net(inputs, tag)

    inputs, tag = inputs.detach().numpy(), tag.detach().numpy()

    print(tag.shape)
    plt.subplot(121)
    plt.imshow(inputs[0][0])
    plt.subplot(122)
    plt.imshow(tag[0][0])
    plt.show()