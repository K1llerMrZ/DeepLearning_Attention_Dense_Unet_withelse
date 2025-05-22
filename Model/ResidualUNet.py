import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()

        self.W1 = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
        )

        self.W2 = nn.Sequential(
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x1 = self.W1(x)
        out = self.W2(x1)

        return out


class UpDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpDoubleConv, self).__init__()

        self.W1 = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        )

        self.W2 = nn.Sequential(
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x1 = self.W1(x)
        out = self.W2(x1)

        return out


class InputConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InputConv, self).__init__()

        self.W = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return self.W(x)


class Conv1m1(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(Conv1m1, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, padding=0),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class ResConv(nn.Module):
    def __init__(self, in_ch, out_ch, net, stride=1):
        super(ResConv, self).__init__()

        self.c1 = Conv1m1(in_ch, out_ch, stride)
        self.conv = net(in_ch, out_ch)

    def forward(self, x):
        return self.conv(x) + self.c1(x)


class ResidualUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super(ResidualUNet, self).__init__()
        self.upsamp = nn.Upsample(scale_factor=2)

        self.convIn = ResConv(in_ch, 64, InputConv)

        self.conv1 = ResConv(64, 128, DoubleConv, stride=2)
        self.conv2 = ResConv(128, 256, DoubleConv, stride=2)

        self.conv3 = ResConv(256, 512, DoubleConv, stride=2)

        self.upconv1 = ResConv(512 + 256, 256, UpDoubleConv)
        self.upconv2 = ResConv(256 + 128, 128, UpDoubleConv)
        self.upconv3 = ResConv(128 + 64, 64, UpDoubleConv)

        self.convOut = nn.Sequential(
            nn.Conv2d(64, out_ch, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.convIn(x)

        x2 = self.conv1(x1)
        x3 = self.conv2(x2)

        xb = self.conv3(x3)

        u1 = self.upsamp(xb)
        u1 = torch.cat((u1, x3), dim=1)
        u2 = self.upconv1(u1)

        u2 = self.upsamp(u2)
        u2 = torch.cat((u2, x2), dim=1)
        u3 = self.upconv2(u2)

        u3 = self.upsamp(u3)
        u3 = torch.cat((u3, x1), dim=1)
        ut = self.upconv3(u3)

        out = self.convOut(ut)
        return out


if __name__ == "__main__":
    net = ResidualUNet(1, 1)
    print(net)

    inputs = torch.randn(1, 1, 512, 512)
    output = net(inputs)

    print(output)
    print(output.size())

    plt.subplot(121)
    plt.imshow(inputs[0][0].numpy())
    plt.subplot(122)
    plt.imshow(output[0][0].detach().numpy())

    plt.show()
