"""
Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko,
 Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam.

MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications

https://arxiv.org/abs/1704.04861
"""

import torch
import torch.nn as nn


class SeparableConv(nn.Module):
    def __init__(self, in_channel, out_channel, alpha, down_sample=False):
        super(SeparableConv, self).__init__()

        stride = 2 if down_sample else 1
        in_channel = int(in_channel * alpha)
        out_channel = int(out_channel * alpha)

        self.dw = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, stride, 1, groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True))

        self.pw = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.pw(self.dw(x))


def conv_block(in_channel, out_channel, alpha, stride):
    return nn.Sequential(nn.Conv2d(in_channel, int(out_channel * alpha), 3, stride, 1, bias=False),
                         nn.BatchNorm2d(out_channel),
                         nn.ReLU(inplace=True))


class MobileNet(nn.Module):
    def __init__(self, in_channel, n_classes, alpha=1):
        super(MobileNet, self).__init__()

        if alpha not in [0.25, 0.5, 0.75, 1.0]:
            raise ValueError('alpha can be one of 0.25,0.5,0.75 or 1.0 only')

        self.model = nn.Sequential(
            conv_block(in_channel, 32, alpha, 2),
            SeparableConv(32, 64, alpha),
            SeparableConv(64, 128, alpha, down_sample=True),
            SeparableConv(128, 128, alpha),
            SeparableConv(128, 256, alpha, down_sample=True),
            SeparableConv(256, 256, alpha),
            SeparableConv(256, 512, alpha, down_sample=True),
            SeparableConv(512, 512, alpha),
            SeparableConv(512, 512, alpha),
            SeparableConv(512, 512, alpha),
            SeparableConv(512, 512, alpha),
            SeparableConv(512, 512, alpha),
            SeparableConv(512, 1024, alpha, down_sample=True),
            SeparableConv(1024, 1024, alpha),
            nn.AvgPool2d(7)
        )

        self.linear = nn.Linear(int(1024 * alpha), n_classes)

        self.model.apply(self.init_weights)
        self.linear.apply(self.init_weights)

    def init_weights(self, layer):
        if type(layer) == nn.Conv2d:
            nn.init.kaiming_normal_(layer.weight, mode='fan_out')
        if type(layer) == nn.Linear:
            nn.init.normal_(layer.weight, std=1e-3)
        if type(layer) == nn.BatchNorm2d:
            nn.init.normal_(layer.weight, 1)
            nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        x = self.model(x)
        x = self.linear(x.view(x.size(0), -1))
        return x


if __name__ == '__main__':
    net = MobileNet(3, 10)
    y = net(torch.randn(4, 3, 224, 224))
    print(y.size())  # torch.Size([4, 10])
