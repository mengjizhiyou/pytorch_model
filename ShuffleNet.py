"""
Xiangyu Zhang, Xinyu Zhou, Mengxiao Lin, Jian Sun.

ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices

https://arxiv.org/abs/1707.01083v2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Shuffle(nn.Module):
    def __init__(self, groups):
        super(Shuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        """[N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]"""
        N, C, H, W = x.size()
        x = x.view(N, self.groups, C // self.groups, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(N, C, H, W)
        return x


def conv_block(in_channel, out_channel, relu_last=True, **kwargs):
    layers = [nn.Conv2d(in_channel, out_channel, bias=False, **kwargs),
              nn.BatchNorm2d(out_channel)]
    if relu_last:
        layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


class ShuffleUnit(nn.Module):
    def __init__(self, in_channel, out_channel, stride, groups):
        super(ShuffleUnit, self).__init__()

        self.stride = stride

        """stage2 未分组逐点卷积，因为in_channel太小"""
        g = 1 if in_channel == 24 else groups
        inner_channel = int(out_channel / 4)

        """Group convolution 将瓶颈的输出通道设置为shuffleunit的输出通道的1/4"""
        self.GConv1 = conv_block(in_channel, inner_channel, kernel_size=1, groups=g)
        self.shuffle = Shuffle(g)

        """Depthwise convolution"""
        self.DWConv = conv_block(inner_channel, inner_channel, kernel_size=3, stride=stride, padding=1,
                                 groups=inner_channel)
        self.GConv2 = conv_block(inner_channel, out_channel, relu_last=False, kernel_size=1, groups=g)
        self.shortcut = nn.Sequential()
        #
        if stride == 2:
            self.shortcut = nn.Sequential(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, x):
        out = self.shuffle(self.GConv1(x))
        out = self.GConv2(self.DWConv(out))
        res = self.shortcut(x)
        out = F.relu(torch.cat([out, res], 1)) if self.stride == 2 else F.relu(out + res)
        return out


class ShuffleNet(nn.Module):
    def __init__(self, in_channel, num_class, num_blocks, groups):
        super(ShuffleNet, self).__init__()

        out_channels = self.out_channels(groups)
        self.in_channel = out_channels[0]
        self.conv1 = conv_block(in_channel, out_channels[0], kernel_size=3, padding=1, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage2 = self._make_stage(out_channels[1], num_blocks[0], groups=groups)
        self.stage3 = self._make_stage(out_channels[2], num_blocks[1], groups=groups)
        self.stage4 = self._make_stage(out_channels[3], num_blocks[2], groups=groups)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(out_channels[-1], num_class)

    def out_channels(self, groups):
        channels = {1: [24, 144, 288, 567],
                    2: [24, 100, 400, 800],
                    3: [24, 240, 480, 960],
                    4: [24, 272, 544, 1088],
                    8: [24, 384, 768, 1536]}
        return channels[groups]

    def _make_stage(self, out_channel, num_block, groups):
        layers = []
        for i in range(num_block):
            """每个stage的第一个卷积块stride=2"""
            stride = 2 if i == 0 else 1
            cat_channel = self.in_channel if i == 0 else 0
            layers.append(ShuffleUnit(self.in_channel, out_channel - cat_channel, stride=stride, groups=groups))
            self.in_channel = out_channel
        return nn.Sequential(*layers)

    def forward(self, x):
        for m in self.children():
            if isinstance(m, nn.Linear):
                x = x.view(x.size(0), -1)
            x = m(x)
        return x


if __name__ == '__main__':
    net = ShuffleNet(3, 10, [4, 8, 4], 3)
    y = net(torch.randn(4, 3, 224, 224))
    print(y.size())  # torch.Size([4, 10])
