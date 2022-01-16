"""
François Chollet.

Xception: Deep Learning with Depthwise Separable Convolutions

https://arxiv.org/abs/1610.02357
"""

import torch.nn as nn
import torch


class SeperableConv(nn.Module):
    def __init__(self, in_channel, out_channel, **kwargs):
        super(SeperableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channel, in_channel, groups=in_channel, bias=False, **kwargs)
        self.pointwise = nn.Conv2d(in_channel, out_channel, 1, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


def residual_block(in_channel, out_channel, relu_first=False, is_pooling=False):
    block = []
    if relu_first:
        block.append(nn.ReLU(inplace=True))
    block.append(SeperableConv(in_channel, out_channel, kernel_size=3, padding=1))
    block.append(nn.BatchNorm2d(out_channel))
    block.append(nn.ReLU(inplace=True))
    block.append(SeperableConv(out_channel, out_channel, kernel_size=3, padding=1))
    block.append(nn.BatchNorm2d(out_channel))
    if is_pooling:
        block.append(nn.MaxPool2d(3, 2, padding=1))
    else:
        block.append(nn.ReLU(inplace=True))
        block.append(SeperableConv(out_channel, out_channel, kernel_size=3, padding=1))
        block.append(nn.BatchNorm2d(out_channel))
    return nn.Sequential(*block)


def shortcut_block(in_channel, out_channel, relu_last=False, **kwargs):
    block = [nn.Conv2d(in_channel, out_channel, **kwargs),
             nn.BatchNorm2d(out_channel)]
    if relu_last:
        block.append(nn.ReLU(inplace=True))
    return nn.Sequential(*block)


class EntryFlow(nn.Module):
    def __init__(self, in_channel):
        super(EntryFlow, self).__init__()
        self.conv1 = shortcut_block(in_channel, 32, relu_last=True, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = shortcut_block(32, 64, relu_last=True, kernel_size=3, padding=1, bias=False)
        self.conv3_residual = residual_block(64, 128, is_pooling=True)
        self.conv3_shortcut = shortcut_block(64, 128, kernel_size=1, stride=2)
        self.conv4_residual = residual_block(128, 256, relu_first=True, is_pooling=True)
        self.conv4_shortcut = shortcut_block(128, 256, kernel_size=1, stride=2)
        self.conv5_residual = residual_block(256, 728, relu_first=True, is_pooling=True)
        self.conv5_shortcut = shortcut_block(256, 728, kernel_size=1, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        residual = self.conv3_residual(x)
        shortcut = self.conv3_shortcut(x)
        x = residual + shortcut
        residual = self.conv4_residual(x)
        shortcut = self.conv4_shortcut(x)
        x = residual + shortcut
        residual = self.conv5_residual(x)
        shortcut = self.conv5_shortcut(x)
        x = residual + shortcut
        return x


class MiddleFlow(nn.Module):
    def __init__(self, in_channel, repeat_times):
        super(MiddleFlow, self).__init__()
        self.conv_residual = []
        for i in range(repeat_times):
            self.conv_residual.append(residual_block(in_channel, in_channel, relu_first=True))
        self.conv_residual = nn.Sequential(*self.conv_residual)

    def forward(self, x):
        return self.conv_residual(x) + x


class ExitFlow(nn.Module):
    def __init__(self, in_channel):
        super(ExitFlow, self).__init__()
        self.conv1_residual = nn.Sequential(
            nn.ReLU(inplace=True),
            SeperableConv(in_channel, in_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(728),
            nn.ReLU(inplace=True),
            SeperableConv(in_channel, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        self.shortcut = shortcut_block(in_channel, 1024, kernel_size=1, stride=2)
        self.conv2 = nn.Sequential(SeperableConv(1024, 1536, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(1536),
                                   nn.ReLU(inplace=True),
                                   SeperableConv(1536, 2048, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(2048),
                                   nn.ReLU(inplace=True))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        residual = self.conv1_residual(x)
        shorcut = self.shortcut(x)
        x = residual + shorcut
        return self.avgpool(self.conv2(x))


class Xception(nn.Module):
    def __init__(self, in_channel, n_classes):
        super(Xception, self).__init__()
        self.entry_flow = EntryFlow(in_channel)
        self.middle_flow = MiddleFlow(728, 8)
        self.exit_flow = ExitFlow(728)
        self.linear = nn.Linear(2048, n_classes)

    def forward(self, x):
        x = self.entry_flow(x)
        x = self.middle_flow(x)
        x = self.exit_flow(x)
        return self.linear(x.view(x.size(0), -1))


if __name__ == '__main__':
    net = Xception(in_channel=3, n_classes=10)
    y = net(torch.randn(4, 3, 299, 299))
    print(y.size())  # torch.Size([4, 10])
