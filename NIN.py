"""
Min Lin, Qiang Chen, Shuicheng Yan.

Network In Network

https://arxiv.org/abs/1312.4400
"""

import torch
import torch.nn as nn


def block(in_channel, out_channel, **kwargs):
    block = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, **kwargs),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channel, out_channel, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channel, out_channel, kernel_size=1),
        nn.ReLU(inplace=True)
    )
    return block


class NIN(nn.Module):
    def __init__(self, in_channel, n_classes):
        super(NIN, self).__init__()
        self.layers = nn.Sequential(
            block(in_channel, 96, kernel_size=11, stride=4, padding=0),
            nn.MaxPool2d(3, stride=2),
            block(96, 256, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(3, stride=2),
            block(256, 384, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(0.5),
            block(384, n_classes, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.reset_parameter()

    def reset_parameter(self):
        # 初始化可学习的参数
        for m in self.modules():
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        return self.layers(x)


if __name__ == '__main__':
    net = NIN(in_channel=3, n_classes=10)
    y = net(torch.randn(4, 3, 224, 224))
    print(y.size())  # torch.Size([4, 10])
