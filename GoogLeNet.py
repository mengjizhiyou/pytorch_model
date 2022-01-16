"""
Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.

Going Deeper with Convolutions

https://arxiv.org/abs/1409.4842v1
"""

import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_channel, out_channel, **kwargs):
        super(Block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, **kwargs),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Inception(nn.Module):
    def __init__(self, in_channel, out_1, red_3, out_3, red_5, out_5, out_pool):
        """
        :param in_channel: 输入通道数
        :param out_1: 1*1卷积的输出通道数
        :param red_3: 3*3卷积的reduce通道数
        :param out_3: 3*3卷积的输出通道数
        """
        super(Inception, self).__init__()
        self.branch1 = Block(in_channel, out_1, kernel_size=1)
        self.branch2 = nn.Sequential(
            Block(in_channel, red_3, kernel_size=1),
            Block(red_3, out_3, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            Block(in_channel, red_5, kernel_size=1),
            Block(red_5, out_5, kernel_size=5, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            Block(in_channel, out_pool, kernel_size=1)
        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)


class GoogleNet(nn.Module):
    def __init__(self, in_channel, n_classes, aux_logits=False, is_training=False):
        super(GoogleNet, self).__init__()
        self.n_classes = n_classes
        self.aux_logits = aux_logits
        self.is_training = is_training
        self.first = nn.Sequential(
            Block(in_channel, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            Block(64, 192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

        )

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        self.last = nn.Sequential(
            nn.AvgPool2d(kernel_size=7, stride=1),
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(1024, self.n_classes)
        )

        if self.aux_logits:
            self.aux1 = InceptionAux(512, n_classes)
            self.aux2 = InceptionAux(528, n_classes)
        else:
            self.aux1 = self.aux2 = None

    def forward(self, x):
        x = self.first(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)

        # Auxiliary Softmax classifier 1
        if self.aux_logits and self.training:
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        # Auxiliary Softmax classifier 2
        if self.aux_logits and self.is_training:
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.last(x)

        if self.aux_logits and self.is_training:
            return aux1, aux2, x

        return x


class InceptionAux(nn.Module):
    def __init__(self, in_channel, n_classes):
        super(InceptionAux, self).__init__()
        self.aux = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=3),
            Block(in_channel, 128, kernel_size=1),
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.7),
            nn.Linear(1024, n_classes)
        )

    def forward(self, x):
        return self.aux(x)


if __name__ == "__main__":
    # N = 3 (Mini batch size)
    x = torch.randn(3, 3, 224, 224)
    model = GoogleNet(in_channel=3, n_classes=10, aux_logits=True, is_training=True)
    print(model(x)[2].size())  # torch.Size([3, 10])
