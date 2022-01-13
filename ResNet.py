# Deep Residual Learning for Image Recognition

import torch
import torch.nn as nn
import torch.nn.functional as F


class block(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, is_dash=False):
        super(block, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.is_dash = is_dash  # 虚线部分

        if is_dash:  # 是否对x进行卷积，使其与残差维度一致
            self.identity = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.is_dash:
            x = self.identity(x)
        return F.relu(x + out)


class ResNet(nn.Module):
    def __init__(self, in_channel, num_classes, num_blocks):
        super(ResNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.block2 = self._make_layer(64, 64, num_blocks[0], 1)
        self.block3 = self._make_layer(64, 128, num_blocks[1], 2)
        self.block4 = self._make_layer(128, 256, num_blocks[2], 2)
        self.block5 = self._make_layer(256, 512, num_blocks[3], 2)

        self.block6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes))

    def _make_layer(self, in_channel, out_channel, num_block, stride):
        layers = []

        # 如果将图像维度减半，如 56x56 -> 28x28(stride=2),或通道数改变
        # 则需要调整恒等映射(跳过连接)，这样它就能够与后面的残差卷积块相加

        for i in range(num_block):
            if i == 0 and (stride != 1 or in_channel != out_channel):
                layers.append(block(in_channel, out_channel, stride=2, is_dash=True))
            else:
                layers.append(block(out_channel, out_channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.block1(x)  # (4,64,56,56)
        out = self.block2(out)  # (4,64,56,56)
        out = self.block3(out)  # (4,128,28,28)
        out = self.block4(out)  # (4,256,14,14)
        out = self.block5(out)  # (4,512,7,7)
        out = self.block6(out)  # (4,10)
        return out


def ResNet34(in_channel, num_classes):
    return ResNet(in_channel, num_classes, [3, 4, 6, 3])


if __name__ == '__main__':
    net = ResNet34(in_channel=3, num_classes=10)
    y = net(torch.randn(4, 3, 224, 224))
    print(y.size())  # torch.Size([4, 10])
