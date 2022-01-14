"""
Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger.
Densely Connected Convolutional Networks
https://arxiv.org/abs/1608.06993v5
"""

import torch
import torch.nn as  nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, in_channel, k, use_bottleneck=True):
        super(Block, self).__init__()

        if use_bottleneck:
            inner_channel = int(4 * k)
            self.block = nn.Sequential(
                nn.BatchNorm2d(in_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channel, inner_channel, kernel_size=1, bias=False),
                nn.BatchNorm2d(inner_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(inner_channel, k, kernel_size=3, padding=1, bias=False)
            )
        else:
            self.block = nn.Sequential(
                nn.BatchNorm2d(in_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channel, k, kernel_size=3, padding=1, bias=False)
            )

    def forward(self, x):
        return torch.cat((x, self.block(x)), 1)  # 按channel连接


class Transition(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, in_channel, k, num_blocks, theta, n_classes, block, use_bottleneck):
        super(DenseNet, self).__init__()
        self.k = k
        self.n_classes = n_classes
        self.use_bottleneck = use_bottleneck

        inner_channel = int(2 * k)  # 第一个稠密层前的卷积层输出通道为2*k
        # 这里的kernel_size和padding是特征映射维度保持不变
        self.conv1 = nn.Conv2d(3, inner_channel, kernel_size=3, padding=1, bias=False)

        self.features = nn.Sequential()
        for i, num_block in enumerate(num_blocks):
            self.features.add_module(f'dense_block_layer_{i}',
                                     self._make_dense_layer(inner_channel, num_block, block))
            inner_channel += k * num_block
            if i != len(num_blocks) - 1:
                """对图像通道数进行压缩，压缩因子为theta，(0-1]"""
                out_channel = int(inner_channel * theta)
                self.features.add_module(f'transition_layer_{i}', Transition(inner_channel, out_channel))
                inner_channel = out_channel
        self.features.add_module('bn', nn.BatchNorm2d(inner_channel))
        self.features.add_module('relu', nn.ReLU(inplace=True))

        self.linear = nn.Linear(inner_channel, n_classes)
        self.reset_params()

    def reset_params(self):
        # 初始化可学习的参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)  # m.weight.data.fill_(1)
                nn.init.constant_(m.bias, 0)  # m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def _make_dense_layer(self, in_channel, num_block, block):
        dense_block = nn.Sequential()
        for i in range(num_block):
            dense_block.add_module(f'bottleneck_layer_{i}', block(in_channel, self.k, self.use_bottleneck))
            in_channel += self.k
        return dense_block

    def forward(self, x):
        out = self.conv1(x)
        out = self.features(out)
        out = torch.squeeze(F.adaptive_avg_pool2d(out, 1))
        return F.log_softmax(self.linear(out))


def densenet121(use_bottleneck=True):
    return DenseNet(3, 32, [6, 12, 24, 16], 0.5, 10, Block, use_bottleneck)


if __name__ == '__main__':
    net = densenet121()
    y = net(torch.randn(4, 3, 32, 32))
    print(y.size())  # torch.Size([4, 10])
