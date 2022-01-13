# ImageNet Classification with Deep Convolutional Neural Networks
import torch.nn as nn
import torch

class AlexNet(nn.Module):
    def __init__(self, input_channel, n_classes):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Sequential(
            # transforming (bsize x 1 x 224 x 224) to (bsize x 96 x 54 x 54)
            # From floor((n_h - k_s + p + s)/s), floor((224 - 11 + 3 + 4) / 4) => floor(219/4) => floor(55.5) => 55
            nn.Conv2d(input_channel, 96, kernel_size=11, stride=4, padding=3),  # (batch_size * 96 * 55 * 55)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2))  # (batch_size * 96 * 27 * 27)
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, padding=2),  # (batch_size * 256 * 27 * 27)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2))  # (batch_size * 256 * 13 * 13)
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, padding=1),  # (batch_size * 384 * 13 * 13)
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),  # (batch_size * 384 * 13 * 13)
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # (batch_size * 256 * 13 * 13)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (batch_size * 256 * 6 * 6)
            nn.Flatten())
        self.fc = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),  # (batch_size * 4096)
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),  # (batch_size * 4096)
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, n_classes))  # (batch_size * 10)

        self.conv1.apply(self.init_weights)
        self.conv2.apply(self.init_weights)
        self.conv3.apply(self.init_weights)
        self.fc.apply(self.init_weights)

    def init_weights(self, layer):
        if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.fc(out)
        return out


if __name__ == '__main__':
    data = torch.randn(size=(16, 3, 224, 224))
    model = AlexNet(input_channel=3, n_classes=10)
    classes = model(data)
    print(classes.size())  # (16,10)
