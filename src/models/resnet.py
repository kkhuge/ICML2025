import torch
import torch.nn as nn
import torch.nn.functional as F



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.downsample = downsample


    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)

        out += residual
        return out



class ResNet(nn.Module):
    def __init__(self, block, k=None, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)

        self.layer1 = self._make_layer(block, 16 * k, 1)
        self.layer2 = self._make_layer(block, 16 * k, 1, stride=2)
        self.layer3 = self._make_layer(block, 16 * k, 1, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16 * k, num_classes)


    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def resnet_1(num_classes=10):
    return ResNet(ResidualBlock, k=1, num_classes=num_classes)

def resnet_2(num_classes=10):
    return ResNet(ResidualBlock, k=2, num_classes=num_classes)

def resnet_4(num_classes=10):
    return ResNet(ResidualBlock, k=4, num_classes=num_classes)

def resnet_8(num_classes=10):
    return ResNet(ResidualBlock, k=8, num_classes=num_classes)

def resnet_16(num_classes=10):
    return ResNet(ResidualBlock, k=16, num_classes=num_classes)

def resnet_32(num_classes=10):
    return ResNet(ResidualBlock, k=32, num_classes=num_classes)

def resnet_64(num_classes=10):
    return ResNet(ResidualBlock, k=64, num_classes=num_classes)

