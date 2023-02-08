# import torch
# from torch import nn
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms, models
# from torch.nn import functional as F
# import torchvision.models as models
#
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print('Using {} device'.format(device))
#
# class ResBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu1 = nn.ReLU()
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.relu2 = nn.ReLU()
#
#         if stride != 1 or in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
#                 nn.BatchNorm2d(out_channels)
#             )
#         else:
#             self.shortcut = lambda x: x
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu1(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu2(out)
#         out += self.shortcut(x)
#         out = self.relu2(out)
#         return out
#
# class ResNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu1 = nn.ReLU()
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.resblock1 = ResBlock(64, 64, 1)
#         self.resblock2 = ResBlock(64, 128, 2)
#         self.resblock3 = ResBlock(128, 256, 2)
#         self.resblock4 = ResBlock(256, 512, 2)
#         self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.flatten = nn.Flatten()
#         self.fc = nn.Linear(512, 2)
#         self.sigmoid = nn.Sigmoid()
#         # self.double()
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu1(out)
#         out = self.maxpool(out)
#         out = self.resblock1(out)
#         out = self.resblock2(out)
#         out = self.resblock3(out)
#         out = self.resblock4(out)
#         out = self.global_avg_pool(out)
#         out = self.flatten(out)
#         out = self.fc(out)
#         out = self.sigmoid(out)
#         return out
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.nn import functional as F
import torchvision.models as models

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))


class ResBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, stride):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # For downsampling
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):

        output = self.conv1(x)
        output = F.relu(self.bn1(output))

        output = self.conv2(output)
        output = self.bn2(output)

        if self.in_channels != self.out_channels:
            # For downsampling
            res = self.conv3(x)
            res = self.bn3(res)

            output += res
        else:
            output += x

        output = F.relu(output)

        return output


class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pooling = nn.MaxPool2d(kernel_size=3, stride=2)

        self.block1 = ResBlock(64, 64, 1)
        self.block2 = ResBlock(64, 128, 2)
        self.block3 = ResBlock(128, 256, 2)
        self.block4 = ResBlock(256, 512, 2)

        self.global_avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pooling(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.global_avg_pooling(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.sigmoid(x)

        return x

