import torch
import torch.nn as nn
import torch.nn.functional as F

from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_enum(name = "modelSize", default = "small", enum_values = ["small", "medium", "large"], help="")

class NetworkBlock(nn.Module):
    def __init__(self, inDimension, outDimension, dropRate=0.0, batchNorm = False):
        super(NetworkBlock, self).__init__()

        self.batchNorm = batchNorm

        self.conv = nn.Conv2d(inDimension, outDimension, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        if self.batchNorm:
            self.bn = nn.BatchNorm2d(outDimension)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        if self.batchNorm:
            x = self.bn(x)
        return x


class VanillaNet(nn.Module):
    sizesDict = {
        "small"  : [32 , 32 , 64 , 64 , 128],
        "medium" : [64 , 64 , 128, 128, 256],
        "large"  : [256, 256, 512, 512, 1024]
    }
    #@TODO determine inputSize automatically
    def __init__(self, num_classes, inputSize = 32):
        super(VanillaNet, self).__init__()

        self.batchNorm = FLAGS.BN
        self.droprate = FLAGS.dropout

        self.sizes = self.sizesDict[FLAGS.modelSize]

        self.block1 = NetworkBlock(3            , self.sizes[0], dropRate = self.droprate, batchNorm = self.batchNorm)
        self.block2 = NetworkBlock(self.sizes[0], self.sizes[1], dropRate = self.droprate, batchNorm = self.batchNorm)
        self.block3 = NetworkBlock(self.sizes[1], self.sizes[2], dropRate = self.droprate, batchNorm = self.batchNorm)
        self.block4 = NetworkBlock(self.sizes[2], self.sizes[3], dropRate = self.droprate, batchNorm = self.batchNorm)

        self.fc1 = nn.Linear(self.sizes[3] * (inputSize//4)**2, self.sizes[4])
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(self.sizes[4], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)

        x = F.max_pool2d(x,  kernel_size = 2, stride = 2, padding = 0)
        if self.droprate > 0:
            x = F.dropout(x, p=self.droprate, training=self.training)

        x = self.block3(x)
        x = self.block4(x)
        x = F.max_pool2d(x,  kernel_size = 2, stride = 2, padding = 0)

        x = x.flatten(start_dim = 1)
        x = self.fc1(x)
        x = self.relu(x)
        if self.droprate > 0:
            x = F.dropout(x, p=self.droprate, training=self.training)
        
        x = self.fc2(x)
        return x

