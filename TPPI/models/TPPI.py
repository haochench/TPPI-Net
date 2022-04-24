"""
Attention:
All the Following TPPI-Nets are designed for patch samples with a size of 5*5*C !!!
"""
import torch
import torch.nn as nn
import numpy as np
import torchsnooper
from TPPI.models.utils import *


# @torchsnooper.snoop()
class CNN_1D_TPPI(nn.Module):
    """
    input shape in training: NCHW-->[N, C=spectral_channel, H=1, W=1]  -->Note: we treat spectral vector as a special HSI cube
    input shape in testing: NCHW-->[N, C=spectral_channel, H=any_size, W=any_size]
    """
    def __init__(self, dataset):
        super(CNN_1D_TPPI, self).__init__()
        self.dataset = dataset
        self.C2 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=6, kernel_size=(3, 1, 1)),
            nn.BatchNorm3d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 1, 1), stride=(2, 1, 1)),
        )
        self.C4 = nn.Sequential(
            nn.Conv3d(in_channels=6, out_channels=12, kernel_size=(3, 1, 1)),
            nn.BatchNorm3d(12),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1)),
        )
        self.C6 = nn.Sequential(
            nn.Conv3d(in_channels=12, out_channels=24, kernel_size=(3, 1, 1)),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1)),
        )
        self.C8 = nn.Sequential(
            nn.Conv3d(in_channels=24, out_channels=48, kernel_size=(3, 1, 1)),
            nn.BatchNorm3d(48),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1)),
        )
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=get_fc_in(dataset, 'CNN_1D'), out_channels=128, kernel_size=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
        )
        self.classifier = nn.Conv2d(in_channels=128, out_channels=get_class_num(dataset), kernel_size=(1, 1))

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        C2 = self.C2(x)
        C4 = self.C4(C2)
        last = self.C6(C4)
        if self.dataset in ("PU", "SV"):
            last = self.C8(last)
        last = torch.reshape(last, (last.shape[0], -1, last.shape[3], last.shape[4]))
        FC = self.fc(last)
        out = self.classifier(FC)
        out = torch.squeeze(out)
        return out


# @torchsnooper.snoop()
class CNN_2D_TPPI_old(nn.Module):
    """
    input shape in training: NCHW-->[N, C=spectral_channel, H=5, W=5]
    input shape in testing: NCHW-->[N, C=spectral_channel, H=any_size, W=any_size]
    """
    def __init__(self, dataset):
        super(CNN_2D_TPPI_old, self).__init__()
        self.dataset = dataset
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=get_in_channel(dataset), out_channels=30, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(30),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=30, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.3),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.3),
        )
        self.avg = nn.AvgPool2d(kernel_size=(5, 5), stride=(1, 1))
        self.classifier = nn.Conv2d(128, get_class_num(dataset), kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        avg = self.avg(conv4)
        out = self.classifier(avg)
        out = torch.squeeze(out)
        return out


# @torchsnooper.snoop()
class CNN_2D_TPPI_new(nn.Module):
    """
        input shape in training: NCHW-->[N, C=spectral_channel, H=5, W=5]
        input shape in testing: NCHW-->[N, C=spectral_channel, H=any_size, W=any_size]
        """
    def __init__(self, dataset):
        super(CNN_2D_TPPI_new, self).__init__()
        self.dataset = dataset
        self.FE = nn.Sequential(
            nn.Conv2d(in_channels=get_in_channel(dataset), out_channels=get_in_planes(dataset), kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(get_in_planes(dataset)),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=get_in_planes(dataset), out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode="replicate"),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode="replicate"),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
        )
        self.avg = nn.AvgPool2d(kernel_size=(5, 5), stride=(1, 1))
        self.classifier = nn.Conv2d(in_channels=128, out_channels=get_class_num(dataset), kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        FE = self.FE(x)
        layer1 = self.layer1(FE)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        avg = self.avg(layer3)
        out = self.classifier(avg)
        out = torch.squeeze(out)
        return out


# @torchsnooper.snoop()
class CNN_3D_TPPI_old(nn.Module):
    """
    input shape in training: NCHW-->[N, C=spectral_channel, H=5, W=5]
    input shape in testing: NCHW-->[N, C=spectral_channel, H=any_size, W=any_size]
    """
    def __init__(self, dataset):
        super(CNN_3D_TPPI_old, self).__init__()
        self.dataset = dataset
        self.conv1_w = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=20, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                      padding=(1, 0, 0)),
            nn.ReLU(inplace=True),
        )
        self.conv2_w = nn.Sequential(
            nn.Conv3d(in_channels=20, out_channels=2, kernel_size=(3, 1, 1), stride=(2, 1, 1),
                      padding=(1, 0, 0)),
            nn.ReLU(inplace=True),
        )
        self.conv3_w = nn.Sequential(
            nn.Conv3d(in_channels=2, out_channels=35, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                      padding=(1, 0, 0)),
            # 3D Conv ....i frozen..,his source code for 8L with 5*5 set padding=(1,1,1) so the input of 5*5 after here is 3*3!!!....
            # i change it to padding=(1, 0, 0) 3*3*C-->1*1*C
            nn.ReLU(inplace=True),
        )
        self.conv4_w = nn.Sequential(
            nn.Conv3d(in_channels=35, out_channels=2, kernel_size=(2, 1, 1), stride=(2, 1, 1),
                      padding=(1, 0, 0)),
            nn.ReLU(inplace=True),
        )
        self.conv5_w = nn.Sequential(
            nn.Conv3d(in_channels=2, out_channels=35, kernel_size=(3, 1, 1), stride=(1, 1, 1),
                      padding=(1, 0, 0)),
            nn.ReLU(inplace=True),
        )
        self.conv6_w = nn.Sequential(
            nn.Conv3d(in_channels=35, out_channels=2, kernel_size=(1, 1, 1), stride=(2, 1, 1),
                      padding=(1, 0, 0)),
            nn.ReLU(inplace=True),
        )
        self.conv7_w = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=35, kernel_size=(3, 1, 1), stride=(1, 1, 1),
                      padding=(1, 0, 0)),
            nn.ReLU(inplace=True),
        )
        self.conv8_w = nn.Conv3d(in_channels=35, out_channels=4, kernel_size=(1, 1, 1), stride=(2, 1, 1),
                                 padding=(0, 0, 0))
        self.classifier = nn.Conv2d(in_channels=get_fc_in(dataset, 'CNN_3D'), out_channels=get_class_num(dataset), kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        x = torch.unsqueeze(x, 1)  # input's shape of the 3DCNN should be:(N,C,D,H,W)
        c1 = self.conv1_w(x)
        c2 = self.conv2_w(c1)
        c3 = self.conv3_w(c2)
        c4 = self.conv4_w(c3)
        c5 = self.conv5_w(c4)
        c6 = self.conv6_w(c5)
        c7 = self.conv7_w(c6)
        c8 = self.conv8_w(c7)
        c8 = torch.reshape(c8, (c8.shape[0], -1, c8.shape[3], c8.shape[4]))
        out = self.classifier(c8)
        out = torch.squeeze(out)
        return out


# @torchsnooper.snoop()
class CNN_3D_TPPI_new(nn.Module):
    """
        input shape in training: NCHW-->[N, C=spectral_channel, H=5, W=5]
        input shape in testing: NCHW-->[N, C=spectral_channel, H=any_size, W=any_size]
        """
    def __init__(self, dataset):
        super(CNN_3D_TPPI_new, self).__init__()
        self.dataset = dataset
        out_channels = [32, 64, 128]
        self.layer1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=out_channels[0], kernel_size=(32, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), padding_mode="replicate"),
            nn.BatchNorm3d(out_channels[0]),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv3d(in_channels=out_channels[0], out_channels=out_channels[1], kernel_size=(32, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), padding_mode="replicate"),
            nn.BatchNorm3d(out_channels[1]),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.5),
        )
        self.layer3 = nn.Sequential(
            nn.Conv3d(in_channels=out_channels[1], out_channels=out_channels[2], kernel_size=(get_CNN3D_new_layer3_channel(dataset), 1, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(out_channels[2]),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.5),
        )
        self.avg = nn.AvgPool2d(kernel_size=(5, 5), stride=(1, 1))
        self.classifier = nn.Conv2d(in_channels=get_fc_in(dataset, 'CNN_3D_new'), out_channels=get_class_num(dataset), kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer3 = torch.squeeze(layer3, dim=2)
        avg = self.avg(layer3)
        out = self.classifier(avg)
        out = torch.squeeze(out)
        return out


# @torchsnooper.snoop()
class HybridSN_TPPI(nn.Module):
    """
    input shape in training: NCHW-->[N, C=spectral_channel, H=5, W=5]
    input shape in testing: NCHW-->[N, C=spectral_channel, H=any_size, W=any_size]
    """
    def __init__(self, dataset):
        super(HybridSN_TPPI, self).__init__()
        self.dataset = dataset
        self.FE = nn.Sequential(
            nn.Conv2d(in_channels=get_in_channel(dataset), out_channels=30, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(30),
        )
        self.conv1 = nn.Sequential(
            # Notice:cause input shape is [N,C,D,H,W]，kernel_size here should be (D,H,W)
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(7, 3, 3), padding=(0, 1, 1), padding_mode="replicate"),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(5, 3, 3), padding=(0, 1, 1), padding_mode="replicate"),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3), padding=(0, 1, 1), padding_mode="replicate"),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=576, out_channels=512, kernel_size=(3, 3), padding=(1, 1), padding_mode="replicate"),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.avg = nn.AvgPool2d(kernel_size=(5, 5), stride=(1, 1))
        self.FC1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.4),
        )
        self.FC2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.4),
        )
        self.classifier = nn.Conv2d(128, get_class_num(dataset), kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        fe = self.FE(x)
        fe = torch.unsqueeze(fe, 1)
        conv1 = self.conv1(fe)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv3 = torch.reshape(conv3, (conv3.shape[0], -1, conv3.shape[3], conv3.shape[4]))  # change shape: [N,C,D,H,W]->[N,C*D,H,W]
        conv4 = self.conv4(conv3)
        fc1 = self.FC1(conv4)
        fc2 = self.FC2(fc1)
        out = self.classifier(fc2)
        out = self.avg(out)
        out = torch.squeeze(out)
        return out


# @torchsnooper.snoop()
class SSAN_TPPI(nn.Module):
    """
    I change the non-local attention module of SSAN to LR(local relation) layer
    Input shape:[N,C=spectral_channel,H=5,W=5]
    """
    def __init__(self, dataset):
        super(SSAN_TPPI, self).__init__()
        self.dataset = dataset
        self.spe1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(7, 1, 1), stride=(1, 1, 1), padding=(3, 0, 0)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.CF1 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=128, kernel_size=(get_in_channel(dataset), 1, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )
        self.spe_AM1 = LocalRelationalLayer(channels=128, k=5, stride=1, padding=2, padding_mode='replicate', m=8)

        self.spa1 = nn.Sequential(
            nn.Conv2d(in_channels=get_SSAN_gate_channel(dataset), out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='replicate'),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.spa_AM1 = LocalRelationalLayer(channels=64, k=5, stride=1, padding=2, padding_mode='replicate', m=8)

        self.spa2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='replicate'),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.spa_AM2 = LocalRelationalLayer(channels=64, k=5, stride=1, padding=2, padding_mode='replicate', m=8)

        self.FC2 = nn.Conv2d(in_channels=64, out_channels=get_class_num(dataset), kernel_size=(1, 1), stride=(1, 1))
        self.avg = nn.AvgPool2d(kernel_size=(5, 5), stride=(1, 1))

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        spe1 = self.spe1(x)
        CF1 = self.CF1(spe1)
        CF1 = torch.squeeze(CF1, dim=2)
        spe_AM1 = self.spe_AM1(CF1)

        spa1 = self.spa1(spe_AM1)
        spa_AM1 = self.spa_AM1(spa1)

        spa2 = self.spa2(spa_AM1)
        spa_AM2 = self.spa_AM2(spa2)

        FC2 = self.FC2(spa_AM2)
        out = self.avg(FC2)
        out = torch.squeeze(out)
        return out


# @torchsnooper.snoop()
class pResNet_TPPI(nn.Module):
    """
    input shape in training: NCHW-->[N, C=spectral_channel, H=5, W=5]
    input shape in testing: NCHW-->[N, C=spectral_channel, H=any_size, W=any_size]
    """
    def __init__(self, dataset):
        super(pResNet_TPPI, self).__init__()
        self.dataset = dataset
        self.in_planes = get_in_planes(dataset)
        # self.spe_layer1 = nn.Sequential(
        #     nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(7, 3, 3), padding=(0, 1, 1)),
        #     nn.BatchNorm3d(8),
        #     nn.ReLU(inplace=True),
        # )
        # self.spe_layer2 = nn.Sequential(
        #     nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(5, 3, 3), padding=(0, 1, 1)),
        #     nn.BatchNorm3d(16),
        #     nn.ReLU(inplace=True),
        # )
        # self.CF = nn.Conv3d(in_channels=16, out_channels=100, kernel_size=(194, 1, 1), stride=(1, 1, 1))
        self.FE = nn.Sequential(
            nn.Conv2d(get_in_channel(dataset), self.in_planes, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(self.in_planes),
        )
        self.layer1 = nn.Sequential(
            Bottleneck_TPPI(self.in_planes, 43),
            Bottleneck_TPPI(43 * 4, 54),
        )
        # self.reduce1 = Bottleneck_TPPI(54*4, 54, reduce=True, downsample=nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1)))
        # self.reduce1 = nn.Conv2d(54*4, 54*4, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        self.reduce1 = Bottleneck_TPPI(54 * 4, 54)
        self.layer2 = nn.Sequential(
            Bottleneck_TPPI(54 * 4, 65),
            Bottleneck_TPPI(65 * 4, 76),
        )
        # self.reduce2 = nn.Conv2d(76 * 4, 76 * 4, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        # self.reduce2 = Bottleneck_TPPI(76*4, 76, reduce=True, downsample=nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1)))
        self.reduce2 = Bottleneck_TPPI(76 * 4, 76)
        self.layer3 = nn.Sequential(
            Bottleneck_TPPI(76 * 4, 87),
            Bottleneck_TPPI(87 * 4, 98),
        )
        self.avgpool = nn.AvgPool2d(5, stride=1)
        self.classifier = nn.Conv2d(in_channels=98 * 4, out_channels=get_class_num(dataset), kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        FE = self.FE(x)
        layer1 = self.layer1(FE)
        reduce1 = self.reduce1(layer1)
        layer2 = self.layer2(reduce1)
        reduce2 = self.reduce2(layer2)
        layer3 = self.layer3(reduce2)
        avg = self.avgpool(layer3)
        out = self.classifier(avg)
        out = torch.squeeze(out)
        return out


# @torchsnooper.snoop()
class SSRN_TPPI(nn.Module):
    """
    Based on paper:Zhong, Z. Spectral-Spatial Residual Network for Hyperspectral Image Classification: A 3-D Deep Learning Framework. TGRS
    Input shape:[N,C=spectral_channel,H=5,W=5]
    """
    def __init__(self, dataset):
        super(SSRN_TPPI, self).__init__()
        self.dataset = dataset
        self.FE1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=24, kernel_size=(7, 1, 1), stride=(2, 1, 1)),
            nn.BatchNorm3d(24),
        )
        self.spe_conv1 = nn.Sequential(
            nn.Conv3d(in_channels=24, out_channels=24, kernel_size=(7, 1, 1), stride=(1, 1, 1), padding=(3, 0, 0)),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=24, out_channels=24, kernel_size=(7, 1, 1), stride=(1, 1, 1), padding=(3, 0, 0)),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True),
        )
        self.spe_conv2 = nn.Sequential(
            nn.Conv3d(in_channels=24, out_channels=24, kernel_size=(7, 1, 1), stride=(1, 1, 1), padding=(3, 0, 0)),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=24, out_channels=24, kernel_size=(7, 1, 1), stride=(1, 1, 1), padding=(3, 0, 0)),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True),
        )
        self.CF = nn.Sequential(
            nn.Conv3d(in_channels=24, out_channels=128, kernel_size=(get_SSRN_channel(dataset), 1, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(128),
        )
        self.FE2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=24, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(24),
        )
        self.spa_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode="replicate"),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.3),
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode="replicate"),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.3),
        )
        # self.spa_conv2 = nn.Sequential(
        #     nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode="replicate"),
        #     nn.BatchNorm2d(24),
        #     nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode="replicate"),
        #     nn.BatchNorm2d(24),
        # )
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1)
        self.classifier = nn.Conv2d(in_channels=24, out_channels=get_class_num(dataset), kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        FE1 = self.FE1(x)
        spe_conv1 = self.spe_conv1(FE1)
        spe_conv1_new = spe_conv1 + FE1
        spe_conv2 = self.spe_conv2(spe_conv1_new)
        spe_conv2_new = spe_conv2 + spe_conv1_new
        CF = self.CF(spe_conv2_new)
        CF = torch.squeeze(CF, dim=2)
        FE2 = self.FE2(CF)
        spa_conv1 = self.spa_conv1(FE2)
        spa_conv1_new = spa_conv1 + FE2
        avg = self.avgpool(spa_conv1_new)
        out = self.classifier(avg)
        out = torch.squeeze(out)
        return out


if __name__ == "__main__":
    """
       open torchsnooper-->test the shape change of each model
       """
    # model = CNN_1D_TPPI('IP')
    # a = np.random.random((2, 200, 1, 1))  # NCHW
    # a = torch.from_numpy(a).float()
    # b = model(a)

    # model = CNN_2D_TPPI('SV')
    # a = np.random.random((2, 204, 200, 200))  # NCHW
    # a = torch.from_numpy(a).float()
    # b = model(a)

    # model = CNN_3D_TPPI('IP')
    # a = np.random.random((2, 200, 5, 5))  # NCHW
    # a = torch.from_numpy(a).float()
    # b = model(a)

    # model = HybridSN_TPPI('SV')
    # a = np.random.random((2, 204, 100, 100))  # NCHW
    # a = torch.from_numpy(a).float()
    # b = model(a)

    # model = SSAN_TPPI('PU')
    # a = np.random.random((2, 103, 30, 30))  # NCHW
    # a = torch.from_numpy(a).float()
    # b = model(a)

    # model = pResNet_TPPI('PU')
    # a = np.random.random((2, 103, 5, 5))  # NCHW
    # a = torch.from_numpy(a).float()
    # a = a.cuda()
    # model = model.cuda()
    # b = model(a)

    # model = SSRN_TPPI('PU')
    # a = np.random.random((2, 103, 5, 5))  # NCHW
    # a = torch.from_numpy(a).float()
    # b = model(a)


