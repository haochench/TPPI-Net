import torch
import torch.nn as nn
import numpy as np
import torchsnooper
# from TPPI.models.utils import *
from TPPI.models.utils import *


@torchsnooper.snoop()
class CNN_1D(nn.Module):
    """
    Based on Paper:Chen, Y. Deep Feature Extraction and Classification of Hyperspectral Images Based on Convolutional Neural Networks. TGRS
    input shape:[N,C=1,L=spectral_channel]
    """
    def __init__(self, dataset):
        super(CNN_1D, self).__init__()
        self.dataset = dataset
        self.C2 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=6, kernel_size=(3,)),
            nn.BatchNorm1d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.C4 = nn.Sequential(
            nn.Conv1d(in_channels=6, out_channels=12, kernel_size=(3,)),
            nn.BatchNorm1d(12),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.C6 = nn.Sequential(
            nn.Conv1d(in_channels=12, out_channels=24, kernel_size=(3,)),
            nn.BatchNorm1d(24),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.C8 = nn.Sequential(
            nn.Conv1d(in_channels=24, out_channels=48, kernel_size=(3,)),
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),  # for IP ,p=0.3, else(PU and SV) is 0.1
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(get_fc_in(dataset, 'CNN_1D'), 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
        )
        self.classifier = nn.Linear(128, get_class_num(dataset))

    def forward(self, x):
        x = torch.squeeze(x, dim=2).transpose(1, 2)
        C2 = self.C2(x)
        C4 = self.C4(C2)
        last = self.C6(C4)
        if self.dataset in ("PU", "SV"):
            last = self.C8(last)
        last = torch.reshape(last, (last.shape[0], -1))
        FC = self.fc(last)
        out = self.classifier(FC)
        return out


# @torchsnooper.snoop()
class CNN_2D_old(nn.Module):
    """
    Based on Papers:
    Xiaofei Yang, Hyperspectral image classification with deep learning models. TGRS
    Chen, Y. Deep Feature Extraction and Classification of Hyperspectral Images Based on Convolutional Neural Networks. TGRS
    Input shape:[N,C=spectral_channel,H=5,W=5]
    Specially, the first conv in Paper1 using for expanding the spectral dimension. Paper2 using PCA to reduce the spectral dimension. I using the first conv to reduce spectral dimension.
    """
    def __init__(self, dataset):
        super(CNN_2D_old, self).__init__()
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
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.3),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.3),
        )
        self.classifier = nn.Linear(128, get_class_num(dataset))

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv4 = conv4.view(conv4.size(0), -1)
        out = self.classifier(conv4)
        return out


# @torchsnooper.snoop()
class CNN_2D_new(nn.Module):
    """
    Based on paper: Chen, Y. Deep Feature Extraction and Classification of Hyperspectral Images Based on Convolutional Neural Networks. TGRS
    Input shape:[N,C=spectral_channel,H=5,W=5]
    """
    def __init__(self, dataset):
        super(CNN_2D_new, self).__init__()
        self.dataset = dataset
        self.FE = nn.Sequential(
            nn.Conv2d(in_channels=get_in_channel(dataset), out_channels=get_in_planes(dataset), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(get_in_planes(dataset)),
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=get_in_planes(dataset), out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Dropout2d(p=0.5),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
        )
        self.classifier = nn.Linear(get_fc_in(dataset, 'CNN_2D_new'), get_class_num(dataset))

    def forward(self, x):
        FE = self.FE(x)
        layer1 = self.layer1(FE)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer3 = torch.reshape(layer3, (layer3.shape[0], -1))
        out = self.classifier(layer3)
        return out


# @torchsnooper.snoop()
class CNN_3D_old(nn.Module):
    """
    Based on:Ben Hamida. 3-D deep learning approach for remote sensing image classification. TGRS
    Input shape:[N,C=spectral_channel,H=5,W=5]
    """
    def __init__(self, dataset):
        super(CNN_3D_old, self).__init__()
        self.dataset = dataset
        self.conv1_w = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=20, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 0, 0)),
            nn.ReLU(inplace=True),
        )  # 5*5->3*3
        self.conv2_w = nn.Sequential(
            nn.Conv3d(in_channels=20, out_channels=2, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(inplace=True),
        )  # 3*3->2*2
        self.conv3_w = nn.Sequential(
            nn.Conv3d(in_channels=2, out_channels=35, kernel_size=(3, 1, 1), stride=(1, 1, 1)),
            nn.ReLU(inplace=True),
        )
        self.conv4_w = nn.Sequential(
            nn.Conv3d(in_channels=35, out_channels=2, kernel_size=(2, 1, 1), stride=(2, 2, 2)),  # 2*2->1*1
            nn.ReLU(inplace=True),
        )
        self.conv5_w = nn.Sequential(
            nn.Conv3d(in_channels=2, out_channels=35, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
            nn.ReLU(inplace=True),
        )
        self.conv6_w = nn.Sequential(
            nn.Conv3d(in_channels=35, out_channels=4, kernel_size=(3, 1, 1), stride=(2, 2, 2), padding=(1, 0, 0)),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(get_fc_in(dataset, 'CNN_3D_old'), get_class_num(dataset))

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        c1 = self.conv1_w(x)
        c2 = self.conv2_w(c1)
        c3 = self.conv3_w(c2)
        c4 = self.conv4_w(c3)
        c5 = self.conv5_w(c4)
        c6 = self.conv6_w(c5)
        c6 = torch.reshape(c6, (c6.shape[0], -1))
        out = self.classifier(c6)
        return out


# @torchsnooper.snoop()
class CNN_3D_new(nn.Module):
    """
     Based on: Chen, Y. Deep Feature Extraction and Classification of Hyperspectral Images Based on Convolutional Neural Networks. TGRS
    Input shape:[N,C=spectral_channel,H=5,W=5]
    """
    def __init__(self, dataset):
        super(CNN_3D_new, self).__init__()
        self.dataset = dataset
        out_channels = [32, 64, 128]
        self.layer1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=out_channels[0], kernel_size=(32, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(out_channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )
        self.layer2 = nn.Sequential(
            nn.Conv3d(in_channels=out_channels[0], out_channels=out_channels[1], kernel_size=(32, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(out_channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Dropout3d(p=0.5),

        )
        self.layer3 = nn.Sequential(
            nn.Conv3d(in_channels=out_channels[1], out_channels=out_channels[2], kernel_size=(get_CNN3D_new_layer3_channel(dataset), 1, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(out_channels[2]),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.5),

        )
        self.classifier = nn.Linear(get_fc_in(dataset, 'CNN_3D_new'), get_class_num(dataset))

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer3 = torch.reshape(layer3, (layer3.shape[0], -1))
        out = self.classifier(layer3)
        return out


# @torchsnooper.snoop()
class HybridSN(nn.Module):
    """
    Based on paper:HybridSN: Exploring 3-D-2-D CNN Feature Hierarchy for Hyperspectral Image Classification. IEEE Geoscience and Remote Sensing Letters
    Input shape:[N,C=spectral_channel,H=5,W=5]
    """
    def __init__(self, dataset):
        super(HybridSN, self).__init__()
        self.dataset = dataset
        self.FE = nn.Sequential(
            nn.Conv2d(in_channels=get_in_channel(dataset), out_channels=30, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(30),
        )
        self.conv1 = nn.Sequential(
            # Notice:cause input shape is [N,C,D,H,W]，kernel_size here should be (D,H,W)
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(7, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(5, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=576, out_channels=64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.FC1 = nn.Sequential(
            nn.Linear(get_fc_in(dataset, 'HybridSN'), 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
        )
        self.FC2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
        )
        self.classifier = nn.Linear(128, get_class_num(dataset))

    def forward(self, x):
        fe = self.FE(x)
        fe = torch.unsqueeze(fe, 1)
        conv1 = self.conv1(fe)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv3 = torch.reshape(conv3, (
        conv3.shape[0], -1, conv3.shape[3], conv3.shape[4]))
        conv4 = self.conv4(conv3)
        conv4 = torch.reshape(conv4, (conv4.shape[0], -1))
        fc1 = self.FC1(conv4)
        fc2 = self.FC2(fc1)
        out = self.classifier(fc2)
        return out


# @torchsnooper.snoop()
class SSAN(nn.Module):
    """
    Based on paper: Sun, H. et.al,Spectral-Spatial Attention Network for Hyperspectral Image Classification. TGRS
    Input shape:[N,C=spectral_channel,H=5,W=5]
    """
    def __init__(self, dataset):
        super(SSAN, self).__init__()
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
        self.spe_AM1 = Attention_gate(gate_channels=get_SSAN_gate_channel(dataset), gate_depth=64)

        self.spa1 = nn.Sequential(
            nn.Conv2d(in_channels=get_SSAN_gate_channel(dataset), out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.spa_AM1 = Attention_gate(gate_channels=64, gate_depth=64)

        self.spa2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.spa_AM2 = Attention_gate(gate_channels=64, gate_depth=64)

        self.FC1 = nn.Sequential(
            nn.Linear(get_fc_in(dataset, 'SSAN'), 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )
        self.FC2 = nn.Linear(256, get_class_num(dataset))

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        spe1 = self.spe1(x)
        CF1 = self.CF1(spe1)
        CF1 = torch.squeeze(CF1)
        spe_AM1 = self.spe_AM1(CF1)

        spa1 = self.spa1(spe_AM1)
        spa_AM1 = self.spa_AM1(spa1)

        spa2 = self.spa2(spa_AM1)
        spa_AM2 = self.spa_AM2(spa2)

        spa_AM2 = torch.reshape(spa_AM2, (spa_AM2.shape[0], -1))
        FC1 = self.FC1(spa_AM2)
        out = self.FC2(FC1)
        return out


# @torchsnooper.snoop()
class pResNet(nn.Module):
    """
    Based on paper:Paoletti. Deep pyramidal residual networks for spectral-spatial hyperspectral image classification. TGRS
    Input shape:[N,C=spectral_channel,H=5,W=5]
    In source code, each layer have 3 bottlenecks, i change to 2 bottlenecks each layer, but still with 3 layer
    """
    def __init__(self, dataset):
        super(pResNet, self).__init__()
        self.dataset = dataset
        self.in_planes = get_in_planes(dataset)
        self.FE = nn.Sequential(
            nn.Conv2d(get_in_channel(dataset), self.in_planes, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(self.in_planes),
        )
        self.layer1 = nn.Sequential(
            Bottleneck_TPPP(self.in_planes, 43),
            Bottleneck_TPPP(43*4, 54),
        )
        self.reduce1 = Bottleneck_TPPP(54 * 4, 54, stride=2, downsample=nn.AvgPool2d((2, 2), (2, 2)))
        self.layer2 = nn.Sequential(
            Bottleneck_TPPP(54*4, 65),
            Bottleneck_TPPP(65*4, 76),
        )
        self.reduce2 = Bottleneck_TPPP(76*4, 76, stride=2, downsample=nn.AvgPool2d((2, 2), (2, 2)))
        self.layer3 = nn.Sequential(
            Bottleneck_TPPP(76*4, 87),
            Bottleneck_TPPP(87*4, 98),
        )
        self.avgpool = nn.AvgPool2d(get_avgpoosize(dataset))
        self.classifier = nn.Linear(98*4, get_class_num(dataset))

    def forward(self, x):
        FE = self.FE(x)  # 降维
        layer1 = self.layer1(FE)
        reduce1 = self.reduce1(layer1)
        layer2 = self.layer2(reduce1)
        reduce2 = self.reduce2(layer2)
        layer3 = self.layer3(reduce2)
        avg = self.avgpool(layer3)
        avg = avg.view(avg.size(0), -1)
        out = self.classifier(avg)
        return out


# @torchsnooper.snoop()
class SSRN(nn.Module):
    """
    Based on paper:Zhong, Z. Spectral-Spatial Residual Network for Hyperspectral Image Classification: A 3-D Deep Learning Framework. TGRS
    Input shape:[N,C=spectral_channel,H=5,W=5]
    """
    def __init__(self, dataset):
        super(SSRN, self).__init__()
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
        )  # 5*5*128->3*3*24
        self.spa_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.3),
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.3),
        )
        self.avgpool = nn.AvgPool2d(kernel_size=3)
        self.classifier = nn.Linear(24, get_class_num(dataset))

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
        # spa_conv2 = self.spa_conv2(spa_conv1_new)
        # spa_conv2_new = spa_conv2 + spa_conv1_new
        avg = self.avgpool(spa_conv1_new)
        avg = torch.squeeze(avg)
        out = self.classifier(avg)
        return out


if __name__ == "__main__":
    """
    open torchsnooper-->test the shape change of each model
    """
    model = CNN_1D('IP')
    a = np.random.random((1, 200, 1, 1))  # NCL
    a = torch.from_numpy(a).float()
    b = model(a)

    # model = CNN_2D('IP')
    # a = np.random.random((2, 200, 5, 5))  # NCHW
    # a = torch.from_numpy(a).float()
    # b = model(a)

    # model = CNN_3D('SV')
    # a = np.random.random((2, 204, 5, 5))  # NCHW
    # a = torch.from_numpy(a).float()
    # b = model(a)

    # model = HybridSN('IP')
    # a = np.random.random((2, 200, 5, 5))  # NCHW
    # a = torch.from_numpy(a).float()
    # b = model(a)

    # model = SSAN('IP')
    # a = np.random.random((2, 200, 5, 5))  # NCHW
    # a = torch.from_numpy(a).float()
    # b = model(a)

    # model = pResNet('IP')
    # a = np.random.random((2, 200, 5, 5))  # NCHW
    # a = torch.from_numpy(a).float()
    # a = a.cuda()
    # model = model.cuda()
    # b = model(a)

    # model = SSRN('SV')
    # a = np.random.random((2, 204, 5, 5))  # NCHW
    # a = torch.from_numpy(a).float()
    # b = model(a)

