import torch.nn as nn
import torch
import torch.nn.functional as F
from models.dropblock import DropBlock
from clock_driven import neuron, layer, functional


def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, 3, padding=1, bias=False)


def conv1x1(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, 1, bias=False)


def norm_layer(planes):
    return nn.BatchNorm2d(planes)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, downsample, T, v_threshold=1.0, v_reset=0.0):
        super(BasicBlock, self).__init__()
        self.T = T

        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.neuron1 = neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.neuron2 = neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset)

        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.neuron3 = neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset)

        self.maxpool = nn.MaxPool2d(2)
        self.downsample = downsample

    def forward(self, x):

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.neuron1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.neuron2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.neuron3(out)
        out = self.maxpool(out)

        return out


class ResNet(nn.Module):

    def __init__(self, channels, tau=2.0, T=16, v_threshold=1.0, v_reset=0.0,
                 img_channel=3, img_size=80):
        super().__init__()

        self.inplanes = 3
        self.T = T
        self.conv1 = nn.Conv2d(img_channel, self.inplanes, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.neuron = neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset) #######################################

        self.layer1 = self._make_layer(channels[0])
        self.layer2 = self._make_layer(channels[1])
        self.layer3 = self._make_layer(channels[2])
        self.layer4 = self._make_layer(channels[3])

        self.out_dim = self.calculate_out_shape(img_channel, img_size)
        self.neuron_fc = neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def calculate_out_shape(self, img_channel, img_size):
        input = torch.randn(1, img_channel, img_size, img_size)
        x = self.conv1(input)  # 输入层
        x = self.bn1(x)
        out_x = self.neuron(x)  # 编码
        out1 = self.layer1(out_x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        output = self.layer4(out3)
        print(f"The input image shape is: {img_channel, img_size, img_size}, the output shape is: {output.shape}")
        functional.reset_net(self)
        return output.reshape(1, -1).shape[-1]

    def _make_layer(self, planes):
        downsample = nn.Sequential(
            conv1x1(self.inplanes, planes),
            norm_layer(planes),
        )
        block = BasicBlock(self.inplanes, planes, downsample, self.T)
        self.inplanes = planes
        return block

    def forward(self, x):
        # print('neuron:', x.shape)
        x = self.conv1(x)  # 输入层
        x = self.bn1(x)
        out_x = self.neuron(x)  # 编码

        out1 = self.layer1(out_x)

        out2 = self.layer2(out1)

        out3 = self.layer3(out2)

        out4 = self.layer4(out3)

        # out = out4.view(out4.shape[0], out4.shape[1], -1).mean(dim=2)

        out_spikes_counter = out4  # 全连接层结果

        for t in range(1, self.T):
            out_x = self.neuron(x)  # 重新编码

            out1 = self.layer1(out_x)
            out2 = self.layer2(out1)
            out3 = self.layer3(out2)
            out4 = self.layer4(out3)

            # out = out4.view(out4.shape[0], out4.shape[1], -1).mean(dim=2)
            out_spikes_counter += out4   # 全连接层结果， 用于返回分类信息

        return out_spikes_counter/self.T, None


def resnet12(img_channel=3, img_size=80, **kwargs):
    """Constructs a ResNet-12 model.
    """
    model = ResNet([64, 128, 256, 512], img_channel=int(img_channel), img_size=int(img_size))
    return model
