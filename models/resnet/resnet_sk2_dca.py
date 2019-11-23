import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import torch
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
from collections import OrderedDict
import math

#TODO sknet for basicblocks

__all__ = ['dca_sk2_resnet18', 'dca_sk2_resnet34', 'dca_sk2_resnet50', 'dca_sk2_resnet101',
           'dca_sk2_resnet152']


def conv3x3(in_planes, out_planes, stride=1, groups=1,dilation=1,padding =1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, bias=False, groups=groups,dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2g = conv3x3(planes, planes, stride, dilation=2,padding=2)
        self.bn2g   = nn.BatchNorm2d(planes)

        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_fc1 = nn.Conv2d(planes, planes//16, 1, bias=False)
        self.bn_fc1   = nn.BatchNorm2d(planes//16)
        self.conv_fc2 = nn.Conv2d(planes//16, 2 * planes, 1, bias=False)

        self.D = planes

        # connected attention
        if inplanes == planes*2 :
            self.att_fc_1 = nn.Sequential(
                nn.Linear(planes//2, planes ),
                nn.LayerNorm(planes),
                nn.ReLU(inplace=True)
            )
            self.att_fc_2 = nn.Sequential(
                nn.Linear(planes // 2, planes),
                nn.LayerNorm(planes),
                nn.ReLU(inplace=True)
            )
        self.att_conv1 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1),
            nn.LayerNorm(planes),
            nn.ReLU(inplace=True)
        )
        self.att_conv2 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1),
            nn.LayerNorm(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        x= inputs[0]
        pre_att = inputs[1]
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        d1 = self.conv2(out)
        d1 = self.bn2(d1)
        d1 = self.relu(d1)

        d2 = self.conv2g(out)
        d2 = self.bn2g(d2)
        d2 = self.relu(d2)

        gap1 = self.avg_pool(d1)
        gap2 = self.avg_pool(d2)
        b, c, _, _ = d1.size()
        if pre_att is not None:
            pre_gap1 = pre_att[:, :, :, 0].squeeze(-1).squeeze(-1)
            pre_gap2 = pre_att[:, :, :, 1].squeeze(-1).squeeze(-1)
            if hasattr(self, 'att_fc_1'):
                pre_gap1 = self.att_fc_1(pre_gap1)
                pre_gap2 = self.att_fc_2(pre_gap2)
            gap1 = torch.cat((gap1.view(b, 1, 1, c), pre_gap1.view(b, 1, 1, c)), dim=1)
            gap2 = torch.cat((gap2.view(b, 1, 1, c), pre_gap2.view(b, 1, 1, c)), dim=1)
            gap1 = self.att_conv1(gap1).view(b, c, 1, 1)
            gap2 = self.att_conv2(gap2).view(b, c, 1, 1)

        d  = gap1 + gap2

        d = F.relu(self.bn_fc1(self.conv_fc1(d)))
        d = self.conv_fc2(d)
        d = torch.unsqueeze(d, 1).view(-1, 2, self.D, 1, 1)
        d = F.softmax(d, 1)
        d1 = d1 * d[:, 0, :, :, :].squeeze(1)
        d2 = d2 * d[:, 1, :, :, :].squeeze(1)
        d  = d1 + d2

        out = self.conv3(d)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        att = torch.cat((self.avg_pool(d1),self.avg_pool(d2)),-1)
        return {0:out,1:att}


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        att = None
        x = {0: x, 1: att}
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x[0])
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def dca_sk2_resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def dca_sk2_resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def dca_sk2_resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def dca_sk2_resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model

def dca_sk2_resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def demo():
    net = dca_sk2_resnet50(num_classes=1000)
    y = net(torch.randn(10, 3, 224,224))
    print(y.size())

# demo()
