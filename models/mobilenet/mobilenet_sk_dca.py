from torch import nn
import torch
import torch.nn.functional as F


__all__ = ['dca_sk_mobilenet_v2']

class SKLayer(nn.Module):
    def __init__(self,pre_planes,in_planes, out_planes, kernel_size=3, stride=1, groups=1, reduction = 8):
        super(SKLayer, self).__init__()
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size, stride, 1, groups=groups, bias=False)
        self.conv5 = nn.Conv2d(in_planes, out_planes, kernel_size, stride, 2, groups=groups, bias=False,dilation=2)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.bn5 = nn.BatchNorm2d(out_planes)
        self.active =  nn.ReLU6(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_fc1 = nn.Conv2d(in_planes, out_planes // reduction, 1, bias=False)
        self.bn_fc1 = nn.BatchNorm2d(out_planes // reduction)
        self.conv_fc2 = nn.Conv2d(out_planes // reduction, 2 * out_planes, 1, bias=False)
        self.D = out_planes
        if pre_planes != in_planes:
            self.att_fc1 = nn.Sequential(
                nn.Linear(pre_planes, in_planes // reduction),
                nn.LayerNorm(in_planes // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(in_planes // reduction, in_planes),
                nn.LayerNorm(in_planes),
                nn.ReLU(inplace=True)
            )
            self.att_fc2 = nn.Sequential(
                nn.Linear(pre_planes, in_planes // reduction),
                nn.LayerNorm(in_planes // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(in_planes // reduction, in_planes),
                nn.LayerNorm(in_planes),
                nn.ReLU(inplace=True)
            )
        self.att_conv1 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1),
            nn.LayerNorm(in_planes),
            nn.ReLU(inplace=True)
        )
        self.att_conv2 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1),
            nn.LayerNorm(in_planes),
            nn.ReLU(inplace=True)
        )



    def forward(self, input):
        x = input[0]
        n,c,_,_ = x.size()

        d1 = self.conv3(x)
        d1 = self.bn3(d1)
        d1 = self.active(d1)

        d2 = self.conv5(x)
        d2 = self.bn5(d2)
        d2 = self.active(d2)

        if input[1] is None:
            gap1 = self.avg_pool(d1)
            gap2 = self.avg_pool(d2)
        else:
            pre_att = input[1].squeeze(-1).squeeze(-1)
            pre_att1 = pre_att[:, 0]
            pre_att2 = pre_att[:, 1]
            if hasattr(self, 'att_fc1'):
                pre_att1 = self.att_fc1(pre_att1)
                pre_att2 = self.att_fc2(pre_att2)

            gap1 = torch.cat((self.avg_pool(d1).view(n,1,1,c),pre_att1.view(n,1,1,c)),dim=1)
            gap2 = torch.cat((self.avg_pool(d2).view(n,1,1,c), pre_att2.view(n,1,1,c)), dim=1)
            gap1 = self.att_conv1(gap1)
            gap2 = self.att_conv2(gap2)

        d = (gap1+gap2).view(n,c,1,1)
        d = F.relu(self.bn_fc1(self.conv_fc1(d)))
        d = self.conv_fc2(d)
        d = torch.unsqueeze(d, 1).view(-1, 2, self.D, 1, 1)
        d = F.softmax(d, 1)
        att= d
        d1 = d1 * d[:, 0, :, :, :].squeeze(1)
        d2 = d2 * d[:, 1, :, :, :].squeeze(1)
        d = d1 + d2
        return {0:d,1:att}


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self,pre_hidden_dim, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        # print("{}-{}-{}".format( inp, oup,pre_hidden_dim))

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio != 1:
            # pw
            self.layer1 = ConvBNReLU(inp, hidden_dim, kernel_size=1)

        # dw
        # ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
        self.layersk = SKLayer(pre_hidden_dim,hidden_dim, hidden_dim, stride=stride, groups=hidden_dim)
        # pw-linear
        self.layer3 = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
        )

    def forward(self, input):
        x=input[0]
        pre_att = input[1]
        out = self.layer1(x) if hasattr(self, 'layer1') else x
        out = self.layersk({0:out,1:pre_att})
        if self.use_res_connect:
            return {0:x + self.layer3(out[0]),1:out[1]}
        else:
            return {0:self.layer3(out[0]),1:out[1]}


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features1 = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        features2 = []
        pre_hidden_dim = 32
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features2.append(block(pre_hidden_dim, input_channel, output_channel, stride, expand_ratio=t))
                pre_hidden_dim = input_channel * t
                input_channel = output_channel

        # building last several layers
        features3=[ConvBNReLU(input_channel, self.last_channel, kernel_size=1)]
        # make it nn.Sequential
        self.features1 = nn.Sequential(*features1)
        self.features2 = nn.Sequential(*features2)
        self.features3 = nn.Sequential(*features3)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features1(x)
        x = self.features2({0:x,1:None})
        x = self.features3(x[0])
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x


def dca_sk_mobilenet_v2(pretrained=False, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(**kwargs)
    return model



def demo():
    net = dca_sk_mobilenet_v2(num_classes=1000)
    y = net(torch.randn(2, 3, 224,224))
    print(y.size())

# demo()
