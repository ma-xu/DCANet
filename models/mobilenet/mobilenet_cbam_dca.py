from torch import nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter


__all__ = ['dca_cbam_mobilenet_v2']


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0 \
            , dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size \
                , stride=stride, padding=padding, dilation=dilation \
                , groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01 \
                , affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, in_channels, gate_channels, reduction_ratio=8 \
            , pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()

        if in_channels != gate_channels:
            self.att_fc = nn.Sequential(
                nn.Conv2d(in_channels,gate_channels, bias=False, kernel_size=1),
                nn.BatchNorm2d(gate_channels),
                nn.ReLU(inplace=True)
            )
        self.alpha = nn.Sequential(
            nn.Conv2d(2, 1,bias=False, kernel_size=1),
            nn.LayerNorm(gate_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        x = inputs[0]
        b, c, _, _ = x.size()
        pre_att = inputs[1]
        channel_att_sum = None
        if pre_att is not None:
            if hasattr(self, 'att_fc'):
                pre_att = self.att_fc(pre_att)
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = self.avgpool(x)
                if pre_att is not None:
                    avg_pool = torch.cat((avg_pool.view(b, 1, 1, c), self.avgpool(pre_att).view(b, 1, 1, c)), dim=1)
                    avg_pool = self.alpha(avg_pool).view(b, c)
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = self.maxpool(x)
                if pre_att is not None:
                    max_pool = torch.cat((max_pool.view(b, 1, 1, c), self.maxpool(pre_att).view(b, 1, 1, c)), dim=1)
                    max_pool = self.alpha(max_pool).view(b, c)
                channel_att_raw = self.mlp( max_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = self.sigmoid(channel_att_sum).unsqueeze(2) \
                .unsqueeze(3).expand_as(x)

        out = x*scale
        # It can be only one, we did not optimize it due to lazy.
        return {0:out,1:out}

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1) \
                .unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self,in_channel, gate_channels):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1 \
                , padding=(kernel_size-1) // 2, relu=False)
        self.sigmoid = nn.Sigmoid()
        self.p1 = Parameter(torch.ones(1))
        self.p2 = Parameter(torch.zeros(1))
        self.bnrelu = nn.Sequential(
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        if x[1] is None:
            x_compress = self.compress(x[0])
        else:
            if x[1].size()[2]!=x[0].size()[2]:
                extent = (x[1].size()[2])//(x[0].size()[2])
                pre_spatial_att = F.avg_pool2d(x[1],kernel_size=extent,stride=extent)
            else:
                pre_spatial_att = x[1]
            x_compress = self.bnrelu(self.p1*self.compress(x[0])+self.p2*self.compress(pre_spatial_att))
        x_out = self.spatial(x_compress)
        scale = self.sigmoid(x_out) # broadcasting
        # It can be only one, we did not optimize it due to lazy.
        return {0:x[0] * scale,1:x[0] * scale}

class CBAM(nn.Module):
    def __init__(self, in_channel,gate_channels, reduction_ratio=8 \
            , pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(in_channel,gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate(in_channel,gate_channels)
    def forward(self, x):
        x_out = self.ChannelGate({0:x[0],1:x[1]})
        channel_att = x_out[1]
        if not self.no_spatial:
            x_out = self.SpatialGate({0:x_out[0],1:x[2]})
        return {0:x_out[0],1:channel_att,2:x_out[1]}

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


class ConvBNReLUBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLUBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )
    def forward(self, inputs):
        return {0:self.conv(inputs[0]),1:inputs[1],2:inputs[2]}


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)
        self.cbam = CBAM(inp,oup,8)

    def forward(self, inputs):
        x = inputs[0]
        cbam_out = self.cbam({0: self.conv(x), 1: inputs[1],2:inputs[2]})
        if self.use_res_connect:
            return {0: x + cbam_out[0], 1: cbam_out[1], 2: cbam_out[2]}
        else:
            return cbam_out


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
        features = [ConvBNReLUBlock(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLUBlock(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

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
        x = {0: x, 1: None,2:None}
        x = self.features(x)
        x = x[0]
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x


def dca_cbam_mobilenet_v2(pretrained=False, **kwargs):
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
    net = dca_cbam_mobilenet_v2(num_classes=1000)
    y = net(torch.randn(2, 3, 224,224))
    print(y.size())

# demo()
