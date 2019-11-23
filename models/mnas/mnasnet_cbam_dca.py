import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

__all__ = [ 'dca_cbam_mnasnet0_5', 'dca_cbam_mnasnet0_75', 'dca_cbam_mnasnet1_0','dca_cbam_mnasnet1_0r8', 'dca_cbam_mnasnet1_3']


# Paper suggests 0.9997 momentum, for TensorFlow. Equivalent PyTorch momentum is
# 1.0 - tensorflow.
_BN_MOMENTUM = 1 - 0.9997

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

class _InvertedResidual(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride, expansion_factor,
                 bn_momentum=0.1,reduction=16):
        super(_InvertedResidual, self).__init__()
        assert stride in [1, 2]
        assert kernel_size in [3, 5]
        mid_ch = in_ch * expansion_factor
        self.apply_residual = (in_ch == out_ch and stride == 1)
        self.layers = nn.Sequential(
            # Pointwise
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            # Depthwise
            nn.Conv2d(mid_ch, mid_ch, kernel_size, padding=kernel_size // 2,
                      stride=stride, groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            # Linear pointwise. Note that there's no activation.
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch, momentum=bn_momentum))
        self.cbam = CBAM(in_ch, out_ch,reduction_ratio=reduction)

    def forward(self, x):
        input = x[0]
        pre_att = x[1]
        cbam_out = self.cbam({0: self.layers(input), 1: pre_att, 2:x[2]})
        if self.apply_residual:
            return {0: input + cbam_out[0], 1: cbam_out[1],2:cbam_out[2]}
        else:
            return cbam_out


def _stack(in_ch, out_ch, kernel_size, stride, exp_factor, repeats,
           bn_momentum,reduction=16):
    """ Creates a stack of inverted residuals. """
    assert repeats >= 1
    # First one has no skip, because feature map size changes.
    first = _InvertedResidual(in_ch, out_ch, kernel_size, stride, exp_factor,
                              bn_momentum=bn_momentum,reduction=reduction)
    remaining = []
    for _ in range(1, repeats):
        remaining.append(
            _InvertedResidual(out_ch, out_ch, kernel_size, 1, exp_factor,
                              bn_momentum=bn_momentum,reduction=reduction))
    return nn.Sequential(first, *remaining)


def _round_to_multiple_of(val, divisor, round_up_bias=0.9):
    """ Asymmetric rounding to make `val` divisible by `divisor`. With default
    bias, will round up, unless the number is no more than 10% greater than the
    smaller divisible value, i.e. (83, 8) -> 80, but (84, 8) -> 88. """
    assert 0.0 < round_up_bias < 1.0
    new_val = max(divisor, int(val + divisor / 2) // divisor * divisor)
    return new_val if new_val >= round_up_bias * val else new_val + divisor


def _scale_depths(depths, alpha):
    """ Scales tensor depths as in reference MobileNet code, prefers rouding up
    rather than down. """
    return [_round_to_multiple_of(depth * alpha, 8) for depth in depths]


class MNASNet(torch.nn.Module):
    """ MNASNet, as described in https://arxiv.org/pdf/1807.11626.pdf.
    """

    def __init__(self, alpha, num_classes=1000, dropout=0.2,reduction=16):
        super(MNASNet, self).__init__()
        depths = _scale_depths([24, 40, 80, 96, 192, 320], alpha)
        layers1 = [
            # First layer: regular conv.
            nn.Conv2d(3, 32, 3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(32, momentum=_BN_MOMENTUM),
            nn.ReLU(inplace=True),
            # Depthwise separable, no skip.
            nn.Conv2d(32, 32, 3, padding=1, stride=1, groups=32, bias=False),
            nn.BatchNorm2d(32, momentum=_BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(16, momentum=_BN_MOMENTUM)
        ]
        layers2 = [
            # MNASNet blocks: stacks of inverted residuals.
            _stack(16, depths[0], 3, 2, 3, 3, _BN_MOMENTUM, reduction=reduction),
            _stack(depths[0], depths[1], 5, 2, 3, 3, _BN_MOMENTUM, reduction=reduction),
            _stack(depths[1], depths[2], 5, 2, 6, 3, _BN_MOMENTUM, reduction=reduction),
            _stack(depths[2], depths[3], 3, 1, 6, 2, _BN_MOMENTUM, reduction=reduction),
            _stack(depths[3], depths[4], 5, 2, 6, 4, _BN_MOMENTUM, reduction=reduction),
            _stack(depths[4], depths[5], 3, 1, 6, 1, _BN_MOMENTUM, reduction=reduction)
        ]
        layers3=[
            # Final mapping to classifier input.
            nn.Conv2d(depths[5], 1280, 1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(1280, momentum=_BN_MOMENTUM),
            nn.ReLU(inplace=True)
        ]
        self.layers1 = nn.Sequential(*layers1)
        self.layers2 = nn.Sequential(*layers2)
        self.layers3 = nn.Sequential(*layers3)
        self.classifier = nn.Sequential(nn.Dropout(p=dropout, inplace=True),
                                        nn.Linear(1280, num_classes))
        self._initialize_weights()

    def forward(self, x):
        x = self.layers1(x)
        x = self.layers2({0:x,1:None,2:None})
        x = self.layers3(x[0])
        # Equivalent to global avgpool and removing H and W dimensions.
        x = x.mean([2, 3])
        return self.classifier(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.01)
                nn.init.zeros_(m.bias)


def dca_cbam_mnasnet0_5(pretrained=False, progress=True, **kwargs):
    """MNASNet with depth multiplier of 0.5 from
    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    <https://arxiv.org/pdf/1807.11626.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MNASNet(0.5, **kwargs)
    return model


def dca_cbam_mnasnet0_75(pretrained=False, **kwargs):
    """MNASNet with depth multiplier of 0.75 from
    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    <https://arxiv.org/pdf/1807.11626.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MNASNet(0.75, **kwargs)
    return model


def dca_cbam_mnasnet1_0(pretrained=False, **kwargs):
    """MNASNet with depth multiplier of 1.0 from
    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    <https://arxiv.org/pdf/1807.11626.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MNASNet(1.0, **kwargs)
    return model

def dca_cbam_mnasnet1_0r8(pretrained=False, reduction=8,**kwargs):
    """MNASNet with depth multiplier of 1.0 from
    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    <https://arxiv.org/pdf/1807.11626.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MNASNet(1.0,reduction=reduction, **kwargs)
    return model


def dca_cbam_mnasnet1_3(pretrained=False, **kwargs):
    """MNASNet with depth multiplier of 1.3 from
    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    <https://arxiv.org/pdf/1807.11626.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MNASNet(1.3, **kwargs)
    return model

def demo():
    net = dca_cbam_mnasnet1_0r8(num_classes=1000)
    y = net(torch.randn(2, 3, 224,224))
    print(y.size())

# demo()
