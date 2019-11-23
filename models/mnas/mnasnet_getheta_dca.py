import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

__all__ = [ 'dca_getheta_mnasnet0_5', 'dca_getheta_mnasnet0_75', 'dca_getheta_mnasnet1_0','dca_getheta_mnasnet1_0r8', 'dca_getheta_mnasnet1_3']


# Paper suggests 0.9997 momentum, for TensorFlow. Equivalent PyTorch momentum is
# 1.0 - tensorflow.
_BN_MOMENTUM = 1 - 0.9997

class GEBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        # If extent is zero, assuming global.
        super(GEBlock, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        if in_planes != out_planes:
            self.att_fc = nn.Sequential(
                nn.Linear(in_planes, out_planes),
                nn.LayerNorm(out_planes),
                nn.ReLU(inplace=True)
            )
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1),
            nn.LayerNorm(out_planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b, c, _, _ = x[0].size()
        gap = self.gap(x[0]).view(b, c)
        if x[1] is None:
            all_att = self.sigmoid(gap)
        else:
            pre_att = self.att_fc(x[1]) if hasattr(self, 'att_fc') else x[1]
            all_att = torch.cat((gap.view(b, 1, 1, c), pre_att.view(b, 1, 1, c)), dim=1)
            all_att = self.conv(all_att).view(b, c)
            all_att = self.sigmoid(all_att)
        return {0: x[0] * all_att.view(b, c, 1, 1), 1: gap * all_att}



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
        self.ge = GEBlock(in_ch,out_ch)

    def forward(self, x):
        input = x[0]
        pre_att = x[1]
        ge_out = self.ge({0: self.layers(input), 1: pre_att})
        if self.apply_residual:
            return {0: input + ge_out[0], 1: ge_out[1]}
        else:
            return ge_out


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
                              bn_momentum=bn_momentum))
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
        x = self.layers2({0:x,1:None})
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


def dca_getheta_mnasnet0_5(pretrained=False, progress=True, **kwargs):
    """MNASNet with depth multiplier of 0.5 from
    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    <https://arxiv.org/pdf/1807.11626.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MNASNet(0.5, **kwargs)
    return model


def dca_getheta_mnasnet0_75(pretrained=False, **kwargs):
    """MNASNet with depth multiplier of 0.75 from
    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    <https://arxiv.org/pdf/1807.11626.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MNASNet(0.75, **kwargs)
    return model


def dca_getheta_mnasnet1_0(pretrained=False, **kwargs):
    """MNASNet with depth multiplier of 1.0 from
    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    <https://arxiv.org/pdf/1807.11626.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MNASNet(1.0, **kwargs)
    return model

def dca_getheta_mnasnet1_0r8(pretrained=False, reduction=8,**kwargs):
    """MNASNet with depth multiplier of 1.0 from
    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    <https://arxiv.org/pdf/1807.11626.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MNASNet(1.0,reduction=reduction, **kwargs)
    return model


def dca_getheta_mnasnet1_3(pretrained=False, **kwargs):
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
    net = dca_getheta_mnasnet1_0r8(num_classes=1000)
    y = net(torch.randn(2, 3, 224,224))
    print(y.size())

# demo()
