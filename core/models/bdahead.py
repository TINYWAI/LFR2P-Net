import sys
import smp
import smp.base
import torch
import torch.nn as nn
import torch.nn.functional as F

current_module = sys.modules[__name__]


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

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


class ConcatBlock(nn.Module):
    def __init__(self, in_channels, out_channels, r=1.0):
        super().__init__()
        mid_chans = int(in_channels // r)
        self.conv1 = smp.base.Conv2dReLU(in_channels=in_channels, out_channels=mid_chans, kernel_size=3, padding=1)
        self.conv2 = smp.base.Conv2dReLU(in_channels=in_channels, out_channels=mid_chans, kernel_size=3, padding=1)
        self.fuse_conv = smp.base.Conv2dReLU(in_channels=mid_chans * 2, out_channels=out_channels, kernel_size=3,
                                             padding=1)

    def forward(self, m1, m2):
        m1 = self.conv1(m1)
        m2 = self.conv2(m2)
        fuse = self.fuse_conv(torch.cat([m1, m2], dim=1))

        return fuse


class ProgressiveBDAHeadV2(torch.nn.Module):
    def __init__(self, in_channels, num_bda_classes, proj_method, guide_method, r=1.0):
        super(ProgressiveBDAHeadV2, self).__init__()
        proj_module = getattr(current_module, proj_method)
        self.bi_proj = proj_module(in_channels, in_channels)

        self.loc_head = torch.nn.Conv2d(in_channels, 2, kernel_size=1)
        self.binary_head = torch.nn.Conv2d(in_channels, 2, kernel_size=1)
        self.bda_head = torch.nn.Conv2d(in_channels, num_bda_classes, kernel_size=1)

        guide_module = getattr(current_module, guide_method)
        self.bda_guide_module = guide_module(in_channels, in_channels, r=r)
        self.init_weight()

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

    def forward(self, loc_features, dam_features):
        loc_output = self.loc_head(loc_features)
        binary_features = self.bi_proj(dam_features.detach())
        binary_output = self.binary_head(binary_features)

        bda_features = self.bda_guide_module(binary_features, dam_features)  # + dam_features
        bda_output = self.bda_head(bda_features)
        return binary_features, bda_features, loc_output, binary_output, bda_output