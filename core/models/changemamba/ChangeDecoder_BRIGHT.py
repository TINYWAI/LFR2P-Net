import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .vmamba import VSSM, LayerNorm2d, VSSBlock, Permute

current_module = sys.modules[__name__]


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):

    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)

        return self.sigmoid(x)


class AdaptiveRadialLPFV2(nn.Module):
    def __init__(self, C, sharpness=20.0, init_cutoff=0.25):
        super().__init__()
        self.sharpness = sharpness

        hidden = max(16, C // 4)
        out_dim = 1
        self.gate = nn.Sequential(
            nn.Conv2d(C + 1, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_dim, 1)
        )

        target = float(init_cutoff / 0.70710678)
        target = min(max(target, 1e-4), 1 - 1e-4)
        init_logit = torch.log(torch.tensor(target) / (1 - torch.tensor(target)))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, init_logit)

        self._r_cache = {}

    @torch.no_grad()
    def _get_r(self, H, W, device, dtype):
        key = (H, W, device, dtype)
        if key in self._r_cache:
            return self._r_cache[key]
        fy = torch.fft.fftshift(torch.fft.fftfreq(H, device=device, dtype=dtype))  # [-0.5,0.5)
        fx = torch.fft.fftshift(torch.fft.fftfreq(W, device=device, dtype=dtype))
        gy, gx = torch.meshgrid(fy, fx, indexing="ij")
        r = torch.sqrt(gx ** 2 + gy ** 2)[None, None, :, :]  # (1,1,H,W)
        self._r_cache[key] = r
        return r

    def forward(self, x):
        B, C, H, W = x.shape
        X = torch.fft.fft2(x, dim=(-2, -1), norm="ortho")
        Xs = torch.fft.fftshift(X, dim=(-2, -1))

        spec_energy = torch.mean(torch.abs(Xs), dim=(-2, -1), keepdim=True)  # (B, C, 1, 1)
        spec_energy = torch.mean(spec_energy, dim=1, keepdim=True)  # (B, 1, 1, 1)
        x_avg = torch.mean(x, dim=(-2, -1), keepdim=True)  # (B, C, 1, 1)
        gate_input = torch.cat([x_avg, spec_energy], dim=1)  # (B, C+1, 1, 1)

        cutoff_logit = self.gate(gate_input)  # (B,1,1,1) or (B,C,1,1)
        cutoff = torch.sigmoid(cutoff_logit) * 0.70710678  # 映射到真实最大半径

        r = self._get_r(H, W, x.device, x.real.dtype)  # (1,1,H,W)

        M = torch.sigmoid((cutoff - r) * self.sharpness)  # (B,1,H,W)
        M = M.expand(B, C, H, W)

        X_low = torch.fft.ifftshift(Xs * M, dim=(-2, -1))
        x_low = torch.fft.ifft2(X_low, dim=(-2, -1), norm="ortho").real
        x_high = x - x_low
        return x_low, x_high, M, cutoff


class LPDiffAttConvV3(nn.Module):
    def __init__(self, in_channels, out_channels, eps=1e-8, **kwargs):
        super().__init__()
        self.m1_proj = conv1x1(in_channels, out_channels)
        self.m2_proj = conv1x1(in_channels, out_channels)
        self.m1_lp_filter = AdaptiveRadialLPFV2(out_channels)
        self.m2_lp_filter = AdaptiveRadialLPFV2(out_channels)
        self.m1_low_se_layer = ChannelAttention(out_channels)
        self.m2_low_se_layer = ChannelAttention(out_channels)
        self.spatial_attention = SpatialAttention()
        self.eps = eps

    def forward(self, m1, m2, building_prob=None):
        m1_proj = self.m1_proj(m1)
        m2_proj = self.m2_proj(m2)

        m1_low, m1_high, m1_lp, m1_lp_r = self.m1_lp_filter(m1_proj)
        m2_low, m2_high, m2_lp, m2_lp_r = self.m2_lp_filter(m2_proj)

        m1_low = self.m1_low_se_layer(m1_low) * m1_low
        m2_low = self.m2_low_se_layer(m2_low) * m2_low
        diff = abs(m1_low - m2_low)

        if building_prob is not None:
            diff = diff * building_prob
        att_map = self.spatial_attention(diff)

        output = diff * att_map
        return output, m1_proj, m2_proj, m1_low, m1_high, m2_low, m2_high, m1_lp, m2_lp, diff


class LocRelationChangeDecoder(nn.Module):
    def __init__(self, encoder_dims, proj_dim, relation_method, channel_first, norm_layer, ssm_act_layer, mlp_act_layer,
                 **kwargs):
        super(LocRelationChangeDecoder, self).__init__()
        relation_method = getattr(current_module, relation_method[len('loc_guide_'):])

        self.relation_layer4 = relation_method(encoder_dims[-1], proj_dim, **kwargs)
        self.relation_layer3 = relation_method(encoder_dims[-2], proj_dim, **kwargs)
        self.relation_layer2 = relation_method(encoder_dims[-3], proj_dim, **kwargs)
        self.relation_layer1 = relation_method(encoder_dims[-4], proj_dim, **kwargs)

        # Define the VSS Block for Spatio-temporal relationship modelling
        self.st_block_41 = nn.Sequential(*[
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=proj_dim, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                     ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'],
                     ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                     ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'],
                     ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                     forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer,
                     mlp_drop_rate=kwargs['mlp_drop_rate'],
                     gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        ])
        self.st_block_31 = nn.Sequential(*[
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=proj_dim, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                     ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'],
                     ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                     ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'],
                     ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                     forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer,
                     mlp_drop_rate=kwargs['mlp_drop_rate'],
                     gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        ])
        self.st_block_21 = nn.Sequential(*[
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=proj_dim, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                     ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'],
                     ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                     ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'],
                     ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                     forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer,
                     mlp_drop_rate=kwargs['mlp_drop_rate'],
                     gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        ])
        self.st_block_11 = nn.Sequential(*[
            nn.Conv2d(kernel_size=1, in_channels=proj_dim, out_channels=proj_dim),
            nn.BatchNorm2d(proj_dim), nn.ReLU(inplace=True)
        ])

        # Fuse layer
        self.fuse_layer_4 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=proj_dim, out_channels=proj_dim),
                                          nn.BatchNorm2d(proj_dim), nn.ReLU(inplace=True))
        self.fuse_layer_3 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=proj_dim, out_channels=proj_dim),
                                          nn.BatchNorm2d(proj_dim), nn.ReLU(inplace=True))
        self.fuse_layer_2 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=proj_dim, out_channels=proj_dim),
                                          nn.BatchNorm2d(proj_dim), nn.ReLU(inplace=True))
        self.fuse_layer_1 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=proj_dim, out_channels=proj_dim),
                                          nn.BatchNorm2d(proj_dim), nn.ReLU(inplace=True))

        # Smooth layer
        self.smooth_layer_3 = ResBlock(in_channels=proj_dim, out_channels=proj_dim, stride=1)
        self.smooth_layer_2 = ResBlock(in_channels=proj_dim, out_channels=proj_dim, stride=1)
        self.smooth_layer_1 = ResBlock(in_channels=proj_dim, out_channels=proj_dim, stride=1)

        # loc head
        self.loc_head_4 = nn.Conv2d(in_channels=proj_dim, out_channels=2, kernel_size=1)
        self.loc_head_3 = nn.Conv2d(in_channels=proj_dim, out_channels=2, kernel_size=1)
        self.loc_head_2 = nn.Conv2d(in_channels=proj_dim, out_channels=2, kernel_size=1)

    def get_loc_logits(self, head, loc_feat):
        logits = head(loc_feat)
        prob = F.softmax(logits, dim=1)
        building_prob = prob[:, 1, ...].unsqueeze(1)
        return logits, building_prob

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, pre_features, post_features, loc_features):
        pre_feat_1, pre_feat_2, pre_feat_3, pre_feat_4 = pre_features
        post_feat_1, post_feat_2, post_feat_3, post_feat_4 = post_features
        loc_feat_4, loc_feat_3, loc_feat_2, loc_feat_1 = loc_features

        '''
            Stage I
        '''
        aux_bd_logits_4, bd_prob_4 = self.get_loc_logits(self.loc_head_4, loc_feat_4)
        relation_4, m1_proj_4, m2_proj_4, m1_low_4, m1_high_4, m2_low_4, m2_high_4, m1_lp_4, m2_lp_4, diff_4 \
            = self.relation_layer4(pre_feat_4, post_feat_4, building_prob=bd_prob_4)
        p41 = self.st_block_41(relation_4)
        p4 = self.fuse_layer_4(p41)

        '''
            Stage II
        '''
        aux_bd_logits_3, bd_prob_3 = self.get_loc_logits(self.loc_head_3, loc_feat_3)
        relation_3, m1_proj_3, m2_proj_3, m1_low_3, m1_high_3, m2_low_3, m2_high_3, m1_lp_3, m2_lp_3, diff_3 \
            = self.relation_layer3(pre_feat_3, post_feat_3, building_prob=bd_prob_3)
        p31 = self.st_block_31(relation_3)
        p3 = self.fuse_layer_3(p31)
        p3 = self._upsample_add(p4, p3)
        p3 = self.smooth_layer_3(p3)

        '''
            Stage III
        '''
        aux_bd_logits_2, bd_prob_2 = self.get_loc_logits(self.loc_head_2, loc_feat_2)
        relation_2, m1_proj_2, m2_proj_2, m1_low_2, m1_high_2, m2_low_2, m2_high_2, m1_lp_2, m2_lp_2, diff_2 \
            = self.relation_layer2(pre_feat_2, post_feat_2, building_prob=bd_prob_2)
        p21 = self.st_block_21(relation_2)
        p2 = self.fuse_layer_2(p21)
        p2 = self._upsample_add(p3, p2)
        p2 = self.smooth_layer_2(p2)

        '''
            Stage IV
        '''
        relation_1, m1_proj_1, m2_proj_1, m1_low_1, m1_high_1, m2_low_1, m2_high_1, m1_lp_1, m2_lp_1, diff_1 \
            = self.relation_layer1(pre_feat_1, post_feat_1)
        p11 = self.st_block_11(relation_1)
        p1 = self.fuse_layer_1(p11)
        p1 = self._upsample_add(p2, p1)
        p1 = self.smooth_layer_1(p1)

        aux_loc_logits = [aux_bd_logits_2, aux_bd_logits_3, aux_bd_logits_4]
        return p1, aux_loc_logits


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
