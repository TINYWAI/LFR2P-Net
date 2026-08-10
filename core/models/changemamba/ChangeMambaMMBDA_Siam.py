import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.models.changemamba.mamba_backbone import Backbone_VSSM
from core.models.changemamba.vmamba import VSSM, LayerNorm2d, VSSBlock, Permute
from core.models.changemamba.ChangeDecoder_BRIGHT import LocRelationChangeDecoder
from core.models.changemamba.SemanticDecoder import SemanticDecoder
from core.models.bdahead import ProgressiveBDAHeadV2


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


class STMambaBDA(nn.Module):
    def __init__(self, cfg, pretrained, in_chans, pre_num_classes, post_num_classes, **kwargs):
        super(STMambaBDA, self).__init__()
        self.encoder = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=pretrained, in_chans=in_chans, **kwargs)
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        _ACTLAYERS = dict(
            silu=nn.SiLU,
            gelu=nn.GELU,
            relu=nn.ReLU,
            sigmoid=nn.Sigmoid,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(kwargs['norm_layer'].lower(), None)
        ssm_act_layer: nn.Module = _ACTLAYERS.get(kwargs['ssm_act_layer'].lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(kwargs['mlp_act_layer'].lower(), None)
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['norm_layer', 'ssm_act_layer', 'mlp_act_layer']}

        decoder_channels = cfg.decoder_channels
        self.relation_method = cfg.get('relation_method')
        self.decoder_building = SemanticDecoder(
            encoder_dims=self.encoder.dims,
            proj_dim=decoder_channels,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )

        self.decoder_damage = LocRelationChangeDecoder(
            encoder_dims=self.encoder.dims,
            proj_dim=decoder_channels,
            relation_method=self.relation_method,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )

        self.bda_head = ProgressiveBDAHeadV2(
            in_channels=decoder_channels,
            num_bda_classes=post_num_classes,
            proj_method=cfg.get('proj_method', 'ResBlock'),
            guide_method=cfg.get('guide_method', 'ConcatBlock'),
        )

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, pre_data, post_data):
        # Encoder processing
        pre_features = self.encoder(pre_data)
        post_features = self.encoder(post_data)

        # Decoder processing - passing encoder outputs to the decoder
        loc_features_list = self.decoder_building(pre_features)
        dam_features, aux_loc_logits = self.decoder_damage(pre_features, post_features, loc_features_list)
        loc_features = loc_features_list[-1]
        binary_features, bda_features, loc_output, binary_output, bda_output = self.bda_head(loc_features, dam_features)

        loc_output = F.interpolate(loc_output, size=pre_data.size()[-2:], mode='bilinear')
        binary_output = F.interpolate(binary_output, size=post_data.size()[-2:], mode='bilinear')
        bda_output = F.interpolate(bda_output, size=post_data.size()[-2:], mode='bilinear')

        feat_dict = {
            'pre_features': pre_features,
            'post_features': post_features,
            'dam_features': bda_features,
            'binary_features': binary_features,
        }
        logits_dict = {
            'loc_logits': [loc_output],
            'clf_logits': [bda_output],
            'aux_binary_logits': binary_output,
        }
        logits_dict['loc_logits'].extend(aux_loc_logits)

        return feat_dict, logits_dict


def ChangeMambaSiamMMBDA(cfg):
    pretrained_weight_path = cfg.pretrained_weights
    model_type = cfg.encoder_name
    if model_type == 'vssm_tiny':
        DROP_PATH_RATE = 0.2
        EMBED_DIM = 96
        DEPTH = [2, 2, 4, 2]
        SSM_D_STATE = 1
        SSM_DT_RANK = "auto"
        SSM_RATIO = 2.0
        SSM_CONV = 3
        SSM_CONV_BIAS = False
        SSM_FORWARDTYPE = "v3noz"
        MLP_RATIO = 4.0
        DOWNSAMPLE = "v3"
        PATCHEMBED = "v2"
    else:
        raise NotImplementedError()

    deep_model = STMambaBDA(
        cfg,
        pretrained=pretrained_weight_path,
        in_chans=cfg.in_channels,  # config.MODEL.VSSM.IN_CHANS,
        patch_size=4,  # config.MODEL.VSSM.PATCH_SIZE,
        pre_num_classes=2,  # config.MODEL.NUM_CLASSES,
        post_num_classes=cfg.da_num_classes,  # config.MODEL.NUM_CLASSES,
        depths=DEPTH,  # config.MODEL.VSSM.DEPTHS,
        dims=EMBED_DIM,  # config.MODEL.VSSM.EMBED_DIM,
        # ===================
        ssm_d_state=SSM_D_STATE,  # config.MODEL.VSSM.SSM_D_STATE,
        ssm_ratio=SSM_RATIO,  # config.MODEL.VSSM.SSM_RATIO,
        ssm_rank_ratio=2.0,  # config.MODEL.VSSM.SSM_RANK_RATIO,
        ssm_dt_rank=("auto" if SSM_DT_RANK == "auto" else int(SSM_DT_RANK)),
        ssm_act_layer="silu",  # config.MODEL.VSSM.SSM_ACT_LAYER,
        ssm_conv=SSM_CONV,  # config.MODEL.VSSM.SSM_CONV,
        ssm_conv_bias=SSM_CONV_BIAS,  # config.MODEL.VSSM.SSM_CONV_BIAS,
        ssm_drop_rate=0.0,  # config.MODEL.VSSM.SSM_DROP_RATE,
        ssm_init="v0",  # config.MODEL.VSSM.SSM_INIT,
        forward_type=SSM_FORWARDTYPE,  # config.MODEL.VSSM.SSM_FORWARDTYPE,
        # ===================
        mlp_ratio=MLP_RATIO,  # config.MODEL.VSSM.MLP_RATIO,
        mlp_act_layer="gelu",  # config.MODEL.VSSM.MLP_ACT_LAYER,
        mlp_drop_rate=0.0,  # config.MODEL.VSSM.MLP_DROP_RATE,
        # ===================
        drop_path_rate=DROP_PATH_RATE,  # config.MODEL.DROP_PATH_RATE,
        patch_norm=True,  # config.MODEL.VSSM.PATCH_NORM,
        norm_layer="ln",  # config.MODEL.VSSM.NORM_LAYER,
        downsample_version=DOWNSAMPLE,  # config.MODEL.VSSM.DOWNSAMPLE,
        patchembed_version=PATCHEMBED,  # config.MODEL.VSSM.PATCHEMBED,
        gmlp=False,  # config.MODEL.VSSM.GMLP,
        use_checkpoint=False,  # config.TRAIN.USE_CHECKPOINT,
    )
    return deep_model
