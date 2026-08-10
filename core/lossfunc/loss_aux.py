import torch
from torch import nn
import torch.nn.functional as F


class PrototypeBuffer(nn.Module):
    def __init__(self, feat_dim, num_classes, momentum=0.9):
        super().__init__()
        self.momentum = momentum
        self.num_classes = num_classes

        self.register_buffer('prototypes', torch.zeros(num_classes, feat_dim))
        self.register_buffer('initialized', torch.zeros(num_classes).bool())

    @torch.no_grad()
    def update(self, feats, labels):
        for c in range(self.num_classes):
            mask = (labels == c)
            if mask.sum() == 0:
                continue

            mean_feat = feats[mask].mean(dim=0)

            if not self.initialized[c]:
                self.prototypes[c] = mean_feat
                self.initialized[c] = True
            else:
                self.prototypes[c] = self.prototypes[c] * self.momentum + mean_feat * (1 - self.momentum)


class PrototypeAlignmentLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, feats, labels, buffer):
        B, C, H, W = feats.shape
        feats = feats.permute(0, 2, 3, 1).reshape(-1, C)
        labels = labels.view(-1)  # [N]

        valid = labels != self.ignore_index
        feats = feats[valid]
        labels = labels[valid]

        protos = buffer.prototypes[labels]  # [num_classes, feat_dim]

        feats = F.normalize(feats, dim=1)
        protos = F.normalize(protos, dim=1)

        loss = 1 - (feats * protos).sum(dim=1)

        return loss.mean()


class AuxBinaryAlignCrossEntropyLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.WEIGHT is not None:
            weight = torch.Tensor(cfg.WEIGHT).cuda()
        else:
            weight = None
        ignore_index = cfg.IGNORE
        self.change_idx = cfg.CHANGE_INDEX
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        self.buffer = PrototypeBuffer(feat_dim=cfg.FEAT_DIM, num_classes=cfg.NUM_CLASSES, momentum=cfg.get('MOMENTUM', 0.9))
        self.proto_align_loss = PrototypeAlignmentLoss(ignore_index=ignore_index)
        self.loss_weight = cfg.get('LOSS_WEIGHT', 1.0)
        if self.loss_weight == 'learn':
            self.alpha = nn.Parameter(torch.tensor(0.0))
            self.beta = nn.Parameter(torch.tensor(0.0))

    def mask2binary(self, labels):
        mask = torch.zeros_like(labels, dtype=torch.bool)
        for c in self.change_idx:
            mask |= (labels == c)
        mask = mask.long()
        mask[labels == 255] = 255
        return mask

    def forward(self, feat_dict, logits_dict, pre_labels, post_labels):
        logits = logits_dict['aux_binary_logits']

        B, C, H, W = logits.size()
        labels_ = post_labels.clone().unsqueeze(1).float()
        labels_ = F.interpolate(labels_, size=(H, W), mode='nearest')
        labels_ = labels_.squeeze(1).long()

        binary_labels = self.mask2binary(labels_)
        coarse_ce_loss = self.ce_loss(logits, binary_labels)

        fine_feats = feat_dict['dam_features']

        B, C, H, W = fine_feats.size()
        labels_ = post_labels.clone().unsqueeze(1).float()
        labels_ = F.interpolate(labels_, size=(H, W), mode='nearest')
        labels_ = labels_.squeeze(1).long()

        self.buffer.update(fine_feats, labels_)
        proto_align_loss = self.proto_align_loss(fine_feats, labels_, self.buffer)

        if self.loss_weight == 'learn':
            alpha = torch.exp(-self.alpha)
            beta = torch.exp(-self.beta)
            loss = alpha * coarse_ce_loss + beta * proto_align_loss
            return loss
        else:
            loss = coarse_ce_loss + proto_align_loss
            return loss * self.loss_weight