import torch
import torch.nn as nn
from core.models.changemamba.vmamba import VSSM, LayerNorm2d
import pdb


class Backbone_VSSM(VSSM):
    """
    vis
    """
    def __init__(self, out_indices=(0, 1, 2, 3), pretrained=None, norm_layer='ln2d', **kwargs):
        # norm_layer='ln'
        kwargs.update(norm_layer=norm_layer)
        super().__init__(**kwargs)
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.out_indices = out_indices
        for i in out_indices:
            layer = norm_layer(self.dims[i])
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)
            permute_name = f'permute{i}'
            self.add_module(permute_name, Permute(0, 3, 1, 2))

        del self.classifier
        self.load_pretrained(pretrained)

    def load_pretrained(self, ckpt=None, key="model"):
        if ckpt is None:
            return

        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            print(f"Successfully load ckpt {ckpt}")
            incompatibleKeys = self.load_state_dict(_ckpt[key], strict=False)
            print(incompatibleKeys)
        except Exception as e:
            print(f"Failed loading checkpoint form {ckpt}: {e}")

    def forward(self, x):
        def layer_forward(l, x):
            x = l.blocks(x)
            y = l.downsample(x)
            return x, y

        x = self.patch_embed(x)
        # x = self.patch_embed_(x)
        outs = []
        for i, layer in enumerate(self.layers):
            o, x = layer_forward(layer, x)  # (B, H, W, C)
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(o)
                if not self.channel_first:
                    # out = out.permute(0, 3, 1, 2).contiguous()
                    permute_layer = getattr(self, f'permute{i}')
                    out = permute_layer(out)
                outs.append(out)

        if len(self.out_indices) == 0:
            return x

        return outs

    def get_stages(self):
        return [
            nn.Sequential(
                self.patch_embed,
                MambaStage(self.layers[0], norm_layer=getattr(self, f'outnorm0'))
            ),
            nn.Sequential(
                Permute(0, 2, 3, 1),
                MambaStage(self.layers[1], norm_layer=getattr(self, f'outnorm1')),
            ),
            nn.Sequential(
                Permute(0, 2, 3, 1),
                MambaStage(self.layers[2], norm_layer=getattr(self, f'outnorm2')),
            ),
            nn.Sequential(
                Permute(0, 2, 3, 1),
                MambaStage(self.layers[3], norm_layer=getattr(self, f'outnorm3'))
            ),
        ]


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args).contiguous()


class MambaStage(nn.Module):
    def __init__(self, layer, norm_layer):
        super().__init__()
        self.layer = layer
        self.norm_layer = norm_layer

    def forward(self, x):
        def layer_forward(l, x):
            x = l.blocks(x)
            y = l.downsample(x)
            return x, y

        o, nxt_input = layer_forward(self.layer, x)
        out = self.norm_layer(o)
        out = out.permute(0, 3, 1, 2).contiguous()
        nxt_input = nxt_input.permute(0, 3, 1, 2).contiguous()

        return out, nxt_input
