import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from src.model.transformer import pair, Transformer


class ViT3D(nn.Module):
    def __init__(self,
                 *,
                 image_size,
                 image_patch_size,
                 frames,
                 frame_patch_size,
                 num_classes,
                 dim,
                 depth,
                 heads,
                 mlp_dim,
                 pool='cls',
                 channels=3,
                 dim_head=64,
                 dropout=0.,
                 emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'

        num_patches = (image_height // patch_height) * (image_width // patch_width) * (frames // frame_patch_size)
        patch_dim = channels * patch_height * patch_width * frame_patch_size

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)', p1=patch_height, p2=patch_width,
                      pf=frame_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, video):
        x = self.to_patch_embedding(video)
        b, n, _ = x.shape

        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x, hidden_states_out = self.transformer(x)

        return x, hidden_states_out


def vit_3d_base(**kwargs):
    model = ViT3D(
        depth=12, dim=768, mlp_dim=3072, heads=12, **kwargs
    )
    return model


def vit_3d_large(**kwargs):
    model = ViT3D(
        depth=24, dim=1024, mlp_dim=4096, heads=16, **kwargs
    )
    return model


def vit_3d_huge(**kwargs):
    model = ViT3D(
        depth=32, dim=1280, mlp_dim=5120, heads=16, **kwargs
    )
    return model
