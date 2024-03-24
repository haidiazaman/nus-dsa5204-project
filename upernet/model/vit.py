import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from .transformer import pair, Transformer


class ViT(nn.Module):
    def __init__(self,
                 *,
                 image_size,
                 patch_size,
                 num_classes,
                 dim,
                 depth,
                 heads,
                 mlp_dim,
                 pool='cls',
                 channels=3,
                 dim_head=64,
                 dropout=0.,
                 emb_dropout=0.,
                 fpn=False):
        super().__init__()
        self.fpn = fpn
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.layer_scale_05 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer_scale_2 = nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)
        self.layer_scale_4 = nn.Sequential(
            nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
            nn.GroupNorm(32,  dim // 2),
            nn.GELU(),
            nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, fpn)
        
        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img, **kwargs):
        img = torch.squeeze(img)
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)

        x[0] = self.layer_scale_4(x[0])
        x[1] = self.layer_scale_2(x[1])
        x[3] = self.layer_scale_05(x[3])
        return x

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
    

def vit_base(**kwargs):
    model = ViT(
        depth=12, dim=768, mlp_dim=3072, heads=12, **kwargs
    )
    return model


def vit_large(**kwargs):
    model = ViT(
        depth=24, dim=1024, mlp_dim=4096, heads=16, **kwargs
    )
    return model


def vit_huge(**kwargs):
    model = ViT(
        depth=32, dim=1280, mlp_dim=5120, heads=16, **kwargs
    )
    return model