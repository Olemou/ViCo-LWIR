import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.checkpoint as cp
from timm.models.vision_transformer import trunc_normal_

# -------------------------
# Vision Transformer Presets
# -------------------------
VIT_CONFIGS = {
    "tiny": dict(embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0),
    "small": dict(embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0),
    "base": dict(embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0),
    "large": dict(embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4.0),
    "huge": dict(embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4.0),
}


# =====================
# Multi-head Attention
# =====================
class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads=12, qkv_bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dim_per_head = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=qkv_bias)
        self.scale = self.dim_per_head**-0.5
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_map = None  # for visualization/debugging

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.dim_per_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = torch.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        self.attn_map = attn.detach()
        return self.proj(out)


# =====================
# Patch Dropout
# =====================
class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob=0.1, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.0
        self.prob = prob
        self.exclude_first_token = exclude_first_token

    def forward(self, x):
        if not self.training or self.prob == 0.0:
            return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        B, N, _ = x.size()
        num_keep = max(1, int(N * (1 - self.prob)))
        rand = torch.randn(B, N)
        keep_idx = rand.topk(num_keep, dim=-1).indices
        batch_idx = torch.arange(B)[:, None]

        x = x[batch_idx, keep_idx]
        return torch.cat((cls_tokens, x), dim=1) if self.exclude_first_token else x


# =====================
# Feed Forward Network
# =====================
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.1):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


# =====================
# Transformer Block
# =====================
class Block(nn.Module):
    def __init__(self, dim, num_heads=12, mlp_ratio=4.0, qkv_bias=True, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# =====================
# Patch Embedding
# =====================
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.num_patches = (img_size // patch_size) ** 2

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


# =====================
# Vision Transformer (Adaptable)
# =====================
class VisionTransformer(nn.Module):
    def __init__(
        self,
        variant: str = "base",  # 'tiny', 'small', 'base', 'large', 'huge'
        img_size=224,
        patch_size=16,
        in_chans=3,
        patch_dropout_prob=0.1,
        drop_rate=0.1,
        qkv_bias=True,
        use_checkpoint=True,
        head_hidden_dim=None
    ):
        super().__init__()
        assert (
            variant in VIT_CONFIGS
        ), f"Unknown variant '{variant}', choose from {list(VIT_CONFIGS.keys())}"

        # Load configuration
        cfg = VIT_CONFIGS[variant]
        embed_dim, depth, num_heads, mlp_ratio = (
            cfg["embed_dim"],
            cfg["depth"],
            cfg["num_heads"],
            cfg["mlp_ratio"],
        )

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.use_checkpoint = use_checkpoint

        # CLS token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Patch dropout
        self.patch_dropout = PatchDropout(patch_dropout_prob)

        # Transformer encoder
        self.blocks = nn.Sequential(
            *[
                Block(embed_dim, num_heads, mlp_ratio, qkv_bias, drop_rate)
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        head_dim = head_hidden_dim or embed_dim
        self.head = nn.Linear(embed_dim, head_dim)

        # Initialization
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.patch_dropout(x)

        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                x = cp.checkpoint(blk, x,use_reentrant=False)
            else:
                x = blk(x)

        x = self.norm(x)

        return self.head(x[:, 1:,:])  # return patch embeddings only
