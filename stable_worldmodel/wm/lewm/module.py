import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v


def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift


class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        """
        x : (B, T, D)
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(
            3, dim=-1
        )  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (
            rearrange(t, 'b t (h d) -> b h t d', h=self.heads) for t in qkv
        )
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=drop, is_causal=causal
        )
        out = rearrange(out, 'b h t d -> b t (h d)')
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(
            dim, heads=heads, dim_head=dim_head, dropout=dropout
        )
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(
            dim, heads=heads, dim_head=dim_head, dropout=dropout
        )
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttention(nn.Module):
    """Scaled dot-product attention with Q from x and K, V from context."""

    def __init__(self, dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        context_dim = context_dim or dim
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, context):
        """
        x : (B, T, D) query tokens (Δx patches + CLS)
        context : (B, N, D) key/value feature map H_t
        """
        x = self.norm(x)
        context = self.norm_context(context)
        drop = self.dropout if self.training else 0.0
        q = rearrange(self.to_q(x), 'b t (h d) -> b h t d', h=self.heads)
        k, v = (
            rearrange(t, 'b n (h d) -> b h n d', h=self.heads)
            for t in self.to_kv(context).chunk(2, dim=-1)
        )
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=drop, is_causal=False
        )
        out = rearrange(out, 'b h t d -> b t (h d)')
        return self.to_out(out)


class CrossAttnBlock(nn.Module):
    """ViT block with self-attention over D then cross-attention to H_t.

    D ← D + SelfAttn(D)
    D ← D + CrossAttn(Q=D, K=H_t, V=H_t)
    D ← D + FFN(D)
    """

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(
            dim, heads=heads, dim_head=dim_head, dropout=dropout
        )
        self.cross_attn = CrossAttention(
            dim, heads=heads, dim_head=dim_head, dropout=dropout
        )
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, context):
        x = x + self.attn(self.norm1(x), causal=False)
        x = x + self.cross_attn(self.norm2(x), context)
        x = x + self.mlp(self.norm3(x))
        return x


class MotionEncoder(nn.Module):
    """ViT-T that maps (Δx_t, H_t) to a latent residual δz_t.

    Patch-embeds Δx_t, prepends a learned CLS token and position embeddings,
    then runs transformer blocks that self-attend over Δx tokens and
    cross-attend to the encoder feature map H_t. Returns the CLS token.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=14,
        dim=192,
        depth=12,
        heads=3,
        dim_head=64,
        mlp_dim=768,
        in_chans=3,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        assert img_size % patch_size == 0, (
            'img_size must be divisible by patch_size'
        )
        num_patches = (img_size // patch_size) ** 2
        self.num_patches = num_patches
        self.dim = dim
        self.patch_embed = nn.Conv2d(
            in_chans, dim, kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.layers = nn.ModuleList(
            [
                CrossAttnBlock(dim, heads, dim_head, mlp_dim, dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, delta_x, features):
        """
        delta_x: (B, C, H, W) pixel difference x_{t+1} - x_t
        features: (B, N, D) encoder feature map H_t (CLS + patches)
        returns: (B, D) CLS token δz_t
        """
        x = self.patch_embed(delta_x.to(self.patch_embed.weight.dtype))
        x = rearrange(x, 'b d h w -> b (h w) d')
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.pos_embedding[:, : x.size(1)]
        x = self.dropout(x)
        for block in self.layers:
            x = block(x, features)
        x = self.norm(x)
        return x[:, 0]


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None):
        x = self.input_proj(x)

        if c is not None:
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)
        x = self.output_proj(x)
        return x


class Embedder(nn.Module):
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.smoothed_dim = smoothed_dim
        self.emb_dim = emb_dim
        self.mlp_scale = mlp_scale
        self.patch_embed = nn.Conv1d(
            input_dim, smoothed_dim, kernel_size=1, stride=1
        )
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or input_dim
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.net(x)


class Predictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or input_dim
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        self.mlp_dim = mlp_dim
        self.emb_dropout = emb_dropout
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_frames, input_dim)
        )
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x, c):
        """
        x: (B, T, d)
        c: (B, T, act_dim)
        """
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x, c)
        return x
