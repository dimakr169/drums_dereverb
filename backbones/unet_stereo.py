import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Sinusoidal timestep embeddings
# -----------------------------
def _sinusoidal_embedding_var(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Your 'sinusoidal_embedding' variant:
      frequencies = exp(linspace(log(1), log(1000), dim//2))
      emb = [sin(2π f t), cos(2π f t)]
    timesteps: (B,) int or float
    returns: (B, dim)
    """
    device = timesteps.device
    half = dim // 2
    freqs = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), half, device=device))
    t = timesteps.float().unsqueeze(1)          # (B,1)
    ang = 2.0 * math.pi * t * freqs.unsqueeze(0)  # (B, half)
    emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)  # (B, dim or dim-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0,1))
    return emb

def _get_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Your 'get_timestep_embedding' variant (DDPM-style):
      emb = exp(arange(half) * (-log(10000)/(half-1)))
      emb = t * emb
      emb = [sin(emb), cos(emb)]
    """
    device = timesteps.device
    half = dim // 2
    freqs = torch.exp(torch.arange(half, device=device, dtype=torch.float32) *
                      (-math.log(10000.0) / max(half - 1, 1)))
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (B, 2*half)
    if dim % 2 == 1:
        emb = F.pad(emb, (0,1))
    return emb

class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int, var: bool = False):
        super().__init__()
        self.dim = dim
        self.var = var
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return _sinusoidal_embedding_var(t, self.dim) if self.var else _get_timestep_embedding(t, self.dim)

# -----------------------------
# Blocks
# -----------------------------
class AttentionBlock(nn.Module):
    """
    Channels-last LayerNorm in TF; here we use GroupNorm(1,C) before/after,
    and run MHA on (B, HW, C) then project back.
    """
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels, eps=1e-5)
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W)
        b, c, h, w = x.shape
        h_in = self.norm(x)                         # (B,C,H,W)
        seq = h_in.permute(0, 2, 3, 1).reshape(b, h*w, c)  # (B, HW, C)
        attn, _ = self.mha(seq, seq, seq)          # (B, HW, C)
        attn = attn.reshape(b, h, w, c).permute(0, 3, 1, 2)  # (B,C,H,W)
        return x + self.proj(attn)

class Downsample(nn.Module):
    """
    TF: AvgPool2D(k=3,s=1,'same') -> Conv2D(stride=2)
    Kept identical, but you may simplify to Conv2d(stride=2) directly.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.avg = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.avg(x))

class Upsample(nn.Module):
    """
    TF: resize (nearest) -> Conv2D
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        return self.conv(x)

class ResNetBlock(nn.Module):
    """
    FiLM conditioning: h = h*(1+s) + b with s,b = Linear(t-emb)
    """
    def __init__(self, in_ch: int, out_ch: int = None, emb_ch: int = None, num_groups: int = 4,
                 dropout: float = 0.1, use_norm: bool = True, conv_shortcut: bool = True):
        super().__init__()
        out_ch = out_ch or in_ch
        self.in_ch, self.out_ch = in_ch, out_ch
        self.use_norm = use_norm

        self.norm1 = nn.GroupNorm(num_groups, in_ch, eps=1e-5) if use_norm else nn.Identity()
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.scale = nn.Linear(emb_ch, out_ch)
        self.shift = nn.Linear(emb_ch, out_ch)

        self.norm2 = nn.GroupNorm(num_groups, out_ch, eps=1e-5) if use_norm else nn.Identity()
        self.act2 = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1) if conv_shortcut else nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W), temb: (B, Cemb) -> projected inside
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        s = self.scale(F.silu(temb)).unsqueeze(-1).unsqueeze(-1)  # (B,out_ch,1,1)
        b = self.shift(F.silu(temb)).unsqueeze(-1).unsqueeze(-1)
        h = h * (1 + s) + b

        h = self.norm2(h)
        h = self.act2(h)
        h = self.drop(h)
        h = self.conv2(h)

        return self.shortcut(x) + 0.1 * h


class UNetRI(nn.Module):
    def __init__(self, config):
        super().__init__()
        C = config.channels
        ch_mult = config.ch_mult
        num_res = len(ch_mult)
        emb_ch = C * 4  # <-- embedding width everywhere
        self.create_mask = config.create_mask

        # ---- timestep embedding MLP ----
        self.t_embed = nn.Sequential(
            TimestepEmbedding(C * 2, var=config.continuous_emb),
            nn.Linear(C * 2, C * 4),
            nn.SiLU(),
            nn.Linear(C * 4, C * 4),
            nn.SiLU(),
        )

        # ---- input conv ----
        # TF used Conv2D(C, (9,1)); in NCHW that's kernel=(9,1), padding=(4,0)
        self.input_conv = nn.Conv2d(config.in_chans, C, kernel_size=(9,1), padding=(4,0))

        # ---- down path ----
        self.down_res = nn.ModuleList()
        self.down_samp = nn.ModuleList()
        in_ch = C
        for i, mult in enumerate(ch_mult):
            out_ch = C * mult
            blocks = nn.ModuleList()
            for _ in range(config.num_res_blocks):
                blocks.append(ResNetBlock(in_ch, out_ch, emb_ch, config.num_groups,
                                           config.dropout, config.use_norm))
                in_ch = out_ch
            self.down_res.append(blocks)
            if i < num_res - 1:
                self.down_samp.append(Downsample(out_ch))

        # ---- middle ----
        mid = []
        for _ in range(2):
            mid.append(ResNetBlock(in_ch, in_ch, emb_ch, config.num_groups, 
                                   config.dropout, config.use_norm))
        if config.use_attention:
            mid.append(AttentionBlock(in_ch))
        self.mid = nn.ModuleList(mid)

        # ---- up path ----
        self.up_res = nn.ModuleList()
        self.up_samp = nn.ModuleList()

        up_in_ch = in_ch  # channels at the bottleneck output (from the mid)
        skip_chs = [C * m for m in ch_mult]  # channels of each saved skip from down path

        for i, mult in enumerate(reversed(ch_mult)):
            # 1) Upsample for *current* incoming channels (before changing up_in_ch)
            if i < len(ch_mult) - 1:
                self.up_samp.append(Upsample(up_in_ch))

            # 2) First block takes concat([upsampled h, skip]) -> out_ch
            out_ch = C * mult
            blocks = nn.ModuleList()
            skip_ch = skip_chs[len(ch_mult) - 1 - i]  # match the correct skip for this level
            blocks.append(ResNetBlock(up_in_ch + skip_ch, out_ch, emb_ch, config.num_groups,
                                    config.dropout, config.use_norm))

            # 3) Remaining blocks are out_ch -> out_ch (no extra concats)
            for _ in range(config.num_res_blocks):
                blocks.append(ResNetBlock(out_ch, out_ch, emb_ch, config.num_groups,
                                        config.dropout, config.use_norm))

            self.up_res.append(blocks)

            # 4) Update incoming channels for the *next* level
            up_in_ch = out_ch

        # ---- output convs ----
        out_c = 4 if config.ri_inp else 1   # stereo RI = 4 channels
        self.out_conv1 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.out_conv2 = nn.Conv2d(C, out_c, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_chans=4, F, T)
        t: (B,)
        returns: (B, out_chans=4, F, T)
        """
        # timestep embedding
        temb = self.t_embed(t)  # (B, 4C)

        # input conv
        h = self.input_conv(x)  # (B, C, F, T)

        # down path
        skips = []
        si = 0
        for i, blocks in enumerate(self.down_res):
            for block in blocks:
                h = block(h, temb)
            skips.append(h)
            if i < len(self.down_samp):
                h = self.down_samp[i](h)

        # middle
        for block in self.mid:
            if isinstance(block, ResNetBlock):
                h = block(h, temb)
            else:
                h = block(h)  # attention

        # up path
        for i, blocks in enumerate(self.up_res):
            if i < len(self.up_samp):
                h = self.up_samp[i](h)

            skip = skips.pop()

            # spatial align if needed
            if skip.shape[-2:] != h.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode='nearest')

            # 1st block with concat once
            h = blocks[0](torch.cat([h, skip], dim=1), temb)

            # remaining blocks without extra concat
            for b in blocks[1:]:
                h = b(h, temb)

        # output
        h = self.out_conv1(h)
        h = self.out_conv2(h)
        return torch.sigmoid(h) if self.create_mask else h