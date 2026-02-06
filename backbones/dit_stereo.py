import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# =============================================================
# Timestep embeddings (DDPM or Sine) + MLP
# =============================================================
def _ddpm_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    device = timesteps.device
    half = dim // 2
    freqs = torch.exp(torch.arange(half, device=device, dtype=torch.float32) *
                      (-math.log(10000.0) / max(half - 1, 1)))
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (B, 2*half)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb

def _sine_timestep_embedding(timesteps: torch.Tensor, dim: int, max_freq: float = 1000.0) -> torch.Tensor:
    device = timesteps.device
    half = dim // 2
    freqs = torch.exp(torch.linspace(math.log(1.0), math.log(max_freq), half, device=device))
    ang = 2.0 * math.pi * timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb

class DiffusionTimeEmbedding(nn.Module):
    def __init__(self, dim: int, use_sine: bool = False, max_freq: float = 1000.0):
        super().__init__()
        self.dim = dim
        self.use_sine = use_sine
        self.max_freq = max_freq
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )

        # Per-t learnable step scale s(t)
        self.scale_head = nn.Sequential(nn.SiLU(), nn.Linear(dim, 1))
        nn.init.zeros_(self.scale_head[-1].weight); nn.init.zeros_(self.scale_head[-1].bias)

        self.s_logit_cap = 3.0  # ~sigmoid(±3) ≈ [0.047,0.953]

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        base = _sine_timestep_embedding(t, self.dim, self.max_freq) if self.use_sine else _ddpm_timestep_embedding(t, self.dim)
        emb = self.mlp(base)                      # (B,D)
        s_logits = self.scale_head(emb).clamp(-self.s_logit_cap, self.s_logit_cap)
        s = torch.sigmoid(s_logits)

        return emb, s


# =============================================================
# (Optional) Rotary Positional Embedding (1D over token index)
# Light-weight; enabled by cfg.use_rope
# =============================================================
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_pos: int = 8192):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_pos).float().unsqueeze(1) * inv_freq.unsqueeze(0)
        self.register_buffer("cos_cached", torch.cos(t).repeat(1, 2), persistent=False)  # (max_pos, dim)
        self.register_buffer("sin_cached", torch.sin(t).repeat(1, 2), persistent=False)

    def rotate(self, x, cos, sin):
        x1, x2 = x[..., ::2], x[..., 1::2]
        y = torch.stack([-x2, x1], dim=-1).reshape_as(x)
        return (x * cos) + (y * sin)

    def forward(self, q, k):
        # q,k: (B, h, N, d)
        N = q.shape[2]
        cos = self.cos_cached[:N].unsqueeze(0).unsqueeze(0).to(q.device)  # (1,1,N,d)
        sin = self.sin_cached[:N].unsqueeze(0).unsqueeze(0).to(q.device)
        return self.rotate(q, cos, sin), self.rotate(k, cos, sin)



# =============================================================
# 2D Sin-Cos positional embedding for patch tokens
# =============================================================
def _build_2d_sincos_pos_embed(Fp: int, Tp: int, dim: int, device) -> torch.Tensor:
    """Return (Fp*Tp, dim) sin-cos 2D positional encodings."""
    assert dim % 2 == 0
    dim_h = dim // 2
    def _pe(n, d):
        pos = torch.arange(n, device=device).float().unsqueeze(1)  # (n,1)
        i = torch.arange(d // 2, device=device).float().unsqueeze(0)  # (1,d/2)
        div = torch.exp(i * (-math.log(10000.0) / max(d // 2 - 1, 1)))
        ang = pos * div
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)  # (n,d)
    pe_f = _pe(Fp, dim_h)  # (Fp, dim/2)
    pe_t = _pe(Tp, dim_h)  # (Tp, dim/2)
    pe_2d = (pe_f[:, None, :].repeat(1, Tp, 1), pe_t[None, :, :].repeat(Fp, 1, 1))
    pe = torch.cat(pe_2d, dim=2).reshape(Fp * Tp, dim)  # (Fp*Tp, dim)
    return pe

# =============================================================
# Patch embed / unembed (supports optional overlap in time)
# =============================================================
class PatchEmbed(nn.Module):
    def __init__(self, in_ch: int, embed_dim: int, patch_f: int, patch_t: int, time_stride: int | None = None):
        super().__init__()
        self.patch_f = patch_f
        self.patch_t = patch_t
        if time_stride is None:
            time_stride = patch_t
        self.time_stride = time_stride

        # Exact geometric extractor (no learnable mixing)
        self.unfold = nn.Unfold(kernel_size=(patch_f, patch_t),
                                stride=(patch_f, time_stride))

        self.in_ch = in_ch
        self.patch_vec = in_ch * patch_f * patch_t

        # Learnable token projection (patch_vec -> D)
        self.proj = nn.Linear(self.patch_vec, embed_dim, bias=True)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor):
        # x: (B, C, F, T)
        B, C, F, T = x.shape
        cols = self.unfold(x)  # (B, C*pf*pt, L) where L = Fp*Tp
        B, PV, L = cols.shape
        tokens = self.proj(cols.transpose(1, 2))  # (B, L, D)
        # Recover the 2D grid
        Fp = F // self.patch_f
        Tp = 1 + (T - self.patch_t) // self.time_stride
        return tokens, (Fp, Tp)
    

class PatchUnembed(nn.Module):
    def __init__(self, out_ch, embed_dim, patch_f, patch_t, time_stride=None):
        super().__init__()
        if time_stride is None: time_stride = patch_t
        self.out_ch = out_ch
        self.embed_dim = embed_dim
        self.patch_f = patch_f
        self.patch_t = patch_t
        self.time_stride = time_stride

        self.patch_vec = out_ch * patch_f * patch_t

        # Token -> patch pixels
        self.proj = nn.Linear(embed_dim, self.patch_vec, bias=True)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, tokens, grid_shape):
        B, N, D = tokens.shape
        Fp, Tp = grid_shape
        assert N == Fp * Tp

        patches = self.proj(tokens)  # (B, N, C*pf*pt)
        patches = patches.transpose(1, 2)  # (B, C*pf*pt, N)

        out = F.fold(
            patches, 
            output_size=(Fp * self.patch_f, (Tp - 1) * self.time_stride + self.patch_t),
            kernel_size=(self.patch_f, self.patch_t),
            stride=(self.patch_f, self.time_stride)
        )
        # Average overlaps
        ones = torch.ones_like(patches)
        denom = F.fold(
            ones, 
            output_size=(Fp * self.patch_f, (Tp - 1) * self.time_stride + self.patch_t),
            kernel_size=(self.patch_f, self.patch_t),
            stride=(self.patch_f, self.time_stride)
        ).clamp_min(1e-8)
        out = out / denom  # (B, C, F*, T*)
        return out

# =============================================================
# Flash/SDPA Attention (fast path via PyTorch 2.9 kernels)
# =============================================================
class SDPAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.h = num_heads
        self.d = embed_dim
        self.dh = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

        # NEW: per-head log-temperature, init ~1.2
        self.log_tau = nn.Parameter(torch.log(torch.ones(num_heads) * 1.2))

    def forward(self, x, rope: RotaryEmbedding | None = None, rel_bias: torch.Tensor | None = None):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B,h,N,dh)

        if rope is not None:
            q, k = rope(q, k)

        # --- Remove for ablation
        # --- Cosine attention: L2 normalize q,k along dh
        # eps = 1e-6
        # q = q / (q.norm(dim=-1, keepdim=True) + eps)
        # k = k / (k.norm(dim=-1, keepdim=True) + eps)

        # Per-head temperature (broadcast to B,N,dh)
        # tau = torch.nn.functional.softplus(self.log_tau).view(1, self.h, 1, 1).clamp(max=1.5)  # (1,h,1,1)
        # q = q * tau * (self.dh ** 0.5)  # keep overall scale similar to SDPA default


        # Prepare additive bias (B*h,N,N)
        attn_mask = None
        if rel_bias is not None:
            rb = rel_bias.to(dtype=q.dtype, device=q.device)    # (h,N,N)
            attn_mask = rb.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * self.h, N, N)

        q_ = q.reshape(B * self.h, N, self.dh)
        k_ = k.reshape(B * self.h, N, self.dh)
        v_ = v.reshape(B * self.h, N, self.dh)

        y = F.scaled_dot_product_attention(
            q_, k_, v_,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False
        )
        y = y.reshape(B, self.h, N, self.dh).permute(0, 2, 1, 3).contiguous().reshape(B, N, D)
        return self.proj(y)


class RelPosBias2D(nn.Module):
    """
    Bounded, zero-centered 2D relative position bias.
    rel_bias = scale * tanh(raw_bias_f ⊕ raw_bias_t), centered per-head.
    """
    def __init__(self, num_heads: int, Fp: int, Tp: int, init_scale: float = 0.5):
        super().__init__()
        self.num_heads = num_heads
        self.Fp = Fp
        self.Tp = Tp

        # Raw (unbounded) 1D biases
        self.raw_f = nn.Parameter(torch.zeros(num_heads, 2 * Fp - 1))
        self.raw_t = nn.Parameter(torch.zeros(num_heads, 2 * Tp - 1))
        nn.init.zeros_(self.raw_f); nn.init.zeros_(self.raw_t)

        # Learnable global scale (softplus to keep positive, small init)
        self.log_scale = nn.Parameter(torch.log(torch.tensor(init_scale)))

        # Precompute index maps
        df = torch.arange(Fp)[:, None] - torch.arange(Fp)[None, :]
        dt = torch.arange(Tp)[:, None] - torch.arange(Tp)[None, :]
        self.register_buffer("df_index", (df + (Fp - 1)).long(), persistent=False)
        self.register_buffer("dt_index", (dt + (Tp - 1)).long(), persistent=False)

    def forward(self) -> torch.Tensor:
        h, Fp, Tp = self.num_heads, self.Fp, self.Tp
        bf = torch.tanh(self.raw_f)[:, self.df_index]          # (h,Fp,Fp), bounded
        bt = torch.tanh(self.raw_t)[:, self.dt_index]          # (h,Tp,Tp), bounded
        b = bf[:, :, None, :, None] + bt[:, None, :, None, :]  # (h,Fp,Tp,Fp,Tp)
        b = b.view(h, Fp * Tp, Fp * Tp)                        # (h,N,N)
        # zero-center per head (don’t shift all logits)
        b = b - b.mean(dim=(1, 2), keepdim=True)
        scale = torch.nn.functional.softplus(self.log_scale).clamp(max=0.2)   # >0
        return b * scale

class CrossAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.h = num_heads
        self.d = embed_dim
        self.dh = embed_dim // num_heads
        self.q = nn.Linear(embed_dim, embed_dim)
        self.kv = nn.Linear(embed_dim, 2 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        # inits: let the output start as zero (identity-through-residual)
        nn.init.xavier_uniform_(self.q.weight);  nn.init.zeros_(self.q.bias)
        nn.init.xavier_uniform_(self.kv.weight); nn.init.zeros_(self.kv.bias)
        nn.init.zeros_(self.proj.weight);        nn.init.zeros_(self.proj.bias)

    def forward(self, x, cond):
        # x: (B,N,D) -> Q ; cond: (B,M,D) -> K,V
        B, N, D = x.shape
        M = cond.shape[1]
        q = self.q(x).view(B, N, self.h, self.dh).permute(0,2,1,3)               # (B,h,N,dh)
        kv = self.kv(cond).view(B, M, 2, self.h, self.dh).permute(2,0,3,1,4)     # (2,B,h,M,dh)
        k, v = kv[0], kv[1]                                                      # (B,h,M,dh)
        q = q.reshape(B*self.h, N, self.dh)
        k = k.reshape(B*self.h, M, self.dh)
        v = v.reshape(B*self.h, M, self.dh)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        y = y.reshape(B, self.h, N, self.dh).permute(0,2,1,3).contiguous().view(B, N, D)
        return self.proj(y)


# =============================================================
# Conditional Transformer block with AdaLN-style conditioning
# =============================================================
class TransformerBlockCond(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float=0.0, use_rope: bool=False, 
                 ffn_activation: str = "gelu",  norm_type: str = "layernorm", groupnorm_groups: int = 8, 
                 use_adaln: bool = True):
        super().__init__()

        self.ffn_activation = ffn_activation
        self.norm_type = norm_type
        self.use_adaln = use_adaln
        self.groupnorm_groups = groupnorm_groups

        # --- Norms (token-wise, but GroupNorm needs a helper) ---
        if norm_type == "groupnorm":
            # We'll wrap the call; C = embed_dim, H*W = tokens
            self.norm1 = nn.GroupNorm(groupnorm_groups, embed_dim)
            self.norm2 = nn.GroupNorm(groupnorm_groups, embed_dim)
        else:
            self.norm1 = nn.LayerNorm(embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)


        self.attn  = SDPAttention(embed_dim, num_heads)
        self.ffn_up = nn.Linear(embed_dim, 8 * embed_dim)
        self.ffn_dn = nn.Linear(4 * embed_dim,  embed_dim)

        # zero-init residual readouts (AdaLN-zero style)
        nn.init.zeros_(self.attn.proj.weight); nn.init.zeros_(self.attn.proj.bias)
        nn.init.zeros_(self.ffn_dn.weight);    nn.init.zeros_(self.ffn_dn.bias)

        # time-cond MLPs (as you already have)
        def zero_linear(i,o):
            lin = nn.Linear(i,o); nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias); return lin
        self.cond1 = nn.Sequential(nn.SiLU(), zero_linear(embed_dim, 3*embed_dim))
        self.cond2 = nn.Sequential(nn.SiLU(), zero_linear(embed_dim, 3*embed_dim))

        with torch.no_grad():
            D = embed_dim
            self.cond1[-1].bias[2*D:3*D].fill_(0.05)
            self.cond2[-1].bias[2*D:3*D].fill_(0.05)

        self.res_scale = 1.0 / (2**0.5)
        self.use_rope = use_rope
        self.rope = RotaryEmbedding(embed_dim // num_heads) if use_rope else None

        # NEW: input-conditioning cross-attention
        self.xattn = CrossAttention(embed_dim, num_heads)

    def _apply_norm(self, norm, x):
        """
        x: (B, N, D)
        - LayerNorm: apply directly.
        - GroupNorm: treat D as channels, N as spatial positions.
        """
        if isinstance(norm, nn.GroupNorm):
            B, N, D = x.shape
            # (B, N, D) -> (B, D, N)
            x = x.transpose(1, 2)          # channels = D
            x = norm(x)                    # GroupNorm over D
            x = x.transpose(1, 2)          # back to (B, N, D)
            return x
        else:
            return norm(x)


    def _mod(self, x, tproj):
        if (not self.use_adaln) or (tproj is None):
            B, N, D = x.shape
            gate = x.new_ones(B, D)
            return x, gate
        shift, scale, gate = tproj.chunk(3, dim=-1)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return x, gate

    def forward(self, x, t_emb, grid_shape=None, cond_tokens=None):
        # Self-attention
        y = self._apply_norm(self.norm1, x)
        if self.use_adaln:
            t1 = self.cond1(t_emb)
        else:
            t1 = None
        y, g1 = self._mod(y, t1)
        y = self.attn(y, self.rope if hasattr(self, "rope") else None)
        x = x + self.res_scale * (y * g1.unsqueeze(1))

        # NEW: cross-attention to input tokens
        if cond_tokens is not None:
            y2 = self.xattn(x, cond_tokens)    # starts as 0-contribution
            x  = x + self.res_scale * y2

        # FFN
        y = self._apply_norm(self.norm2, x)
        if self.use_adaln:
            t2 = self.cond2(t_emb)
        else:
            t2 = None
        y, g2 = self._mod(y, t2)
        u, g = self.ffn_up(y).chunk(2, dim=-1)
        if self.ffn_activation.lower() == "swiglu":
            h = F.silu(u) * g           # SwiGLU
        else:
            h = F.gelu(u) * g           # current behavior (GEGLU-like)
        y = self.ffn_dn(h)
        x = x + self.res_scale * (y * g2.unsqueeze(1))
        return x

    

# =============================================================
# Full Diffusion Transformer 
# =============================================================
class TransformerDiffuser(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.residual_prediction = config.residual_prediction
        self.use_conditioner = config.use_conditioner
        self.use_smap = config.use_smap
        self.use_stereo_mask = config.use_stereo_mask
        self.shared_stereo_mask = config.shared_stereo_mask

        self.patch_embed = PatchEmbed(config.in_chans, config.embed_dim,
                                      config.patch_f, config.patch_t,
                                      time_stride=config.time_stride)
        self.t_embed = DiffusionTimeEmbedding(config.embed_dim,
                                              use_sine=config.continuous_emb,
                                              max_freq=config.max_freq)
        self.blocks = nn.ModuleList([
                        TransformerBlockCond(
                            config.embed_dim,
                            config.num_heads,
                            config.dropout,
                            use_rope=config.use_rope,
                            ffn_activation=config.ffn_activation,
                            norm_type=config.norm_type,
                            groupnorm_groups=config.groupnorm_groups,
                            use_adaln=config.use_adaln
                        )
                        for _ in range(config.num_layers)
                    ])
                
        self.patch_unembed = PatchUnembed(config.in_chans, config.embed_dim,
                                          config.patch_f, config.patch_t,
                                          time_stride=config.time_stride)

        self.register_buffer("_pe_cache", torch.zeros(1), persistent=False)
        self._pe_shape = None
        self.block_embed = nn.Embedding(config.num_layers, config.embed_dim)

        gain_init = 0.1 if self.residual_prediction else 1.0  # 0.05–0.2 also fine for residual
        self.out_gain = nn.Parameter(torch.ones(1, self.cfg.in_chans, 1, 1) * gain_init)

        # NEW: slot for relative position bias per grid
        self._relpos = None                         # NEW

        # NEW: conditioner head (optional)
        if self.use_conditioner:
            self.conditioner = nn.Sequential(
                nn.Conv2d(config.in_chans, config.in_chans, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(config.in_chans, config.in_chans, kernel_size=3, padding=1)
                )

        # NEW: s_map head (optional)
        if self.use_smap:
            self.s_out = nn.Sequential(
                nn.Conv2d(config.in_chans, config.in_chans, kernel_size=1),
                nn.Sigmoid()
            )

    def _pos_embed(self, grid_shape, device):
        if self.cfg.pos_embed == "none" or self.cfg.use_rope:
            return None
        Fp, Tp = grid_shape
        if self._pe_shape != grid_shape or (not torch.is_tensor(self._pe_cache)) or self._pe_cache.numel() <= 1:
            pe = _build_2d_sincos_pos_embed(Fp, Tp, self.cfg.embed_dim, device)
            self._pe_cache = pe
            self._pe_shape = grid_shape
        return self._pe_cache

    def forward(self, x: torch.Tensor, t: torch.Tensor, sc: torch.Tensor | None = None) -> torch.Tensor:
        B, C, F_in, T_in = x.shape
        tokens, grid = self.patch_embed(x)           # (B, N, D), grid=(Fp,Tp)

        base_cond = tokens.detach()         # freeze conditioning branch (stable)

        assert tokens.shape[1] == grid[0] * grid[1], f"N={tokens.shape[1]} vs Fp*Tp={grid}"
        Fp, Tp = grid

        # if self-conditioning provided, concat its tokens as extra condition
        extra_cond = None
        if sc is not None:
            extra_cond = self.patch_embed(sc)[0].detach()
            cond_tokens = torch.cat([base_cond, extra_cond], dim=1)
        else:
            cond_tokens = base_cond

        # Conditioner head: provide coarse residual estimate from x
        if self.use_conditioner:
            x_coarse = self.conditioner(x)  # (B, 4, F, T)
            coarse_tokens, _ = self.patch_embed(x_coarse)
            cond_tokens = torch.cat([cond_tokens, coarse_tokens.detach()], dim=1)

        # 2D absolute PE (if enabled)
        pe = self._pos_embed(grid, tokens.device)
        if pe is not None:
            tokens = tokens + pe.unsqueeze(0)

        # NEW: build / refresh relative 2D bias for this grid
        if (self._relpos is None) or (self._relpos.Fp != Fp) or (self._relpos.Tp != Tp):
            self._relpos = RelPosBias2D(self.cfg.num_heads, Fp, Tp).to(tokens.device)

        # time embedding
        t_emb, s_scalar = self.t_embed(t)


        # transformer blocks (pass rel_bias to attention)
        for i, blk in enumerate(self.blocks):
            t_blk = t_emb + self.block_embed.weight[i].unsqueeze(0)
            tokens = blk(tokens, t_blk, grid_shape=grid, cond_tokens=cond_tokens)   # pass (Fp,Tp)

        # light readout → unembed
        B, N, D = tokens.shape
        out = self.patch_unembed(tokens, grid)       # (B, 4, F*, T*)

        # crop/pad to input size
        out = out[..., :F_in, :T_in]
        F_out, T_out = out.shape[-2], out.shape[-1]
        if (F_out != F_in) or (T_out != T_in):
            pad_f = max(0, F_in - F_out)
            pad_t = max(0, T_in - T_out)
            if pad_f or pad_t:
                out = F.pad(out, (0, pad_t, 0, pad_f))
            out = out[..., :F_in, :T_in]

        out = out * self.out_gain  # (B,4,F,T) scaled per channel

        # residual: predict v_hat and s_map or s_scalar
        if self.residual_prediction:
            if self.use_smap:
                s = self.s_out(out)
            else:
                s = s_scalar.view(B, 1, 1, 1)
            return out, s
        return out, s_scalar


def reinit_projections_orthonormal(model):
    """
    Rebuilds patch_embed.proj and patch_unembed.proj to match current cfg,
    sets encoder weight to orthonormal rows (shape D x patch_vec),
    and decoder weight to its pseudo-inverse (shape patch_vec x D).
    Also makes PatchEmbed.forward derive Tp from Unfold's L.
    """
    device = next(model.parameters()).device
    pf = model.cfg.patch_f
    pt = model.cfg.patch_t
    ts = model.cfg.time_stride
    in_ch = model.cfg.in_chans
    D = model.cfg.embed_dim

    patch_vec = in_ch * pf * pt  # e.g., 4*19*11 = 836

    # --- Recreate Linear layers to exact shapes ---
    model.patch_embed.proj = nn.Linear(patch_vec, D, bias=True).to(device)      # weight: (D, patch_vec)
    model.patch_unembed.proj = nn.Linear(D, patch_vec, bias=True).to(device)    # weight: (patch_vec, D)

    # --- Orthonormal encoder; decoder = pseudo-inverse ---
    with torch.no_grad():
        # Encoder W: (D, patch_vec) with orthonormal rows when D <= patch_vec
        W = torch.empty(D, patch_vec, device=device)
        nn.init.orthogonal_(W)                             # safe for any shape
        model.patch_embed.proj.weight.copy_(W)             # (D, patch_vec)
        model.patch_embed.proj.bias.zero_()

        # Pseudo-inverse: (patch_vec, D)
        W_pinv = torch.linalg.pinv(W)                      # (patch_vec, D)
        model.patch_unembed.proj.weight.copy_(W_pinv)      # <-- NO .T
        model.patch_unembed.proj.bias.zero_()

    # --- Ensure PatchEmbed derives Tp from Unfold's output L (avoids off-by-one) ---
    def patched_forward(x: torch.Tensor):
        B, C, Freq, Time = x.shape
        cols = model.patch_embed.unfold(x)                 # (B, C*pf*pt, L)
        L = cols.shape[-1]
        tokens = model.patch_embed.proj(cols.transpose(1, 2))  # (B, L, D)
        Fp = Freq // pf
        Tp = L // Fp                                       # derive Tp from L
        return tokens, (Fp, Tp)
    model.patch_embed.forward = patched_forward




