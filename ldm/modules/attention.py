from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
import pickle
import os

from ldm.modules.diffusionmodules.util import checkpoint


def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


def calc_token_mean_std(feat: torch.Tensor, eps: float = 1e-5):
    mean = feat.mean(dim=2, keepdim=True)
    var = feat.var(dim=2, unbiased=False, keepdim=True)
    std = torch.sqrt(var + eps)
    return mean, std


def token_mean_variance_norm(feat: torch.Tensor) -> torch.Tensor:
    mean, std = calc_token_mean_std(feat)
    return (feat - mean) / std


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

        self.attn = None
        self.Q_c = None
        self.K_c = None
        self.V_c = None
        self.Q_s = None
        self.K_s = None
        self.V_s = None
        self.Q_cs = None
        self.K_cs = None
        self.V_cs = None
        self.Q_hat_cs = None
        self.ada_gate = nn.Parameter(torch.tensor(0.2))
        # toggle hooks for staged AdaAttN behaviour
        self.use_mean_shift = True      # Step 1: mean-only modulation
        self.use_std_scale =   True      # Step 2: add std-based affine scaling
        self.use_residual_gate = True   # Step 3: gated blend with base attention
        self.style_scale = 1.0          # Scaling factor for style values
        self.lambda_max = 2.5
        self.style_topk = 128

    def forward(self,
                x,
                context=None,
                mask=None,
                Q_c_injected=None,
                K_s_injected=None,
                V_s_injected=None,
                injection_config=None,):
        self.attn = None
        h = self.heads
        b = x.shape[0]
        attn_matrix_scale = 1.0
        q_mix = 0.
        if injection_config is not None:
            attn_matrix_scale = injection_config['T']
            q_mix = injection_config['gamma']

        context = default(context, x)

        Q_c = None
        K_c = None
        V_c = None
        Q_s = None
        K_s = None
        V_s = None
        Q_cs = None
        K_cs = None
        V_cs = None

        Q_hat_cs = None

        if Q_c_injected is None:
            Q_cs = self.to_q(x)
            Q_cs = rearrange(Q_cs, 'b n (h d) -> (b h) n d', h=h)
            Q_hat_cs = Q_cs
        else:
            Q_c = Q_c_injected
            Q_c = torch.cat([Q_c]*b)

            Q_cs = self.to_q(x)
            Q_cs = rearrange(Q_cs, 'b n (h d) -> (b h) n d', h=h)



            Q_hat_cs = Q_c * q_mix + Q_cs * (1. - q_mix)

            

        K_cs = self.to_k(context)
        K_cs = rearrange(K_cs, 'b m (h d) -> (b h) m d', h=h)
        V_cs = self.to_v(context)
        V_cs = rearrange(V_cs, 'b m (h d) -> (b h) m d', h=h)

        q = Q_hat_cs

        self.Q_c = Q_c if Q_c_injected is not None else Q_hat_cs
        self.K_c = K_c
        self.V_c = V_c
        self.Q_s = Q_s
        self.K_s = K_s if K_s is not None else K_cs
        self.V_s = V_s if V_s is not None else V_cs
        self.Q_cs = Q_cs
        self.K_cs = K_cs
        self.V_cs = V_cs
        self.Q_hat_cs = Q_hat_cs

        # Joint softmax over style + content tokens when style keys are injected; fallback to original path otherwise
        K_s_expanded = None
        V_s_expanded = None
        if K_s_injected is not None:
            K_s = K_s_injected
            K_s_expanded = torch.cat([K_s_injected] * b, dim=0)
        if V_s_injected is not None:
            V_s = V_s_injected
            V_s_expanded = torch.cat([V_s_injected] * b, dim=0)

        if K_s_expanded is not None:
            style_logits = einsum('b i d, b j d -> b i j', q, K_s_expanded)
            content_logits = einsum('b i d, b j d -> b i j', q, K_cs)

            style_logits = style_logits * attn_matrix_scale
            style_logits = style_logits * self.scale
            content_logits = content_logits * self.scale

            # --- Adaptive per-query lambda_map (ONLY new param: self.lambda_max) ---
            s = style_logits
            c = content_logits

            n_q = style_logits.shape[1]
            n_s = style_logits.shape[2]
            n_c = content_logits.shape[2]

            s = s.reshape(b, h, n_q, n_s)
            c = c.reshape(b, h, n_q, n_c)

            r = torch.logsumexp(s, dim=-1) - torch.logsumexp(c, dim=-1)
            r = r.clamp(-20.0, 20.0)
            r = r.mean(dim=1)

            gate = torch.sigmoid(r)
            lambda_map = 1.0 + (self.lambda_max - 1.0) * gate
            lambda_map = lambda_map[:, None, :, None].to(dtype=style_logits.dtype)

            style_logits = style_logits.reshape(b, h, n_q, n_s) * lambda_map
            style_logits = style_logits.reshape(b * h, n_q, n_s)
            # --- end adaptive lambda_map ---

            # --- Top-k sparsify style logits (only new param: self.style_topk) ---
            k = int(getattr(self, "style_topk", 0) or 0)
            if k > 0:
                Ns = style_logits.shape[-1]
                if k < Ns:
                    # do topk in fp32 for stability under fp16/bf16
                    sl = style_logits.float()
                    topk_vals, topk_idx = torch.topk(sl, k=k, dim=-1, largest=True, sorted=False)
                    # fill with very negative number (safer than -inf in some kernels)
                    neg = torch.finfo(style_logits.dtype).min
                    sparse_style = style_logits.new_full(style_logits.shape, neg)
                    sparse_style.scatter_(-1, topk_idx.to(device=sparse_style.device), topk_vals.to(dtype=sparse_style.dtype))
                    style_logits = sparse_style
            # --- end Top-k ---

            logits = torch.cat([style_logits, content_logits], dim=-1)

            if exists(mask):
                mask = rearrange(mask, 'b ... -> b (...)')
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                style_mask = torch.ones(mask.shape[0], 1, style_logits.shape[-1], device=mask.device, dtype=mask.dtype)
                full_mask = torch.cat([style_mask, mask], dim=-1)
                max_neg_value = -torch.finfo(logits.dtype).max
                logits.masked_fill_(~full_mask, max_neg_value)

            attn = logits.softmax(dim=-1)
            self.attn = attn

            v_style = V_s_expanded if V_s_expanded is not None else V_cs
            v_style = v_style * self.style_scale
            v_content = V_cs
            v = torch.cat([v_style, v_content], dim=1)
            out = einsum('b i j, b j d -> b i d', attn, v)
        else:
            k = K_cs
            v = V_s_expanded if V_s_expanded is not None else V_cs
            if V_s_expanded is not None:
                v = v * self.style_scale
            sim = einsum('b i d, b j d -> b i j', q, k)
            sim *= self.scale

            if exists(mask):
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                sim.masked_fill_(~mask, max_neg_value)

            attn = sim.softmax(dim=-1)
            self.attn = attn
            out = einsum('b i j, b j d -> b i d', attn, v)

        # adaattn-inspired modulation
        Attn_cs_head = rearrange(attn, '(b h) n m -> b h n m', h=h)
        V_s_head = rearrange(v, '(b h) m d -> b h m d', h=h)
        Q_hat_cs_head = rearrange(q, '(b h) n d -> b h n d', h=h)

        V_cs_mean_head = torch.einsum('bhnm,bhmd->bhnd', Attn_cs_head, V_s_head)
        Q_hat_cs_norm_head = token_mean_variance_norm(Q_hat_cs_head)
        V_cs_mean_tokens = Q_hat_cs_norm_head + V_cs_mean_head
        V_cs_mean_tokens = rearrange(V_cs_mean_tokens, 'b h n d -> (b h) n d')

        out_mod = out
        if self.use_mean_shift:
            out_mod = V_cs_mean_tokens

        if self.use_std_scale:
            V_cs_sq_mean_head = torch.einsum('bhnm,bhmd->bhnd', Attn_cs_head, V_s_head * V_s_head)
            V_cs_std_head = torch.sqrt(torch.clamp(V_cs_sq_mean_head - V_cs_mean_head ** 2, min=1e-5))
            V_cs_affine_head = V_cs_std_head * Q_hat_cs_norm_head + V_cs_mean_head
            V_cs_affine_tokens = rearrange(V_cs_affine_head, 'b h n d -> (b h) n d')
            out_mod = V_cs_affine_tokens

        if self.use_residual_gate:
            gate = torch.tanh(self.ada_gate)
            out_mod = out + gate * (out_mod - out)  # out_mod = out * (1 - gate) + gate * out_mod

        out = out_mod

        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint
        
    def forward(self,
                x,
                context=None,
                self_attn_Q_c_injected=None,
                self_attn_K_s_injected=None,
                self_attn_V_s_injected=None,
                injection_config=None,
                ):
        return checkpoint(self._forward, (x,
                                          context,
                                          self_attn_Q_c_injected,
                                          self_attn_K_s_injected,
                                          self_attn_V_s_injected,
                                          injection_config,), self.parameters(), self.checkpoint)

    def _forward(self,
                 x,
                 context=None,
                 self_attn_Q_c_injected=None,
                 self_attn_K_s_injected=None,
                 self_attn_V_s_injected=None,
                 injection_config=None):
        x_ = self.attn1(self.norm1(x),
                       Q_c_injected=self_attn_Q_c_injected,
                       K_s_injected=self_attn_K_s_injected,
                       V_s_injected=self_attn_V_s_injected,
                       injection_config=injection_config,)
        x = x_ + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self,
                x,
                context=None,
                self_attn_Q_c_injected=None,
                self_attn_K_s_injected=None,
                self_attn_V_s_injected=None,
                injection_config=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')

        for block in self.transformer_blocks:
            x = block(x,
                      context=context,
                      self_attn_Q_c_injected=self_attn_Q_c_injected,
                      self_attn_K_s_injected=self_attn_K_s_injected,
                      self_attn_V_s_injected=self_attn_V_s_injected,
                      injection_config=injection_config)

            
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in
