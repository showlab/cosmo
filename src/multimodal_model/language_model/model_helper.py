"""
Taken from https://github.com/lucidrains/flamingo-pytorch
"""

import torch
from einops import rearrange, repeat
from einops_exts import rearrange_many
from torch import einsum, nn
import torch.nn.functional as F
import random



# part implement is from https://github.com/lucidrains/CoCa-pytorch/blob/main/coca_pytorch/coca_pytorch.py
# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def rotate_half(x):
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())


def extend_instance(obj, mixin):
    """Apply mixins to a class instance after creation"""
    base_cls = obj.__class__
    base_cls_name = obj.__class__.__name__
    obj.__class__ = type(
        base_cls_name, (mixin, base_cls), {}
    )  # mixin needs to go first for our forward() logic to work


def getattr_recursive(obj, att):
    """
    Return nested attribute of obj
    Example: getattr_recursive(obj, 'a.b.c') is equivalent to obj.a.b.c
    """
    if att == "":
        return obj
    i = att.find(".")
    if i < 0:
        return getattr(obj, att)
    else:
        return getattr_recursive(getattr(obj, att[:i]), att[i + 1 :])


def setattr_recursive(obj, att, val):
    """
    Set nested attribute of obj
    Example: setattr_recursive(obj, 'a.b.c', val) is equivalent to obj.a.b.c = val
    """
    if "." in att:
        obj = getattr_recursive(obj, ".".join(att.split(".")[:-1]))
    setattr(obj, att.split(".")[-1], val)

# gated cross attention

def FeedForward(dim, mult=4, compress_ratio=4):
    inner_dim = int(dim * mult) // compress_ratio
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


# normalization
# they use layernorm without bias, something that pytorch does not offer


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# residual


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

# to latents



# rotary positional embedding
# https://arxiv.org/abs/2104.09864


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len, *, device):
        seq = torch.arange(max_seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = einsum("i , j -> i j", seq, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)
    

class MaskedCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_visual,
        dim_head=64,
        heads=8,
        only_attend_immediate_media=True,
        compress_ratio=4,
        qv_norm=False,
    ):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads // compress_ratio

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_visual, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # whether for text to only attend to immediate preceding image, or all previous images
        self.only_attend_immediate_media = only_attend_immediate_media

        # add QK layer norm
        self.qv_norm = qv_norm
        if self.qv_norm:
            self.q_norm = nn.LayerNorm(inner_dim)
            self.k_norm = nn.LayerNorm(inner_dim)

    def forward(self, x, media, media_locations=None, use_cached_media=False):
        """
        Args:
            x (torch.Tensor): text features
                shape (B, T_txt, D_txt)
            media (torch.Tensor): image features
                shape (B, T_img, n, D_img) where n is the dim of the latents
            media_locations: boolean mask identifying the media tokens in x
                shape (B, T_txt)
            use_cached_media: bool
                If true, treat all of x as if they occur after the last media
                registered in media_locations. T_txt does not need to exactly
                equal media_locations.shape[1] in this case
        """
        if not use_cached_media:
            assert (
                media_locations.shape[1] == x.shape[1]
            ), f"media_location.shape is {media_locations.shape} but x.shape is {x.shape}"
        T_txt = x.shape[1]
        _, T_img, n = media.shape[:3]
        h = self.heads

        x = self.norm(x)

        q = self.to_q(x)
        media = rearrange(media, "b t n d -> b (t n) d")

        k, v = self.to_kv(media).chunk(2, dim=-1)

        # follow paper https://arxiv.org/pdf/2302.05442.pdf and file:///C:/Users/jinpeng.wang/Desktop/Paper/Datasets/arxiv_23_6_obelics.pdf, introduce q, k normalization here

        if self.qv_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=h)

        q = q * self.scale

        sim = einsum("... i d, ... j d -> ... i j", q, k)
        # size: (batch_size, heads, sequence_length_text, sequence_length_media * n)
        #  n is the dim of the latents

        if exists(media_locations):
            # at each boolean of True, increment the time counter (relative to media time)
            media_time = torch.arange(T_img, device=x.device) + 1
            if use_cached_media:
                # text time is set to the last cached media location
                text_time = repeat(
                    torch.count_nonzero(media_locations, dim=1),
                    "b -> b i",
                    i=T_txt,
                )
            else:
                # at each boolean of True, increment the time counter (relative to media time)
                text_time = media_locations.cumsum(dim=-1)
            # >>> x = torch.tensor([0,0,1,0,1])
            # >>> print(x.cumsum(dim=-1))
            # tensor([0, 0, 1, 1, 2])
            # example:
            # media_locations = torch.tensor([[1, 0, 1, 0, 0], [0, 1, 0, 0, 1]])
            # text_time = media_locations.cumsum(dim=-1)
            # text_time
            #    tensor([[1, 1, 2, 2, 2],
            #            [0, 1, 1, 1, 2]])
            # If T_img equals 5, the command torch.arange(T_img, device=x.device) + 1 would output a tensor [1, 2, 3, 4, 5]
            # text time must equal media time if only attending to most immediate image
            # otherwise, as long as text time is greater than media time (if attending to all previous images / media)
            mask_op = torch.eq if self.only_attend_immediate_media else torch.ge
            # torch.ge: element-wise comparison of two tensors using the greater than or equal to (>=) operator
            # x = torch.tensor([1, 2, 3, 4, 5])
            # y = torch.tensor([3, 3, 3, 3, 3])
            # torch.ge(x, y) = tensor([False, False,  True,  True,  True])
            # so if text_time >= media_time, then text_to_media_mask = True
            # in other words, text attend to all previous images
            text_to_media_mask = mask_op(
                rearrange(text_time, "b i -> b 1 i 1"),
                repeat(media_time, "j -> 1 1 1 (j n)", n=n),
            )
            # creating a mask that specifies which text tokens should attend to which media tokens
            # The shape of this mask is (batch_size, 1, sequence_length (txt_len), sequence_length * n).
            # can not give zero directly to masked_fill, so use a very large negative number
            # so that softmax will give close to zero
            sim = sim.masked_fill(~text_to_media_mask, -torch.finfo(sim.dtype).max)

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # masked_fill as zero after softmax to keep accuracy
        # equals not to predict all text tokens before the first <visual> if 
        if exists(media_locations) and self.only_attend_immediate_media:
            # any text without a preceding media needs to have attention zeroed out
            text_without_media_mask = text_time == 0
            text_without_media_mask = rearrange(
                text_without_media_mask, "b i -> b 1 i 1"
            )
            attn = attn.masked_fill(text_without_media_mask, 0.0)

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class GatedCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_visual,
        compress_ratio=1,
        dim_head=64,
        heads=8,
        ff_mult=4,
        only_attend_immediate_media=True,
        qv_norm=False,
    ):
        super().__init__()
        self.attn = MaskedCrossAttention(
            dim=dim,
            dim_visual=dim_visual,
            dim_head=dim_head,
            heads=heads,
            compress_ratio=compress_ratio,
            only_attend_immediate_media=only_attend_immediate_media,
            qv_norm=qv_norm,
        )
        self.attn_gate = nn.Parameter(torch.tensor([0.0]))
        self.dim = dim
        self.dim_head = dim_head
        self.ff = FeedForward(dim, mult=ff_mult, compress_ratio=compress_ratio)
        self.ff_gate = nn.Parameter(torch.tensor([0.0]))

    def forward(
        self,
        x,
        media,
        media_locations=None,
        use_cached_media=False,
    ):
        x = (
            self.attn(
                x,
                media,
                media_locations=media_locations,
                use_cached_media=use_cached_media,
            )
            * self.attn_gate.tanh()
            + x
        )
        x = self.ff(x) * self.ff_gate.tanh() + x
        # if random.random() < 0.1:
        #     print("gate value in gated cross attention, attn_gate, ff_gate", self.attn_gate.tanh().item(), self.ff_gate.tanh().item())
        return x