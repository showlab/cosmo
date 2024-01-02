# --------------------------------------------------------
# MediaSparseFormer
# Copyright 2023 Ziteng Gao
# Licensed under The MIT License
# Written by Ziteng Gao
# --------------------------------------------------------

from functools import partial
import math
import random
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.nn.modules.normalization import _shape_t


LAYER_NORM = partial(nn.LayerNorm, eps=1e-6)
RESERVE_ROI = []
RESERVE_SP = []
RESERVE_ATTN = []

try:
    from torch.utils.checkpoint import checkpoint_sequential
except:
    pass


def is_debug() -> bool:
    '''
    check whether debug or not
    '''
    import os
    return 'DEBUG' in os.environ


def _maybe_promote(x: torch.Tensor) -> torch.Tensor:
    """
    Credits to Meta's xformers (xformers/components/attention/favor.py)
    """
    # Only promote fp16 buffers, bfloat16 would be fine for instance
    return x
    return x.float() if x.dtype == torch.float16 else x


@torch.no_grad()
def init_layer_norm_unit_norm(layer: nn.LayerNorm, gamma=1.0):
    assert len(layer.normalized_shape) == 1
    width = layer.normalized_shape[0]
    nn.init.ones_(layer.weight)
    layer.weight.data.mul_(gamma * (width ** -0.5))


@torch.no_grad()
def sin_coord_map(x: torch.Tensor, temperature=10000):
    B, C, H, W = x.size()
    num_feats = C//2

    assert H == W
    division = 1000

    coord_y = torch.linspace(0.0, math.pi*0.5, H, device=x.device).view(
        1, 1, -1, 1).repeat(B, 1, 1, W)
    coord_x = torch.linspace(0.0, math.pi*0.5, W, device=x.device).view(
        1, 1, 1, -1).repeat(B, 1, H, 1)

    dim_t = torch.arange(
        num_feats, dtype=torch.float32, device=coord_y.device)

    dim_t = (temperature**(2 * (dim_t // 2) / num_feats)).view(1, -1, 1, 1)

    coord_y = coord_y/dim_t
    coord_x = coord_x/dim_t
    pos = torch.cat([
        coord_x[:, 0::2].sin(), coord_x[:, 1::2].cos(),
        coord_y[:, 0::2].sin(), coord_y[:, 1::2].cos()
    ], dim=1)

    return pos


@torch.no_grad()
def init_linear_params(weight, bias):
    std = .02
    nn.init.trunc_normal_(weight, std=std)
    if bias is not None:
        nn.init.constant_(bias, 0)


class RestrictGradNorm(torch.autograd.Function):
    GRAD_SCALE = 1.0  # variable tracking grad scale

    @staticmethod
    def forward(ctx, x, norm=0.1):
        ctx.norm = norm
        ctx.save_for_backward(x)
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # amp training scales up the grad scale for the entire network
        norm = grad_output.new_tensor(ctx.norm).reshape(1, 1, 1, 1, 3) * RestrictGradNorm.GRAD_SCALE
        grad_x = grad_output.clone().clamp(-norm, norm)
        return grad_x, None


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Copyright 2020 Ross Wightman
    """
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    ** IMPORTANT **
    Modified (Jun. 16, by Ziteng Gao):
    since we use the second dimension as the batch dimension, the random tensor shape is
    actually `(1, x.shape[1],) + (1,) * (x.ndim - 2)` (not the originally `(x.shape[0],)
    +(1,) * (x.ndim - 2)` in timm).
    Sorry for this bug since I simply adopted timm code in the code reorganization
    without further investigation.
    ** This corrected version aligns with the paper **
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (1, x.shape[1],) + (1,) * (x.ndim - 2)
    random_tensor = keep_prob + \
        torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob, exclude_cls_token=True):
        super().__init__()
        self.prob = prob
        self.exclude_cls_token = exclude_cls_token

    def forward(self, x, roi):
        if not self.training or (isinstance(self.prob, (float, int)) and self.prob == 0.):
            return x, roi

        x = x.transpose(0, 1)
        roi = roi.transpose(0, 1)

        if self.exclude_cls_token:
            cls_token, x = x[:, -1:], x[:, :-1]
            cls_roi, roi = roi[:, -1:], roi[:, :-1]

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        if isinstance(self.prob, (tuple, list)):
            prob = random.random()*(self.prob[1]-self.prob[0]) + self.prob[0]
        else:
            prob = self.prob

        keep_prob = 1 - prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]
        roi = roi[batch_indices, patch_indices_keep]

        if self.exclude_cls_token:
            x = torch.cat((x, cls_token), dim=1)
            roi = torch.cat((roi, cls_roi), dim=1)

        x = x.transpose(0, 1)
        roi = roi.transpose(0, 1)

        return x, roi


class DropPath(nn.Module):
    """
    Copyright 2020 Ross Wightman
    """
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


@autocast(enabled=False)
def roi_adjust(token_roi: torch.Tensor, token_adjust: torch.Tensor):
    token_xy = (token_roi[..., 3:]+token_roi[..., :3]) * 0.5
    token_wh = (token_roi[..., 3:]-token_roi[..., :3]).abs()

    token_xy = token_xy + token_adjust[..., :3]
    token_wh = token_wh * token_adjust[..., 3:].exp()

    token_roi_new = torch.cat(
        [token_xy-0.5*token_wh, token_xy+0.5*token_wh], dim=-1)
    return token_roi_new


@autocast(enabled=False)
def translate_to_absolute_coordinates(
        token_roi: torch.Tensor,
        relative_offset_xyz: torch.Tensor,
        num_heads: int,
        num_points: int) -> torch.Tensor:
    batch, num_tokens, _ = relative_offset_xyz.shape

    relative_offset_xyz = relative_offset_xyz.view(batch, num_tokens, 1, num_heads*num_points, 3)

    offset_mean = relative_offset_xyz.mean(-2, keepdim=True)
    offset_std = relative_offset_xyz.std(-2, keepdim=True)+1e-7
    relative_offset_xyz = (relative_offset_xyz - offset_mean)/(3*offset_std)

    relative_offset_xyz = relative_offset_xyz.view(batch, num_tokens, 1, num_heads*num_points, 3)

    token_xyz = (token_roi[:, :, 3:]+token_roi[:, :, :3])/2.0
    roi_wht = token_roi[:, :, 3:]-token_roi[:, :, :3]
    relative_offset_xyz = relative_offset_xyz[..., :3] * \
        roi_wht.view(batch, num_tokens, 1, 1, 3)
    absolute_xyz = token_xyz.view(batch, num_tokens, 1, 1, 3) \
        + relative_offset_xyz
    
    return absolute_xyz


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, dilation=dilation, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


@autocast(enabled=False)
def feat_sampling_3d(
        sample_points: torch.Tensor,
        value: torch.Tensor,
        n_points=1):
    batch, Hq, Wq, num_points_per_head, _ = sample_points.shape
    batch, channel, temporal, height, width = value.shape

    n_heads = num_points_per_head//n_points

    if is_debug():
        RESERVE_SP.append(sample_points.detach())
    sample_points = sample_points.view(batch, Hq, Wq, n_heads, n_points, 3) \
        .permute(0, 3, 1, 2, 4, 5).contiguous().flatten(0, 1)
    # We truncate the grad for sampling coordinates to the unit length (e.g., 1.0/height)
    # to avoid inaccurate gradients due to bilinear sampling. In other words, we restrict
    # gradients to be local.
    sample_points = RestrictGradNorm.apply(sample_points, (1.0/width, 1.0/height, 1.0/temporal))
    sample_points = sample_points.flatten(2, 3)
    sample_points = sample_points*2.0-1.0
    sample_points = sample_points.unsqueeze(1)
    value = value.view(batch*n_heads, channel//n_heads, temporal, height, width)
    out = F.grid_sample(
        value, sample_points,
        mode='bilinear', padding_mode='border', align_corners=False,
    )

    out = out.view(batch, n_heads, channel//n_heads, Hq, Wq, n_points)

    return out.permute(0, 3, 4, 1, 5, 2).flatten(1, 2)


@autocast(enabled=False)
def layer_norm_by_dim(x: torch.Tensor, dim=-1):
    mean = x.mean(dim, keepdim=True)
    x = x - mean
    std = (x.var(dim=dim, keepdim=True)+1e-7).sqrt()
    return x / std


class AdaptiveMixing(nn.Module):
    def __init__(self, in_dim, in_points, n_groups, query_dim=None,
                 out_dim=None, out_points=None, out_query_dim=None):
        super(AdaptiveMixing, self).__init__()
        out_dim = out_dim if out_dim is not None else in_dim
        out_points = out_points if out_points is not None else in_points
        out_points = out_points if out_points > 32 else 32
        query_dim = query_dim if query_dim is not None else in_dim
        out_query_dim = out_query_dim if out_query_dim is not None else query_dim

        self.query_dim = query_dim
        self.out_query_dim = out_query_dim
        self.in_dim = in_dim
        self.in_points = in_points
        self.n_groups = n_groups
        self.out_dim = out_dim
        self.out_points = out_points

        self.eff_in_dim = in_dim//n_groups
        self.eff_out_dim = self.eff_in_dim

        self.channel_param_count = (self.eff_in_dim * self.eff_out_dim)
        self.spatial_param_count = (self.in_points * self.out_points)

        self.total_param_count = self.channel_param_count + self.spatial_param_count

        self.parameter_generator = nn.Sequential(
            LAYER_NORM(self.query_dim),
            nn.Linear(self.query_dim, 128),
            nn.Linear(128, self.total_param_count*self.n_groups),
        )

        self.m_beta = nn.Parameter(torch.zeros(self.eff_out_dim))
        self.s_beta = nn.Parameter(torch.zeros(self.out_points))

        # the name should be `out_proj` but ...
        self.tuo_proj = nn.Linear(n_groups*self.eff_out_dim*self.out_points,
                                  self.out_query_dim)

        self.act = nn.GELU()

    @torch.no_grad()
    def init_layer(self):
        init_layer_norm_unit_norm(self.parameter_generator[0], gamma=1.)
        nn.init.zeros_(self.parameter_generator[-1].weight)

        bias = self.parameter_generator[-1].bias
        nn.init.xavier_uniform_(
            bias[:self.eff_in_dim*self.eff_out_dim].view(
                self.eff_in_dim, self.eff_out_dim), gain=1
        )
        nn.init.xavier_uniform_(
            bias[self.eff_in_dim*self.eff_out_dim:].view(
                self.in_points, self.out_points), gain=1
        )

    def forward(self, x, query):
        B, N, g, P, C = x.size()
        assert g == 1
        # batch, num_query, group, point, channel
        G = self.n_groups
        out = x.reshape(B*N*self.n_groups, P, C)

        params: torch.Tensor = self.parameter_generator(query)
        params = params.reshape(B*N*G, -1)

        channel_mixing, spatial_mixing = params.split_with_sizes(
            [self.eff_in_dim*self.eff_out_dim, self.out_points*self.in_points],
            dim=-1
        )
        channel_mixing = channel_mixing.reshape(B*N*G, self.eff_in_dim, self.eff_out_dim)
        spatial_mixing = spatial_mixing.reshape(B*N*G, self.out_points, self.in_points)

        channel_bias = self.m_beta.view(1, 1, self.eff_out_dim)
        spatial_bias = self.s_beta.view(1, self.out_points, 1)

        out = torch.baddbmm(channel_bias, out, channel_mixing)
        out = self.act(out)

        out = torch.baddbmm(spatial_bias, spatial_mixing, out)
        out = self.act(out)
        out = out.reshape(B, N, -1)
        out = self.tuo_proj(out)

        return out


class SFUnit(nn.Module):
    def __init__(self,
                 dim,
                 conv_dim,
                 num_sampling_points,
                 sampling_enabled=False,
                 adjusting_enabled=False,
                 final_adjusting=False,
                 only_sampling=False,
                 mlp_ratio=4,
                 repeats=1,
                 drop_path=0.0):
        super(SFUnit, self).__init__()
        self.dim = dim
        self.conv_dim = dim

        self.num_sampling_points = num_sampling_points

        self.sampling_enabled = sampling_enabled
        self.adjusting_enabled = adjusting_enabled
        self.final_adjusting = final_adjusting
        self.repeats = repeats
        self.only_sampling = only_sampling

        if self.sampling_enabled:
            self.adaptive_mixing = AdaptiveMixing(
                conv_dim, num_sampling_points, 1, query_dim=dim, out_query_dim=dim
            )
        else:
            self.adaptive_mixing = None

        if not self.only_sampling:
            self.ffn = nn.Sequential(
                LAYER_NORM(dim),
                nn.Linear(dim, dim*mlp_ratio),
                nn.GELU(),
                nn.Linear(dim*mlp_ratio, dim),
            )

            self.attn = nn.MultiheadAttention(
                dim,
                64 if dim >= 64 else dim,
                dropout=0.0
            )

            self.ln_attn = LAYER_NORM(dim)

        if self.sampling_enabled:
            self.roi_offset_module = nn.Sequential(
                LAYER_NORM(dim),
                nn.Linear(dim, num_sampling_points*3, bias=False),
            )
            self.roi_offset_bias = nn.Parameter(
                torch.randn(num_sampling_points*3))

        if self.adjusting_enabled or self.final_adjusting:
            self.roi_adjust_module = nn.Sequential(
                LAYER_NORM(dim),
                nn.Linear(dim, 6, bias=False),
            )
        self.drop_path = drop_path
        self.dropout = DropPath(
            drop_path if isinstance(drop_path, (float, int)) else .0
        )

    @torch.no_grad()
    def init_layer(self):
        if not self.only_sampling:
            init_linear_params(self.attn.in_proj_weight, self.attn.in_proj_bias)

        if self.adjusting_enabled:
            init_layer_norm_unit_norm(self.roi_adjust_module[0],  1.)
            nn.init.zeros_(self.roi_adjust_module[-1].weight)
            pass

        if self.sampling_enabled:
            init_layer_norm_unit_norm(self.roi_offset_module[0],  1.)
            nn.init.zeros_(self.roi_offset_module[-1].weight)

            # root = int(self.num_sampling_points**0.5)
            # x = torch.linspace(0.5/root, 1-0.5/root, root)\
            #     .view(1, -1, 1).repeat(root, 1, 1)
            # y = torch.linspace(0.5/root, 1-0.5/root, root)\
            #     .view(-1, 1, 1).repeat(1, root, 1)
            # grid = torch.cat([x, y], dim=-1).view(root**2, -1)
            # bias = self.roi_offset_bias.view(-1, 2)
            # bias.data[:root**2] = grid - 0.5
            nn.init.uniform_(self.roi_offset_bias, -0.5, 0.5)

    def shift_token_roi(self, token_embedding: torch.Tensor, token_roi: torch.Tensor):
        roi_adjust_logit = self.roi_adjust_module(token_embedding)
        roi_adjust_logit = self.dropout(roi_adjust_logit)
        token_roi = roi_adjust(token_roi, roi_adjust_logit)
        if is_debug():
            RESERVE_ROI.append(token_roi.detach().transpose(0, 1))
        return token_roi

    def sampling_mixing(self, feat, token_embedding: torch.Tensor, token_roi: torch.Tensor):
        roi_offset_a = self.roi_offset_module(
            token_embedding
        )
        roi_offset_base = self.roi_offset_bias.reshape(1, 1, -1).repeat(
            token_embedding.size(0), token_embedding.size(1), 1)
        roi_offset = roi_offset_a*0.0 + roi_offset_base

        sampling_points = translate_to_absolute_coordinates(
            token_roi.transpose(0, 1),
            roi_offset.transpose(0, 1),
            1,
            self.num_sampling_points
        )

        sampled_feat = feat_sampling_3d(
            sampling_points,
            feat,
            n_points=self.num_sampling_points,
        )

        src = self.adaptive_mixing(
            sampled_feat, token_embedding.transpose(0, 1))
        src = src.transpose(0, 1)
        # src = self.dropout(src)
        token_embedding = token_embedding + src

        return token_embedding

    def ffn_forward(self, token_embedding):
        src = self.ffn(token_embedding)
        # src = self.dropout(src)
        token_embedding = token_embedding + src
        return token_embedding

    def self_attention_forward(self, token_embedding):
        src = self.ln_attn(token_embedding)
        src, attn_map = self.attn(src, src, src)
        if is_debug():
            RESERVE_ATTN.append(attn_map.detach())
        # src = self.dropout(src)
        token_embedding = token_embedding + src
        return token_embedding

    def forward_inner(self,
                      img_feat: torch.Tensor,
                      token_embedding: torch.Tensor,
                      token_roi: torch.Tensor,
                      drop_path=None,):
        if drop_path is not None:
            self.dropout.drop_prob = drop_path
        if not self.only_sampling:
            token_embedding = self.self_attention_forward(token_embedding)
        if self.adjusting_enabled:
            token_roi = self.shift_token_roi(token_embedding, token_roi)
        if self.sampling_enabled:
            token_embedding = self.sampling_mixing(
                img_feat, token_embedding, token_roi)
        if not self.only_sampling:
            token_embedding = self.ffn_forward(token_embedding)

        return token_embedding, token_roi

    def forward(self,
                img_feat: torch.Tensor,
                token_embedding: torch.Tensor,
                token_roi: torch.Tensor,
                drop_path=None,):
        for i in range(self.repeats):
            drop_path = self.drop_path
            _drop_path = drop_path if isinstance(
                drop_path, float) else drop_path[i]
            token_embedding, token_roi = self.forward_inner(
                img_feat,
                token_embedding,
                token_roi,
                _drop_path,
            )

        return token_embedding, token_roi


class AvgTokenHead(nn.Module):
    def __init__(self, in_dim, dim=0, num_classes=1000):
        super(AvgTokenHead, self).__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.norm = LAYER_NORM(in_dim)
        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        x = x.mean(dim=self.dim)
        return self.classifier(self.norm(x))


class EarlyConvolution(nn.Module):
    def __init__(self, conv_dim: int):
        super(EarlyConvolution, self).__init__()
        self.conv_dim = conv_dim
        self.conv1 = nn.Conv3d(3, self.conv_dim,
                               kernel_size=(1, 6, 6), stride=(1, 2, 2), padding=(0, 2, 2),
                               bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d((1, 2, 2), (1, 2, 2), (0, 0, 0))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = layer_norm_by_dim(x, 1)
        return x


class SparseFormer(nn.Module):
    def __init__(self,
                 conv_dim=96,
                 num_latent_tokens=36,
                 token_sampling_points=25,
                 width_configurations=[384, 384],
                 block_sizes=[1, 1],
                 repeats=[4, 1],
                 drop_path_rate=0.0,
                 parent_vit_model=None):
        super(SparseFormer, self).__init__()
        self.num_latent_tokens = num_latent_tokens
        self.width_configurations = width_configurations
        self.block_sizes = block_sizes
        self.repeats = repeats
        self.token_sampling_points = token_sampling_points
        self.use_cls_token = True

        self.feat_extractor = EarlyConvolution(conv_dim=conv_dim)

        start_dim = width_configurations[0]
        end_dim = width_configurations[-1]
        self.init_token_roi_learnable = True
        if self.init_token_roi_learnable:
            self.token_roi = nn.Embedding(num_latent_tokens, 6)
        else:
            self.register_buffer("token_roi", torch.zeros(num_latent_tokens, 6))
        self.token_embedding = nn.Embedding(num_latent_tokens, start_dim)

        self.layers = nn.ModuleList()

        if self.use_cls_token:
            self.cls_token_embedding = nn.Embedding(1, end_dim)
        # Preparing the list of drop path rate
        nums = []
        for i, width in enumerate(width_configurations):
            repeat = repeats[i]
            block_size = block_sizes[i]
            nums += [block_size * repeat]
        lin_drop_path = list(torch.linspace(0.0, drop_path_rate, sum(nums)))
        lin_drop_path = [p.item() for p in lin_drop_path]

        block_wise_idx = 0
        for i, width in enumerate(width_configurations):
            repeat = repeats[i]
            block_size = block_sizes[i]
            nums += [block_size * repeat]
            if i > 0 and width_configurations[i-1] != width_configurations[i]:
                transition = nn.Sequential(
                    nn.Linear(
                        width_configurations[i-1], width_configurations[i]),
                    LAYER_NORM(width_configurations[i])
                )
                self.layers.append(transition)

            if repeat > 1:
                assert block_size == 1

            for block_idx in range(block_size):
                is_leading_block = (block_idx == 0)
                is_only_sampling = (i == (len(width_configurations) - 1))
                module = SFUnit(
                    width,
                    conv_dim=conv_dim,
                    num_sampling_points=token_sampling_points,
                    sampling_enabled=is_leading_block,
                    adjusting_enabled=is_leading_block,
                    repeats=repeat,
                    only_sampling=is_only_sampling,
                    drop_path=lin_drop_path[block_wise_idx:block_wise_idx+repeat]
                )
                self.layers.append(module)
                block_wise_idx += repeat

        # self.head = AvgTokenHead(end_dim, dim=0)

        # self.vit_model_name = "vit_base_patch16_224"
        # self.vit_model_name = "vit_large_patch16_224"
        # import timm
        # vit = timm.create_model(self.vit_model_name, pretrained=True)
        # self.blocks = vit.blocks
        # self.blocks.requires_grad_(False)
        # self.vit_norm = LAYER_NORM(width_configurations[-1])

        # assert parent_vit_model is not None
        self.parent_vit_model = parent_vit_model

        self.srnet_init()

    def srnet_init(self):
        # first recursively initialize transformer-related weights
        def _init_transformers_weights(m):
            if isinstance(m, nn.Linear):
                init_linear_params(m.weight, m.bias)
            elif isinstance(m, nn.LayerNorm) and m.elementwise_affine:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.apply(_init_transformers_weights)

        # then special case
        for n, m in self.named_modules():
            if hasattr(m, 'init_layer'):
                type_ = str(type(m))
                m.init_layer()

    def no_weight_decay(self):
        return ['token_roi']

    @torch.no_grad()
    def init_layer(self):
        nn.init.trunc_normal_(self.token_embedding.weight, std=0.1)

        # if self.init_token_roi_learnable:
        #     token_roi = self.token_roi.weight
        # else:
        #     token_roi = self.token_roi

        # token_roi.data[..., :3].fill_(0.0)
        # token_roi.data[..., 3:].fill_(1.0)

        # def init_grid(root, offset, unit_width=0.5):
        #     grid = torch.arange(root).float()/(root-1)
        #     grid = grid.view(root, -1)
        #     grid_x = grid.view(root, 1, 1).repeat(1, root, 1)
        #     grid_y = grid.view(1, root, 1).repeat(root, 1, 1)
        #     grid = torch.cat([grid_x, grid_y], dim=-1)
        #     token_roi_reshaped = token_roi[offset:offset+root**2].view(
        #         root, root, -1)
        #     token_roi_reshaped.data[..., 0:2] = grid * (1-unit_width) + 0.00
        #     token_roi_reshaped.data[..., 2:4] = grid * (1-unit_width) + unit_width

        # def init_random(token_number, unit_width=0.5):
        #     token_roi_reshaped = token_roi[:token_number]
        #     token_roi_reshaped.data[..., 0:2] = torch.rand_like(
        #         token_roi_reshaped.data[..., 0:2]) * (1-unit_width)
        #     token_roi_reshaped.data[..., 2:4] = token_roi_reshaped.data[..., 0:2] + unit_width

        # size = int(self.num_latent_tokens**0.5)
        # size = 1.0/size
        # # init_random(self.num_latent_tokens-1)
        # if self.use_cls_token:
        #     init_random(self.num_latent_tokens-1, size)
        # else:
        #     init_random(self.num_latent_tokens, size)

        vit_teacher = self.parent_vit_model
        if vit_teacher is not None:
            self.ln_pre = vit_teacher.ln_pre
            self.blocks = vit_teacher.transformer
            self.blocks.grad_checkpointing = True
            self.blocks.resblocks = self.blocks.resblocks[8:]
            self.ln_post = vit_teacher.ln_post
            self.parent_vit_model = None
        self.patch_drop = PatchDropout(0.0, exclude_cls_token=self.use_cls_token)

    def inflate_times(self, t):
        if t == 1:
            return
        N = self.token_embedding.weight.size(0)
        dim = self.token_embedding.weight.size(1)
        new_N = N * t
        self.token_embedding = nn.Embedding(new_N, dim)
        self.token_roi = nn.Embedding(new_N, 6)
        

    def load_2d_state_dict(self, state_dict):
        source_state_dict = state_dict
        target_state_dict = self.state_dict()
        modified_state_dict = dict()

        for key in source_state_dict:
            if not key in target_state_dict:
                continue
            if target_state_dict[key].shape == source_state_dict[key].shape:
                modified_state_dict[key] = source_state_dict[key]
                continue
            # then inflate things
            s = source_state_dict[key]
            tshape = target_state_dict[key].shape
            if 'conv' in key:
                modified_state_dict[key] = s.unsqueeze(2)
                continue
            if 'roi_offset_bias' in key:
                m = torch.zeros(tshape)
                m.data.view(-1, 3)[:, :2] = s.view(-1, 2)
                modified_state_dict[key] = m
                continue
            if 'roi_offset_module.1.weight' in key or 'roi_adjust_module.1.weight' in key:
                m = torch.zeros(tshape)
                dim = tshape[1]
                m.data.view(-1, 3, dim)[:, :2] = s.view(-1, 2, dim)
                modified_state_dict[key] = m
                continue
            if 'token_roi' in key:
                times = tshape[0]//s.shape[0]
                print(s.shape, tshape, key)
                m = torch.zeros(tshape).view(times, s.shape[0], 2, 3)
                s = s.view(1, s.shape[0], 2, 2)
                centers = torch.linspace(0.5/times, 1.0-0.5/times, times).view(times, 1)
                zs = torch.cat([centers-0.5/times, centers+0.5/times], dim=1)
                m.data[:, :, :, :2] = s
                m.data[:, :, :, 2:] = zs.reshape(times, 1, 2, 1)
                modified_state_dict[key] = m.reshape(*tshape)
                continue
            if 'token_embedding' in key:
                times = tshape[0]//s.shape[0]
                modified_state_dict[key] = s.repeat(times, 1)
                continue

        print(self.load_state_dict(modified_state_dict))

    def normalize_final_feature(self, x):
        return F.layer_norm(x, [x.size(-1)])

    def forward(self, x: torch.Tensor,
                scale=1.,
                train_ratio=0.):
        assert len(x.shape) == 5
        # [B, C, T, H, W]
        RestrictGradNorm.GRAD_SCALE = scale
        img_feat = self.feat_extractor(x)
        img_feat = _maybe_promote(img_feat)

        batch_size = img_feat.size(0)

        token_embedding = self.token_embedding.weight\
            .unsqueeze(1).repeat(1, batch_size, 1)

        if self.init_token_roi_learnable:
            token_roi = self.token_roi.weight\
                .unsqueeze(1).repeat(1, batch_size, 1)
        else:
            token_roi = self.token_roi\
                .unsqueeze(1).repeat(1, batch_size, 1)

        token_embedding = _maybe_promote(token_embedding)
        token_roi = _maybe_promote(token_roi)

        for layer in self.layers:
            if isinstance(layer, SFUnit):
                token_embedding, token_roi = layer(
                    img_feat, token_embedding, token_roi)
            else:
                token_embedding = layer(token_embedding)

        self.blocks.grad_checkpointing = self.training
        if self.use_cls_token:
            cls_token_embedding = self.cls_token_embedding.weight.unsqueeze(
                1).repeat(1, batch_size, 1)
            token_embedding = torch.cat([token_embedding, cls_token_embedding], dim=0)

        token_embedding = self.ln_pre(token_embedding)
        token_embedding = self.blocks(token_embedding)

        token_embedding = self.ln_post(token_embedding)
        if self.use_cls_token:
            cls_token = token_embedding[-1]
            token_embedding = token_embedding[:-1]
        else:
            cls_token = token_embedding.mean(0)
            cls_token = self.ln_post(cls_token)
        return cls_token, token_embedding

    def visual(self, x, *args, **kwargs):
        return self.forward(x, *args, **kwargs)


if __name__ == '__main__':
    net = SparseFormer(width_configurations=[384, 1024])

    net.eval()

    IMG_SIZE = 224
    input = torch.randn(1, 3, 8, IMG_SIZE, IMG_SIZE)
    # print(net)
    # net(input)

    from fvcore.nn import FlopCountAnalysis, flop_count_table
    flops = FlopCountAnalysis(net, input)
    print(flop_count_table(flops, max_depth=2))