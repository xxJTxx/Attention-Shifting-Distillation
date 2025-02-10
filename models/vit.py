""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929

The official jax code is released and available at https://github.com/google-research/vision_transformer

Status/TODO:
* Models updated to be compatible with official impl. Args added to support backward compat for old PyTorch weights.
* Weights ported from official jax impl for 384x384 base and small models, 16x16 and 32x32 patches.
* Trained (supervised on ImageNet-1k) my custom 'small' patch model to 77.9, 'base' to 79.4 top-1 with this code.
* Hopefully find time and GPUs for SSL or unsupervised pretraining on OpenImages w/ ImageNet fine-tune in future.

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2020 Ross Wightman
"""
from enum import Enum

import torch
import torch.nn as nn
from functools import partial
from itertools import repeat
import collections.abc as container_abcs
import torch
import math
import warnings


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Debug_Flags(Enum):
    L2_HEAD_CTX_NORMS = 1  # return the l2 norms of the context heads
    ATTN_MASKS = 2  # return the attention masks
    HEAD_CTX = 3  # return the head contexts
    LAYER_CTX = 4  # return the aggregated head contexts
    HEAD_OUTPUT = 5  # return the attention output from each head
    LAYER_OUTPUT = 6  # return the output from aggregated heads
    RESIDUAL_CTX_VEC = 7  # residual before adding layer output
    RESIDUAL_CTX_ADD = 8  # ctx right before adding to residual
    RESIDUAL_CTX_FINAL = 9  # RESIDUAL_VEC + RESIDUAL_ADD
    FINAL_LATENT_VECTOR = 10  # final latent vector for the model
    PATCH_EMBED = 11
    RESIDUAL_LAYER_FINAL = 12  # RESIDUAL_VEC + RESIDUAL_ADD but for the whole raster
    NO_POS = 13


class MaskProcessor(nn.Module):
    def __init__(self, patch_size=16):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(patch_size)

    def forward(self, ones_mask):
        B = ones_mask.shape[0]
        ones_mask = ones_mask[0].unsqueeze(0)  # take the first mask
        ones_mask = self.avg_pool(ones_mask)[0]
        ones_mask = torch.where(ones_mask.view(-1) > 0)[0] + 1
        ones_mask = torch.cat([torch.cuda.IntTensor(1).fill_(0), ones_mask]).unsqueeze(0)
        ones_mask = ones_mask.expand(B, -1)
        return ones_mask


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, head_hide_list=None, debug_flags=None):
        debug = {}
        if debug_flags is None:
            debug_flags = {}

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = (attn - torch.max(attn, dim=-1, keepdim=True)[0]).softmax(dim=-1)
        if head_hide_list is not None:
            # we are masking some heads
            if -1 in head_hide_list:
                # mask all in this layer
                attn[:, :, :, 1:] *= 0
                attn[:, :, :, 0] = 1
            else:
                attn[:, head_hide_list, :, 1:] *= 0
                attn[:, head_hide_list, :, 0] = 1

        if Debug_Flags.ATTN_MASKS in debug_flags:
            debug[Debug_Flags.ATTN_MASKS.name] = attn  # store attention mask

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2)  # B, N, Num Heads, head_dim
        if Debug_Flags.L2_HEAD_CTX_NORMS in debug_flags:
            l2_norms = x[:, 0].norm(dim=-1)
            debug[Debug_Flags.L2_HEAD_CTX_NORMS.name] = l2_norms
        if Debug_Flags.HEAD_CTX in debug_flags:
            head_contexts = x[:, 0]
            debug[Debug_Flags.HEAD_CTX.name] = head_contexts
        if Debug_Flags.HEAD_OUTPUT in debug_flags:
            head_outputs = x
            debug[Debug_Flags.HEAD_OUTPUT.name] = head_outputs

        x = x.reshape(B, N, C)
        x = self.proj(x)
        if Debug_Flags.LAYER_CTX in debug_flags:
            layer_contexts = x[:, 0]
            debug[Debug_Flags.LAYER_CTX.name] = layer_contexts
        if Debug_Flags.LAYER_OUTPUT in debug_flags:
            layer_outputs = x
            debug[Debug_Flags.LAYER_OUTPUT.name] = layer_outputs
        x = self.proj_drop(x)
        return x, debug


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, head_hide_list=None, debug_flags=None):
        if debug_flags is None:
            debug_flags = {}
        out, debug = self.attn(self.norm1(x), head_hide_list=head_hide_list, debug_flags=debug_flags)
        out = self.drop_path(out)
        if Debug_Flags.RESIDUAL_CTX_ADD in debug_flags:
            debug[Debug_Flags.RESIDUAL_CTX_ADD.name] = out[:, 0]
        if Debug_Flags.RESIDUAL_CTX_VEC in debug_flags:
            debug[Debug_Flags.RESIDUAL_CTX_VEC.name] = x[:, 0]
        x = x + out
        if Debug_Flags.RESIDUAL_CTX_FINAL in debug_flags:
            debug[Debug_Flags.RESIDUAL_CTX_FINAL.name] = x[:, 0]
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if Debug_Flags.RESIDUAL_LAYER_FINAL in debug_flags:
            debug[Debug_Flags.RESIDUAL_LAYER_FINAL.name] = x
        return x, debug


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, norm_embed=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        # self.norm_embed = norm_layer(embed_dim) if norm_embed else None
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        # self.repr = nn.Linear(embed_dim, representation_size)
        # self.repr_act = nn.Tanh()

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        self.mask_processor = MaskProcessor(patch_size=patch_size)
        self.first_pass = True

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, head_mask, debug_flags, patch_mask):
        if x.shape[1] == 4:
            self.first_pass = False
            assert patch_mask is None
            x, ones_mask = x[:, :3], x[:, 3]
            patch_mask = self.mask_processor(ones_mask)

        B = x.shape[0]
        x = self.patch_embed(x)
        patch_embedding = x
        if head_mask is None:
            head_mask = {}

        # if self.norm_embed:
        #     x = self.norm_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        if debug_flags is not None and Debug_Flags.NO_POS in debug_flags:
            x = x
        else:
            x = x + self.pos_embed
        x = self.pos_drop(x)  # B, N, C
        if patch_mask is not None:
            # patch_mask is B, K
            B, N, C = x.shape
            if len(patch_mask.shape) == 1:  # not a separate one per batch
                x = x[:, patch_mask]
            else:
                patch_mask = patch_mask.unsqueeze(-1).expand(-1, -1, C)
                x = torch.gather(x, 1, patch_mask)

        all_debug = []
        for i, blk in enumerate(self.blocks):
            if i in head_mask:
                x, all_debug_layer = blk(x, head_hide_list=head_mask[i], debug_flags=debug_flags)
            else:
                x, all_debug_layer = blk(x, debug_flags=debug_flags)
            all_debug.append(all_debug_layer)

        consolidated_all_debug = {}
        for e in all_debug[0].keys():
            consolidated_all_debug[e] = torch.stack([d[e] for d in all_debug], 1)

        if debug_flags is not None and Debug_Flags.PATCH_EMBED in debug_flags:
            consolidated_all_debug[Debug_Flags.PATCH_EMBED.name] = patch_embedding

        x = self.norm(x)
        return x[:, 0], consolidated_all_debug

    def forward(self, x, head_mask=None, debug_flags=None, patch_mask=None):
        # dict of layer_index -> list of head indices to turn off. If list just contains -1, turn all heads off in that layer
        x, debug = self.forward_features(x, head_mask=head_mask, debug_flags=debug_flags, patch_mask=patch_mask)
        if debug_flags is not None and Debug_Flags.FINAL_LATENT_VECTOR in debug_flags:
            debug[Debug_Flags.FINAL_LATENT_VECTOR.name] = x
        x = self.head(x)
        if debug_flags is None:
            return x
        else:
            return x, debug


def deit_tiny_patch16_224(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    # model.default_cfg = default_cfgs['deit_tiny_patch16_224']
    # if pretrained:
    # checkpoint = torch.hub.load_state_dict_from_url(
    #     url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
    #     map_location="cpu", check_hash=True
    # )
    checkpoint = torch.load("deit_tiny_patch16_224-a1311bcf.pth")
    model.load_state_dict(checkpoint["model"])
    print('==>[Loaded PyTorch-pretrained deit checkpoint.]')
    return model


def deit_base_patch16_224(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    # if pretrained:
    #     checkpoint = torch.hub.load_state_dict_from_url(
    #         url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
    #         map_location="cpu", check_hash=True
    #     )
    checkpoint = torch.load("ft_local/deit_base_patch16_224-b5f2ef4d.pth")
    model.load_state_dict(checkpoint["model"])
    print('==>[Loaded PyTorch-pretrained deit checkpoint.]')
    return model


