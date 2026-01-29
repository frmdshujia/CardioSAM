# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""
ViTDet backbone adapted from Detectron2.
This module implements Vision Transformer (ViT) backbone for object detection.

Rope embedding code adopted from:
1. https://github.com/meta-llama/codellama/blob/main/llama/model.py
2. https://github.com/naver-ai/rope-vit
3. https://github.com/lucidrains/rotary-embedding-torch
"""

import math
from functools import partial
from itertools import repeat
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

try:
    from timm.layers import DropPath, Mlp, trunc_normal_
except ModuleNotFoundError:
    # compatibility for older timm versions
    from timm.models.layers import DropPath, Mlp, trunc_normal_
from torch import Tensor

from .model_misc import LayerScale


def init_t_xy(
    end_x: int, end_y: int, scale: float = 1.0, offset: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode="floor").float()
    return t_x * scale + offset, t_y * scale + offset


def compute_axial_cis(
    dim: int,
    end_x: int,
    end_y: int,
    theta: float = 10000.0,
    scale_pos: float = 1.0,
    offset: int = 0,
) -> torch.Tensor:
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    t_x, t_y = init_t_xy(end_x, end_y, scale_pos, offset)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[-2], x.shape[-1])
    shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_enc(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    repeat_freqs_k: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = (
        torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        if xk.shape[-2] != 0
        else None
    )
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    if xk_ is None:
        # no keys to rotate, due to dropout
        return xq_out.type_as(xq).to(xq.device), xk
    # repeat freqs along seq_len dim to match k seq_len
    if repeat_freqs_k:
        r = xk_.shape[-2] // xq_.shape[-2]
        freqs_cis = freqs_cis.repeat(*([1] * (freqs_cis.ndim - 2)), r, 1)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


def window_partition(x: Tensor, window_size: int) -> Tuple[Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.
    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        x (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.
    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.reshape(
        B, Hp // window_size, Wp // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :]
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: Tensor) -> Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).
    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
            align_corners=False,
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def get_abs_pos(
    abs_pos: Tensor,
    has_cls_token: bool,
    hw: Tuple[int, int],
    retain_cls_token: bool = False,
    tiling: bool = False,
) -> Tensor:
    """
    Calculate absolute positional embeddings. If needed, resize embeddings and remove cls_token
        dimension for the original embeddings.
    Args:
        abs_pos (Tensor): absolute positional embeddings with (1, num_position, C).
        has_cls_token (bool): If true, has 1 embedding in abs_pos for cls token.
        hw (Tuple): size of input image tokens.
        retain_cls_token: whether to retain the cls_token
        tiling: whether to tile the embeddings, *instead* of interpolation (a la abs_win)
    Returns:
        Absolute positional embeddings after processing with shape (1, H, W, C),
        if retain_cls_token is False, otherwise (1, 1+H*W, C)
    """
    if retain_cls_token:
        assert has_cls_token

    h, w = hw
    if has_cls_token:
        cls_pos = abs_pos[:, :1]
        abs_pos = abs_pos[:, 1:]

    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))
    assert size * size == xy_num

    if size != h or size != w:
        new_abs_pos = abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2)
        if tiling:
            new_abs_pos = new_abs_pos.tile(
                [1, 1] + [x // y + 1 for x, y in zip((h, w), new_abs_pos.shape[2:])]
            )[:, :, :h, :w]
        else:
            new_abs_pos = F.interpolate(
                new_abs_pos,
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            )

        if not retain_cls_token:
            return new_abs_pos.permute(0, 2, 3, 1)
        else:
            # add cls_token back, flatten spatial dims
            assert has_cls_token
            return torch.cat(
                [cls_pos, new_abs_pos.permute(0, 2, 3, 1).reshape(1, h * w, -1)],
                dim=1,
            )

    else:
        if not retain_cls_token:
            return abs_pos.reshape(1, h, w, -1)
        else:
            assert has_cls_token
            return torch.cat([cls_pos, abs_pos], dim=1)


def concat_rel_pos(
    q: Tensor,
    k: Tensor,
    q_hw: Tuple[int, int],
    k_hw: Tuple[int, int],
    rel_pos_h: Tensor,
    rel_pos_w: Tensor,
    rescale: bool = False,
    relative_coords: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Concatenate rel pos coeffs to the q & k tensors, so that qk^T is now
    effectively including rel pos biases.
    Args:
        q (Tensor): q tensor with shape (B, L_q, C).
        k (Tensor): k tensor with shape (B, L_k, C).
        q_hw, k_hw: These are spatial size of q & k tensors.
        rel_pos_h, rel_pos_w: These are relative pos embeddings/params of height, width.
        rescale (bool): whether to rescale. e.g. for use when using sdpa, pytorch will
            scale by the wrong factor due to the concat.
    Returns:
        q, k: But, padded so that qk^T accounts for rel pos biases
    """
    q_h, q_w = q_hw
    k_h, k_w = k_hw

    assert (q_h == q_w) and (k_h == k_w), "only square inputs supported"

    if relative_coords is not None:
        Rh = rel_pos_h[relative_coords]
        Rw = rel_pos_w[relative_coords]
    else:
        Rh = get_rel_pos(q_h, k_h, rel_pos_h)
        Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)

    old_scale = dim**0.5
    new_scale = (dim + k_h + k_w) ** 0.5 if rescale else old_scale  # for sdpa
    # attn will be divided by new_scale, but we want to divide q by old_scale
    scale_ratio = new_scale / old_scale

    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh) * new_scale  # (B, q_h, q_w, k_h)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw) * new_scale  # (B, q_h, q_w, k_w)

    eye_h = torch.eye(k_h, dtype=q.dtype, device=q.device)
    eye_w = torch.eye(k_w, dtype=q.dtype, device=q.device)

    eye_h = eye_h.view(1, k_h, 1, k_h).expand([B, k_h, k_w, k_h])
    eye_w = eye_w.view(1, 1, k_w, k_w).expand([B, k_h, k_w, k_w])

    q = torch.cat([r_q * scale_ratio, rel_h, rel_w], dim=-1).view(B, q_h * q_w, -1)
    k = torch.cat([k.view(B, k_h, k_w, -1), eye_h, eye_w], dim=-1).view(
        B, k_h * k_w, -1
    )

    return q, k


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
        bias: bool = True,
    ):
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int):  embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x


'''
to_2tuple 只是个辅助小工具，核心作用是让 “高 / 宽” 参数成对出现，不用反复写；
OverlapPatchEmbed 是核心工具，本质是 “带重叠的切图 + 编码”：用卷积（智能切刀）把图片切成重叠的小补丁，再把每个补丁转成 AI 能看懂的数字串；
重叠切图的好处是保留更多图片细节，让 AI 识别更准确（比如能看清图片边缘的小物体）。
'''
def to_2tuple(x):
    if isinstance(x, tuple):
        return x
    if isinstance(x, list):
        return tuple(x)
    if isinstance(x, Tensor) and x.numel() == 1:
        v = x.item()
        return (v, v)
    if hasattr(x, "__iter__") and not isinstance(x, (str, bytes)):
        return tuple(x)
    return tuple(repeat(x, 2))


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(
        self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

'''
整个类就干三件事：① 造一个 “高斯核”（磨皮用的权重模板）；② 把核存起来；③ 用这个核给图片做滤波（磨皮）。
'''
class GaussianFilter(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("kernel", self._gauss_kernel(), persistent=False)

    def _gauss_kernel(self, channels=3):
        kernel = torch.tensor(
            [
                [1.0, 4.0, 6.0, 4.0, 1.0],
                [4.0, 16.0, 24.0, 16.0, 4.0],
                [6.0, 24.0, 36.0, 24.0, 6.0],
                [4.0, 16.0, 24.0, 16.0, 4.0],
                [1.0, 4.0, 6.0, 4.0, 1.0],
            ]
        )
        kernel /= 256.0
        kernel = kernel.repeat(channels, 1, 1, 1)
        return kernel

    def conv_gauss(self, img):
        kernel = self.kernel.to(device=img.device, dtype=img.dtype)
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode="reflect")
        out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
        return out

'''
SRMFilter 是图片 “瑕疵 / 细节检测工具”，核心是 3 个固定的 5×5 滤波模板，专门突出图片里的边缘、噪点、伪影；
'''
class SRMFilter(nn.Module):
    def __init__(self):
        super().__init__()
        self.srm_layer = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2)
        filter1 = [
            [0, 0, 0, 0, 0],
            [0, -1 / 4, 2 / 4, -1 / 4, 0],
            [0, 2 / 4, -4 / 4, 2 / 4, 0],
            [0, -1 / 4, 2 / 4, -1 / 4, 0],
            [0, 0, 0, 0, 0],
        ]
        filter2 = [
            [-1 / 12, 2 / 12, -2 / 12, 2 / 12, -1 / 12],
            [2 / 12, -6 / 12, 8 / 12, -6 / 12, 2 / 12],
            [-2 / 12, 8 / 12, -12 / 12, 8 / 12, -2 / 12],
            [2 / 12, -6 / 12, 8 / 12, -6 / 12, 2 / 12],
            [-1 / 12, 2 / 12, -2 / 12, 2 / 12, -1 / 12],
        ]
        filter3 = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1 / 2, -2 / 2, 1 / 2, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        self.srm_layer.weight.data = torch.Tensor(
            [[filter1, filter1, filter1], [filter2, filter2, filter2], [filter3, filter3, filter3]]
        )
        for param in self.srm_layer.parameters():
            param.requires_grad = False

    def conv_srm(self, img):
        return self.srm_layer(img)

'''
PromptGenerator 是 AI 模型的 “提示生成器”，核心是预处理图片 + 切图编码 + 特征适配，给模型喂针对性的特征；
预处理阶段：支持高斯磨皮、SRM 细节提取、傅里叶高低频滤波，按需选择；
特征编码：手工调优（OverlapPatchEmbed 切重叠补丁）+ 嵌入调优（Linear 降维），双路特征融合；
适配器阶段：分 3 种类型（部分共享 / 完全共享 / 完全不共享），把降维后的特征还原回模型能识别的维度；
最终输出：原特征 + 提示特征，帮模型聚焦关键信息，提升任务效果。
'''
class PromptGenerator(nn.Module):
    def __init__(
        self,
        scale_factor,
        prompt_type,
        embed_dims,
        tuning_stage,
        depths,
        input_type,
        freq_nums,
        handcrafted_tune,
        embedding_tune,
        adaptor,
        img_size,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.prompt_type = prompt_type
        self.embed_dims = embed_dims
        self.input_type = input_type
        self.freq_nums = freq_nums
        self.tuning_stage = tuning_stage
        self.depths = depths
        self.handcrafted_tune = handcrafted_tune
        self.embedding_tune = embedding_tune
        self.adaptor = adaptor

        if self.input_type == "gaussian":
            self.gaussian_filter = GaussianFilter()
        if self.input_type == "srm":
            self.srm_filter = SRMFilter()
        if self.input_type == "all":
            self.prompt = nn.Parameter(
                torch.zeros(3, img_size, img_size), requires_grad=False
            )

        if self.handcrafted_tune:
            if "1" in self.tuning_stage:
                self.handcrafted_generator1 = OverlapPatchEmbed(
                    img_size=img_size,
                    patch_size=7,
                    stride=4,
                    in_chans=3,
                    embed_dim=self.embed_dims[0] // self.scale_factor,
                )
            if "2" in self.tuning_stage:
                self.handcrafted_generator2 = OverlapPatchEmbed(
                    img_size=img_size // 4,
                    patch_size=3,
                    stride=2,
                    in_chans=self.embed_dims[0] // self.scale_factor,
                    embed_dim=self.embed_dims[1] // self.scale_factor,
                )
            if "3" in self.tuning_stage:
                self.handcrafted_generator3 = OverlapPatchEmbed(
                    img_size=img_size // 8,
                    patch_size=3,
                    stride=2,
                    in_chans=self.embed_dims[1] // self.scale_factor,
                    embed_dim=self.embed_dims[2] // self.scale_factor,
                )
            if "4" in self.tuning_stage:
                self.handcrafted_generator4 = OverlapPatchEmbed(
                    img_size=img_size // 16,
                    patch_size=3,
                    stride=2,
                    in_chans=self.embed_dims[2] // self.scale_factor,
                    embed_dim=self.embed_dims[3] // self.scale_factor,
                )

        if self.embedding_tune:
            if "1" in self.tuning_stage:
                self.embedding_generator1 = nn.Linear(
                    self.embed_dims[0], self.embed_dims[0] // self.scale_factor
                )
            if "2" in self.tuning_stage:
                self.embedding_generator2 = nn.Linear(
                    self.embed_dims[1], self.embed_dims[1] // self.scale_factor
                )
            if "3" in self.tuning_stage:
                self.embedding_generator3 = nn.Linear(
                    self.embed_dims[2], self.embed_dims[2] // self.scale_factor
                )
            if "4" in self.tuning_stage:
                self.embedding_generator4 = nn.Linear(
                    self.embed_dims[3], self.embed_dims[3] // self.scale_factor
                )

        if self.adaptor == "adaptor":
            if "1" in self.tuning_stage:
                for i in range(self.depths[0] + 1):
                    lightweight_mlp = nn.Sequential(
                        nn.Linear(
                            self.embed_dims[0] // self.scale_factor,
                            self.embed_dims[0] // self.scale_factor,
                        ),
                        nn.GELU(),
                    )
                    setattr(self, f"lightweight_mlp1_{i}", lightweight_mlp)
                self.shared_mlp1 = nn.Linear(
                    self.embed_dims[0] // self.scale_factor, self.embed_dims[0]
                )

            if "2" in self.tuning_stage:
                for i in range(self.depths[1] + 1):
                    lightweight_mlp = nn.Sequential(
                        nn.Linear(
                            self.embed_dims[1] // self.scale_factor,
                            self.embed_dims[1] // self.scale_factor,
                        ),
                        nn.GELU(),
                    )
                    setattr(self, f"lightweight_mlp2_{i}", lightweight_mlp)
                self.shared_mlp2 = nn.Linear(
                    self.embed_dims[1] // self.scale_factor, self.embed_dims[1]
                )

            if "3" in self.tuning_stage:
                for i in range(self.depths[2] + 1):
                    lightweight_mlp = nn.Sequential(
                        nn.Linear(
                            self.embed_dims[2] // self.scale_factor,
                            self.embed_dims[2] // self.scale_factor,
                        ),
                        nn.GELU(),
                    )
                    setattr(self, f"lightweight_mlp3_{i}", lightweight_mlp)
                self.shared_mlp3 = nn.Linear(
                    self.embed_dims[2] // self.scale_factor, self.embed_dims[2]
                )

            if "4" in self.tuning_stage:
                for i in range(self.depths[3] + 1):
                    lightweight_mlp = nn.Sequential(
                        nn.Linear(
                            self.embed_dims[3] // self.scale_factor,
                            self.embed_dims[3] // self.scale_factor,
                        ),
                        nn.GELU(),
                    )
                    setattr(self, f"lightweight_mlp4_{i}", lightweight_mlp)
                self.shared_mlp4 = nn.Linear(
                    self.embed_dims[3] // self.scale_factor, self.embed_dims[3]
                )

        elif self.adaptor == "fully_shared":
            self.fully_shared_mlp1 = nn.Sequential(
                nn.Linear(
                    self.embed_dims[0] // self.scale_factor,
                    self.embed_dims[0] // self.scale_factor,
                ),
                nn.GELU(),
                nn.Linear(self.embed_dims[0] // self.scale_factor, self.embed_dims[0]),
            )
            self.fully_shared_mlp2 = nn.Sequential(
                nn.Linear(
                    self.embed_dims[1] // self.scale_factor,
                    self.embed_dims[1] // self.scale_factor,
                ),
                nn.GELU(),
                nn.Linear(self.embed_dims[1] // self.scale_factor, self.embed_dims[1]),
            )
            self.fully_shared_mlp3 = nn.Sequential(
                nn.Linear(
                    self.embed_dims[2] // self.scale_factor,
                    self.embed_dims[2] // self.scale_factor,
                ),
                nn.GELU(),
                nn.Linear(self.embed_dims[2] // self.scale_factor, self.embed_dims[2]),
            )
            self.fully_shared_mlp4 = nn.Sequential(
                nn.Linear(
                    self.embed_dims[3] // self.scale_factor,
                    self.embed_dims[3] // self.scale_factor,
                ),
                nn.GELU(),
                nn.Linear(self.embed_dims[3] // self.scale_factor, self.embed_dims[3]),
            )

        elif self.adaptor == "fully_unshared":
            for i in range(self.depths[0]):
                fully_unshared_mlp1 = nn.Sequential(
                    nn.Linear(
                        self.embed_dims[0] // self.scale_factor,
                        self.embed_dims[0] // self.scale_factor,
                    ),
                    nn.GELU(),
                    nn.Linear(
                        self.embed_dims[0] // self.scale_factor, self.embed_dims[0]
                    ),
                )
                setattr(self, f"fully_unshared_mlp1_{i}", fully_unshared_mlp1)
            for i in range(self.depths[1]):
                fully_unshared_mlp2 = nn.Sequential(
                    nn.Linear(
                        self.embed_dims[1] // self.scale_factor,
                        self.embed_dims[1] // self.scale_factor,
                    ),
                    nn.GELU(),
                    nn.Linear(
                        self.embed_dims[1] // self.scale_factor, self.embed_dims[1]
                    ),
                )
                setattr(self, f"fully_unshared_mlp2_{i}", fully_unshared_mlp2)
            for i in range(self.depths[2]):
                fully_unshared_mlp3 = nn.Sequential(
                    nn.Linear(
                        self.embed_dims[2] // self.scale_factor,
                        self.embed_dims[2] // self.scale_factor,
                    ),
                    nn.GELU(),
                    nn.Linear(
                        self.embed_dims[2] // self.scale_factor, self.embed_dims[2]
                    ),
                )
                setattr(self, f"fully_unshared_mlp3_{i}", fully_unshared_mlp3)
            for i in range(self.depths[3]):
                fully_unshared_mlp4 = nn.Sequential(
                    nn.Linear(
                        self.embed_dims[3] // self.scale_factor,
                        self.embed_dims[3] // self.scale_factor,
                    ),
                    nn.GELU(),
                    nn.Linear(
                        self.embed_dims[3] // self.scale_factor, self.embed_dims[3]
                    ),
                )
                setattr(self, f"fully_unshared_mlp4_{i}", fully_unshared_mlp4)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_handcrafted(self, x):
        if self.input_type == "fft":
            x = self.fft(x, self.freq_nums, self.prompt_type)
        elif self.input_type == "all":
            x = self.prompt.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        elif self.input_type == "gaussian":
            x = self.gaussian_filter.conv_gauss(x)
        elif self.input_type == "srm":
            x = self.srm_filter.srm_layer(x)

        B = x.shape[0]
        if "1" in self.tuning_stage:
            handcrafted1, H1, W1 = self.handcrafted_generator1(x)
        else:
            handcrafted1 = None

        if "2" in self.tuning_stage:
            handcrafted2, H2, W2 = self.handcrafted_generator2(
                handcrafted1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
            )
        else:
            handcrafted2 = None

        if "3" in self.tuning_stage:
            handcrafted3, H3, W3 = self.handcrafted_generator3(
                handcrafted2.reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()
            )
        else:
            handcrafted3 = None

        if "4" in self.tuning_stage:
            handcrafted4, H4, W4 = self.handcrafted_generator4(
                handcrafted3.reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous()
            )
        else:
            handcrafted4 = None

        return handcrafted1, handcrafted2, handcrafted3, handcrafted4

    def init_prompt(self, embedding_feature, handcrafted_feature, block_num):
        if self.embedding_tune:
            embedding_generator = getattr(self, f"embedding_generator{block_num}")
            embedding_feature = embedding_generator(embedding_feature)
        if self.handcrafted_tune:
            handcrafted_feature = handcrafted_feature
        return handcrafted_feature, embedding_feature

    def get_prompt(self, x, prompt, block_num, depth_num):
        feat = 0
        B, H, W = prompt[1].shape[0], prompt[1].shape[1], prompt[1].shape[2]
        if self.handcrafted_tune:
            feat += prompt[0].reshape(B, H, W, -1)
        if self.embedding_tune:
            feat = feat + prompt[1]

        if self.adaptor == "adaptor":
            lightweight_mlp = getattr(
                self, f"lightweight_mlp{block_num}_{depth_num}"
            )
            shared_mlp = getattr(self, f"shared_mlp{block_num}")
            feat = lightweight_mlp(feat)
            feat = shared_mlp(feat)
        elif self.adaptor == "fully_shared":
            fully_shared_mlp = getattr(self, f"fully_shared_mlp{block_num}")
            feat = fully_shared_mlp(feat)
        elif self.adaptor == "fully_unshared":
            fully_unshared_mlp = getattr(
                self, f"fully_unshared_mlp{block_num}_{depth_num}"
            )
            feat = fully_unshared_mlp(feat)

        return x + feat

    def fft(self, x, rate, prompt_type):
        mask = torch.zeros_like(x)
        w, h = x.shape[-2:]
        line = int((w * h * rate) ** 0.5 // 2)
        mask[:, :, w // 2 - line : w // 2 + line, h // 2 - line : h // 2 + line] = 1

        fft = torch.fft.fftshift(torch.fft.fft2(x, norm="forward"))

        if prompt_type == "highpass":
            fft = fft * (1 - mask)
        elif prompt_type == "lowpass":
            fft = fft * mask
        fr = fft.real
        fi = fft.imag

        fft_hires = torch.fft.ifftshift(torch.complex(fr, fi))
        inv = torch.fft.ifft2(fft_hires, norm="forward").real
        inv = torch.abs(inv)
        return inv


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings and 2d-rope."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
        cls_token: bool = False,
        use_rope: bool = False,
        rope_theta: float = 10000.0,
        rope_pt_size: Optional[Tuple[int, int]] = None,
        rope_interp: bool = False,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size or rope size.
            attn_type: Type of attention operation, e.g. "vanilla", "vanilla-xformer".
            cls_token: whether a cls_token is present.
            use_rope: whether to use rope 2d (indep of use_rel_pos, as it can be used together)
            rope_theta: control frequencies of rope
            rope_pt_size: size of rope in previous stage of training, needed for interpolation or tiling
            rope_interp: whether to interpolate (or extrapolate) rope to match input size
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.cls_token = cls_token

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        # rel_pos embeddings and rope
        self.use_rel_pos = use_rel_pos
        self.input_size = input_size

        self.use_rope = use_rope
        self.rope_theta = rope_theta
        self.rope_pt_size = rope_pt_size
        self.rope_interp = rope_interp

        # init rel_pos embeddings and rope
        self._setup_rel_pos(rel_pos_zero_init)
        self._setup_rope_freqs()

    def _setup_rel_pos(self, rel_pos_zero_init: bool = True) -> None:
        if not self.use_rel_pos:
            self.rel_pos_h = None
            self.rel_pos_w = None
            return

        assert self.input_size is not None
        assert self.cls_token is False, "not supported"
        # initialize relative positional embeddings
        self.rel_pos_h = nn.Parameter(
            torch.zeros(2 * self.input_size[0] - 1, self.head_dim)
        )
        self.rel_pos_w = nn.Parameter(
            torch.zeros(2 * self.input_size[1] - 1, self.head_dim)
        )

        if not rel_pos_zero_init:
            trunc_normal_(self.rel_pos_h, std=0.02)
            trunc_normal_(self.rel_pos_w, std=0.02)

        # Precompute the relative coords
        H, W = self.input_size
        q_coords = torch.arange(H)[:, None]
        k_coords = torch.arange(W)[None, :]
        relative_coords = (q_coords - k_coords) + (H - 1)
        self.register_buffer("relative_coords", relative_coords.long())

    def _setup_rope_freqs(self) -> None:
        if not self.use_rope:
            self.freqs_cis = None
            return

        assert self.input_size is not None
        # determine rope input size
        if self.rope_pt_size is None:
            self.rope_pt_size = self.input_size

        # initialize 2d rope freqs
        self.compute_cis = partial(
            compute_axial_cis,
            dim=self.head_dim,
            theta=self.rope_theta,
        )

        # interpolate rope
        scale_pos = 1.0
        if self.rope_interp:
            scale_pos = self.rope_pt_size[0] / self.input_size[0]
        # get scaled freqs_cis
        freqs_cis = self.compute_cis(
            end_x=self.input_size[0],
            end_y=self.input_size[1],
            scale_pos=scale_pos,
        )
        if self.cls_token:
            t = torch.zeros(
                self.head_dim // 2,
                dtype=torch.float32,
                device=freqs_cis.device,
            )
            cls_freqs_cis = torch.polar(torch.ones_like(t), t)[None, :]
            freqs_cis = torch.cat([cls_freqs_cis, freqs_cis], dim=0)

        self.register_buffer("freqs_cis", freqs_cis)

    def _apply_rope(self, q, k) -> Tuple[Tensor, Tensor]:
        if not self.use_rope:
            return q, k

        assert self.freqs_cis is not None
        return apply_rotary_enc(q, k, freqs_cis=self.freqs_cis)

    def forward(self, x: Tensor) -> Tensor:
        s = 1 if self.cls_token else 0  # used to exclude cls_token
        if x.ndim == 4:
            B, H, W, _ = x.shape
            assert s == 0  # no cls_token
            L = H * W
            ndim = 4
        else:
            assert x.ndim == 3
            B, L, _ = x.shape
            ndim = 3
            H = W = math.sqrt(L - s)

        # qkv with shape (3, B, nHead, L, C)
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, -1)
        # q, k, v with shape (B, nHead, L, C)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        # handle rope and rel pos embeddings
        q, k = self._apply_rope(q, k)
        if self.use_rel_pos:
            q, k = concat_rel_pos(
                q.flatten(0, 1),
                k.flatten(0, 1),
                (H, W),
                x.shape[1:3],
                self.rel_pos_h,
                self.rel_pos_w,
                rescale=True,
                relative_coords=self.relative_coords,
            )

            # sdpa expects [B, nheads, H*W, C] so we transpose back
            q = q.reshape(B, self.num_heads, H * W, -1)
            k = k.reshape(B, self.num_heads, H * W, -1)

        x = F.scaled_dot_product_attention(q, k, v)

        if ndim == 4:
            x = (
                x.view(B, self.num_heads, H, W, -1)
                .permute(0, 2, 3, 1, 4)
                .reshape(B, H, W, -1)
            )
        else:
            x = x.view(B, self.num_heads, L, -1).permute(0, 2, 1, 3).reshape(B, L, -1)

        x = self.proj(x)

        return x


class Block(nn.Module):
    """Transformer blocks with support of window attention"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
        use_rope: bool = False,
        rope_pt_size: Optional[Tuple[int, int]] = None,
        rope_tiled: bool = False,
        rope_interp: bool = False,
        use_ve_rope: bool = False,
        cls_token: bool = False,
        dropout: float = 0.0,
        init_values: Optional[float] = None,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then not
                use window attention.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
            dropout (float): Dropout rate.
            cls_token: whether a cls_token is present.
            use_rope: whether to use rope 2d (indep of use_rel_pos, as it can be used together)
            rope_pt_size: size of rope in previous stage of training, needed for interpolation or tiling
            rope_interp: whether to interpolate (or extrapolate) rope to match target input size,
                expected to specify source size as rope_pt_size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
            use_rope=use_rope,
            rope_pt_size=rope_pt_size,
            rope_interp=rope_interp,
            cls_token=cls_token,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=(dropout, 0.0),
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.dropout = nn.Dropout(dropout)
        self.window_size = window_size

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.ls1(self.attn(x))
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + self.dropout(self.drop_path(x))
        x = x + self.dropout(self.drop_path(self.ls2(self.mlp(self.norm2(x)))))

        return x


class ViT(nn.Module):
    """
    This module implements Vision Transformer (ViT) backbone in :paper:`vitdet`.
    "Exploring Plain Vision Transformer Backbones for Object Detection",
    https://arxiv.org/abs/2203.16527
    """

    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        norm_layer: Union[Callable[..., nn.Module], str] = "LayerNorm",
        act_layer: Callable[..., nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        tile_abs_pos: bool = True,
        rel_pos_blocks: Union[Tuple[int, ...], bool] = (2, 5, 8, 11),
        rel_pos_zero_init: bool = True,
        window_size: int = 14,
        global_att_blocks: Tuple[int, ...] = (2, 5, 8, 11),
        use_rope: bool = False,
        rope_pt_size: Optional[int] = None,
        use_interp_rope: bool = False,
        pretrain_img_size: int = 224,
        pretrain_use_cls_token: bool = True,
        retain_cls_token: bool = True,
        dropout: float = 0.0,
        return_interm_layers: bool = False,
        init_values: Optional[float] = None,  # for layerscale
        ln_pre: bool = False,
        ln_post: bool = False,
        bias_patch_embed: bool = True,
        compile_mode: Optional[str] = None,
        use_act_checkpoint: bool = True,
        # ================= Adapter 参数 =================
        enable_adapter: bool = False,
        tuning_stage: str = "1234",
        handcrafted_tune: bool = True,
        embedding_tune: bool = True,
        adaptor: str = "adaptor",
        # ===============================================
    ):
        """
        Args:
            img_size (int): Input image size. Only relevant for rel pos or rope.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path_rate (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            tile_abs_pos (bool): If True, tile absolute positional embeddings instead of interpolation.
            rel_pos_blocks (list): Blocks which have rel pos embeddings.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_att_blocks (list): Indexes for blocks using global attention (other blocks use window attention).
            use_rope (bool): whether to use rope 2d (indep of rel_pos_blocks, as it can be used together).
            rope_pt_size (int): size of rope in previous stage of training, needed for interpolation or tiling.
            use_interp_rope: whether to interpolate (or extrapolate) rope to match target input size,
                expected to specify source size as rope_pt_size.
            use_act_checkpoint (bool): If True, use activation checkpointing.
            pretrain_img_size (int): input image size for pretraining models.
            pretrain_use_cls_token (bool): If True, pretraining models use class token.
            retain_cls_token: whether cls_token should be retained.
            dropout (float): Dropout rate. Applied in residual blocks of attn, mlp and inside the mlp.

            return_interm_layers (bool): Whether to return intermediate layers (all global attention blocks).
            init_values: layer scale init, None for no layer scale.

            ln_pre (bool): If True, apply layer norm before transformer blocks.
            ln_post (bool): If True, apply layer norm after transformer blocks.
            bias_patch_embed (bool): bias in conv for patch embed?
            compile_mode (str): mode to compile the forward
        """
        super().__init__()
        self.pretrain_use_cls_token = pretrain_use_cls_token
        self.enable_adapter = enable_adapter

        if self.enable_adapter:
            # 计算每个 stage 的 depth 分配，并初始化 Adapter
            if retain_cls_token:
                raise ValueError(
                    "Adapter injection does not support retain_cls_token=True."
                )
            self.depth_per_stage = depth // 4
            remainder = depth % 4
            depths_list = [self.depth_per_stage] * 4
            if remainder > 0:
                depths_list[-1] += remainder

            self.prompt_generator = PromptGenerator(
                scale_factor=32,
                prompt_type="highpass",
                embed_dims=[embed_dim, embed_dim, embed_dim, embed_dim],
                tuning_stage=tuning_stage,
                depths=depths_list,
                input_type="fft",
                freq_nums=0.25,
                handcrafted_tune=handcrafted_tune,
                embedding_tune=embedding_tune,
                adaptor=adaptor,
                img_size=img_size,
            )

        window_block_indexes = [i for i in range(depth) if i not in global_att_blocks]
        self.full_attn_ids = list(global_att_blocks)
        self.rel_pos_blocks = [False] * depth
        if isinstance(rel_pos_blocks, bool) and rel_pos_blocks:
            self.rel_pos_blocks = [True] * depth
        else:
            for i in rel_pos_blocks:
                self.rel_pos_blocks[i] = True

        self.retain_cls_token = retain_cls_token
        if self.retain_cls_token:
            assert pretrain_use_cls_token
            assert len(window_block_indexes) == 0, (
                "windowing not supported with cls token"
            )

            assert sum(self.rel_pos_blocks) == 0, "rel pos not supported with cls token"

            scale = embed_dim**-0.5
            self.class_embedding = nn.Parameter(scale * torch.randn(1, 1, embed_dim))

        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-5)

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=bias_patch_embed,
        )

        # Handle absolute positional embedding
        self.tile_abs_pos = tile_abs_pos
        self.use_abs_pos = use_abs_pos
        if self.tile_abs_pos:
            assert self.use_abs_pos

        if self.use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            num_patches = (pretrain_img_size // patch_size) * (
                pretrain_img_size // patch_size
            )
            num_positions = (num_patches + 1) if pretrain_use_cls_token else num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
        else:
            self.pos_embed = None

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList()
        cur_stage = 1
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=self.rel_pos_blocks[i],
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i in window_block_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
                use_rope=use_rope,
                rope_pt_size=(
                    (window_size, window_size)
                    if rope_pt_size is None
                    else (rope_pt_size, rope_pt_size)
                ),
                rope_interp=use_interp_rope,
                cls_token=self.retain_cls_token,
                dropout=dropout,
                init_values=init_values,
            )

            if i not in window_block_indexes:
                cur_stage += 1

            self.use_act_checkpoint = use_act_checkpoint

            self.blocks.append(block)

        self.return_interm_layers = return_interm_layers
        self.channel_list = (
            [embed_dim] * len(self.full_attn_ids)
            if return_interm_layers
            else [embed_dim]
        )

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)

        self.ln_pre = norm_layer(embed_dim) if ln_pre else nn.Identity()
        self.ln_post = norm_layer(embed_dim) if ln_post else nn.Identity()

        self.apply(self._init_weights)

        if compile_mode is not None:
            self.forward = torch.compile(
                self.forward, mode=compile_mode, fullgraph=True
            )
            if self.use_act_checkpoint and self.training:
                torch._dynamo.config.optimize_ddp = False

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _resize_handcrafted(self, feature, target_h, target_w):
        """
        将手工特征缩放到 ViT 特征大小，兼容多种维度格式。
        """
        if feature is None:
            return None

        if feature.ndim == 3:
            feature = feature.unsqueeze(1)

        if feature.ndim == 4:
            if feature.shape[-1] < feature.shape[1]:
                feature = feature.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

        if feature.shape[2] != target_h or feature.shape[3] != target_w:
            feature = F.interpolate(
                feature,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )

        return feature.permute(0, 2, 3, 1)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        inp = x
        x = self.patch_embed(x)
        h, w = x.shape[1], x.shape[2]

        s = 0
        if self.retain_cls_token:
            # If cls_token is retained, we don't
            # maintain spatial shape
            x = torch.cat([self.class_embedding, x.flatten(1, 2)], dim=1)
            s = 1

        if self.pos_embed is not None:
            x = x + get_abs_pos(
                self.pos_embed,
                self.pretrain_use_cls_token,
                (h, w),
                self.retain_cls_token,
                tiling=self.tile_abs_pos,
            )

        x = self.ln_pre(x)

        if self.enable_adapter:
            # 预先生成手工特征（如 FFT 高频）
            handcrafted_list = self.prompt_generator.init_handcrafted(inp)

        outputs = []
        for i, blk in enumerate(self.blocks):
            if self.enable_adapter:
                # 根据 block 所在 stage 注入 Adapter prompt
                if i < self.depth_per_stage:
                    stage_idx = 1
                    rel_idx = i
                elif i < self.depth_per_stage * 2:
                    stage_idx = 2
                    rel_idx = i - self.depth_per_stage
                elif i < self.depth_per_stage * 3:
                    stage_idx = 3
                    rel_idx = i - self.depth_per_stage * 2
                else:
                    stage_idx = 4
                    rel_idx = i - self.depth_per_stage * 3

                current_handcrafted = handcrafted_list[stage_idx - 1]
                if str(stage_idx) in self.prompt_generator.tuning_stage:
                    resized_handcrafted = self._resize_handcrafted(
                        current_handcrafted, h, w
                    )
                    prompt_tuple = self.prompt_generator.init_prompt(
                        x, resized_handcrafted, stage_idx
                    )
                    x = self.prompt_generator.get_prompt(
                        x, prompt_tuple, stage_idx, rel_idx
                    )

            if self.use_act_checkpoint and self.training:
                x = checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
            if (i == self.full_attn_ids[-1]) or (
                self.return_interm_layers and i in self.full_attn_ids
            ):
                if i == self.full_attn_ids[-1]:
                    x = self.ln_post(x)

                feats = x[:, s:]
                if feats.ndim == 4:
                    feats = feats.permute(0, 3, 1, 2)
                else:
                    assert feats.ndim == 3
                    # h = w = math.sqrt(feats.shape[1])
                    feats = feats.reshape(
                        feats.shape[0], h, w, feats.shape[-1]
                    ).permute(0, 3, 1, 2)

                outputs.append(feats)

        return outputs

    def get_layer_id(self, layer_name: str) -> int:
        # https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
        num_layers = self.get_num_layers()

        if layer_name.find("rel_pos") != -1:
            return num_layers + 1
        elif layer_name.find("ln_pre") != -1:
            return 0
        elif layer_name.find("pos_embed") != -1 or layer_name.find("cls_token") != -1:
            return 0
        elif layer_name.find("patch_embed") != -1:
            return 0
        elif layer_name.find("blocks") != -1:
            return int(layer_name.split("blocks")[1].split(".")[1]) + 1
        else:
            return num_layers + 1

    def get_num_layers(self) -> int:
        return len(self.blocks)
