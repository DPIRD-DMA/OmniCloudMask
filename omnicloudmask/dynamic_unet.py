"""
DynamicUNet architecture for semantic segmentation.

This module contains code derived from the fastai library.
Original source: https://github.com/fastai/fastai
Licensed under Apache License 2.0

Copyright 2017 onwards, fast.ai, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Modifications made for OmniCloudMask:
- Extracted minimal required components for inference
- Removed fastai/fastcore dependencies
- Simplified for standalone use with timm models
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class NormType(Enum):
    """Normalization type enum."""

    Batch = 1
    BatchZero = 2
    Instance = 3
    InstanceZero = 4


def one_param(m: nn.Module) -> torch.Tensor:
    """Return the first parameter of a module."""
    return next(m.parameters())


def in_channels(m: nn.Module) -> int:
    """Return the number of input channels of first conv layer in model."""
    for layer in m.modules():
        if hasattr(layer, "weight") and layer.weight is not None and len(layer.weight.shape) == 4:
            return layer.weight.shape[1]
    raise ValueError("No conv layer found")


def icnr_init(
    x: torch.Tensor, scale: int = 2, init: Callable = nn.init.kaiming_normal_
) -> torch.Tensor:
    """ICNR init of `x`, with `scale` and `init` function."""
    ni, nf, h, w = x.shape
    ni2 = int(ni / (scale**2))
    k = init(x.new_zeros([ni2, nf, h, w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale**2)
    return k.contiguous().view([nf, ni, h, w]).transpose(0, 1)


def apply_init(m: nn.Module, init_fn: Callable) -> None:
    """Initialize all non-batchnorm layers of `m` with `init_fn`."""
    for module in m.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            continue
        if hasattr(module, "weight") and module.weight is not None:
            if module.weight.ndim >= 2:
                init_fn(module.weight)


class Hook:
    """A hook to capture module output."""

    def __init__(self, m: nn.Module, hook_func: Callable, is_forward: bool = True, detach: bool = True):
        self.hook_func = hook_func
        self.detach = detach
        self.stored: Any = None
        f = m.register_forward_hook if is_forward else m.register_backward_hook
        self.hook = f(self.hook_fn)

    def hook_fn(self, module: nn.Module, input: Any, output: Any) -> None:
        if self.detach:
            output = output.detach() if isinstance(output, torch.Tensor) else output
        self.stored = self.hook_func(module, input, output)

    def remove(self) -> None:
        self.hook.remove()

    def __enter__(self) -> "Hook":
        return self

    def __exit__(self, *args: Any) -> None:
        self.remove()


class Hooks:
    """A collection of hooks."""

    def __init__(self, ms: list[nn.Module], hook_func: Callable, is_forward: bool = True, detach: bool = True):
        self.hooks = [Hook(m, hook_func, is_forward, detach) for m in ms]

    def __getitem__(self, i: int) -> Hook:
        return self.hooks[i]

    def __len__(self) -> int:
        return len(self.hooks)

    def __iter__(self):
        return iter(self.hooks)

    @property
    def stored(self) -> list[Any]:
        return [h.stored for h in self.hooks]

    def remove(self) -> None:
        for h in self.hooks:
            h.remove()

    def __enter__(self) -> "Hooks":
        return self

    def __exit__(self, *args: Any) -> None:
        self.remove()


def hook_outputs(modules: list[nn.Module], detach: bool = True, grad: bool = False) -> Hooks:
    """Return Hooks that store activations of all `modules`."""
    return Hooks(modules, lambda m, i, o: o, is_forward=not grad, detach=detach)


def model_sizes(m: nn.Module, size: tuple[int, int], device: Optional[torch.device] = None) -> list[list[int]]:
    """Pass a dummy input through the model `m` to get sizes of each layer."""
    if device is None:
        device = one_param(m).device

    children = list(m.children())
    if not children:
        with torch.no_grad():
            x = torch.zeros(1, in_channels(m), *size, device=device)
            out = m(x)
            return [list(out.shape)]

    sizes = []
    with hook_outputs(children, detach=True) as hooks:
        with torch.no_grad():
            x = torch.zeros(1, in_channels(m), *size, device=device)
            m(x)
        for hook in hooks:
            if hook.stored is not None:
                sizes.append(list(hook.stored.shape))

    return sizes


def dummy_eval(m: nn.Module, size: tuple[int, int], device: Optional[torch.device] = None) -> torch.Tensor:
    """Evaluate `m` on a dummy input of `size`."""
    if device is None:
        device = one_param(m).device
    return m(torch.zeros(1, in_channels(m), *size, device=device))


def _get_norm(
    nf: int, norm_type: Optional[NormType], ndim: int = 2, zero_bn: bool = False, **kwargs: Any
) -> Optional[nn.Module]:
    """Return a normalization layer."""
    if norm_type is None:
        return None
    if norm_type in (NormType.Batch, NormType.BatchZero):
        bn_cls = getattr(nn, f"BatchNorm{ndim}d")
        bn = bn_cls(nf, **kwargs)
        if zero_bn or norm_type == NormType.BatchZero:
            nn.init.zeros_(bn.weight)
        return bn
    elif norm_type in (NormType.Instance, NormType.InstanceZero):
        return getattr(nn, f"InstanceNorm{ndim}d")(nf, affine=True, **kwargs)
    return None


class ConvLayer(nn.Sequential):
    """Convolution layer with configurable normalization and activation."""

    def __init__(
        self,
        ni: int,
        nf: int,
        ks: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        bias: Optional[bool] = None,
        ndim: int = 2,
        norm_type: Optional[NormType] = NormType.Batch,
        bn_1st: bool = True,
        act_cls: Optional[type] = nn.ReLU,
        transpose: bool = False,
        init: str = "auto",
        xtra: Optional[nn.Module] = None,
        bias_std: float = 0.01,
        groups: int = 1,
        **kwargs: Any,
    ):
        if padding is None:
            padding = (ks - 1) // 2 if not transpose else 0
        bn_type = (
            NormType.BatchZero
            if norm_type == NormType.BatchZero
            else NormType.Batch
            if norm_type in (NormType.Batch,)
            else norm_type
        )
        if norm_type in (NormType.InstanceZero, NormType.Instance):
            bn_type = (
                NormType.InstanceZero
                if norm_type == NormType.InstanceZero
                else NormType.Instance
            )
        if bias is None:
            bias = norm_type not in (NormType.Batch, NormType.BatchZero) if norm_type else True
        conv_cls = getattr(nn, f"Conv{'Transpose' if transpose else ''}{ndim}d")
        conv = conv_cls(
            ni, nf, ks, stride=stride, padding=padding, bias=bias, groups=groups, **kwargs
        )

        if init == "auto":
            if act_cls in (nn.ReLU, nn.LeakyReLU):
                nn.init.kaiming_uniform_(
                    conv.weight, a=0.01 if act_cls == nn.LeakyReLU else 0
                )
            else:
                nn.init.xavier_uniform_(conv.weight)
        if bias and bias_std and conv.bias is not None:
            nn.init.normal_(conv.bias, std=bias_std)

        layers: list[nn.Module] = [conv]
        zero_bn = norm_type in (NormType.BatchZero, NormType.InstanceZero) if norm_type else False
        bn = _get_norm(nf, bn_type, ndim, zero_bn=zero_bn) if norm_type is not None else None
        act = act_cls(inplace=True) if act_cls else None

        if bn_1st:
            if bn:
                layers.append(bn)
            if act:
                layers.append(act)
        else:
            if act:
                layers.append(act)
            if bn:
                layers.append(bn)
        if xtra:
            layers.append(xtra)
        super().__init__(*layers)


class PixelShuffle_ICNR(nn.Sequential):
    """Upsample by `scale` using ICNR init on PixelShuffle."""

    def __init__(
        self,
        ni: int,
        nf: Optional[int] = None,
        scale: int = 2,
        blur: bool = False,
        norm_type: Optional[NormType] = NormType.Batch,
        act_cls: type = nn.ReLU,
    ):
        nf = ni if nf is None else nf
        layers: list[nn.Module] = [
            ConvLayer(
                ni, nf * (scale**2), ks=1, norm_type=norm_type, act_cls=act_cls, bias_std=0
            ),
            nn.PixelShuffle(scale),
        ]
        layers[0][0].weight.data.copy_(icnr_init(layers[0][0].weight.data, scale=scale))
        if blur:
            layers += [nn.ReplicationPad2d((1, 0, 1, 0)), nn.AvgPool2d(2, stride=1)]
        super().__init__(*layers)


class SelfAttention(nn.Module):
    """Self-attention layer for 2D inputs."""

    def __init__(self, n_channels: int):
        super().__init__()
        self.query = nn.Conv1d(n_channels, n_channels // 8, 1)
        self.key = nn.Conv1d(n_channels, n_channels // 8, 1)
        self.value = nn.Conv1d(n_channels, n_channels, 1)
        self.gamma = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.size()
        x_flat = x.view(*size[:2], -1)
        q, k, v = self.query(x_flat), self.key(x_flat), self.value(x_flat)
        beta = F.softmax(torch.bmm(q.transpose(1, 2), k), dim=1)
        out = self.gamma * torch.bmm(v, beta) + x_flat
        return out.view(*size)


class MergeLayer(nn.Module):
    """Merge a shortcut with the result of the module by adding or concatenating."""

    def __init__(self, dense: bool = False):
        super().__init__()
        self.dense = dense

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, x.orig], dim=1) if self.dense else x + x.orig


class ResBlock(nn.Module):
    """Residual block."""

    def __init__(
        self,
        expansion: int,
        ni: int,
        nf: int,
        stride: int = 1,
        groups: int = 1,
        nh1: Optional[int] = None,
        nh2: Optional[int] = None,
        norm_type: Optional[NormType] = NormType.Batch,
        act_cls: type = nn.ReLU,
        ks: int = 3,
        **kwargs: Any,
    ):
        super().__init__()
        norm2 = (
            NormType.BatchZero
            if norm_type == NormType.Batch
            else NormType.InstanceZero
            if norm_type == NormType.Instance
            else norm_type
        )
        if nh2 is None:
            nh2 = nf
        if nh1 is None:
            nh1 = nh2
        self.convpath = nn.Sequential(
            ConvLayer(ni, nh1, ks, norm_type=norm_type, act_cls=act_cls, groups=groups, **kwargs),
            ConvLayer(nh1, nh2, ks, norm_type=norm2, act_cls=None, groups=groups, **kwargs),
        )
        self.idpath = (
            nn.Identity()
            if ni == nf
            else ConvLayer(ni, nf, 1, norm_type=norm_type, act_cls=None)
        )
        self.pool = nn.Identity() if stride == 1 else nn.AvgPool2d(stride, ceil_mode=True)
        self.act = act_cls(inplace=True) if act_cls else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.convpath(x) + self.idpath(self.pool(x)))


class UnetBlock(nn.Module):
    """A U-Net decoder block."""

    def __init__(
        self,
        up_in_c: int,
        x_in_c: int,
        hook: Hook,
        final_div: bool = True,
        blur: bool = False,
        act_cls: type = nn.ReLU,
        self_attention: bool = False,
        init: Callable = nn.init.kaiming_normal_,
        norm_type: Optional[NormType] = NormType.Batch,
        **kwargs: Any,
    ):
        super().__init__()
        self.hook = hook
        self.shuf = PixelShuffle_ICNR(
            up_in_c, up_in_c // 2, blur=blur, act_cls=act_cls, norm_type=norm_type
        )
        self.bn = _get_norm(x_in_c, NormType.Batch)
        ni = up_in_c // 2 + x_in_c
        nf = ni if final_div else ni // 2
        self.conv1 = ConvLayer(ni, nf, norm_type=norm_type, act_cls=act_cls, **kwargs)
        self.conv2 = ConvLayer(
            nf,
            nf,
            norm_type=norm_type,
            act_cls=act_cls,
            xtra=SelfAttention(nf) if self_attention else None,
            **kwargs,
        )
        self.relu = act_cls(inplace=True) if act_cls else nn.Identity()
        apply_init(nn.Sequential(self.conv1, self.conv2), init)

    def forward(self, up_in: torch.Tensor) -> torch.Tensor:
        s = self.hook.stored
        up_out = self.shuf(up_in)
        ssh = s.shape[-2:]
        if ssh != up_out.shape[-2:]:
            up_out = F.interpolate(up_out, s.shape[-2:], mode="nearest")
        cat_x = self.relu(torch.cat([up_out, self.bn(s)], dim=1))
        return self.conv2(self.conv1(cat_x))


class ResizeToOrig(nn.Module):
    """Resize output to match original input size."""

    def __init__(self, mode: str = "nearest"):
        super().__init__()
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.orig.shape[-2:] != x.shape[-2:]:
            return F.interpolate(x, x.orig.shape[-2:], mode=self.mode)
        return x


class ToTensorBase(nn.Module):
    """Identity module for compatibility with fastai."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _get_sz_change_idxs(sizes: list[list[int]]) -> list[int]:
    """Get indices of layers BEFORE the feature map size changes."""
    feature_szs = [s[-1] for s in sizes]
    sz_chg_idxs = []
    for i in range(len(feature_szs) - 1):
        if feature_szs[i] != feature_szs[i + 1]:
            sz_chg_idxs.append(i)
    return sz_chg_idxs


class SequentialEx(nn.Module):
    """Sequential container that passes original input to each module."""

    def __init__(self, *layers: nn.Module):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        for layer in self.layers:
            res.orig = x
            nres = layer(res)
            res.orig = None
            res = nres
        return res


class DynamicUnet(SequentialEx):
    """Create a U-Net from a given encoder architecture."""

    def __init__(
        self,
        encoder: nn.Module,
        n_out: int,
        img_size: tuple[int, int],
        blur: bool = False,
        blur_final: bool = True,
        self_attention: bool = False,
        y_range: Optional[tuple[float, float]] = None,
        last_cross: bool = True,
        bottle: bool = False,
        act_cls: type = nn.ReLU,
        init: Callable = nn.init.kaiming_normal_,
        norm_type: Optional[NormType] = None,
        **kwargs: Any,
    ):
        imsize = img_size
        sizes = model_sizes(encoder, size=imsize)
        sz_chg_idxs = list(reversed(_get_sz_change_idxs(sizes)))
        self.sfs = hook_outputs([encoder[i] for i in sz_chg_idxs], detach=False)
        x = dummy_eval(encoder, imsize).detach()

        ni = sizes[-1][1]
        middle_conv = nn.Sequential(
            ConvLayer(ni, ni * 2, act_cls=act_cls, norm_type=norm_type, **kwargs),
            ConvLayer(ni * 2, ni, act_cls=act_cls, norm_type=norm_type, **kwargs),
        )
        middle_conv.eval()
        x = middle_conv(x)
        layers: list[nn.Module] = [encoder, nn.BatchNorm2d(ni), nn.ReLU(), middle_conv]

        for i, idx in enumerate(sz_chg_idxs):
            not_final = i != len(sz_chg_idxs) - 1
            up_in_c, x_in_c = int(x.shape[1]), int(sizes[idx][1])
            do_blur = blur and (not_final or blur_final)
            sa = self_attention and (i == len(sz_chg_idxs) - 3)
            unet_block = UnetBlock(
                up_in_c,
                x_in_c,
                self.sfs[i],
                final_div=not_final,
                blur=do_blur,
                self_attention=sa,
                act_cls=act_cls,
                init=init,
                norm_type=norm_type,
                **kwargs,
            )
            unet_block.eval()
            layers.append(unet_block)
            x = unet_block(x)

        ni = x.shape[1]
        if imsize != sizes[0][-2:]:
            layers.append(PixelShuffle_ICNR(ni, act_cls=act_cls, norm_type=norm_type))
        layers.append(ResizeToOrig())
        if last_cross:
            layers.append(MergeLayer(dense=True))
            ni += in_channels(encoder)
            layers.append(
                ResBlock(1, ni, ni // 2 if bottle else ni, act_cls=act_cls, norm_type=norm_type, **kwargs)
            )
        layers += [ConvLayer(ni, n_out, ks=1, act_cls=None, norm_type=norm_type, **kwargs)]
        apply_init(nn.Sequential(layers[3], layers[-2] if last_cross else layers[-1]), init)
        if y_range is not None:
            layers.append(_SigmoidRange(*y_range))
        layers.append(ToTensorBase())
        super().__init__(*layers)

    def __del__(self) -> None:
        if hasattr(self, "sfs"):
            self.sfs.remove()


class _SigmoidRange(nn.Module):
    """Sigmoid activation with output range."""

    def __init__(self, low: float, high: float):
        super().__init__()
        self.low = low
        self.high = high

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x) * (self.high - self.low) + self.low


def _is_classifier_head(module: nn.Module) -> bool:
    """Check if a module is a classifier head that should be removed."""
    cls_name = type(module).__name__.lower()
    if any(x in cls_name for x in ["classifier", "head"]):
        return True
    if hasattr(module, "global_pool") and hasattr(module, "fc"):
        return True
    if isinstance(module, (nn.Linear, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d)):
        return True
    return False


def create_unet_model(
    arch: Callable,
    n_out: int,
    img_size: tuple[int, int],
    pretrained: bool = True,
    cut: Optional[int] = None,
    n_in: int = 3,
    act_cls: type = nn.ReLU,
    norm_type: Optional[NormType] = None,
    **kwargs: Any,
) -> DynamicUnet:
    """Create a DynamicUnet from a backbone architecture.

    Args:
        arch: A function that returns the backbone model
        n_out: Number of output channels
        img_size: Input image size (height, width)
        pretrained: Whether to use pretrained weights (passed to arch)
        cut: Index to cut the backbone at (if None, uses default)
        n_in: Number of input channels
        act_cls: Activation class to use
        norm_type: Normalization type
        **kwargs: Additional arguments passed to DynamicUnet

    Returns:
        DynamicUnet model
    """
    body = arch()

    if hasattr(body, "forward_features"):
        children = list(body.children())
        if cut is None:
            cut = len(children)
            for i in range(len(children) - 1, -1, -1):
                child = children[i]
                if _is_classifier_head(child):
                    cut = i
                    continue
                if isinstance(child, (nn.Identity, nn.Flatten)):
                    continue
                break

        body = nn.Sequential(*children[:cut])
    elif cut is not None:
        children = list(body.children())
        body = nn.Sequential(*children[:cut])

    return DynamicUnet(body, n_out, img_size, act_cls=act_cls, norm_type=norm_type, **kwargs)
