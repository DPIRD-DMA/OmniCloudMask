import torch

# This module contains MPS (Apple Silicon) optimizations for PyTorch models:
# 1. Conv2d patch: Replaces Conv2d layers with MPS-safe versions that ensure inputs are contiguous  # noqa: E501
#    - Addresses RuntimeError: "view size is not compatible with input tensor's size and stride"  # noqa: E501
#    - Required for some model architectures (like timm EdgeNeXt) on MPS devices
# 2. Fast argmax: Optimized argmax implementation for MPS devices (~60x faster than torch.argmax)  # noqa: E501
#    - Uses pairwise comparison approach (8.5ms vs 3.3s for standard torch.argmax)
#    - Significantly improves inference performance for cloud mask generation


def fast_argmax_mps(tensor: torch.Tensor) -> torch.Tensor:
    """Fast argmax implementation optimized for MPS devices.

    Uses pairwise comparison approach which is ~60x faster than torch.argmax
    on MPS devices (8.5ms vs 540ms for transferring to NumPy, vs 3.3s for torch.argmax).

    Args:
        tensor: Input tensor of shape (C, H, W) where C is number of classes

    Returns:
        Tensor of shape (1, H, W) with dtype int64 containing argmax indices
    """

    # Pairwise comparison approach optimized for MPS
    # Compare classes 0 vs 1 (use >= to prefer lower index on ties, matching argmax)
    mask_01 = tensor[0] >= tensor[1]
    max_01 = torch.where(mask_01, tensor[0], tensor[1])
    idx_01 = torch.where(
        mask_01,
        torch.zeros_like(mask_01, dtype=torch.int64),
        torch.ones_like(mask_01, dtype=torch.int64),
    )

    # Compare result vs remaining classes (use >= to prefer lower index on ties)
    for i in range(2, tensor.size(0)):
        mask_final = max_01 >= tensor[i]
        idx_01 = torch.where(
            mask_final, idx_01, torch.full_like(idx_01, i, dtype=torch.int64)
        )
        max_01 = torch.where(mask_final, max_01, tensor[i])

    return idx_01.unsqueeze(0)


def requires_mps_fix(
    model: torch.nn.Module, inference_device: torch.device, inference_dtype: torch.dtype
) -> bool:
    """
    Check if MPS fix is needed by testing the model for contiguity issues.
    Returns True if the MPS contiguity issue occurs, False otherwise
    """
    try:
        test_input = torch.randn(
            1, 3, 65, 65, device=inference_device, dtype=inference_dtype
        )

        with torch.no_grad():
            _ = model(test_input)
        return False

    except RuntimeError as e:
        if "view size is not compatible" in str(e):
            return True  # MPS contiguity issue detected
        else:
            raise


class MPSSafeConv2d(torch.nn.Conv2d):
    """MPS-safe version of Conv2d that ensures inputs are contiguous"""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.device.type == "mps" and not input.is_contiguous():
            input = input.contiguous()
        return super().forward(input)


def patch_model_for_mps(
    model: torch.nn.Module, inference_device: torch.device, inference_dtype: torch.dtype
) -> torch.nn.Module:
    """
    Replace all Conv2d layers in a model with MPS-safe versions.
    This modifies the model in-place by changing the class of Conv2d layers.

    Args:
        model: The PyTorch model to patch

    Returns:
        The same model with Conv2d layers made MPS-safe
    """
    if not requires_mps_fix(
        model=model, inference_device=inference_device, inference_dtype=inference_dtype
    ):
        return model

    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and not isinstance(
            module, MPSSafeConv2d
        ):
            module.__class__ = MPSSafeConv2d

    return model


def patch_models_for_mps(
    models: list[torch.nn.Module],
    inference_device: torch.device,
    inference_dtype: torch.dtype,
) -> list[torch.nn.Module]:
    """
    Patch multiple models for MPS compatibility

    Args:
        models: List of PyTorch models to patch

    Returns:
        List of patched models
    """
    if inference_device.type != "mps":
        return models

    for model in models:
        patch_model_for_mps(
            model=model,
            inference_device=inference_device,
            inference_dtype=inference_dtype,
        )
    return models
