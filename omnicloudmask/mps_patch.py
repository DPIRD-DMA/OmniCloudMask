import torch

# This is a patch to fix MPS contiguity issues with certain PyTorch models on Apple Silicon  # noqa: E501
# It replaces Conv2d layers with MPS-safe versions that ensure inputs are contiguous
# This addresses RuntimeError: "view size is not compatible with input tensor's size and stride"  # noqa: E501
# that occurs with some model architectures (like timm EdgeNeXt) on MPS devices


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
