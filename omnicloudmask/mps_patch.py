from typing import Callable, Optional

import timm
import torch

# This is a patch to fix MPS contiguity issues in timm EdgeNeXt models
# It modifies the Conv2d forward method to ensure inputs are contiguous
# This is not needed for torch 2.4 but is required for 2.5, 2.6 and 2.7

# Store the original forward method globally to avoid class attribute issues
_original_conv2d_forward: Optional[Callable] = None


def needs_mps_fix() -> bool:
    """
    Check if MPS fix is needed by testing a timm EdgeNeXt model
    Returns True if the MPS contiguity issue occurs, False otherwise
    """
    try:
        # Only test if MPS is available
        if not torch.backends.mps.is_available():
            return False

        # Create the smallest EdgeNeXt model to minimize overhead
        model = timm.create_model("edgenext_xx_small", pretrained=False)
        model.eval()

        # Move to MPS device
        device = torch.device("mps")
        model = model.to(device)

        # Create a small test input
        test_input = torch.randn(1, 3, 64, 64, device=device)

        # Try to run inference - this should fail with the MPS contiguity issue
        with torch.no_grad():
            _ = model(test_input)

        return False  # If we get here, no fix needed

    except RuntimeError as e:
        if "view size is not compatible" in str(e):
            return True  # This is exactly the error we need to fix
        else:
            # Some other error - re-raise it as it's unexpected
            raise e
    except Exception as e:
        raise Exception(
            f"An unexpected error occurred while checking MPS fix. {e}"
        ) from None


def apply_mps_fix():
    """Apply minimal MPS compatibility fix"""
    global _original_conv2d_forward

    if not needs_mps_fix():
        return

    if _original_conv2d_forward is None:
        _original_conv2d_forward = torch.nn.Conv2d.forward

        def mps_safe_conv_forward(self, input):
            if input.device.type == "mps" and not input.is_contiguous():
                input = input.contiguous()

            assert _original_conv2d_forward is not None
            return _original_conv2d_forward(self, input)

        torch.nn.Conv2d.forward = mps_safe_conv_forward


def remove_mps_fix():
    """Remove MPS compatibility fix"""
    global _original_conv2d_forward

    if _original_conv2d_forward is not None:
        torch.nn.Conv2d.forward = _original_conv2d_forward
        _original_conv2d_forward = None
