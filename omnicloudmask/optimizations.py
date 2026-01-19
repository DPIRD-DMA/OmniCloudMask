import torch

# This module contains performance optimizations for
# PyTorch operations across different devices


def pairwise_argmax(
    tensor: torch.Tensor, dim: int = 0, keepdim: bool = True
) -> torch.Tensor:
    """Argmax using pairwise comparisons - optimized for CPU and MPS devices.

    Args:
        tensor: Input tensor of shape (C, H, W) where C is number of classes
        dim: Dimension to reduce (default: 0, currently only 0 is supported)
        keepdim: Whether to keep the reduced dimension (default: True)

    Returns:
        Tensor of shape (1, H, W) with dtype int64 containing argmax indices
        (if keepdim=True) or (H, W) if keepdim=False

    Raises:
        ValueError: If dim is not 0 (only dim=0 is currently supported)
    """
    if dim != 0:
        raise ValueError(
            f"pairwise_argmax only supports dim=0, got dim={dim}. "
            "Use torch.argmax for other dimensions."
        )

    # Pairwise comparison approach
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

    if keepdim:
        return idx_01.unsqueeze(0)
    else:
        return idx_01


def optimized_argmax(
    tensor: torch.Tensor, dim: int = 0, keepdim: bool = True
) -> torch.Tensor:
    """Device-aware optimized argmax that chooses the best implementation per device.

    Uses pairwise comparison on CPU and MPS,
    falls back to torch.argmax on CUDA (where the standard implementation is faster).

    Args:
        tensor: Input tensor of shape (C, H, W) where C is number of classes
        dim: Dimension to reduce (default: 0)
        keepdim: Whether to keep the reduced dimension (default: True)

    Returns:
        Tensor with argmax indices along the specified dimension
    """
    # Only use pairwise version for dim=0 and keepdim=True on CPU/MPS
    if dim == 0 and keepdim and tensor.device.type in ("cpu", "mps"):
        return pairwise_argmax(tensor)

    # Use standard torch.argmax for CUDA or other configurations
    return torch.argmax(tensor, dim=dim, keepdim=keepdim)
