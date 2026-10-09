import torch

# This module contains performance optimizations for
# PyTorch operations across different devices


def pairwise_argmax(
    tensor: torch.Tensor, dim: int = 0, keepdim: bool = True
) -> torch.Tensor:
    """Argmax using pairwise comparisons - optimized for CPU and MPS devices.

    Args:
        tensor: Input tensor, e.g. (C, H, W) or (B, C, H, W)
        dim: Class dimension to reduce (default: 0)
        keepdim: Whether to keep the reduced dimension (default: True)

    Returns:
        Tensor with dtype int64 containing argmax indices, with the reduced
        dimension kept as size 1 (if keepdim=True) or removed (if keepdim=False)
    """
    dim = dim % tensor.ndim

    # Pairwise comparison approach
    # Compare classes 0 vs 1 (use >= to prefer lower index on ties, matching argmax)
    mask_01 = tensor.select(dim, 0) >= tensor.select(dim, 1)
    max_01 = torch.where(mask_01, tensor.select(dim, 0), tensor.select(dim, 1))
    idx_01 = torch.where(
        mask_01,
        torch.zeros_like(mask_01, dtype=torch.int64),
        torch.ones_like(mask_01, dtype=torch.int64),
    )

    # Compare result vs remaining classes (use >= to prefer lower index on ties)
    for i in range(2, tensor.size(dim)):
        class_i = tensor.select(dim, i)
        mask_final = max_01 >= class_i
        idx_01 = torch.where(
            mask_final, idx_01, torch.full_like(idx_01, i, dtype=torch.int64)
        )
        max_01 = torch.where(mask_final, max_01, class_i)

    if keepdim:
        return idx_01.unsqueeze(dim)
    else:
        return idx_01


def optimized_argmax(
    tensor: torch.Tensor, dim: int = 0, keepdim: bool = True
) -> torch.Tensor:
    """Device-aware optimized argmax that chooses the best implementation per device.

    Uses pairwise comparison on CPU and MPS,
    falls back to torch.argmax on CUDA (where the standard implementation is faster).

    Args:
        tensor: Input tensor, e.g. (C, H, W) or (B, C, H, W)
        dim: Dimension to reduce (default: 0)
        keepdim: Whether to keep the reduced dimension (default: True)

    Returns:
        Tensor with argmax indices along the specified dimension
    """
    # Only use pairwise version for keepdim=True on CPU/MPS
    if keepdim and tensor.device.type in ("cpu", "mps"):
        return pairwise_argmax(tensor, dim=dim)

    # Use standard torch.argmax for CUDA or other configurations
    return torch.argmax(tensor, dim=dim, keepdim=keepdim)
