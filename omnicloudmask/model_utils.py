from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch


def load_model(
    model_path: Union[Path, str],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.nn.Module:
    """Load a PyTorch model from a file and move it to the specified device and dtype."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    try:
        model = torch.load(model_path, map_location="cpu")
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")

    model.eval()
    return model.to(dtype).to(device)


def get_torch_dtype(dtype: Union[torch.dtype, str]) -> torch.dtype:
    """Return a torch.dtype from a string or torch.dtype."""
    if isinstance(dtype, str):
        dtype_mapping = {
            "float16": torch.float16,
            "half": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "float": torch.float32,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        try:
            return dtype_mapping[dtype]
        except KeyError:
            raise ValueError(
                f"Invalid dtype: {dtype}. Must be one of {list(dtype_mapping.keys())}"
            )
    elif isinstance(dtype, torch.dtype):
        return dtype
    else:
        raise TypeError(
            f"Expected dtype to be a str or torch.dtype, but got {type(dtype)}"
        )


def create_gradient_mask(
    patch_size: int, patch_overlap: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Create a gradient mask for a given patch size and overlap."""
    if patch_overlap > 0:
        if patch_overlap * 2 > patch_size:
            patch_overlap = patch_size // 2

        gradient_strength = 1
        gradient = (
            torch.ones((patch_size, patch_size), dtype=torch.int, device=device)
            * patch_overlap
        )
        gradient[:, :patch_overlap] = torch.tile(
            torch.arange(1, patch_overlap + 1),
            (patch_size, 1),
        )
        gradient[:, -patch_overlap:] = torch.tile(
            torch.arange(patch_overlap, 0, -1),
            (patch_size, 1),
        )
        gradient = gradient / patch_overlap
        rotated_gradient = torch.rot90(gradient)
        combined_gradient = rotated_gradient * gradient

        combined_gradient = (combined_gradient * gradient_strength) + (
            1 - gradient_strength
        )
    else:
        combined_gradient = torch.ones(
            (patch_size, patch_size), dtype=torch.int, device=device
        )
    return combined_gradient.to(dtype)


def channel_norm(patch: np.ndarray, nodata_value: Optional[int] = 0) -> np.ndarray:
    """Normalize each band of the input array by subtracting the nonzero mean and dividing
    by the nonzero standard deviation then fill nodata values with 0."""
    out_array = np.zeros(patch.shape).astype(np.float32)
    for id, band in enumerate(patch):
        # Mask for non-zero values
        mask = band != nodata_value
        # Check if there are any non-zero values
        if np.any(mask):
            mean = band[mask].mean()
            std = band[mask].std()
            if std == 0:
                std = 1  # Prevent division by zero
            # Normalize only non-zero values
            out_array[id][mask] = (band[mask] - mean) / std
        else:
            continue
        # Fill original nodata values with 0
        out_array[id][~mask] = 0
    return out_array


def store_results(
    pred_batch: torch.Tensor,
    index_batch: list[tuple],
    pred_tracker: torch.Tensor,
    gradient: torch.Tensor,
    grad_tracker: Optional[torch.Tensor] = None,
) -> None:
    """Store the results of the model inference in the pred_tracker and grad_tracker tensors."""
    # Store the predictions in the pred_tracker tensor
    pred_batch *= gradient[None, None, :, :]

    for pred, index in zip(pred_batch.to(pred_tracker.device), index_batch):
        pred_tracker[:, index[0] : index[1], index[2] : index[3]] += pred
        if grad_tracker is not None:
            grad_tracker[index[0] : index[1], index[2] : index[3]] += gradient.to(
                grad_tracker.device
            )


def inference_and_store(
    models: list[torch.nn.Module],
    patch_batch: torch.Tensor,
    index_batch: list[tuple],
    pred_tracker: torch.Tensor,
    gradient: torch.Tensor,
    grad_tracker: Optional[torch.Tensor] = None,
) -> None:
    """Perform inference on the patch_batch and store the results in the pred_tracker and grad_tracker tensors."""
    # pre-initialize the all_preds tensor to store the predictions from each model
    all_preds = torch.zeros(
        len(models),
        patch_batch.shape[0],
        pred_tracker.shape[0],
        patch_batch.shape[2],
        patch_batch.shape[3],
        device=patch_batch.device,
        dtype=patch_batch.dtype,
    )
    for index, model in enumerate(models):
        with torch.no_grad():
            all_preds[index] = model(patch_batch)

    mean_preds = all_preds.mean(dim=0)

    store_results(
        pred_batch=mean_preds,
        index_batch=index_batch,
        pred_tracker=pred_tracker,
        gradient=gradient,
        grad_tracker=grad_tracker,
    )


def default_device() -> torch.device:
    """Return the default device for model inference"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
