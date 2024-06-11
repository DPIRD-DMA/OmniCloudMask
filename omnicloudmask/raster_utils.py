from typing import Optional

import numpy as np
from .model_utils import channel_norm
import rasterio as rio
from rasterio.profiles import Profile
from pathlib import Path


def get_patch(
    input_array: np.ndarray,
    index: tuple,
    no_data_value: Optional[int] = 0,
) -> tuple[Optional[np.ndarray], tuple[int, int, int, int]]:
    """Extract a patch from a 3D array and normalize it. If the patch is entirely nodata, return None.
    If the patch contains nodata, try to move patches to reduce nodata regions in patches.
    """
    top, bottom, left, right = index
    patch = input_array[:, top:bottom, left:right].astype(np.float32)

    if patch.sum() == 0:
        return None, index

    if no_data_value is None:
        if np.all(patch == no_data_value):
            return None, index

    if np.any(patch == 0):
        max_bottom, max_right = input_array.shape[1:3]

        if np.any(patch[:, 0, :]) or np.any(patch[:, -1, :]):
            while not np.any(patch[:, 0, :]) and bottom < max_bottom:  # check top row
                patch = patch[:, 1:, :]
                top += 1
                bottom += 1

            while not np.any(patch[:, -1, :]) and top > 0:
                patch = patch[:, :-1, :]
                bottom -= 1
                top -= 1

        # Both sides are not zero-filled
        if np.any(patch[:, :, 0]) or np.any(patch[:, :, -1]):
            while not np.any(patch[:, :, 0]) and right < max_right:  # check left column
                patch = patch[:, :, 1:]
                left += 1
                right += 1

            while not np.any(patch[:, :, -1]) and left > 0:  # check right column
                patch = patch[:, :, :-1]
                right -= 1
                left -= 1
        patch = input_array[:, top:bottom, left:right].astype(np.float32)
        index = (top, bottom, left, right)
    return channel_norm(patch, no_data_value), index


def mask_prediction(
    scene: np.ndarray, pred_tracker_np: np.ndarray, no_data_value: int = 0
) -> np.ndarray:
    """Create a no data mask from a raster scene."""
    mask = np.all(scene != no_data_value, axis=0).astype(np.uint8)
    pred_tracker_np *= mask
    return pred_tracker_np


def make_patch_indexes(
    array_width: int,
    array_height: int,
    patch_size: int = 1000,
    patch_overlap: int = 300,
) -> list[tuple[int, int, int, int]]:
    """Create a list of patch indexes for a given shape and patch size."""

    stride = patch_size - patch_overlap

    max_bottom = array_height - patch_size
    max_right = array_width - patch_size

    patch_indexes = []
    for top in range(0, array_height, stride):
        if top > max_bottom:
            top = max_bottom
        bottom = top + patch_size
        for left in range(0, array_width, stride):
            if left > max_right:
                left = max_right
            right = left + patch_size
            patch_indexes.append((top, bottom, left, right))

    return patch_indexes


def save_prediction(
    output_path: Path, export_profile: Profile, pred_tracker_np: np.ndarray
) -> None:
    with rio.open(output_path, "w", **export_profile) as dst:
        dst.write(pred_tracker_np)
