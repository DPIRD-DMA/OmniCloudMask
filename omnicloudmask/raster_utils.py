from pathlib import Path
from typing import Optional, Union

import numpy as np
import rasterio as rio
import torch
from classcounter import count_classes
from rasterio.profiles import Profile

from .constants import CLASS_COLOURS, CLASS_NAMES, CLOUD_CLASSES, class_label
from .model_utils import channel_norm, is_nodata, is_nodata_torch


def get_patch(
    input_array: np.ndarray,
    index: tuple,
    no_data_value: Optional[int | float] = 0,
) -> tuple[Optional[np.ndarray], Optional[tuple[int, int, int, int]]]:
    """Extract a patch from a 3D array and normalize it. If the patch is
    entirely nodata, return None. If the patch contains nodata,
    try to move patches to reduce nodata regions in patches."""

    assert input_array.ndim == 3, "Input array must have 3 dimensions"

    top, bottom, left, right = index
    patch = input_array[:, top:bottom, left:right]

    patch_nodata = is_nodata(patch, no_data_value)

    #  If the entire patch is nodata, return None
    if np.all(patch_nodata):
        return None, None

    # If patch edges include nodata, shift inward until nodata is found on edge
    # Only perform this if there is nodata in the patch
    if np.any(patch_nodata):
        # don't move outside the bounds of the input array
        max_bottom, max_right = input_array.shape[1:3]

        # Used to avoid back-and-forth shifting
        moved_vert = False
        # If nodata is on the top edge, move down until it's not
        if bottom < max_bottom and np.all(is_nodata(patch[:, 0, :], no_data_value)):
            while bottom < max_bottom and np.all(
                is_nodata(patch[:, 0, :], no_data_value)
            ):
                top += 1
                bottom += 1
                patch = input_array[:, top:bottom, left:right]
            moved_vert = True

        # If we didn't move down, see if the bottom edge contains nodata and move up
        if (
            not moved_vert
            and top > 0
            and np.all(is_nodata(patch[:, -1, :], no_data_value))
        ):
            while top > 0 and np.all(is_nodata(patch[:, -1, :], no_data_value)):
                top -= 1
                bottom -= 1
                patch = input_array[:, top:bottom, left:right]

        #  Same logic for left and right edges
        moved_horiz = False

        if right < max_right and np.all(is_nodata(patch[:, :, 0], no_data_value)):
            while right < max_right and np.all(
                is_nodata(patch[:, :, 0], no_data_value)
            ):
                left += 1
                right += 1
                patch = input_array[:, top:bottom, left:right]
            moved_horiz = True

        if (
            not moved_horiz
            and left > 0
            and np.all(is_nodata(patch[:, :, -1], no_data_value))
        ):
            while left > 0 and np.all(is_nodata(patch[:, :, -1], no_data_value)):
                left -= 1
                right -= 1
                patch = input_array[:, top:bottom, left:right]

        patch = input_array[:, top:bottom, left:right]

    index = (top, top + patch.shape[1], left, left + patch.shape[2])
    return channel_norm(patch.astype(np.float32), no_data_value), index


def mask_prediction(
    scene: np.ndarray, pred_tracker_np: np.ndarray, no_data_value: int | float = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Create a no data mask from a raster scene,
    all bands at a pixel location must be no data (NaN or equal to no_data_value),
    apply this mask to the prediction tracker,
    then return the masked prediction tracker and the mask."""
    assert scene.ndim == 3, "Scene must have 3 dimensions"
    assert pred_tracker_np.ndim == 3, "Prediction tracker must have 3 dimensions"
    assert scene.shape[1:] == pred_tracker_np.shape[1:], (
        "Scene and prediction tracker must have the same shape"
    )
    mask = scene_valid_mask(scene, no_data_value).astype(np.uint8)
    pred_tracker_np *= mask
    return pred_tracker_np, mask


def scene_valid_mask(scene: np.ndarray, no_data_value: int | float = 0) -> np.ndarray:
    """Boolean (height, width) mask of valid pixels in a (bands, height, width)
    scene. A pixel is no data if all bands at that location are no data (NaN or
    equal to no_data_value)."""
    return ~np.all(is_nodata(scene, no_data_value), axis=0)


def class_stats_tags(
    class_pred: Union[np.ndarray, torch.Tensor],
    valid: Union[np.ndarray, torch.Tensor],
) -> dict[str, str]:
    """GDAL metadata tags with the percentage of valid pixels in each class.

    class_pred holds class indices and valid is a boolean mask of the same shape
    (or broadcastable to it). Tensors are counted on their device."""
    backend = "torch" if isinstance(class_pred, torch.Tensor) else "numpy"
    counts = count_classes(class_pred, names=CLASS_NAMES, mask=valid, backend=backend)
    total = sum(counts.values())

    def pct(count: float) -> str:
        return f"{count / total * 100 if total else 0:.2f}"

    tags = {f"OCM_{name}_PCT": pct(counts[name]) for name in CLASS_NAMES.values()}
    tags["OCM_CLOUD_PCT"] = pct(sum(counts[name] for name in CLOUD_CLASSES))
    tags["OCM_VALID_PIXELS"] = str(int(total))
    # unlike the class percentages, this is a percentage of all pixels
    pixels = valid.numel() if isinstance(valid, torch.Tensor) else valid.size
    tags["OCM_NODATA_PCT"] = f"{(pixels - total) / pixels * 100 if pixels else 0:.2f}"
    tags["OCM_CLASSES"] = ",".join(
        f"{i}={class_label(name)}" for i, name in CLASS_NAMES.items()
    )
    return tags


def batch_nodata_mask(
    batch: torch.Tensor, no_data_value: int | float = 0
) -> torch.Tensor:
    """Create a (B, 1, H, W) boolean valid-data mask for a (B, C, H, W) batch,
    a pixel is no data if all bands at that location are no data (NaN or equal to
    no_data_value)."""
    assert batch.ndim == 4, "Batch must have 4 dimensions"
    return ~torch.all(is_nodata_torch(batch, no_data_value), dim=1, keepdim=True)


def make_patch_indexes(
    array_width: int,
    array_height: int,
    patch_size: int = 1000,
    patch_overlap: int = 300,
) -> list[tuple[int, int, int, int]]:
    """Create a list of patch indexes for a given shape and patch size."""
    assert patch_size > patch_overlap, "Patch size must be greater than patch overlap"
    assert patch_overlap >= 0, "Patch overlap must be greater than or equal to 0"
    assert patch_size > 0, "Patch size must be greater than 0"
    assert patch_size <= array_width, (
        "Patch size must be less than or equal to array width"
    )
    assert patch_size <= array_height, (
        "Patch size must be less than or equal to array height"
    )

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
    output_path: Path,
    export_profile: Profile,
    pred_tracker_np: np.ndarray,
    nodata_mask: Optional[np.ndarray],
    tags: Optional[dict[str, str]] = None,
    standard_classes: bool = False,
) -> None:
    """Save the prediction tracker to a raster file, optionally also saving the
    nodata mask if not None and GDAL metadata tags if given. If standard_classes,
    the predictions use the OmniCloudMask classes, so masks get a colour table and
    confidence maps get a band description per class."""
    with rio.open(output_path, "w", **export_profile) as dst:
        dst.write(pred_tracker_np)
        if nodata_mask is not None:
            dst.write_mask((nodata_mask * 255).astype("uint8"))
        if tags:
            dst.update_tags(**tags)
        if standard_classes:
            if pred_tracker_np.shape[0] == 1 and pred_tracker_np.dtype == np.uint8:
                dst.write_colormap(1, CLASS_COLOURS)
                dst.set_band_description(
                    1,
                    "OCM class: "
                    + ",".join(f"{i}={class_label(n)}" for i, n in CLASS_NAMES.items()),
                )
            elif pred_tracker_np.shape[0] == len(CLASS_NAMES):
                for i, name in CLASS_NAMES.items():
                    dst.set_band_description(i + 1, class_label(name))
