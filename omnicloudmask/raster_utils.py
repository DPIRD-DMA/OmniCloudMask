from pathlib import Path
from typing import Optional

import numpy as np
import rasterio as rio
from rasterio.profiles import Profile

from .constants import CLASS_NAMES, CLOUD_CLASSES
from .model_utils import channel_norm


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

    #  If the entire patch is nodata, return None
    if np.all(patch == no_data_value):
        return None, None

    # If patch edges include nodata, shift inward until nodata is found on edge
    # Only perform this if no_data_value is in the patch
    if no_data_value is not None and np.any(patch == no_data_value):
        # don't move outside the bounds of the input array
        max_bottom, max_right = input_array.shape[1:3]

        # Used to avoid back-and-forth shifting
        moved_vert = False
        # If nodata is on the top edge, move down until it's not
        if bottom < max_bottom and np.all(patch[:, 0, :] == no_data_value):
            while bottom < max_bottom and np.all(patch[:, 0, :] == no_data_value):
                top += 1
                bottom += 1
                patch = input_array[:, top:bottom, left:right]
            moved_vert = True

        # If we didn't move down, see if the bottom edge contains nodata and move up
        if not moved_vert and top > 0 and np.all(patch[:, -1, :] == no_data_value):
            while top > 0 and np.all(patch[:, -1, :] == no_data_value):
                top -= 1
                bottom -= 1
                patch = input_array[:, top:bottom, left:right]

        #  Same logic for left and right edges
        moved_horiz = False

        if right < max_right and np.all(patch[:, :, 0] == no_data_value):
            while right < max_right and np.all(patch[:, :, 0] == no_data_value):
                left += 1
                right += 1
                patch = input_array[:, top:bottom, left:right]
            moved_horiz = True

        if not moved_horiz and left > 0 and np.all(patch[:, :, -1] == no_data_value):
            while left > 0 and np.all(patch[:, :, -1] == no_data_value):
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
    all bands at a pixel location must be equal to no_data_value,
    apply this mask to the prediction tracker,
    then return the masked prediction tracker and the mask."""
    assert scene.ndim == 3, "Scene must have 3 dimensions"
    assert pred_tracker_np.ndim == 3, "Prediction tracker must have 3 dimensions"
    assert scene.shape[1:] == pred_tracker_np.shape[1:], (
        "Scene and prediction tracker must have the same shape"
    )
    # if all bands at a single pixel are no_data_value,
    # then it is considered no data
    mask = (~np.all(scene == no_data_value, axis=0)).astype(np.uint8)
    pred_tracker_np *= mask
    return pred_tracker_np, mask


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


def compute_class_stats(
    pred: np.ndarray, nodata_mask: Optional[np.ndarray] = None
) -> dict[str, str]:
    """Calculate the percentage of valid pixels in each class."""
    values = pred[0] if nodata_mask is None else pred[0][nodata_mask.astype(bool)]
    counts = np.bincount(values.ravel(), minlength=len(CLASS_NAMES))
    total = counts.sum()
    pct = counts / total * 100 if total else np.zeros(len(counts))

    pct_by_name = {name: pct[i] for i, name in CLASS_NAMES.items()}

    tags = {f"OCM_{name}_PCT": f"{p:.2f}" for name, p in pct_by_name.items()}
    tags["OCM_CLOUD_PCT"] = f"{sum(pct_by_name[n] for n in CLOUD_CLASSES):.2f}"
    tags["OCM_VALID_PIXELS"] = str(int(total))
    tags["OCM_CLASSES"] = ",".join(
        f"{i}={name.replace('_', ' ').title()}" for i, name in CLASS_NAMES.items()
    )
    return tags


def save_prediction(
    output_path: Path,
    export_profile: Profile,
    pred_tracker_np: np.ndarray,
    nodata_mask: Optional[np.ndarray],
    class_pred: Optional[np.ndarray] = None,
    write_class_stats: bool = True,
) -> None:
    """Save the prediction tracker to a raster file,
    optionally also saves the nodata mask if not None.
    If write_class_stats, class statistics tags are written for classified outputs,
    or for confidence outputs when class_pred (the argmax of the confidence maps)
    is provided."""
    if not write_class_stats:
        class_pred = None
    elif class_pred is None and pred_tracker_np.shape[0] == 1:
        class_pred = pred_tracker_np
    with rio.open(output_path, "w", **export_profile) as dst:
        dst.write(pred_tracker_np)
        if nodata_mask is not None:
            dst.write_mask((nodata_mask * 255).astype("uint8"))
        if class_pred is not None:
            tags = compute_class_stats(class_pred, nodata_mask)
            dst.update_tags(**tags)
