"""Helpers for batch_prediction.ipynb: build a demo chip dataset and plot chips."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import planetary_computer
import pystac_client
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.windows import Window
from rasterio.windows import transform as window_transform

ITEM_ID = "S2C_MSIL2A_20260104T015411_R117_T51KYB_20260104T055910"

# Display colours for the four classes, clear is transparent
CLASS_COLOURS = np.array(
    [
        [0, 0, 0, 0],  # clear
        [1.0, 0.5, 0.05, 0.5],  # thick cloud
        [0.17, 0.63, 0.17, 0.5],  # thin cloud
        [0.84, 0.15, 0.16, 0.5],  # cloud shadow
    ]
)


def _download_scene() -> tuple[np.ndarray, dict]:
    """Download Red, Green, Blue and NIR (B8A) of a Sentinel-2 L2A scene at 20 m.
    Returns a (4, H, W) array and the 20 m rasterio profile."""
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    item = catalog.get_collection("sentinel-2-l2a").get_item(ITEM_ID)

    # B8A is natively 20 m, use its grid for every band
    with rio.open(item.assets["B8A"].href) as src:
        profile = src.profile
        shape = (src.height, src.width)

    def read_band(href: str) -> np.ndarray:
        with rio.open(href) as src:
            return src.read(1, out_shape=shape, resampling=Resampling.bilinear)

    hrefs = [item.assets[band].href for band in ["B04", "B03", "B02", "B8A"]]
    with ThreadPoolExecutor() as executor:
        bands = list(executor.map(read_band, hrefs))
    return np.stack(bands), profile


def load_chip_dataset(
    data_dir: Path, chip_size: int = 512, n_chips: int = 25
) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    """Build a demo dataset of chips cut from a Sentinel-2 scene, evenly sampled
    across the scene. Chips are saved as GeoTIFFs in data_dir/chips and reused on
    later runs.

    Returns:
        chips: (N, 3, chip_size, chip_size) Red, Green, NIR array
        rgb_chips: (N, 3, chip_size, chip_size) Red, Green, Blue array for display
        chip_paths: GeoTIFF path of each chip, bands Red, Green, NIR
    """
    chip_dir = data_dir / "chips"
    rgb_cache = data_dir / f"rgb_chips_{chip_size}_{n_chips}.npy"
    chip_paths = sorted(chip_dir.glob(f"chip_{chip_size}_*.tif"))

    if len(chip_paths) != n_chips or not rgb_cache.exists():
        print("Downloading Sentinel-2 scene and cutting chips...")
        _write_chips(chip_dir, rgb_cache, chip_size, n_chips)
        chip_paths = sorted(chip_dir.glob(f"chip_{chip_size}_*.tif"))

    chips = []
    for path in chip_paths:
        with rio.open(path) as src:
            chips.append(src.read())
    chips = np.stack(chips)
    rgb_chips = np.load(rgb_cache)
    print(f"Loaded {len(chips)} chips of {chip_size}x{chip_size} px")
    return chips, rgb_chips, chip_paths


def _write_chips(chip_dir: Path, rgb_cache: Path, chip_size: int, n_chips: int) -> None:
    scene, profile = _download_scene()
    red, green, blue, nir = scene

    rows, cols = scene.shape[1] // chip_size, scene.shape[2] // chip_size
    grid = [(r, c) for r in range(rows) for c in range(cols)]
    positions = [grid[i] for i in np.linspace(0, len(grid) - 1, n_chips, dtype=int)]

    chip_dir.mkdir(parents=True, exist_ok=True)
    rgb_chips = []
    for i, (r, c) in enumerate(positions):
        window = Window(c * chip_size, r * chip_size, chip_size, chip_size)
        rows_slice = slice(r * chip_size, (r + 1) * chip_size)
        cols_slice = slice(c * chip_size, (c + 1) * chip_size)

        chip = np.stack([red, green, nir])[:, rows_slice, cols_slice]
        rgb_chips.append(np.stack([red, green, blue])[:, rows_slice, cols_slice])

        chip_profile = profile.copy()
        chip_profile.update(
            driver="GTiff",
            count=3,
            height=chip_size,
            width=chip_size,
            dtype=chip.dtype,
            transform=window_transform(window, profile["transform"]),
            compress="deflate",
        )
        path = chip_dir / f"chip_{chip_size}_{i:03d}.tif"
        with rio.open(path, "w", **chip_profile) as dst:
            dst.write(chip)

    np.save(rgb_cache, np.stack(rgb_chips))


def show_chips(
    rgb: np.ndarray,
    masks: Optional[np.ndarray] = None,
    titles: Optional[list[str]] = None,
    ncols: int = 5,
    size: float = 2.2,
) -> None:
    """Plot chips in a grid, with non-clear mask classes overlaid if given."""
    if len(rgb) == 0:
        print("No chips to show")
        return

    step = max(1, rgb.shape[-1] // 128)  # downsample to keep figures small
    nrows = int(np.ceil(len(rgb) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * size, nrows * size), dpi=72)
    for i, ax in enumerate(np.atleast_1d(axes).flat):
        ax.axis("off")
        if i >= len(rgb):
            continue
        image = rgb[i, :, ::step, ::step] / 3000
        ax.imshow(np.clip(image, 0, 1).transpose(1, 2, 0))
        if masks is not None:
            ax.imshow(CLASS_COLOURS[masks[i, 0, ::step, ::step]])
        if titles is not None:
            ax.set_title(titles[i], fontsize=9)
    plt.tight_layout()
    plt.show()
