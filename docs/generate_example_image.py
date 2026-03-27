"""Generate docs/_static/example.png from a Planetary Computer Sentinel-2 crop."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import planetary_computer
import pystac_client
import rasterio
from matplotlib.patches import Patch
from rasterio.enums import Resampling
from rasterio.windows import Window
from scipy.ndimage import binary_dilation

from omnicloudmask import predict_from_array

ITEM_ID = "S2B_MSIL2A_20260313T020309_R017_T51KUV_20260313T054018"
CROP_SIZE = 1000
COL_OFF, ROW_OFF = 8500, 3500
OUT_PATH = Path(__file__).parent / "_static" / "example.png"


def read_band_crop(href: str, resolution: float = 10.0) -> np.ndarray:
    with rasterio.open(href) as src:
        scale = src.res[0] / resolution
        native_col = int(COL_OFF / scale)
        native_row = int(ROW_OFF / scale)
        native_size = int(CROP_SIZE / scale)
        window = Window(native_col, native_row, native_size, native_size)  # type: ignore
        return src.read(
            1,
            window=window,
            out_shape=(CROP_SIZE, CROP_SIZE),
            resampling=Resampling.nearest,
        )


def main() -> None:
    # 1. Fetch scene and download crop
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    item = catalog.get_collection("sentinel-2-l2a").get_item(ITEM_ID)
    item = planetary_computer.sign(item)
    print(f"Found: {item.id}  (cloud cover: {item.properties['eo:cloud_cover']:.1f}%)")

    bands = ["B04", "B03", "B02", "B8A"]
    with ThreadPoolExecutor() as executor:
        red, green, blue, nir = executor.map(
            read_band_crop, [item.assets[b].href for b in bands]
        )

    scene = np.stack([red, green, nir])  # (3, H, W) — Red, Green, NIR
    print(f"Scene shape: {scene.shape}")

    # 2. Run OmniCloudMask
    mask = predict_from_array(input_array=scene, inference_device="cpu")
    print(f"Mask shape: {mask.shape}")

    # 3. Generate and save figure
    rgb_raw = np.stack([red, green, blue]).transpose(1, 2, 0).astype(float)
    p2, p98 = np.percentile(rgb_raw, [2, 98])
    rgb = np.clip((rgb_raw - p2) / (p98 - p2), 0, 1)

    mask_sq = mask.squeeze()

    CLOUD = "#f1c40f"
    SHADOW = "#00e5ff"

    cloud_region = np.isin(mask_sq, [1, 2])
    shadow_region = mask_sq == 3
    any_mask = cloud_region | shadow_region

    overlay = np.zeros((*mask_sq.shape, 4), dtype=float)
    overlay[cloud_region] = (*matplotlib.colors.to_rgb(CLOUD), 0.4)
    overlay[shadow_region] = (*matplotlib.colors.to_rgb(SHADOW), 0.35)

    # Outer border only where masked meets clear (not cloud-shadow boundary)
    outer_border = binary_dilation(any_mask, iterations=1) & ~any_mask
    cloud_border = outer_border & binary_dilation(cloud_region, iterations=1)
    shadow_border = outer_border & binary_dilation(shadow_region, iterations=1)
    overlay[cloud_border] = (*matplotlib.colors.to_rgb(CLOUD), 1.0)
    overlay[shadow_border] = (*matplotlib.colors.to_rgb(SHADOW), 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("white")

    axes[0].imshow(rgb)
    axes[0].set_title("Input", fontsize=13, fontweight="bold", pad=8)
    axes[0].axis("off")

    axes[1].imshow(rgb)
    axes[1].imshow(overlay)
    axes[1].set_title("OmniCloudMask", fontsize=13, fontweight="bold", pad=8)
    axes[1].axis("off")

    legend_elements = [
        Patch(facecolor=CLOUD, label="Cloud"),
        Patch(facecolor=SHADOW, label="Cloud Shadow"),
    ]
    axes[1].legend(
        handles=legend_elements,
        loc="lower right",
        ncol=1,
        fontsize=10,
        frameon=True,
        facecolor="white",
        edgecolor="lightgrey",
        framealpha=0.9,
    )

    plt.subplots_adjust(wspace=0.05)
    plt.tight_layout()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUT_PATH, dpi=150, bbox_inches="tight", facecolor="white", pad_inches=0.15
    )
    plt.close(fig)
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
