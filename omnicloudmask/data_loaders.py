import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import rasterio as rio
from rasterio.profiles import Profile


def load_s2(
    input_path: Union[Path, str],
    resolution: float = 10.0,
    required_bands: list[str] = ["B04", "B03", "B8A"],
) -> tuple[np.ndarray, Profile]:
    """Load a Sentinel-2 (L1C or L2A) image from a SAFE folder containing the bands"""
    if not 10 <= resolution <= 50:
        raise ValueError("Resolution must be between 10 and 50")
    input_path = Path(input_path)
    processing_level = find_s2_processing_level(input_path)
    return open_s2_bands(input_path, processing_level, resolution, required_bands)


def find_s2_processing_level(
    input_path: Path,
) -> str:
    """Derive the processing level of a Sentinel-2 image from the folder name."""

    folder_name = Path(input_path).name
    processing_level = folder_name.split("_")[1][3:6]

    if processing_level not in ["L1C", "L2A"]:
        raise ValueError(
            f"Processing level {processing_level} not recognized, expected L1C or L2A"
        )
    return processing_level


def open_s2_bands(
    input_path: Path,
    processing_level: str,
    resolution: float,
    required_bands: list[str],
) -> tuple[np.ndarray, Profile]:
    bands = []
    for band_name in required_bands:
        if processing_level == "L1C":
            try:
                band = list(input_path.rglob(f"*IMG_DATA/*{band_name}.jp2"))[0]

            except IndexError:
                raise ValueError(f"Band {band_name} not found in {input_path}")
        else:
            try:
                for search_resolution in [10, 20, 60]:
                    band_paths = list(
                        input_path.rglob(f"*{band_name}_{search_resolution}m.jp2")
                    )
                    if band_paths:
                        band = band_paths[0]
                        break

            except IndexError:
                raise ValueError(f"Band {band_name} not found in {input_path}")
        with rio.open(band) as src:
            profile = src.profile
            native_resolution = int(src.res[0])
            scale_factor = native_resolution / resolution
            if native_resolution == resolution:
                bands.append(src.read(1))
            else:
                bands.append(
                    src.read(
                        1,
                        out_shape=(
                            int(src.height * scale_factor),
                            int(src.width * scale_factor),
                        ),
                    )
                )
    profile["transform"] = rio.transform.from_origin(  # type: ignore
        profile["transform"][2],
        profile["transform"][5],
        resolution,
        resolution,
    )
    data = np.array(bands)
    profile["height"] = data.shape[1]
    profile["width"] = data.shape[2]
    return data, profile


def load_multiband(
    input_path: Union[Path, str],
    resample_res: Optional[float] = None,
    band_order: Optional[list[int]] = None,
) -> tuple[np.ndarray, Profile]:
    """Load a multiband image and resample it to requested resolution."""
    if band_order is None:
        warnings.warn(
            "No band order provided, using default [1, 2, 3] (RGN)", UserWarning
        )
        band_order = [1, 2, 3]
    input_path = Path(input_path)

    with rio.open(input_path) as src:
        if resample_res:
            current_res = src.res
            desired_res = (resample_res, resample_res)
            scale_factor = (
                current_res[0] / desired_res[0],
                current_res[1] / desired_res[1],
            )
        else:
            scale_factor = (1, 1)

        data = src.read(
            band_order,
            out_shape=(
                len(band_order),
                int(src.height * scale_factor[0]),
                int(src.width * scale_factor[1]),
            ),
            resampling=rio.enums.Resampling.nearest,  # type: ignore
        )
        profile = src.profile

        return data, profile


def load_ls8(
    input_path: Union[Path, str], resolution: int = 30
) -> tuple[np.ndarray, Profile]:
    """Load a Landsat 8 image from a folder containing the bands"""
    if resolution != 30:
        raise ValueError("Resolution must be 30")

    input_path = Path(input_path)

    required_bands = ["B4", "B3", "B5"]

    band_files = {}
    for band_name in required_bands:
        try:
            band = list(input_path.rglob(f"*{band_name}.TIF"))[0]

        except IndexError:
            raise ValueError(f"Band {band_name} not found in {input_path}")
        band_files[band_name] = band

    data = []
    profile = Profile()
    for band_name in required_bands:
        with rio.open(band_files[band_name]) as src:
            if not profile:
                profile = src.profile
            data.append(src.read(1))

    data = np.array(data)
    return data, profile
