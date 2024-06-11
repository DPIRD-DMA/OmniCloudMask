import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import rasterio as rio
from rasterio.profiles import Profile


def load_s2(
    input_path: Union[Path, str],
    resolution: int = 10,
    required_bands: list[str] = ["B04", "B03", "B8A"],
) -> tuple[np.ndarray, Profile]:
    """Load a Sentinel-2 image from a SAFE directory, either L1C or L2A."""
    if resolution not in [10, 20]:
        raise ValueError("Resolution must be either 10 or 20")
    folder_name = Path(input_path).name
    processing_level = folder_name.split("_")[1][3:6]

    if processing_level not in ["L1C", "L2A"]:
        raise ValueError(
            f"Processing level {processing_level} not recognized, expected L1C or L2A"
        )
    if processing_level == "L1C":
        return load_s2_l1c(input_path, resolution, required_bands)
    else:
        return load_s2_l2a(input_path, resolution, required_bands)


def load_s2_l2a(
    input_path: Union[Path, str],
    resolution: int = 10,
    required_bands: list[str] = ["B04", "B03", "B8A"],
) -> tuple[np.ndarray, Profile]:
    """Load a Sentinel-2 L2A image from a SAFE directory"""
    if resolution not in [10, 20]:
        raise ValueError("Resolution must be either 10 or 20")

    input_path = Path(input_path)

    if resolution == 10:
        resolution_list = [10, 10, 20]
    else:
        resolution_list = [20, 20, 20]

    band_files = {}
    for band_name, native_resolution in zip(required_bands, resolution_list):
        try:
            band = list(input_path.rglob(f"*{band_name}_{native_resolution}m.jp2"))[0]

        except IndexError:
            raise ValueError(
                f"Band {band_name} and resolution of {native_resolution} not found in {input_path}"
            )
        band_files[band_name] = band

    data = []
    profile = Profile()
    for band_name in required_bands:
        with rio.open(band_files[band_name]) as src:
            if not profile:
                profile = src.profile
            native_resolution = int(src.res[0])
            scale_factor = native_resolution / resolution
            if native_resolution == resolution:
                data.append(src.read(1))
            else:
                data.append(
                    src.read(
                        1,
                        out_shape=(
                            int(src.height * scale_factor),
                            int(src.width * scale_factor),
                        ),
                    )
                )

    data = np.array(data)
    return data, profile


def load_s2_l1c(
    input_path: Union[Path, str],
    resolution: int = 10,
    required_bands: list[str] = ["B04", "B03", "B8A"],
) -> tuple[np.ndarray, Profile]:
    """Load a Sentinel-2 L1C image from a SAFE directory"""
    if resolution not in [10, 20]:
        raise ValueError("Resolution must be either 10 or 20")

    input_path = Path(input_path)

    band_files = {}
    for band_name in required_bands:
        try:
            band = list(input_path.rglob(f"*IMG_DATA/*{band_name}.jp2"))[0]

        except IndexError:
            raise ValueError(f"Band {band_name} not found in {input_path}")
        band_files[band_name] = band

    data = []
    profile = Profile()
    for band_name in required_bands:
        with rio.open(band_files[band_name]) as src:
            if not profile:
                profile = src.profile
            native_resolution = int(src.res[0])
            scale_factor = native_resolution / resolution
            if native_resolution == resolution:
                data.append(src.read(1))
            else:
                data.append(
                    src.read(
                        1,
                        out_shape=(
                            int(src.height * scale_factor),
                            int(src.width * scale_factor),
                        ),
                    )
                )

    data = np.array(data)
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
