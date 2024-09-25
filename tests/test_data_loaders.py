from pathlib import Path

import numpy as np

from omnicloudmask.data_loaders import (
    load_ls8,
    load_multiband,
    load_s2,
)

s2_l2_path = (
    Path(__file__).parent.resolve()
    / "test data/S2A_MSIL2A_20170725T142751_N9999_R053_T19GBQ_20240410T040247.SAFE"
)

s2_l1_path = (
    Path(__file__).parent.resolve()
    / "test data/S2B_MSIL1C_20180302T150259_N0206_R125_T22WES_20180302T183800.SAFE"
)

ls_path = Path(__file__).parent.resolve() / "test data/LC81960302014022LGN00"

multiband_path = Path(__file__).parent.resolve() / "test data/maxar.tif"


def test_load_s2_basic():
    bands, profile = load_s2(
        input_path=s2_l2_path, resolution=10.0, required_bands=["B04", "B03", "B8A"]
    )
    assert isinstance(bands, np.ndarray)
    assert bands.shape == (3, 10980, 10980)
    assert profile["transform"][0] == 10.0


def test_load_s2_11m():
    bands, profile = load_s2(
        input_path=s2_l2_path, resolution=11.0, required_bands=["B04", "B03", "B8A"]
    )
    assert isinstance(bands, np.ndarray)
    assert profile["transform"][0] == 11.0


def test_load_s2_20m():
    bands, profile = load_s2(
        input_path=s2_l2_path, resolution=20.0, required_bands=["B04", "B03", "B8A"]
    )
    assert isinstance(bands, np.ndarray)
    assert bands.shape == (3, 10980 / 2, 10980 / 2)
    assert profile["transform"][0] == 20.0


def test_load_s2_l1():
    bands, profile = load_s2(
        input_path=s2_l1_path, resolution=11.0, required_bands=["B04", "B03", "B8A"]
    )
    assert isinstance(bands, np.ndarray)
    assert profile["transform"][0] == 11.0


def test_load_multiband():
    bands, profile = load_multiband(
        multiband_path, resample_res=10, band_order=[1, 2, 4]
    )
    assert isinstance(bands, np.ndarray)
    assert bands.shape[0] == 3


def test_ls8():
    bands, profile = load_ls8(ls_path, required_bands=["B4", "B3", "B2"])
    assert isinstance(bands, np.ndarray)
    assert bands.shape[0] == 3
    assert profile["transform"][0] == 30.0
