from pathlib import Path

import numpy as np
import rasterio as rio

from omnicloudmask.cloud_mask import (
    predict_from_array,
    predict_from_load_func,
)
from omnicloudmask.data_loaders import (
    load_ls8,
    load_multiband,
    load_s2,
)


def test_predict_from_array_basic():
    # Create some sample data
    data = np.random.rand(3, 200, 200)
    # Call the function
    result = predict_from_array(data, patch_size=100, patch_overlap=50)
    # Check the result
    assert result.shape == (1, 200, 200), "Unexpected shape for result"
    # make sure we dont have values outside of 0,1,2,3s
    assert np.all(
        np.isin(np.unique(result), [0, 1, 2, 3])
    ), "Unexpected values in result"


def test_predict_from_array_with_confidence():
    # Create some sample data
    data = np.random.rand(3, 200, 200)
    # Call the function
    result = predict_from_array(
        data, patch_size=100, patch_overlap=50, export_confidence=True
    )
    # make sure all values are between 0 and 1
    assert np.all(
        np.logical_and(result >= 0, result <= 1)
    ), "Unexpected values in confidence"
    # Check the result
    assert result.shape == (4, 200, 200), "Unexpected shape for result"


def test_predict_from_array_with_mask():
    # Create some sample data
    data = np.random.rand(3, 200, 200)
    mask_data = (np.random.rand(1, 200, 200) > 0.5).astype(np.uint8)
    data = data * mask_data
    # Call the function
    result = predict_from_array(
        data, patch_size=100, patch_overlap=50, apply_no_data_mask=True, no_data_value=0
    )
    # make sure all mask values are 0
    assert np.all(result[mask_data == 0] == 0), "Unexpected values in mask"
    # make sure all values are between 0 and 1
    assert result.shape == (1, 200, 200), "Unexpected shape for result"


def test_predict_from_load_func_s2():
    s2_l2_path = (
        Path(__file__).parent.resolve()
        / "test data/S2A_MSIL2A_20170725T142751_N9999_R053_T19GBQ_20240410T040247.SAFE"
    )
    # Call the function
    result_paths = predict_from_load_func(
        scene_paths=[s2_l2_path],
        load_func=load_s2,
        patch_size=1000,
        patch_overlap=10,
    )
    assert len(result_paths) == 1, "Unexpected number of results"
    assert result_paths[0].exists(), "Result file not created"
    pred_array = rio.open(result_paths[0]).read()
    # Check the result
    assert pred_array.shape == (1, 10980, 10980), "Unexpected shape for result"
    # make sure we dont have values outside of 0,1,2,3s
    assert np.all(
        np.isin(np.unique(pred_array), [0, 1, 2, 3])
    ), "Unexpected values in result"
