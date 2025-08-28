from functools import partial
from pathlib import Path

import numpy as np
import pytest
import rasterio as rio
import torch

from omnicloudmask.cloud_mask import (
    check_patch_size,
    predict_from_array,
    predict_from_load_func,
)
from omnicloudmask.data_loaders import (
    load_ls8,
    load_multiband,
    load_s2,
)
from omnicloudmask.download_models import get_models
from omnicloudmask.model_utils import load_model_from_weights

test_dir = Path(__file__).parent.resolve() / "test data"
s2_l1_path = (
    test_dir / "S2B_MSIL1C_20180302T150259_N0206_R125_T22WES_20180302T183800.SAFE"
)
s2_l2_path = (
    test_dir / "S2A_MSIL2A_20170725T142751_N9999_R053_T19GBQ_20240410T040247.SAFE"
)
ls_path = test_dir / "LC81960302014022LGN00"

maxar_path = test_dir / "maxar.tif"


def cleanup():
    OCM_outputs = list(test_dir.rglob("*OCM_v*.tif"))
    if len(OCM_outputs) > 3:
        raise ValueError(
            f"Something has gone wrong, found {len(OCM_outputs)} OCM outputs"
            f" in test data directory"
        )
    for file in OCM_outputs:
        try:
            file.unlink()
            print(f"Final cleanup - Deleted: {file}")
        except Exception as e:
            print(f"Final cleanup - Failed to delete {file}: {e}")


@pytest.fixture(scope="function", autouse=True)
def cleanup_ocm_outputs():
    # Setup: This code runs before each test
    cleanup()
    yield
    # Teardown: This code runs after each test
    cleanup()


@pytest.fixture(scope="session", autouse=True)
def final_cleanup(request):
    # This runs at the start of the session
    yield
    # This runs at the end of the session
    request.addfinalizer(cleanup)


def test_predict_from_array_basic():
    # Create some sample data
    data = np.random.rand(3, 200, 200)
    # Call the function
    result = predict_from_array(data, patch_size=100, patch_overlap=50)
    # Check the result
    assert result.shape == (1, 200, 200), "Unexpected shape for result"
    # make sure we dont have values outside of 0,1,2,3s
    assert np.all(np.isin(np.unique(result), [0, 1, 2, 3])), (
        "Unexpected values in result"
    )


def test_predict_from_array_basic_v1():
    # Create some sample data
    data = np.random.rand(3, 200, 200)
    # Call the function
    result = predict_from_array(
        data, patch_size=100, patch_overlap=50, model_version=1.0
    )
    # Check the result
    assert result.shape == (1, 200, 200), "Unexpected shape for result"
    # make sure we dont have values outside of 0,1,2,3s
    assert np.all(np.isin(np.unique(result), [0, 1, 2, 3])), (
        "Unexpected values in result"
    )


def test_predict_from_array_basic_v2():
    # Create some sample data
    data = np.random.rand(3, 200, 200)
    # Call the function
    result = predict_from_array(
        data, patch_size=100, patch_overlap=50, model_version=2.0
    )
    # Check the result
    assert result.shape == (1, 200, 200), "Unexpected shape for result"
    # make sure we dont have values outside of 0,1,2,3s
    assert np.all(np.isin(np.unique(result), [0, 1, 2, 3])), (
        "Unexpected values in result"
    )


def test_predict_from_array_cpu_mosaic():
    # Create some sample data
    data = np.random.rand(3, 200, 200)
    # Call the function
    result = predict_from_array(
        data, patch_size=100, patch_overlap=50, mosaic_device="cpu"
    )
    # Check the result
    assert result.shape == (1, 200, 200), "Unexpected shape for result"
    # make sure we dont have values outside of 0,1,2,3s
    assert np.all(np.isin(np.unique(result), [0, 1, 2, 3])), (
        "Unexpected values in result"
    )


def test_predict_from_array_custom_model():
    # Create some sample data
    data = np.random.rand(3, 200, 200)
    model_detail = get_models()[0]
    model = load_model_from_weights(
        model_name=model_detail["timm_model_name"],
        weights_path=model_detail["Path"],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    # Call the function
    result = predict_from_array(
        data,
        patch_size=100,
        patch_overlap=50,
        mosaic_device="cpu",
        inference_device="cpu",
        custom_models=model,
    )
    # Check the result
    assert result.shape == (1, 200, 200), "Unexpected shape for result"
    # make sure we dont have values outside of 0,1,2,3s
    assert np.all(np.isin(np.unique(result), [0, 1, 2, 3])), (
        "Unexpected values in result"
    )


def test_predict_from_array_with_confidence():
    # Create some sample data
    data = np.random.rand(3, 200, 200)
    # Call the function
    result = predict_from_array(
        data, patch_size=100, patch_overlap=50, export_confidence=True
    )
    # make sure all values are between 0 and 1
    assert np.all(np.logical_and(result >= 0, result <= 1)), (
        "Unexpected values in confidence"
    )
    # Check the result
    assert result.shape == (4, 200, 200), "Unexpected shape for result"


def test_predict_from_array_with_confidence_no_softmax():
    # Create some sample data
    data = np.random.rand(3, 200, 200)
    # Call the function
    result = predict_from_array(
        data,
        patch_size=100,
        patch_overlap=50,
        export_confidence=True,
        softmax_output=False,
    )
    # make sure some values are outside of 0 and 1
    assert result.min() < 0, f"Unexpected values in confidence, min: {result.min()}"
    assert result.max() > 1, f"Unexpected values in confidence, max: {result.max()}"
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


def test_predict_from_load_func_multiband():
    # Call the function
    load_multiband_maxar = partial(
        load_multiband, resample_res=10, band_order=[1, 2, 4]
    )
    # expect warning
    with pytest.warns(UserWarning):
        result_paths = predict_from_load_func(
            scene_paths=[maxar_path],
            load_func=load_multiband_maxar,
        )

    assert len(result_paths) == 1, "Unexpected number of results"
    assert result_paths[0].exists(), "Result file not created"
    pred_array = rio.open(result_paths[0]).read()
    # Check the result
    assert pred_array.shape == (
        1,
        2322,
        2322,
    ), f"Unexpected shape for result {pred_array.shape}"
    # make sure we dont have values outside of 0,1,2,3s
    assert np.all(np.isin(np.unique(pred_array), [0, 1, 2, 3])), (
        "Unexpected values in result"
    )


def test_predict_from_load_func_s2_L2():
    # Call the function
    result_paths = predict_from_load_func(
        scene_paths=[s2_l2_path],
        load_func=load_s2,
        inference_dtype="bf16",
        batch_size=2,
    )
    assert len(result_paths) == 1, "Unexpected number of results"
    assert result_paths[0].exists(), "Result file not created"
    pred_array = rio.open(result_paths[0]).read()
    # Check the result
    assert pred_array.shape == (1, 10980, 10980), "Unexpected shape for result"
    # make sure we dont have values outside of 0,1,2,3s
    assert np.all(np.isin(np.unique(pred_array), [0, 1, 2, 3])), (
        "Unexpected values in result"
    )
    expected_output_path = (
        test_dir.parent
        / "S2A_MSIL2A_20170725T142751_N9999_R053_T19GBQ_20240410T040247_expected_output.tif"  # noqa
    )
    assert expected_output_path.exists(), "Expected output file not found"
    expected_output_array = rio.open(expected_output_path).read()

    difference = np.abs(pred_array - expected_output_array).sum()
    # Check that is within 0.1% of the expected output
    assert difference < (10980 * 10980) * 0.01, (
        f"Unexpected difference between expected and actual output: {difference}"
    )


def test_predict_from_load_func_s2_L2_with_cpu_mosaic():
    # Call the function
    result_paths = predict_from_load_func(
        scene_paths=[s2_l2_path],
        load_func=load_s2,
        patch_size=1000,
        patch_overlap=10,
        mosaic_device="cpu",
    )
    assert len(result_paths) == 1, "Unexpected number of results"
    assert result_paths[0].exists(), "Result file not created"
    pred_array = rio.open(result_paths[0]).read()
    # Check the result
    assert pred_array.shape == (1, 10980, 10980), "Unexpected shape for result"
    # make sure we dont have values outside of 0,1,2,3s
    assert np.all(np.isin(np.unique(pred_array), [0, 1, 2, 3])), (
        "Unexpected values in result"
    )


def test_predict_from_load_func_s2_L2_and_L1():
    # Call the function
    result_paths = predict_from_load_func(
        scene_paths=[s2_l2_path, s2_l1_path],
        load_func=load_s2,
        patch_size=1000,
        patch_overlap=10,
    )
    assert len(result_paths) == 2, "Unexpected number of results"

    for result_path in result_paths:
        assert result_path.exists(), "Result file not created"
        pred_array = rio.open(result_path).read()
        # Check the result
        assert pred_array.shape == (1, 10980, 10980), "Unexpected shape for result"
        # make sure we dont have values outside of 0,1,2,3s
        assert np.all(np.isin(np.unique(pred_array), [0, 1, 2, 3])), (
            "Unexpected values in result"
        )


def test_predict_from_load_func_s2_L1():
    # Call the function
    result_paths = predict_from_load_func(
        scene_paths=[s2_l1_path],
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
    assert np.all(np.isin(np.unique(pred_array), [0, 1, 2, 3])), (
        "Unexpected values in result"
    )


def test_predict_from_load_func_s2_L1_small_patch():
    # Call the function
    result_paths = predict_from_load_func(
        scene_paths=[s2_l1_path],
        load_func=load_s2,
        patch_size=300,
        patch_overlap=10,
    )
    assert len(result_paths) == 1, "Unexpected number of results"
    assert result_paths[0].exists(), "Result file not created"
    pred_array = rio.open(result_paths[0]).read()
    # Check the result
    assert pred_array.shape == (1, 10980, 10980), "Unexpected shape for result"
    # make sure we dont have values outside of 0,1,2,3s
    assert np.all(np.isin(np.unique(pred_array), [0, 1, 2, 3])), (
        "Unexpected values in result"
    )


def test_predict_from_load_func_ls():
    # Call the function
    result_paths = predict_from_load_func(
        scene_paths=[ls_path],
        load_func=load_ls8,
    )
    assert len(result_paths) == 1, "Unexpected number of results"
    assert result_paths[0].exists(), "Result file not created"
    pred_array = rio.open(result_paths[0]).read()
    # Check the result
    assert pred_array.shape == (
        1,
        7811,
        7681,
    ), f"Unexpected shape for result {pred_array.shape}"
    # make sure we dont have values outside of 0,1,2,3s
    assert np.all(np.isin(np.unique(pred_array), [0, 1, 2, 3])), (
        "Unexpected values in result"
    )


def test_check_patch_size():
    patch_size = 300
    patch_overlap = 50
    no_data_value = 0
    channels = 3
    test_array = np.random.rand(channels, 10000, 10000)
    check_patch_size(
        input_array=test_array,
        no_data_value=no_data_value,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
    )


def test_check_patch_size_small_input():
    patch_size = 300
    patch_overlap = 50
    no_data_value = 0
    channels = 3
    test_array_size = 5
    test_array = np.random.rand(channels, test_array_size, test_array_size)
    with pytest.raises(ValueError):
        check_patch_size(
            input_array=test_array,
            no_data_value=no_data_value,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
        )


def test_patch_size_too_small():
    patch_size = 31
    patch_overlap = 15
    no_data_value = 0
    channels = 3
    test_array_size = 300
    test_array = np.random.rand(channels, test_array_size, test_array_size)
    with pytest.raises(ValueError):
        check_patch_size(
            input_array=test_array,
            no_data_value=no_data_value,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
        )


def test_warning_for_small_patch_size():
    patch_size = 40
    patch_overlap = 15
    no_data_value = 0
    channels = 3
    test_array_size = 300
    test_array = np.random.rand(channels, test_array_size, test_array_size)
    with pytest.warns(UserWarning):
        check_patch_size(
            input_array=test_array,
            no_data_value=no_data_value,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
        )


def test_check_patch_size_wrong_dim_input():
    patch_size = 300
    patch_overlap = 50
    no_data_value = 0
    channels = 3
    test_array_size = 10000
    test_array = np.random.rand(1, channels, test_array_size, test_array_size)
    with pytest.raises(ValueError):
        check_patch_size(
            input_array=test_array,
            no_data_value=no_data_value,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
        )


def test_check_patch_size_small_input_warning():
    patch_size = 300
    patch_overlap = 50
    no_data_value = 0
    channels = 3
    test_array_size = 49
    test_array = np.random.rand(channels, test_array_size, test_array_size)
    with pytest.warns(UserWarning):
        check_patch_size(
            input_array=test_array,
            no_data_value=no_data_value,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
        )


def test_check_patch_size_large_nodata_warning():
    patch_size = 600
    patch_overlap = 300
    no_data_value = 0
    channels = 3
    test_array_size = 1000
    test_array = np.random.rand(channels, test_array_size, test_array_size)
    test_array[:, :700, :700] = no_data_value
    with pytest.warns(UserWarning):
        new_patch_overlap, new_patch_size = check_patch_size(
            input_array=test_array,
            no_data_value=no_data_value,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
        )
    assert new_patch_overlap == 250, (
        f"Unexpected patch overlap, got {new_patch_overlap}"
    )
    assert new_patch_size == 500, f"Unexpected patch size, got {new_patch_size}"


def test_check_patch_size_too_large_patch_warning():
    patch_size = 600
    patch_overlap = 50
    no_data_value = 0
    channels = 3
    test_array_size = 500
    test_array = np.random.rand(channels, test_array_size, test_array_size)
    with pytest.warns(UserWarning):
        new_patch_overlap, new_patch_size = check_patch_size(
            input_array=test_array,
            no_data_value=no_data_value,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
        )
    assert new_patch_overlap == 50, f"Unexpected patch overlap, got {new_patch_overlap}"
    assert new_patch_size == 500, f"Unexpected patch size, got {new_patch_size}"


def test_check_patch_size_overlap_larger_than_patch_size():
    patch_size = 300
    patch_overlap = 500
    no_data_value = 0
    channels = 3
    test_array_size = 1000
    test_array = np.random.rand(channels, test_array_size, test_array_size)
    with pytest.raises(ValueError):
        check_patch_size(
            input_array=test_array,
            no_data_value=no_data_value,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
        )
