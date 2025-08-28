from typing import Union
from unittest.mock import patch

import numpy as np
import pytest
import torch

from omnicloudmask.download_models import get_models
from omnicloudmask.model_utils import (
    channel_norm,
    create_gradient_mask,
    default_device,
    get_torch_dtype,
    # inference_and_store,
    load_model_from_weights,
    store_results,
)


@pytest.mark.parametrize(
    "input_dtype, expected_dtype",
    [
        ("float16", torch.float16),
        ("half", torch.float16),
        ("fp16", torch.float16),
        ("float32", torch.float32),
        ("float", torch.float32),
        ("bfloat16", torch.bfloat16),
        ("bf16", torch.bfloat16),
        ("FLOAT32", torch.float32),
        ("FloAt16", torch.float16),
        (torch.float16, torch.float16),
        (torch.float32, torch.float32),
        (torch.bfloat16, torch.bfloat16),
    ],
)
def test_get_torch_dtype_valid_inputs(
    input_dtype: Union[str, torch.dtype], expected_dtype: torch.dtype
):
    result = get_torch_dtype(input_dtype)
    assert result == expected_dtype, f"Expected {expected_dtype}, but got {result}"


@pytest.mark.parametrize(
    "invalid_input",
    [
        "invalid_dtype",
        "float64",
        123,
        3.14,
        None,
        [],
        {},
    ],
)
def test_get_torch_dtype_invalid_inputs(invalid_input):
    with pytest.raises((ValueError, TypeError)):
        get_torch_dtype(invalid_input)


def test_get_torch_dtype_error_message():
    invalid_dtype = "invalid_dtype"
    with pytest.raises(ValueError) as exc_info:
        get_torch_dtype(invalid_dtype)

    expected_error_msg = f"Invalid dtype: {invalid_dtype}. Must be one of "
    assert expected_error_msg in str(exc_info.value)


def test_get_torch_dtype_type_error_message():
    invalid_type = 123
    with pytest.raises(TypeError) as exc_info:
        get_torch_dtype(invalid_type)  # type: ignore

    expected_error_msg = (
        f"Expected dtype to be a str or torch.dtype, but got {type(invalid_type)}"
    )
    assert expected_error_msg in str(exc_info.value)


@pytest.mark.parametrize(
    "patch_size, patch_overlap, expected_shape",
    [
        (100, 20, (100, 100)),
        (64, 16, (64, 64)),
        (32, 8, (32, 32)),
        (50, 0, (50, 50)),  # Test with no overlap
    ],
)
def test_create_gradient_mask_shape(patch_size, patch_overlap, expected_shape):
    device = torch.device("cpu")
    dtype = torch.float32
    mask = create_gradient_mask(patch_size, patch_overlap, device, dtype)
    assert mask.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {mask.shape}"
    )


@pytest.mark.parametrize(
    "patch_size, patch_overlap, dtype",
    [
        (100, 20, torch.float32),
        (64, 16, torch.float16),
        (32, 8, torch.int32),
    ],
)
def test_create_gradient_mask_dtype(patch_size, patch_overlap, dtype):
    device = torch.device("cpu")
    mask = create_gradient_mask(patch_size, patch_overlap, device, dtype)
    assert mask.dtype == dtype, f"Expected dtype {dtype}, got {mask.dtype}"


def test_create_gradient_mask_device():
    patch_size, patch_overlap = 64, 16
    dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask = create_gradient_mask(patch_size, patch_overlap, device, dtype)
    assert mask.device.type == device.type, (
        f"Expected device {device.type}, got {mask.device.type}"
    )


def test_create_gradient_mask_values():
    patch_size, patch_overlap = 100, 20
    device = torch.device("cpu")
    dtype = torch.float32
    mask = create_gradient_mask(patch_size, patch_overlap, device, dtype)

    # Check corners
    assert (
        mask[0, 0]
        == mask[-1, 0]
        == mask[0, -1]
        == mask[-1, -1]
        == pytest.approx(0.0025)
    )

    # Check center
    assert mask[50, 50] == pytest.approx(1.0)

    # Check edges
    assert (
        mask[0, 50]
        == mask[50, 0]
        == mask[-1, 50]
        == mask[50, -1]
        == pytest.approx(0.05)
    )


def test_create_gradient_mask_symmetry():
    patch_size, patch_overlap = 64, 16
    device = torch.device("cpu")
    dtype = torch.float32
    mask = create_gradient_mask(patch_size, patch_overlap, device, dtype)

    assert torch.allclose(mask, mask.flip(0)), "Mask should be symmetric vertically"
    assert torch.allclose(mask, mask.flip(1)), "Mask should be symmetric horizontally"


def test_create_gradient_mask_no_overlap():
    patch_size = 50
    patch_overlap = 0
    device = torch.device("cpu")
    dtype = torch.float32
    mask = create_gradient_mask(patch_size, patch_overlap, device, dtype)

    assert torch.all(mask == 1.0), "Mask should be all ones when there's no overlap"


def test_create_gradient_mask_large_overlap():
    patch_size = 100
    patch_overlap = 60  # Greater than patch_size // 2
    device = torch.device("cpu")
    dtype = torch.float32
    mask = create_gradient_mask(patch_size, patch_overlap, device, dtype)

    assert mask.shape == (100, 100), "Shape should still be correct with large overlap"
    assert torch.max(mask) <= 1.0 and torch.min(mask) > 0.0, (
        "Values should be between 0 and 1"
    )
    assert torch.allclose(mask, mask.flip(0)), "Mask should be symmetric vertically"
    assert torch.allclose(mask, mask.flip(1)), "Mask should be symmetric horizontally"


#
def test_channel_norm_basic():
    patch = np.array([[1, 2, 3, 0], [4, 5, 6, 0]])
    result = channel_norm(patch)

    expected = np.array(
        [[-1.22474487, 0.0, 1.22474487, 0.0], [-1.22474487, 0.0, 1.22474487, 0.0]]
    )

    np.testing.assert_array_almost_equal(result, expected)


def test_channel_norm_with_custom_nodata():
    patch = np.array([[1, 2, 3, -9999], [4, 5, 6, -9999]])
    result = channel_norm(patch, nodata_value=-9999)

    expected = np.array(
        [[-1.22474487, 0.0, 1.22474487, 0.0], [-1.22474487, 0.0, 1.22474487, 0.0]]
    )

    np.testing.assert_array_almost_equal(result, expected)


def test_channel_norm_all_nodata():
    patch = np.array([[0, 0, 0], [0, 0, 0]])
    result = channel_norm(patch)

    expected = np.zeros_like(patch, dtype=np.float32)

    np.testing.assert_array_equal(result, expected)


def test_channel_norm_single_value():
    patch = np.array([[5, 5, 5, 0], [5, 5, 5, 0]])
    result = channel_norm(patch)

    expected = np.array([[0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float32)

    np.testing.assert_array_equal(result, expected)


def test_channel_norm_3d_array():
    patch = np.array([[[1, 2, 3, 0], [4, 5, 6, 0]], [[7, 8, 9, 0], [10, 11, 12, 0]]])
    result = channel_norm(patch=patch, nodata_value=0)
    # print(result)
    expected = np.array(
        [
            [
                [-1.4638501, -0.8783101, -0.29277003, 0.0],
                [0.29277003, 0.8783101, 1.4638501, 0.0],
            ],
            [
                [-1.4638501, -0.8783101, -0.29277003, 0.0],
                [0.29277003, 0.8783101, 1.4638501, 0.0],
            ],
        ],
    )

    np.testing.assert_array_almost_equal(result, expected)


def test_channel_norm_float_input():
    patch = np.array([[1.5, 2.5, 3.5, 0.0], [4.5, 5.5, 6.5, 0.0]])
    result = channel_norm(patch)
    print(result)

    expected = np.array(
        [
            [
                -1.2247449,
                0.0,
                1.2247449,
                0.0,
            ],
            [
                -1.2247449,
                0.0,
                1.2247449,
                0.0,
            ],
        ]
    )

    np.testing.assert_array_almost_equal(result, expected)


def test_channel_norm_negative_values():
    patch = np.array([[-3, -2, -1, 0], [1, 2, 3, 0]])
    result = channel_norm(patch)

    expected = np.array(
        [
            [
                -1.2247449,
                0.0,
                1.2247449,
                0.0,
            ],
            [
                -1.2247449,
                0.0,
                1.2247449,
                0.0,
            ],
        ]
    )

    np.testing.assert_array_almost_equal(result, expected)


def test_channel_norm_output_type():
    patch = np.array([[1, 2, 3, 0], [4, 5, 6, 0]])
    result = channel_norm(patch)

    assert result.dtype == np.float32, f"Expected dtype float32, got {result.dtype}"


def test_channel_norm_all_zeros():
    patch = np.array([[0, 0, 0], [0, 0, 0]])
    result = channel_norm(patch)

    expected = np.zeros_like(patch, dtype=np.float32)

    np.testing.assert_array_equal(result, expected)


def test_default_device_cuda_available():
    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.backends.mps.is_available", return_value=False):
            device = default_device()
            assert device == torch.device("cuda"), (
                "Should return CUDA device when CUDA is available"
            )


def test_default_device_mps_available():
    with patch("torch.cuda.is_available", return_value=False):
        with patch("torch.backends.mps.is_available", return_value=True):
            device = default_device()
            assert device == torch.device("mps"), (
                "Should return MPS device when MPS is available and CUDA is not"
            )


def test_default_device_cpu_fallback():
    with patch("torch.cuda.is_available", return_value=False):
        with patch("torch.backends.mps.is_available", return_value=False):
            device = default_device()
            assert device == torch.device("cpu"), (
                "Should return CPU device when neither CUDA nor MPS is available"
            )


def test_default_device_cuda_priority():
    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.backends.mps.is_available", return_value=True):
            device = default_device()
            assert device == torch.device("cuda"), (
                "Should prioritize CUDA over MPS when both are available"
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_default_device_real_cuda():
    device = default_device()
    assert device == torch.device("cuda"), (
        "Should return CUDA device on a system with CUDA available"
    )


def test_default_device_real_cpu():
    with patch("torch.cuda.is_available", return_value=False):
        with patch("torch.backends.mps.is_available", return_value=False):
            device = default_device()
            assert device == torch.device("cpu"), (
                "Should return CPU device when no accelerator is available"
            )


#


@pytest.fixture
def setup_tensors():
    pred_batch = torch.tensor(
        [
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
                [[19.0, 20.0, 21.0], [22.0, 23.0, 24.0], [25.0, 26.0, 27.0]],
                [[28.0, 29.0, 30.0], [31.0, 32.0, 33.0], [34.0, 35.0, 36.0]],
            ],
            [
                [[37.0, 38.0, 39.0], [40.0, 41.0, 42.0], [43.0, 44.0, 45.0]],
                [[46.0, 47.0, 48.0], [49.0, 50.0, 51.0], [52.0, 53.0, 54.0]],
                [[55.0, 56.0, 57.0], [58.0, 59.0, 60.0], [61.0, 62.0, 63.0]],
                [[64.0, 65.0, 66.0], [67.0, 68.0, 69.0], [70.0, 71.0, 72.0]],
            ],
        ]
    )
    index_batch = [(0, 3, 0, 3), (1, 4, 1, 4)]
    pred_tracker = torch.zeros((4, 5, 5))
    gradient = torch.tensor([[0.5, 1.0, 0.5], [1.0, 0.5, 0.5], [1.0, 0.5, 0.5]])
    grad_tracker = torch.zeros((5, 5))
    return pred_batch, index_batch, pred_tracker, gradient, grad_tracker


def test_store_results_basic(setup_tensors):
    pred_batch, index_batch, pred_tracker, gradient, _ = setup_tensors
    store_results(pred_batch, index_batch, pred_tracker, gradient)

    expected = torch.tensor(
        [
            [
                [0.5, 2.0, 1.5, 0.0, 0.0],
                [4.0, 21.0, 41.0, 19.5, 0.0],
                [7.0, 44.0, 25.0, 21.0, 0.0],
                [0.0, 43.0, 22.0, 22.5, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [5.0, 11.0, 6.0, 0.0, 0.0],
                [13.0, 30.0, 54.5, 24.0, 0.0],
                [16.0, 57.5, 34.0, 25.5, 0.0],
                [0.0, 52.0, 26.5, 27.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [9.5, 20.0, 10.5, 0.0, 0.0],
                [22.0, 39.0, 68.0, 28.5, 0.0],
                [25.0, 71.0, 43.0, 30.0, 0.0],
                [0.0, 61.0, 31.0, 31.5, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [14.0, 29.0, 15.0, 0.0, 0.0],
                [31.0, 48.0, 81.5, 33.0, 0.0],
                [34.0, 84.5, 52.0, 34.5, 0.0],
                [0.0, 70.0, 35.5, 36.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        ]
    )

    assert torch.allclose(pred_tracker, expected), (
        f"Expected {expected}, got {pred_tracker}"
    )


def test_store_results_with_grad_tracker(setup_tensors):
    pred_batch, index_batch, pred_tracker, gradient, grad_tracker = setup_tensors
    store_results(pred_batch, index_batch, pred_tracker, gradient, grad_tracker)
    print(grad_tracker)
    expected_grad_tracker = torch.tensor(
        [
            [0.5, 1.0, 0.5, 0.0, 0.0],
            [1.0, 1.0, 1.5, 0.5, 0.0],
            [1.0, 1.5, 1.0, 0.5, 0.0],
            [0.0, 1.0, 0.5, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    assert torch.allclose(grad_tracker, expected_grad_tracker), (
        f"Expected {expected_grad_tracker}, got {grad_tracker}"
    )


def test_store_results_accumulation(setup_tensors):
    pred_batch, index_batch, pred_tracker, gradient, _ = setup_tensors

    # Call store_results twice
    store_results(pred_batch, index_batch, pred_tracker, gradient)
    store_results(pred_batch, index_batch, pred_tracker, gradient)

    expected = torch.tensor(
        [
            [
                [0.75, 4.00, 2.25, 0.00, 0.00],
                [8.00, 31.50, 80.50, 29.25, 0.00],
                [14.00, 86.00, 37.50, 31.50, 0.00],
                [0.00, 86.00, 33.00, 33.75, 0.00],
                [0.00, 0.00, 0.00, 0.00, 0.00],
            ],
            [
                [7.50, 22.00, 9.00, 0.00, 0.00],
                [26.00, 45.00, 105.25, 36.00, 0.00],
                [32.00, 110.75, 51.00, 38.25, 0.00],
                [0.00, 104.00, 39.75, 40.50, 0.00],
                [0.00, 0.00, 0.00, 0.00, 0.00],
            ],
            [
                [14.25, 40.00, 15.75, 0.00, 0.00],
                [44.00, 58.50, 130.00, 42.75, 0.00],
                [50.00, 135.50, 64.50, 45.00, 0.00],
                [0.00, 122.00, 46.50, 47.25, 0.00],
                [0.00, 0.00, 0.00, 0.00, 0.00],
            ],
            [
                [21.00, 58.00, 22.50, 0.00, 0.00],
                [62.00, 72.00, 154.75, 49.50, 0.00],
                [68.00, 160.25, 78.00, 51.75, 0.00],
                [0.00, 140.00, 53.25, 54.00, 0.00],
                [0.00, 0.00, 0.00, 0.00, 0.00],
            ],
        ]
    )

    assert torch.allclose(pred_tracker, expected), (
        f"Expected {expected}, got {pred_tracker}"
    )


def test_store_results_device_handling():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred_batch = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], device=device)
    print(pred_batch.shape)
    index_batch = [(0, 2, 0, 2)]
    pred_tracker = torch.zeros((1, 2, 2), device=device)
    gradient = torch.tensor([[1.0, 1.0], [1.0, 1.0]], device=device)

    store_results(pred_batch, index_batch, pred_tracker, gradient)

    expected = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], device=device)

    assert torch.allclose(pred_tracker, expected), (
        f"Expected {expected}, got {pred_tracker}"
    )
    assert pred_tracker.device.type == device.type, (
        f"Expected device type {device.type}, got {pred_tracker.device.type}"
    )
    if device.type == "cuda":
        assert pred_tracker.is_cuda, "Expected pred_tracker to be on CUDA device"
    elif device.type == "cpu":
        assert not pred_tracker.is_cuda, "Expected pred_tracker to be on CPU device"


def test_store_results_empty_batch():
    pred_batch = torch.tensor([])
    index_batch = []
    pred_tracker = torch.zeros((1, 4, 4))
    gradient = torch.tensor([[1.0, 1.0], [1.0, 1.0]])

    with pytest.raises(AssertionError):
        store_results(pred_batch, index_batch, pred_tracker, gradient)


def test_store_results_wrong_shapes():
    pred_batch = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    index_batch = [(0, 2, 0, 2)]
    pred_tracker = torch.zeros((1, 2, 2))
    gradient = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])  # Wrong shape

    with pytest.raises(AssertionError):
        store_results(pred_batch, index_batch, pred_tracker, gradient)


def test_load_model_from_weights():
    models = []
    for model_details in get_models():
        models.append(
            load_model_from_weights(
                model_name=model_details["timm_model_name"],
                weights_path=model_details["Path"],
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
        )
    assert len(models) == 2
    for model in models:
        assert isinstance(model, torch.nn.Module)


if __name__ == "__main__":
    pytest.main()
