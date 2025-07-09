import numpy as np
import pytest

from omnicloudmask.model_utils import channel_norm
from omnicloudmask.raster_utils import get_patch, make_patch_indexes, mask_prediction


def test_get_patch_within_bounds():
    input_array = np.random.rand(4, 100, 100)
    no_data_value = 0
    index = (0, 50, 0, 50)

    patch, returned_index = get_patch(
        input_array=input_array, index=index, no_data_value=no_data_value
    )

    assert patch is not None, "Patch should not be None for valid input"
    assert patch.shape == (4, 50, 50), f"Expected shape (4, 50, 50), got {patch.shape}"
    assert returned_index == index, f"Expected index {index}, got {returned_index}"


def test_get_patch_exceeding_bounds():
    input_array = np.random.rand(4, 100, 100)
    no_data_value = 0
    index = (0, 150, 0, 150)

    patch, returned_index = get_patch(
        input_array=input_array, index=index, no_data_value=no_data_value
    )

    assert patch is not None, "Patch should not be None even when exceeding bounds"
    assert patch.shape == (
        4,
        100,
        100,
    ), f"Expected shape (4, 100, 100), got {patch.shape}"
    assert returned_index == (
        0,
        100,
        0,
        100,
    ), f"Expected index (0, 100, 0, 100), got {returned_index}"


def test_get_patch_with_nodata():
    input_array = np.zeros((4, 100, 100))
    no_data_value = 0
    index = (0, 50, 0, 50)
    patch, returned_index = get_patch(
        input_array=input_array, index=index, no_data_value=no_data_value
    )
    assert patch is None, "Patch should be None when entirely nodata"
    assert returned_index is None, "Index should be None when entirely nodata"


def test_get_patch_wrong_dimensions():
    input_array = np.random.rand(1, 3, 100, 100)
    no_data_value = 0
    index = (0, 50, 0, 50)

    with pytest.raises(AssertionError):
        get_patch(input_array=input_array, index=index, no_data_value=no_data_value)


def test_get_patch_get_correct_patch():
    input_array = np.random.rand(3, 100, 100)
    no_data_value = 0
    index = (0, 50, 0, 50)

    patch, returned_index = get_patch(
        input_array=input_array, index=index, no_data_value=no_data_value
    )

    assert patch is not None, "Patch should not be None for valid input"
    assert patch.shape == (3, 50, 50), f"Expected shape (3, 50, 50), got {patch.shape}"
    assert returned_index == index, f"Expected index {index}, got {returned_index}"
    expected_patch = channel_norm(
        input_array[:, index[0] : index[1], index[2] : index[3]], no_data_value
    )
    assert (
        patch.shape == expected_patch.shape
    ), "Patch shape should match expected patch"
    assert np.allclose(
        patch, expected_patch, rtol=1e-5, atol=1e-5
    ), f"Patch should equal slice; got {patch} vs {expected_patch}"


def test_get_patch_move_away_from_nodata():
    input_array = np.random.rand(3, 100, 100)
    # set first row and col to 0
    input_array[:, 0, :] = 0
    input_array[:, :, 0] = 0
    no_data_value = 0
    index = (0, 50, 0, 50)

    patch, returned_index = get_patch(
        input_array=input_array, index=index, no_data_value=no_data_value
    )
    assert returned_index is not None, "Index should not be None"
    assert returned_index == (
        1,
        51,
        1,
        51,
    ), f"Expected index (1, 51, 1, 51), got {returned_index}"

    expected_patch = channel_norm(
        input_array[
            :,
            returned_index[0] : returned_index[1],
            returned_index[2] : returned_index[3],
        ],
        no_data_value,
    )
    assert patch is not None, "Patch should not be None for valid input"
    assert (
        patch.shape == expected_patch.shape
    ), "Patch shape should match expected patch"
    assert np.allclose(
        patch, expected_patch, rtol=1e-5, atol=1e-5
    ), f"Patch should equal slice; got {patch} vs {expected_patch}"


def test_make_patch_indexes_count_first_and_last():
    array_width = 100
    array_height = 100
    patch_size = 50
    patch_overlap = 10

    indexes = make_patch_indexes(
        array_width=array_width,
        array_height=array_height,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
    )

    assert len(indexes) == 9, f"Expected 9 patches got {len(indexes)}"
    assert indexes[0] == (0, 50, 0, 50), "Expected first patch to be (0, 50, 0, 50)"
    assert indexes[-1] == (
        50,
        100,
        50,
        100,
    ), "Expected last patch to be (50, 100, 50, 100)"


def test_make_patch_indexes_no_overlap():
    array_width = 100
    array_height = 100
    patch_size = 50
    patch_overlap = 0

    indexes = make_patch_indexes(
        array_width=array_width,
        array_height=array_height,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
    )

    assert len(indexes) == 4, f"Expected 4 patches got {len(indexes)}"
    assert indexes[0] == (0, 50, 0, 50), "Expected first patch to be (0, 50, 0, 50)"
    assert indexes[-1] == (
        50,
        100,
        50,
        100,
    ), "Expected last patch to be (50, 100, 50, 100)"


def test_make_patch_indexes_overlap_same_as_patch_size():
    array_width = 100
    array_height = 100
    patch_size = 50
    patch_overlap = 50

    with pytest.raises(AssertionError):
        make_patch_indexes(
            array_width=array_width,
            array_height=array_height,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
        )


def test_make_patch_indexes_overlap_larger_than_patch_size():
    array_width = 100
    array_height = 100
    patch_size = 50
    patch_overlap = 60

    with pytest.raises(AssertionError):
        make_patch_indexes(
            array_width=array_width,
            array_height=array_height,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
        )


def test_make_patch_indexes_patch_size_larger_than_array():
    array_width = 100
    array_height = 100
    patch_size = 150
    patch_overlap = 50

    with pytest.raises(AssertionError):
        make_patch_indexes(
            array_width=array_width,
            array_height=array_height,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
        )


def test_make_patch_indexes_patch_size_zero():
    array_width = 100
    array_height = 100
    patch_size = 0
    patch_overlap = 50

    with pytest.raises(AssertionError):
        make_patch_indexes(
            array_width=array_width,
            array_height=array_height,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
        )


#
def test_mask_prediction_basic():
    scene = np.array([[[2, 0, 2], [2, 2, 0]], [[2, 0, 0], [2, 2, 2]]])

    pred_tracker = np.ones((1, 2, 3))
    no_data_value = 0

    masked_pred_tracker, mask = mask_prediction(scene, pred_tracker, no_data_value)

    expected = np.array([[[1.0, 0.0, 1.0], [1.0, 1.0, 1.0]]])

    # check the correct pixels are masked
    np.testing.assert_array_equal(masked_pred_tracker, expected)

    # check that the mask is correct
    expected_mask = np.array([[1, 0, 1], [1, 1, 1]])
    np.testing.assert_array_equal(mask, expected_mask)


def test_mask_prediction_all_valid():
    scene = np.ones((3, 2, 2))  # All 1s, no no_data values
    pred_tracker = np.ones((1, 2, 2))
    no_data_value = 0

    masked_pred_tracker, mask = mask_prediction(scene, pred_tracker, no_data_value)

    np.testing.assert_array_equal(masked_pred_tracker, pred_tracker)

    expected_mask = np.ones((2, 2), dtype=np.uint8)
    np.testing.assert_array_equal(mask, expected_mask)


def test_mask_prediction_all_no_data():
    scene = np.zeros((3, 2, 2))  # All 0s, all no_data values
    pred_tracker = np.ones((1, 2, 2))

    no_data_value = 0

    masked_pred_tracker, mask = mask_prediction(scene, pred_tracker, no_data_value)

    expected = np.zeros((1, 2, 2))
    np.testing.assert_array_equal(masked_pred_tracker, expected)

    expected_mask = np.zeros((2, 2), dtype=np.uint8)
    np.testing.assert_array_equal(mask, expected_mask)


def test_mask_prediction_custom_no_data_value():
    scene = np.array([[[1, -9999, 3], [4, 5, -9999]], [[7, 8, -9999], [10, 11, -9999]]])
    pred_tracker = np.ones((1, 2, 3))
    no_data_value = -9999

    masked_pred_tracker, mask = mask_prediction(scene, pred_tracker, no_data_value)

    expected = np.array([[[1, 1, 1], [1, 1, 0]]])

    np.testing.assert_array_equal(masked_pred_tracker, expected)

    expected_mask = np.array([[1, 1, 1], [1, 1, 0]], dtype=np.uint8)
    np.testing.assert_array_equal(mask, expected_mask)


def test_mask_prediction_float_values():
    scene = np.array(
        [[[1.1, 0.0, 3.3], [4.4, 5.5, 0.0]], [[7.7, 8.8, 0.0], [10.0, 11.1, 0.0]]]
    )
    pred_tracker = np.ones((1, 2, 3))
    no_data_value = 0

    masked_pred_tracker, mask = mask_prediction(scene, pred_tracker, no_data_value)

    expected = np.array([[[1, 1, 1], [1, 1, 0]]])

    np.testing.assert_array_equal(masked_pred_tracker, expected)

    expected_mask = np.array([[1, 1, 1], [1, 1, 0]], dtype=np.uint8)
    np.testing.assert_array_equal(mask, expected_mask)


def test_mask_prediction_wrong_shapes():
    scene = np.ones((3, 2, 2))
    pred_tracker = np.ones((1, 3, 3))  # Wrong shape
    no_data_value = 0

    with pytest.raises(AssertionError):
        mask_prediction(scene, pred_tracker, no_data_value)


if __name__ == "__main__":
    pytest.main()
