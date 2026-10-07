import numpy as np
import pytest
import rasterio as rio
from rasterio.profiles import Profile

from omnicloudmask.model_utils import channel_norm
from omnicloudmask.raster_utils import (
    compute_class_stats,
    get_patch,
    make_patch_indexes,
    mask_prediction,
    save_prediction,
)


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
    assert patch.shape == expected_patch.shape, (
        "Patch shape should match expected patch"
    )
    assert np.allclose(patch, expected_patch, rtol=1e-5, atol=1e-5), (
        f"Patch should equal slice; got {patch} vs {expected_patch}"
    )


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
    assert patch.shape == expected_patch.shape, (
        "Patch shape should match expected patch"
    )
    assert np.allclose(patch, expected_patch, rtol=1e-5, atol=1e-5), (
        f"Patch should equal slice; got {patch} vs {expected_patch}"
    )


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


def test_compute_class_stats_basic():
    # 10 pixels: 4 clear, 3 thick cloud, 2 thin cloud, 1 cloud shadow
    pred = np.array([[[0, 0, 0, 0, 1], [1, 1, 2, 2, 3]]], dtype=np.uint8)

    tags = compute_class_stats(pred)

    assert tags["OCM_CLEAR_PCT"] == "40.00"
    assert tags["OCM_THICK_CLOUD_PCT"] == "30.00"
    assert tags["OCM_THIN_CLOUD_PCT"] == "20.00"
    assert tags["OCM_CLOUD_SHADOW_PCT"] == "10.00"
    assert tags["OCM_CLOUD_PCT"] == "50.00", "Cloud should be thick + thin"
    assert tags["OCM_VALID_PIXELS"] == "10"


def test_compute_class_stats_all_values_are_strings():
    pred = np.zeros((1, 2, 2), dtype=np.uint8)

    tags = compute_class_stats(pred)

    assert all(isinstance(v, str) for v in tags.values()), (
        "All tag values must be strings to be written as GDAL metadata"
    )


def test_compute_class_stats_missing_classes_are_zero():
    pred = np.ones((1, 3, 3), dtype=np.uint8)  # all thick cloud

    tags = compute_class_stats(pred)

    assert tags["OCM_CLEAR_PCT"] == "0.00"
    assert tags["OCM_THICK_CLOUD_PCT"] == "100.00"
    assert tags["OCM_THIN_CLOUD_PCT"] == "0.00"
    assert tags["OCM_CLOUD_SHADOW_PCT"] == "0.00"
    assert tags["OCM_CLOUD_PCT"] == "100.00"


def test_compute_class_stats_excludes_nodata():
    # nodata pixels are set to 0 by mask_prediction, which is also the clear class,
    # so they must not be counted as clear
    pred = np.array([[[0, 0, 1], [1, 0, 0]]], dtype=np.uint8)
    nodata_mask = np.array([[0, 0, 1], [1, 1, 1]], dtype=np.uint8)

    tags = compute_class_stats(pred, nodata_mask)

    assert tags["OCM_VALID_PIXELS"] == "4"
    assert tags["OCM_CLEAR_PCT"] == "50.00"
    assert tags["OCM_THICK_CLOUD_PCT"] == "50.00"


def test_compute_class_stats_after_mask_prediction():
    scene = np.array([[[2, 0, 2], [2, 2, 0]], [[2, 0, 2], [2, 2, 0]]])
    pred = np.array([[[1, 1, 3], [2, 1, 1]]], dtype=np.uint8)

    masked_pred, mask = mask_prediction(scene, pred, no_data_value=0)
    tags = compute_class_stats(masked_pred, mask)

    assert tags["OCM_VALID_PIXELS"] == "4"
    assert tags["OCM_CLEAR_PCT"] == "0.00"
    assert tags["OCM_THICK_CLOUD_PCT"] == "50.00"
    assert tags["OCM_THIN_CLOUD_PCT"] == "25.00"
    assert tags["OCM_CLOUD_SHADOW_PCT"] == "25.00"


def test_compute_class_stats_all_nodata():
    pred = np.zeros((1, 2, 2), dtype=np.uint8)
    nodata_mask = np.zeros((2, 2), dtype=np.uint8)

    tags = compute_class_stats(pred, nodata_mask)

    assert tags["OCM_VALID_PIXELS"] == "0"
    assert tags["OCM_CLEAR_PCT"] == "0.00"
    assert tags["OCM_CLOUD_PCT"] == "0.00"


def test_compute_class_stats_rounding():
    # 1 of 3 pixels in each of 3 classes
    pred = np.array([[[0, 1, 3]]], dtype=np.uint8)

    tags = compute_class_stats(pred)

    assert tags["OCM_CLEAR_PCT"] == "33.33"
    assert tags["OCM_THICK_CLOUD_PCT"] == "33.33"
    assert tags["OCM_CLOUD_SHADOW_PCT"] == "33.33"


def _export_profile(pred: np.ndarray) -> Profile:
    profile = Profile()
    profile.update(
        driver="GTiff",
        dtype=pred.dtype,
        count=pred.shape[0],
        height=pred.shape[1],
        width=pred.shape[2],
        compress="lzw",
        nodata=None,
    )
    return profile


def test_save_prediction_writes_class_stats_tags(tmp_path):
    pred = np.array([[[0, 0, 0, 0, 1], [1, 1, 2, 2, 3]]], dtype=np.uint8)
    output_path = tmp_path / "mask.tif"

    save_prediction(output_path, _export_profile(pred), pred, nodata_mask=None)

    with rio.open(output_path) as src:
        tags = src.tags()
    assert tags["OCM_CLEAR_PCT"] == "40.00"
    assert tags["OCM_CLOUD_PCT"] == "50.00"
    assert tags["OCM_CLOUD_SHADOW_PCT"] == "10.00"
    assert tags["OCM_VALID_PIXELS"] == "10"
    assert "OCM_CLASSES" in tags


def test_save_prediction_tags_respect_nodata_mask(tmp_path):
    pred = np.array([[[0, 0, 1], [1, 0, 0]]], dtype=np.uint8)
    nodata_mask = np.array([[0, 0, 1], [1, 1, 1]], dtype=np.uint8)
    output_path = tmp_path / "mask.tif"

    save_prediction(output_path, _export_profile(pred), pred, nodata_mask)

    with rio.open(output_path) as src:
        tags = src.tags()
    assert tags["OCM_VALID_PIXELS"] == "4"
    assert tags["OCM_CLEAR_PCT"] == "50.00"


def test_save_prediction_confidence_output_has_no_class_tags(tmp_path):
    pred = np.random.rand(4, 3, 3).astype(np.float32)
    output_path = tmp_path / "confidence.tif"

    save_prediction(output_path, _export_profile(pred), pred, nodata_mask=None)

    with rio.open(output_path) as src:
        tags = src.tags()
    assert not any(key.startswith("OCM_") for key in tags), (
        "Confidence maps should not have class percentage tags"
    )


def test_save_prediction_confidence_output_with_class_pred(tmp_path):
    pred = np.random.rand(4, 2, 5).astype(np.float32)
    class_pred = np.array([[[0, 0, 0, 0, 1], [1, 1, 2, 2, 3]]], dtype=np.uint8)
    output_path = tmp_path / "confidence.tif"

    save_prediction(
        output_path,
        _export_profile(pred),
        pred,
        nodata_mask=None,
        class_pred=class_pred,
    )

    with rio.open(output_path) as src:
        tags = src.tags()
        assert src.count == 4
    assert tags["OCM_CLEAR_PCT"] == "40.00"
    assert tags["OCM_CLOUD_PCT"] == "50.00"
    assert tags["OCM_VALID_PIXELS"] == "10"


def test_compute_class_stats_classes_legend():
    pred = np.zeros((1, 2, 2), dtype=np.uint8)

    tags = compute_class_stats(pred)

    assert tags["OCM_CLASSES"] == ("0=Clear,1=Thick Cloud,2=Thin Cloud,3=Cloud Shadow")


def test_class_names_exported_from_package():
    from omnicloudmask import CLASS_NAMES

    assert CLASS_NAMES == {
        0: "CLEAR",
        1: "THICK_CLOUD",
        2: "THIN_CLOUD",
        3: "CLOUD_SHADOW",
    }


def test_save_prediction_write_class_stats_false(tmp_path):
    pred = np.array([[[0, 0, 0, 0, 1], [1, 1, 2, 2, 3]]], dtype=np.uint8)
    output_path = tmp_path / "mask.tif"

    save_prediction(
        output_path,
        _export_profile(pred),
        pred,
        nodata_mask=None,
        class_pred=pred,
        write_class_stats=False,
    )

    with rio.open(output_path) as src:
        tags = src.tags()
    assert not any(key.startswith("OCM_") for key in tags), (
        "No class stats tags should be written when write_class_stats is False"
    )


if __name__ == "__main__":
    pytest.main()
