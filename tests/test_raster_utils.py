import threading
import time
from typing import Callable

import numpy as np
import pytest
import rasterio as rio
import torch
from rasterio.profiles import Profile

from omnicloudmask.model_utils import channel_norm
from omnicloudmask import raster_utils
from omnicloudmask.raster_utils import (
    ReadWriteLock,
    class_stats_tags,
    save_prediction,
    scene_valid_mask,
    batch_nodata_mask,
    get_patch,
    make_patch_indexes,
    mask_prediction,
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


def test_get_patch_all_nan_nodata():
    input_array = np.full((3, 100, 100), np.nan)
    patch, returned_index = get_patch(input_array, (0, 50, 0, 50), np.nan)
    assert patch is None
    assert returned_index is None


@pytest.mark.parametrize("no_data_value", [np.nan, 0])
def test_get_patch_move_away_from_nan_nodata(no_data_value):
    rng = np.random.default_rng(0)
    input_array = rng.random((3, 100, 100)) + 0.1
    input_array[:, :2, :] = np.nan
    input_array[:, :, :3] = np.nan

    patch, returned_index = get_patch(input_array, (0, 50, 0, 50), no_data_value)
    # NaN is always no data, whatever no_data_value is
    assert returned_index == (2, 52, 3, 53)
    assert patch is not None
    assert not np.isnan(patch).any()


def test_get_patch_nan_shifts_like_zero():
    rng = np.random.default_rng(0)
    zero_array = rng.random((3, 100, 100)) + 0.1
    zero_array[:, 70:, :] = 0
    zero_array[:, :, 80:] = 0
    nan_array = np.where(zero_array == 0, np.nan, zero_array)

    for index in [(50, 100, 50, 100), (60, 110, 0, 50), (0, 50, 60, 110)]:
        zero_patch, zero_index = get_patch(zero_array, index, 0)
        nan_patch, nan_index = get_patch(nan_array, index, np.nan)
        assert zero_index == nan_index
        np.testing.assert_allclose(zero_patch, nan_patch)


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


def test_mask_prediction_nan_no_data_value():
    nan = np.nan
    scene = np.array(
        [[[1.1, nan, 3.3], [4.4, 5.5, nan]], [[7.7, 8.8, nan], [10.0, 11.1, nan]]]
    )
    pred_tracker = np.ones((1, 2, 3))
    no_data_value = np.nan

    masked_pred_tracker, mask = mask_prediction(scene, pred_tracker, no_data_value)

    # only pixels where all bands are NaN are masked
    expected = np.array([[[1, 1, 1], [1, 1, 0]]])

    np.testing.assert_array_equal(masked_pred_tracker, expected)

    expected_mask = np.array([[1, 1, 1], [1, 1, 0]], dtype=np.uint8)
    np.testing.assert_array_equal(mask, expected_mask)


def test_mask_prediction_nan_with_zero_no_data_value():
    scene = np.ones((2, 2, 2))
    scene[:, 0, 0] = np.nan
    scene[:, 1, 1] = 0
    scene[0, 0, 1] = np.nan  # only one band is NaN, so still valid

    _, mask = mask_prediction(scene, np.ones((1, 2, 2)), 0)
    np.testing.assert_array_equal(mask, [[0, 1], [1, 0]])

    batch_mask = batch_nodata_mask(torch.from_numpy(scene[None]), 0).numpy()
    np.testing.assert_array_equal(batch_mask[0, 0], mask.astype(bool))

    # so NaN pixels also count towards OCM_NODATA_PCT
    np.testing.assert_array_equal(scene_valid_mask(scene, 0), mask.astype(bool))


def test_batch_nodata_mask_matches_mask_prediction():
    rng = np.random.default_rng(0)
    batch = rng.integers(0, 3, size=(3, 3, 8, 8)).astype(np.float32)
    batch[1] = 0

    result = batch_nodata_mask(torch.from_numpy(batch), 0).numpy()
    assert result.shape == (3, 1, 8, 8)
    for image, mask in zip(batch, result, strict=True):
        _, expected = mask_prediction(image, np.ones((1, 8, 8)), 0)
        np.testing.assert_array_equal(mask[0], expected.astype(bool))


def test_batch_nodata_mask_nan():
    batch = np.ones((1, 2, 2, 2), dtype=np.float32)
    batch[0, :, 0, 0] = np.nan
    batch[0, 0, 1, 1] = np.nan  # only one band is NaN, so still valid

    result = batch_nodata_mask(torch.from_numpy(batch), np.nan).numpy()
    np.testing.assert_array_equal(result[0, 0], [[False, True], [True, True]])


def test_mask_prediction_wrong_shapes():
    scene = np.ones((3, 2, 2))
    pred_tracker = np.ones((1, 3, 3))  # Wrong shape
    no_data_value = 0

    with pytest.raises(AssertionError):
        mask_prediction(scene, pred_tracker, no_data_value)


# 10 pixels: 4 clear, 3 thick cloud, 2 thin cloud, 1 cloud shadow
STATS_PRED = np.array([[[0, 0, 0, 0, 1], [1, 1, 2, 2, 3]]], dtype=np.uint8)


@pytest.mark.parametrize("as_tensor", [False, True])
def test_class_stats_tags_basic(as_tensor: bool) -> None:
    pred = torch.from_numpy(STATS_PRED) if as_tensor else STATS_PRED
    valid = (
        torch.ones(pred.shape, dtype=torch.bool)
        if as_tensor
        else np.ones(pred.shape, dtype=bool)
    )

    tags = class_stats_tags(pred, valid)

    assert tags["OCM_CLEAR_PCT"] == "40.00"
    assert tags["OCM_THICK_CLOUD_PCT"] == "30.00"
    assert tags["OCM_THIN_CLOUD_PCT"] == "20.00"
    assert tags["OCM_CLOUD_SHADOW_PCT"] == "10.00"
    assert tags["OCM_CLOUD_PCT"] == "50.00", "Cloud should be thick + thin"
    assert tags["OCM_VALID_PIXELS"] == "10"
    assert tags["OCM_NODATA_PCT"] == "0.00"
    assert tags["OCM_CLASSES"] == "0=Clear,1=Thick Cloud,2=Thin Cloud,3=Cloud Shadow"
    assert all(isinstance(v, str) for v in tags.values())


@pytest.mark.parametrize("as_tensor", [False, True])
def test_class_stats_tags_excludes_invalid_pixels(as_tensor: bool) -> None:
    # nodata pixels are 0 in the prediction, which is also the clear class, so
    # they must not be counted as clear
    pred = np.array([[[0, 0, 1], [1, 0, 0]]], dtype=np.uint8)
    valid = np.array([[[False, False, True], [True, True, True]]])
    if as_tensor:
        pred, valid = torch.from_numpy(pred), torch.from_numpy(valid)

    tags = class_stats_tags(pred, valid)

    assert tags["OCM_VALID_PIXELS"] == "4"
    assert tags["OCM_CLEAR_PCT"] == "50.00"
    assert tags["OCM_THICK_CLOUD_PCT"] == "50.00"
    assert tags["OCM_NODATA_PCT"] == "33.33", "2 of 6 pixels are no data"


def test_class_stats_tags_missing_classes_are_zero() -> None:
    pred = np.ones((1, 3, 3), dtype=np.uint8)  # all thick cloud

    tags = class_stats_tags(pred, np.ones(pred.shape, dtype=bool))

    assert tags["OCM_CLEAR_PCT"] == "0.00"
    assert tags["OCM_THICK_CLOUD_PCT"] == "100.00"
    assert tags["OCM_CLOUD_PCT"] == "100.00"


def test_class_stats_tags_all_invalid() -> None:
    pred = np.zeros((1, 2, 2), dtype=np.uint8)

    tags = class_stats_tags(pred, np.zeros(pred.shape, dtype=bool))

    assert tags["OCM_VALID_PIXELS"] == "0"
    assert tags["OCM_NODATA_PCT"] == "100.00"
    assert tags["OCM_CLEAR_PCT"] == "0.00"
    assert tags["OCM_CLOUD_PCT"] == "0.00"


def test_class_stats_tags_rounding() -> None:
    pred = np.array([[[0, 1, 3]]], dtype=np.uint8)

    tags = class_stats_tags(pred, np.ones(pred.shape, dtype=bool))

    assert tags["OCM_CLEAR_PCT"] == "33.33"
    assert tags["OCM_THICK_CLOUD_PCT"] == "33.33"
    assert tags["OCM_CLOUD_SHADOW_PCT"] == "33.33"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_class_stats_tags_cuda_matches_numpy() -> None:
    rng = np.random.default_rng(0)
    pred = rng.integers(0, 4, (1, 50, 50), dtype=np.uint8)
    valid = rng.random((1, 50, 50)) > 0.3

    expected = class_stats_tags(pred, valid)
    result = class_stats_tags(
        torch.from_numpy(pred).cuda(), torch.from_numpy(valid).cuda()
    )

    assert result == expected


@pytest.mark.parametrize("no_data_value", [0, -9999, np.nan])
def test_scene_valid_mask_matches_mask_prediction(no_data_value) -> None:
    rng = np.random.default_rng(1)
    scene = rng.random((3, 6, 6)).astype(np.float32) + 1
    scene[:, :2, :] = no_data_value  # all bands no data
    scene[0, 3, :] = no_data_value  # one band only, still valid

    valid = scene_valid_mask(scene, no_data_value)
    _, mask = mask_prediction(scene, np.ones((1, 6, 6)), no_data_value)

    assert valid.dtype == bool
    np.testing.assert_array_equal(valid, mask.astype(bool))
    assert not valid[:2].any() and valid[2:].all()


def test_save_prediction_writes_tags(tmp_path) -> None:
    output_path = tmp_path / "mask.tif"
    profile = Profile(
        driver="GTiff", dtype="uint8", count=1, height=2, width=5, nodata=None
    )

    save_prediction(
        output_path, profile, STATS_PRED, nodata_mask=None, tags={"OCM_VERSION": "9"}
    )

    with rio.open(output_path) as src:
        assert src.tags()["OCM_VERSION"] == "9"


@pytest.mark.parametrize("standard_classes", [True, False])
def test_save_prediction_mask_styling(tmp_path, standard_classes: bool) -> None:
    output_path = tmp_path / "mask.tif"
    profile = Profile(
        driver="GTiff", dtype="uint8", count=1, height=2, width=5, nodata=None
    )

    save_prediction(
        output_path, profile, STATS_PRED, None, standard_classes=standard_classes
    )

    with rio.open(output_path) as src:
        if standard_classes:
            colormap = src.colormap(1)
            assert colormap[0] == (0, 0, 0, 255)
            assert colormap[1] == (255, 128, 13, 255)
            assert colormap[3] == (214, 39, 40, 255)
            assert src.descriptions[0].startswith("OCM class: 0=Clear")
        else:
            with pytest.raises(ValueError):
                src.colormap(1)
            assert src.descriptions == (None,)


def test_save_prediction_confidence_band_descriptions(tmp_path) -> None:
    output_path = tmp_path / "confidence.tif"
    pred = np.random.default_rng(0).random((4, 2, 5)).astype(np.float32)
    profile = Profile(
        driver="GTiff", dtype="float32", count=4, height=2, width=5, nodata=None
    )

    save_prediction(output_path, profile, pred, None, standard_classes=True)

    with rio.open(output_path) as src:
        assert src.descriptions == (
            "Clear",
            "Thick Cloud",
            "Thin Cloud",
            "Cloud Shadow",
        )


def run_threads(*targets: Callable[[], None]) -> None:
    """Run each target in a thread, failing on a hang or re-raising the first error."""
    errors = []

    def wrap(target: Callable[[], None]) -> Callable[[], None]:
        def run() -> None:
            try:
                target()
            except BaseException as e:
                errors.append(e)

        return run

    threads = [threading.Thread(target=wrap(t), daemon=True) for t in targets]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
        assert not thread.is_alive(), "thread hung"
    if errors:
        raise errors[0]


def wait_for(condition: Callable[[], bool], timeout: float = 5) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.001)


def test_read_write_lock_readers_share() -> None:
    lock = ReadWriteLock()
    both_in = threading.Barrier(2, timeout=5)

    def reader() -> None:
        with lock.read():
            # raises unless both readers hold the lock at once
            both_in.wait()

    run_threads(reader, reader)


def test_read_write_lock_writer_waits_for_readers() -> None:
    lock = ReadWriteLock()
    events = []
    release = threading.Event()

    def reader() -> None:
        with lock.read():
            release.wait(5)
            events.append("read done")

    def writer() -> None:
        wait_for(lambda: lock._readers == 1)
        with lock.write():
            events.append("write")

    def releaser() -> None:
        wait_for(lambda: lock._writers_waiting == 1)
        release.set()

    run_threads(reader, writer, releaser)
    assert events == ["read done", "write"]


def test_read_write_lock_waiting_writer_holds_back_new_readers() -> None:
    """So a steady stream of loads can't starve saves."""
    lock = ReadWriteLock()
    events = []
    release = threading.Event()

    def first_reader() -> None:
        with lock.read():
            release.wait(5)

    def writer() -> None:
        wait_for(lambda: lock._readers == 1)
        with lock.write():
            events.append("write")

    def late_reader() -> None:
        wait_for(lambda: lock._writers_waiting == 1)
        with lock.read():
            events.append("late read")

    def releaser() -> None:
        wait_for(lambda: lock._writers_waiting == 1)
        # let the late reader reach its wait before the first reader leaves
        time.sleep(0.05)
        assert events == []
        release.set()

    run_threads(first_reader, writer, late_reader, releaser)
    assert events == ["write", "late read"]


def test_read_write_lock_nested_read_passes_waiting_writer() -> None:
    """A load_func that runs an in-memory prediction, which loads too, mustn't
    deadlock against a save waiting for the outer load."""
    lock = ReadWriteLock()
    events = []

    def reader() -> None:
        with lock.read():
            wait_for(lambda: lock._writers_waiting == 1)
            with lock.read():
                events.append("nested read")

    def writer() -> None:
        wait_for(lambda: lock._readers == 1)
        with lock.write():
            events.append("write")

    run_threads(reader, writer)
    assert events == ["nested read", "write"]


def test_read_write_lock_write_while_reading_raises() -> None:
    lock = ReadWriteLock()
    with lock.read():
        with pytest.raises(RuntimeError, match="while loading"):
            with lock.write():
                pass
    # the lock is still usable
    with lock.write():
        pass


def test_save_prediction_same_path_concurrently(tmp_path) -> None:
    output_path = tmp_path / "mask.tif"

    def save() -> None:
        for _ in range(10):
            save_prediction(output_path, SAVE_PROFILE, STATS_PRED, None)

    run_threads(save, save, save)
    assert [p.name for p in tmp_path.iterdir()] == ["mask.tif"]


SAVE_PROFILE = Profile(
    driver="GTiff", dtype="uint8", count=1, height=2, width=5, nodata=None
)


def test_save_prediction_replaces_truncated_file(tmp_path) -> None:
    """A crash mid-save used to leave a truncated GeoTIFF that GDAL couldn't open,
    so the next run failed to overwrite it (#52)."""
    output_path = tmp_path / "mask.tif"
    output_path.write_bytes(b"II*\x00truncated")
    (tmp_path / "mask.tif.partial").write_bytes(b"II*\x00truncated")

    save_prediction(output_path, SAVE_PROFILE, STATS_PRED, np.ones((2, 5)))

    with rio.open(output_path) as src:
        np.testing.assert_array_equal(src.read(), STATS_PRED)
    assert [p.name for p in tmp_path.iterdir()] == ["mask.tif"]


def test_save_prediction_failure_leaves_no_file(tmp_path, monkeypatch) -> None:
    output_path = tmp_path / "mask.tif"

    def fail_mid_write(path, *args) -> None:
        path.write_bytes(b"II*\x00truncated")
        raise RuntimeError("disk full")

    monkeypatch.setattr(raster_utils, "_write_prediction", fail_mid_write)
    with pytest.raises(RuntimeError, match="disk full"):
        save_prediction(output_path, SAVE_PROFILE, STATS_PRED, None)

    # nothing under the final name, so overwrite=False won't skip the scene
    assert list(tmp_path.iterdir()) == []


def test_save_prediction_mask_is_internal(tmp_path) -> None:
    output_path = tmp_path / "mask.tif"
    nodata_mask = np.array([[1, 1, 0, 0, 1], [1, 1, 1, 1, 0]], dtype=np.uint8)

    # GDAL before 3.9 writes an external .msk unless told otherwise
    with rio.Env(GDAL_TIFF_INTERNAL_MASK=False):
        save_prediction(output_path, SAVE_PROFILE, STATS_PRED, nodata_mask)

    assert [p.name for p in tmp_path.iterdir()] == ["mask.tif"]
    with rio.open(output_path) as src:
        np.testing.assert_array_equal(src.read_masks(1), nodata_mask * 255)


if __name__ == "__main__":
    pytest.main()
