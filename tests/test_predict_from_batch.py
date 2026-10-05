from pathlib import Path

import numpy as np
import pytest
import rasterio as rio
import torch
from rasterio.transform import from_origin

from omnicloudmask import predict_from_array, predict_from_batch
from omnicloudmask.cloud_mask import make_output_path

IMAGE_SIZE = 64


@pytest.fixture(scope="module")
def images() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.random((5, 3, IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)


@pytest.fixture(scope="module")
def reference_masks(images: np.ndarray) -> np.ndarray:
    return predict_from_batch(images, batch_size=2, inference_device="cpu")


def write_tif(path: Path, array: np.ndarray) -> Path:
    profile = {
        "driver": "GTiff",
        "height": array.shape[1],
        "width": array.shape[2],
        "count": array.shape[0],
        "dtype": array.dtype,
        "crs": "EPSG:32750",
        "transform": from_origin(400000, 6500000, 10, 10),
    }
    with rio.open(path, "w", **profile) as dst:
        dst.write(array)
    return path


def load_tif(input_path: Path) -> tuple[np.ndarray, dict]:
    with rio.open(input_path) as src:
        return src.read(), src.profile


def load_tif_array_only(input_path: Path) -> np.ndarray:
    with rio.open(input_path) as src:
        return src.read()


@pytest.fixture
def tif_paths(tmp_path: Path, images: np.ndarray) -> list[Path]:
    return [
        write_tif(tmp_path / f"chip_{i}.tif", image) for i, image in enumerate(images)
    ]


def test_predict_from_batch_mask(reference_masks: np.ndarray) -> None:
    assert reference_masks.shape == (5, 1, IMAGE_SIZE, IMAGE_SIZE)
    assert reference_masks.dtype == np.uint8
    assert np.all(np.isin(np.unique(reference_masks), [0, 1, 2, 3]))


def test_predict_from_batch_confidence(images: np.ndarray) -> None:
    result = predict_from_batch(
        images, batch_size=2, inference_device="cpu", export_confidence=True
    )
    assert result.shape == (5, 4, IMAGE_SIZE, IMAGE_SIZE)
    assert result.dtype == np.float32
    assert np.all((result >= 0) & (result <= 1))


def test_predict_from_batch_confidence_no_softmax(images: np.ndarray) -> None:
    result = predict_from_batch(
        images[:2],
        batch_size=2,
        inference_device="cpu",
        export_confidence=True,
        softmax_output=False,
    )
    assert result.shape == (2, 4, IMAGE_SIZE, IMAGE_SIZE)
    assert result.min() < 0 or result.max() > 1


def test_predict_from_batch_matches_predict_from_array(
    images: np.ndarray, reference_masks: np.ndarray
) -> None:
    expected = predict_from_array(
        images[0],
        patch_size=IMAGE_SIZE,
        patch_overlap=0,
        inference_device="cpu",
    )
    agreement = (expected == reference_masks[0]).mean()
    assert agreement >= 0.999, f"Only {agreement:.4f} of pixels agree"


@pytest.mark.parametrize("batch_size", [1, 3, 8])
def test_predict_from_batch_batch_size_invariant(
    images: np.ndarray, reference_masks: np.ndarray, batch_size: int
) -> None:
    result = predict_from_batch(images, batch_size=batch_size, inference_device="cpu")
    np.testing.assert_array_equal(result, reference_masks)


@pytest.mark.parametrize(
    "make_data",
    [
        pytest.param(lambda x: list(x), id="list_of_arrays"),
        pytest.param(lambda x: (image for image in x), id="generator"),
        pytest.param(lambda x: torch.from_numpy(x), id="tensor"),
        pytest.param(lambda x: [torch.from_numpy(i) for i in x], id="list_of_tensors"),
        pytest.param(lambda x: (x * 10000).astype(np.uint16), id="uint16_array"),
    ],
)
def test_predict_from_batch_input_types(
    images: np.ndarray, reference_masks: np.ndarray, make_data
) -> None:
    result = predict_from_batch(make_data(images), batch_size=2, inference_device="cpu")
    assert result.shape == reference_masks.shape
    agreement = (result == reference_masks).mean()
    assert agreement >= 0.999, f"Only {agreement:.4f} of pixels agree"


def test_predict_from_batch_no_data_mask(images: np.ndarray) -> None:
    data = images[:3].copy()
    data[0, :, :10, :] = 0  # partial no data
    data[1] = 0  # entirely no data

    result = predict_from_batch(data, batch_size=3, inference_device="cpu")
    assert np.all(result[0, :, :10, :] == 0)
    assert np.all(result[1] == 0)

    confidence = predict_from_batch(
        data, batch_size=3, inference_device="cpu", export_confidence=True
    )
    assert np.all(confidence[1] == 0)


def test_predict_from_batch_nan_no_data(images: np.ndarray) -> None:
    data = images[:2].copy()
    data[0, :, :10, :] = np.nan

    result = predict_from_batch(
        data, batch_size=2, inference_device="cpu", no_data_value=np.nan
    )
    assert np.all(result[0, :, :10, :] == 0)


def test_predict_from_batch_load_func(
    tif_paths: list[Path], reference_masks: np.ndarray
) -> None:
    result = predict_from_batch(
        tif_paths, batch_size=2, load_func=load_tif, inference_device="cpu"
    )
    np.testing.assert_array_equal(result, reference_masks)

    result = predict_from_batch(
        tif_paths, batch_size=2, load_func=load_tif_array_only, inference_device="cpu"
    )
    np.testing.assert_array_equal(result, reference_masks)


def test_predict_from_batch_export(
    tif_paths: list[Path], reference_masks: np.ndarray, tmp_path: Path
) -> None:
    output_dir = tmp_path / "out"
    paths = predict_from_batch(
        tif_paths,
        batch_size=2,
        load_func=load_tif,
        inference_device="cpu",
        export_to_disk=True,
        output_dir=output_dir,
    )

    assert paths == [make_output_path(p, output_dir) for p in tif_paths]
    for path, expected, source in zip(paths, reference_masks, tif_paths):
        with rio.open(path) as dst, rio.open(source) as src:
            np.testing.assert_array_equal(dst.read(), expected)
            assert dst.crs == src.crs
            assert dst.transform == src.transform


@pytest.mark.filterwarnings("ignore::rasterio.errors.NotGeoreferencedWarning")
def test_predict_from_batch_export_without_profile(
    tif_paths: list[Path], tmp_path: Path
) -> None:
    paths = predict_from_batch(
        tif_paths[:1],
        load_func=load_tif_array_only,
        inference_device="cpu",
        export_to_disk=True,
        output_dir=tmp_path / "out",
    )
    with rio.open(paths[0]) as dst:
        assert dst.shape == (IMAGE_SIZE, IMAGE_SIZE)


def test_predict_from_batch_export_skip_existing(
    tif_paths: list[Path], tmp_path: Path
) -> None:
    output_dir = tmp_path / "out"
    existing = make_output_path(tif_paths[0], output_dir)
    existing.write_bytes(b"placeholder")

    paths = predict_from_batch(
        tif_paths[:2],
        load_func=load_tif,
        inference_device="cpu",
        export_to_disk=True,
        output_dir=output_dir,
        overwrite=False,
    )

    assert paths == [make_output_path(p, output_dir) for p in tif_paths[:2]]
    assert existing.read_bytes() == b"placeholder"
    assert paths[1].exists()


def test_predict_from_batch_export_all_existing_skips_inference(
    tif_paths: list[Path], tmp_path: Path
) -> None:
    def failing_loader(input_path: Path) -> np.ndarray:
        raise AssertionError("Nothing should be loaded")

    output_dir = tmp_path / "out"
    make_output_path(tif_paths[0], output_dir).write_bytes(b"placeholder")

    paths = predict_from_batch(
        tif_paths[:1],
        load_func=failing_loader,
        export_to_disk=True,
        output_dir=output_dir,
        overwrite=False,
    )
    assert paths == [make_output_path(tif_paths[0], output_dir)]


def test_predict_from_batch_shape_mismatch(images: np.ndarray) -> None:
    data = list(images[:2]) + [np.random.rand(3, IMAGE_SIZE, IMAGE_SIZE + 1)]
    with pytest.raises(ValueError, match="index 2"):
        predict_from_batch(data, batch_size=2, inference_device="cpu")


def test_predict_from_batch_wrong_band_count() -> None:
    data = np.random.rand(2, 2, IMAGE_SIZE, IMAGE_SIZE)
    with pytest.raises(ValueError, match="fill it with zeros"):
        predict_from_batch(data, inference_device="cpu")


def test_predict_from_batch_wrong_dimensions() -> None:
    data = [np.random.rand(IMAGE_SIZE, IMAGE_SIZE)]
    with pytest.raises(ValueError, match="3 dimensions"):
        predict_from_batch(data, inference_device="cpu")


def test_predict_from_batch_image_too_small() -> None:
    data = np.random.rand(2, 3, 16, 16)
    with pytest.raises(ValueError, match="width and height"):
        predict_from_batch(data, inference_device="cpu")


def test_predict_from_batch_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        predict_from_batch([], inference_device="cpu")


def test_predict_from_batch_invalid_batch_size(images: np.ndarray) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        predict_from_batch(images, batch_size=0)


def test_predict_from_batch_export_requires_load_func(images: np.ndarray) -> None:
    with pytest.raises(ValueError, match="load_func"):
        predict_from_batch(images, export_to_disk=True)


def test_predict_from_batch_export_requires_paths(images: np.ndarray) -> None:
    with pytest.raises(ValueError, match="file paths"):
        predict_from_batch(
            images, load_func=lambda input_path: input_path, export_to_disk=True
        )


def test_predict_from_batch_load_func_error_propagates(tif_paths: list[Path]) -> None:
    def loader(input_path: Path) -> np.ndarray:
        if input_path == tif_paths[3]:
            raise RuntimeError("corrupt file")
        return load_tif_array_only(input_path)

    with pytest.raises(RuntimeError, match="corrupt file"):
        predict_from_batch(
            tif_paths, batch_size=2, load_func=loader, inference_device="cpu"
        )


def test_predict_from_batch_custom_model_classes(images: np.ndarray) -> None:
    model = torch.nn.Conv2d(3, 2, kernel_size=3, padding=1)

    result = predict_from_batch(
        images[:2],
        inference_device="cpu",
        custom_models=model,
        export_confidence=True,
    )
    assert result.shape == (2, 2, IMAGE_SIZE, IMAGE_SIZE)
