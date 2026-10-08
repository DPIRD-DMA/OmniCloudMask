import itertools
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import rasterio as rio
import torch
from rasterio.transform import from_origin

from omnicloudmask import predict_from_array, predict_from_batch, predict_from_load_func
from omnicloudmask.cloud_mask import _prefetch, make_output_path

IMAGE_SIZE = 64


def run_with_timeout(func, timeout: float = 120):
    """Run func in a daemon thread, failing the test if it hangs, otherwise
    returning its result or re-raising its error."""
    outcome = {}

    def target():
        try:
            outcome["result"] = func()
        except BaseException as e:
            # Release the frames' locals as a real caller would, so open
            # generators are closed here and any hang during cleanup is caught
            traceback.clear_frames(e.__traceback__)
            outcome["error"] = e

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        pytest.fail(f"Call did not finish within {timeout}s, it may have deadlocked")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["result"]


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
        run_with_timeout(
            lambda: predict_from_batch(
                tif_paths, batch_size=2, load_func=loader, inference_device="cpu"
            )
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


def test_predict_from_batch_model_error_mid_run(images: np.ndarray) -> None:
    class FailOnSecondBatch(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 4, kernel_size=1)
            self.calls = 0

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("model failed")
            return self.conv(x)

    # many small batches keep the prefetch queue full when the error is raised
    data = np.repeat(images, 4, axis=0)
    with pytest.raises(RuntimeError, match="model failed"):
        run_with_timeout(
            lambda: predict_from_batch(
                data,
                batch_size=1,
                inference_device="cpu",
                custom_models=FailOnSecondBatch(),
            )
        )


DEVICES = [
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA not available"
        ),
    ),
    pytest.param(
        "mps",
        marks=pytest.mark.skipif(
            not torch.backends.mps.is_available(), reason="MPS not available"
        ),
    ),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", ["fp32", "fp16"])
def test_predict_from_batch_accelerator_matches_cpu(
    images: np.ndarray, reference_masks: np.ndarray, device: str, dtype: str
) -> None:
    result = predict_from_batch(
        images, batch_size=2, inference_device=device, inference_dtype=dtype
    )
    agreement = (result == reference_masks).mean()
    assert agreement >= 0.99, f"Only {agreement:.4f} of pixels agree"


@pytest.mark.parametrize("device", DEVICES)
def test_predict_from_batch_accelerator_tensor_input(
    images: np.ndarray, device: str
) -> None:
    expected = predict_from_batch(images, batch_size=2, inference_device=device)
    result = predict_from_batch(
        torch.from_numpy(images).to(device), batch_size=2, inference_device=device
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("device", [pytest.param("cpu"), *DEVICES])
@pytest.mark.parametrize("input_type", ["numpy", "cpu_tensor", "device_tensor"])
def test_predict_from_batch_from_threads(
    images: np.ndarray, device: str, input_type: str
) -> None:
    """Calling predict_from_batch from several threads at once must not crash, which
    it did on MPS, and must match a single-threaded call."""
    data = {
        "numpy": images,
        "cpu_tensor": torch.from_numpy(images),
        "device_tensor": torch.from_numpy(images).to(device),
    }[input_type]
    expected = predict_from_batch(data, batch_size=2, inference_device=device)

    def run(_: int) -> np.ndarray:
        return predict_from_batch(data, batch_size=2, inference_device=device)

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = run_with_timeout(lambda: list(executor.map(run, range(8))))

    for result in results:
        np.testing.assert_array_equal(result, expected)


def test_predict_from_batch_default_device(
    images: np.ndarray, reference_masks: np.ndarray
) -> None:
    result = predict_from_batch(images, batch_size=2)
    agreement = (result == reference_masks).mean()
    assert agreement >= 0.99, f"Only {agreement:.4f} of pixels agree"


def test_predict_from_batch_without_no_data_mask(images: np.ndarray) -> None:
    data = images[:3].copy()
    data[0, :, :10, :] = 0  # partial no data
    data[1] = 0  # entirely no data

    kwargs = {"batch_size": 3, "inference_device": "cpu", "export_confidence": True}
    unmasked = predict_from_batch(data, apply_no_data_mask=False, **kwargs)
    masked = predict_from_batch(data, apply_no_data_mask=True, **kwargs)

    # no-data pixels keep their predictions when the mask is not applied
    assert unmasked[0, :, :10, :].min() >= 0.001
    assert np.all(masked[0, :, :10, :] == 0)
    np.testing.assert_array_equal(unmasked[0, :, 10:, :], masked[0, :, 10:, :])
    np.testing.assert_array_equal(unmasked[2], masked[2])
    # images that are entirely no data are never predicted, as in the tiled path
    assert np.all(unmasked[1] == 0)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("softmax_output", [True, False])
def test_predict_from_batch_all_no_data_matches_predict_from_array(
    softmax_output: bool,
) -> None:
    data = np.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    kwargs = {
        "inference_device": "cpu",
        "export_confidence": True,
        "softmax_output": softmax_output,
    }
    result = predict_from_batch(data, **kwargs)
    expected = predict_from_array(
        data[0], patch_size=IMAGE_SIZE, patch_overlap=0, **kwargs
    )
    # zeros with softmax, NaN for raw logits (assert_array_equal treats NaN as equal)
    np.testing.assert_array_equal(result[0], expected)
    assert np.isnan(result).all() != softmax_output


@pytest.mark.parametrize("apply_no_data_mask", [True, False])
def test_predict_from_batch_export_matches_predict_from_load_func(
    tmp_path: Path, images: np.ndarray, apply_no_data_mask: bool
) -> None:
    data = images[0].copy()
    data[:, :10, :] = 0
    path = write_tif(tmp_path / "chip.tif", data)

    [batch_path] = predict_from_batch(
        [path],
        load_func=load_tif,
        inference_device="cpu",
        export_to_disk=True,
        output_dir=tmp_path / "batch",
        apply_no_data_mask=apply_no_data_mask,
    )
    [tiled_path] = predict_from_load_func(
        [path],
        load_tif,
        patch_size=IMAGE_SIZE,
        patch_overlap=0,
        inference_device="cpu",
        output_dir=tmp_path / "tiled",
        apply_no_data_mask=apply_no_data_mask,
    )

    with rio.open(batch_path) as batch, rio.open(tiled_path) as tiled:
        np.testing.assert_array_equal(batch.read(), tiled.read())
        np.testing.assert_array_equal(batch.read_masks(1), tiled.read_masks(1))
        assert batch.profile == tiled.profile
        if apply_no_data_mask:
            assert np.all(batch.read_masks(1)[:10] == 0)
            assert np.all(batch.read_masks(1)[10:] == 255)
        else:
            assert np.all(batch.read_masks(1) == 255)


def _single_image_batches(count: int):
    for i in range(count):
        yield [np.full((3, 4, 4), i, dtype=np.float32)], [None], [i]


def test_prefetch_slow_consumer_keeps_order() -> None:
    received = []
    for batch, profiles, items in _prefetch(
        _single_image_batches(6), torch.device("cpu"), depth=1
    ):
        # slower than the producer, so it retries puts on a full queue
        time.sleep(0.25)
        assert batch.shape == (1, 3, 4, 4)
        assert batch[0, 0, 0, 0] == items[0]
        assert profiles == [None]
        received.append(items[0])
    assert received == list(range(6))


def test_prefetch_close_stops_producer() -> None:
    source_closed = threading.Event()

    def endless_batches():
        try:
            for i in itertools.count():
                yield [np.zeros((3, 4, 4), dtype=np.float32)], [None], [i]
        finally:
            source_closed.set()

    threads_before = threading.active_count()
    prefetched = _prefetch(endless_batches(), torch.device("cpu"), depth=1)
    next(prefetched)
    prefetched.close()

    assert source_closed.is_set()
    assert threading.active_count() == threads_before


def test_prefetch_producer_error_propagates() -> None:
    def failing_batches():
        yield [np.zeros((3, 4, 4), dtype=np.float32)], [None], [0]
        raise RuntimeError("load failed")

    prefetched = _prefetch(failing_batches(), torch.device("cpu"))
    next(prefetched)
    with pytest.raises(RuntimeError, match="load failed"):
        next(prefetched)


# Contract tests: behaviour that refactors of the batch pipeline (zero-copy
# slicing, native dtypes, pinned buffers, async copies) must preserve.

ALL_DEVICES = [pytest.param("cpu"), *DEVICES]


class SeededConv(torch.nn.Module):
    """Small deterministic stand-in for the real models."""

    def __init__(self) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1)
        with torch.no_grad():
            self.conv.weight.copy_(
                torch.randn(self.conv.weight.shape, generator=generator)
            )
            self.conv.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def striped_images(count: int, stripe: int = 4) -> np.ndarray:
    """Random images where image i has a no-data stripe at rows i*stripe, so each
    masked output identifies which input it came from."""
    rng = np.random.default_rng(3)
    data = rng.random((count, 3, IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32) + 0.1
    for i in range(count):
        data[i, :, i * stripe : (i + 1) * stripe, :] = 0
    return data


@pytest.mark.parametrize("device", ALL_DEVICES)
def test_predict_from_batch_outputs_align_with_inputs(device: str) -> None:
    # more batches than any planned ring of reusable buffers
    count, stripe = 12, 4
    result = predict_from_batch(
        striped_images(count, stripe),
        batch_size=2,
        inference_device=device,
        export_confidence=True,
    )
    for i in range(count):
        rows = np.zeros(IMAGE_SIZE, dtype=bool)
        rows[i * stripe : (i + 1) * stripe] = True
        assert np.all(result[i][:, rows] == 0), f"image {i} stripe not masked"
        # softmax confidence is clipped to at least 0.001 on valid pixels
        assert np.all(result[i][:, ~rows] > 0), f"image {i} has another's mask"


@pytest.mark.parametrize(
    "make_data, device",
    [
        pytest.param(lambda x: x, "cpu", id="numpy"),
        pytest.param(torch.from_numpy, "cpu", id="cpu_tensor"),
        pytest.param(
            lambda x: torch.from_numpy(x).cuda(),
            "cuda",
            id="cuda_tensor",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
    ],
)
def test_predict_from_batch_does_not_modify_input(make_data, device: str) -> None:
    data = make_data(striped_images(5))
    before = data.clone() if isinstance(data, torch.Tensor) else data.copy()

    predict_from_batch(
        data, batch_size=2, inference_device=device, export_confidence=True
    )

    if isinstance(data, torch.Tensor):
        assert torch.equal(data, before)
    else:
        np.testing.assert_array_equal(data, before)


@pytest.mark.parametrize(
    "dtype, low, high, no_data_value",
    [
        pytest.param(np.uint8, 1, 255, 0, id="uint8"),
        # values above 32767 catch sign errors if uint16 is reinterpreted as int16
        pytest.param(np.uint16, 30000, 65535, 65535, id="uint16_high"),
        pytest.param(np.int16, -5000, 5000, -9999, id="int16_negative"),
        pytest.param(np.float32, 0, 1, 0, id="float32"),
        pytest.param(np.float64, 0, 1, 0, id="float64"),
    ],
)
def test_predict_from_batch_input_dtypes_match_float32(
    dtype, low, high, no_data_value
) -> None:
    rng = np.random.default_rng(1)
    shape = (3, 3, IMAGE_SIZE, IMAGE_SIZE)
    if np.issubdtype(dtype, np.integer):
        data = rng.integers(low, high, size=shape).astype(dtype)
    else:
        data = rng.uniform(low, high, size=shape).astype(dtype) + 0.01
    data[0, :, :8, :] = no_data_value

    kwargs = {
        "batch_size": 2,
        "inference_device": "cpu",
        "export_confidence": True,
        "no_data_value": no_data_value,
    }
    result = predict_from_batch(data, **kwargs)
    expected = predict_from_batch(data.astype(np.float32), **kwargs)

    np.testing.assert_array_equal(result, expected)
    assert np.all(result[0, :, :8, :] == 0), "no-data value not detected"


def _non_contiguous_cases() -> dict:
    rng = np.random.default_rng(2)
    big = rng.random((5, 3, 2 * IMAGE_SIZE, 2 * IMAGE_SIZE)).astype(np.float32)
    small = big[:, :, :IMAGE_SIZE, :IMAGE_SIZE].copy()
    return {
        "strided": big[:, :, ::2, ::2],
        "slice_of_larger": big[1:4, :, 10 : 10 + IMAGE_SIZE, 20 : 20 + IMAGE_SIZE],
        "negative_stride": small[:, :, ::-1, :],
        "fortran_order": np.asfortranarray(small),
        "transposed_tensor": torch.from_numpy(small).transpose(2, 3),
    }


@pytest.mark.parametrize("case", list(_non_contiguous_cases()))
def test_predict_from_batch_non_contiguous_inputs(case: str) -> None:
    data = _non_contiguous_cases()[case]
    contiguous = (
        data.contiguous()
        if isinstance(data, torch.Tensor)
        else np.ascontiguousarray(data)
    )

    for export_confidence in [False, True]:
        kwargs = {
            "batch_size": 2,
            "inference_device": "cpu",
            "export_confidence": export_confidence,
        }
        result = predict_from_batch(data, **kwargs)
        expected = predict_from_batch(contiguous, **kwargs)
        if export_confidence:
            # memory layout can change which conv kernel runs, so allow float noise
            np.testing.assert_allclose(result, expected, rtol=0, atol=1e-6)
        else:
            np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("softmax_output", [True, False])
def test_predict_from_batch_confidence_matches_predict_from_array(
    images: np.ndarray, softmax_output: bool
) -> None:
    data = images[:3].copy()
    data[1, :, :8, :] = 0
    kwargs = {
        "inference_device": "cpu",
        "export_confidence": True,
        "softmax_output": softmax_output,
    }

    result = predict_from_batch(data, batch_size=2, **kwargs)
    for i, image in enumerate(data):
        expected = predict_from_array(
            image, patch_size=IMAGE_SIZE, patch_overlap=0, **kwargs
        )
        np.testing.assert_allclose(result[i], expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("device", ALL_DEVICES)
@pytest.mark.parametrize("export_confidence", [False, True])
@pytest.mark.parametrize("use_generator", [False, True])
def test_predict_from_batch_results_own_their_memory(
    images: np.ndarray, device: str, export_confidence: bool, use_generator: bool
) -> None:
    def run(data: np.ndarray) -> np.ndarray:
        source = (image for image in data) if use_generator else data
        return predict_from_batch(
            source,
            batch_size=2,
            inference_device=device,
            export_confidence=export_confidence,
        )

    first = run(images)
    snapshot = first.copy()
    run(images[::-1].copy())  # a second run must not overwrite the first result

    np.testing.assert_array_equal(first, snapshot)
    assert first.flags.owndata and first.flags.writeable


def test_predict_from_batch_generator_lookahead_is_bounded() -> None:
    batch_size, count = 2, 40
    state = {"pulled": 0, "processed": 0, "max_lookahead": 0}

    def counting_generator():
        for image in striped_images(count, stripe=1):
            state["pulled"] += 1
            yield image

    class SlowCountingModel(SeededConv):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            lookahead = state["pulled"] - state["processed"]
            state["max_lookahead"] = max(state["max_lookahead"], lookahead)
            time.sleep(0.05)  # slower than loading, so loading runs ahead
            state["processed"] += x.shape[0]
            return super().forward(x)

    predict_from_batch(
        counting_generator(),
        batch_size=batch_size,
        inference_device="cpu",
        custom_models=SlowCountingModel(),
    )

    assert state["processed"] == count
    # a fixed number of batches may be loaded ahead, never the whole input
    assert state["max_lookahead"] <= 8 * batch_size + 1, state["max_lookahead"]


def _wait_for_new_threads(baseline: set, timeout: float = 5) -> list:
    deadline = time.monotonic() + timeout
    while True:
        new = [t for t in threading.enumerate() if t not in baseline and t.is_alive()]
        if not new or time.monotonic() > deadline:
            return new
        time.sleep(0.05)


class FailOnSecondBatch(SeededConv):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("model failed")
        return super().forward(x)


@pytest.mark.parametrize("scenario", ["success", "model_error", "loader_error"])
def test_predict_from_batch_does_not_leak_threads(
    tif_paths: list[Path], scenario: str
) -> None:
    def loader(input_path: Path) -> np.ndarray:
        if scenario == "loader_error" and input_path == tif_paths[3]:
            raise RuntimeError("loader failed")
        return load_tif_array_only(input_path)

    def run() -> np.ndarray:
        model = FailOnSecondBatch() if scenario == "model_error" else SeededConv()
        return predict_from_batch(
            tif_paths,
            batch_size=1,
            load_func=loader,
            inference_device="cpu",
            custom_models=model,
        )

    # warm up once so long-lived threads (e.g. tqdm's monitor) exist beforehand
    predict_from_batch(tif_paths[:1], load_func=loader, inference_device="cpu")
    baseline = set(threading.enumerate())

    if scenario == "success":
        run_with_timeout(run)
    else:
        with pytest.raises(RuntimeError, match="failed"):
            run_with_timeout(run)

    assert _wait_for_new_threads(baseline) == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("scenario", ["success", "model_error"])
def test_predict_from_batch_releases_cuda_memory(scenario: str) -> None:
    data = striped_images(8)
    model = SeededConv()
    # warm up so the model and CUDA context are already allocated
    predict_from_batch(data, batch_size=2, inference_device="cuda", custom_models=model)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()

    if scenario == "success":
        predict_from_batch(
            data, batch_size=2, inference_device="cuda", custom_models=model
        )
    else:
        with pytest.raises(RuntimeError, match="model failed"):
            run_with_timeout(
                lambda: predict_from_batch(
                    data,
                    batch_size=2,
                    inference_device="cuda",
                    custom_models=FailOnSecondBatch(),
                )
            )

    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() <= baseline


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_predict_from_batch_accelerator_confidence_dtype(
    images: np.ndarray, device: str, dtype: str
) -> None:
    result = predict_from_batch(
        images,
        batch_size=2,
        inference_device=device,
        inference_dtype=dtype,
        export_confidence=True,
    )
    assert result.dtype == np.float32
    assert result.shape == (5, 4, IMAGE_SIZE, IMAGE_SIZE)
    assert np.all((result >= 0) & (result <= 1))


def test_predict_from_batch_mixed_numpy_and_tensor_list(
    images: np.ndarray, reference_masks: np.ndarray
) -> None:
    mixed = [
        torch.from_numpy(image) if i % 2 else image for i, image in enumerate(images)
    ]
    result = predict_from_batch(mixed, batch_size=2, inference_device="cpu")
    agreement = (result == reference_masks).mean()
    assert agreement >= 0.999, f"Only {agreement:.4f} of pixels agree"


def test_predict_from_batch_load_func_uint16_files(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    data = rng.integers(30000, 65535, size=(3, 3, IMAGE_SIZE, IMAGE_SIZE)).astype(
        np.uint16
    )
    paths = [
        write_tif(tmp_path / f"chip_{i}.tif", image) for i, image in enumerate(data)
    ]
    kwargs = {"batch_size": 2, "inference_device": "cpu", "export_confidence": True}

    result = predict_from_batch(paths, load_func=load_tif, **kwargs)
    expected = predict_from_batch(data.astype(np.float32), **kwargs)
    np.testing.assert_array_equal(result, expected)


def test_predict_from_batch_compile_warm_up_uses_image_shape(monkeypatch) -> None:
    height, width, batch_size = 56, 80, 3
    warm_up_shapes = []

    class RecordingModel(SeededConv):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            warm_up_shapes.append(tuple(x.shape))
            return super().forward(x)

    monkeypatch.setattr(
        "omnicloudmask.model_utils.torch.compile", lambda model, **kwargs: model
    )
    data = np.random.default_rng(5).random((4, 3, height, width)).astype(np.float32)
    predict_from_batch(
        data,
        batch_size=batch_size,
        inference_device="cpu",
        custom_models=RecordingModel(),
        compile_models=True,
    )

    # warm-up runs every batch size from 1 to batch_size at the image's (H, W)
    expected = [(i, 3, height, width) for i in range(1, batch_size + 1)]
    assert warm_up_shapes[:batch_size] == expected


@pytest.mark.parametrize("device", ALL_DEVICES)
def test_predict_from_batch_export_many_batches_matches_memory(
    tmp_path: Path, device: str
) -> None:
    data = striped_images(12)
    paths = [
        write_tif(tmp_path / f"chip_{i}.tif", image) for i, image in enumerate(data)
    ]
    kwargs = {"batch_size": 2, "inference_device": device, "load_func": load_tif}

    in_memory = predict_from_batch(paths, **kwargs)
    exported = predict_from_batch(
        paths, export_to_disk=True, output_dir=tmp_path / "out", **kwargs
    )

    for path, expected in zip(exported, in_memory):
        with rio.open(path) as src:
            np.testing.assert_array_equal(src.read(), expected)
