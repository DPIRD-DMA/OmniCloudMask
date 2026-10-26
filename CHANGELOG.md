# Changelog

## [Unreleased]

### Fixed
- `predict_from_load_func` and `predict_from_batch` could segfault, abort or hang on multi-scene Sentinel-2 L1C runs, since 1.8.0 made it far more likely. GDAL's JPEG2000 driver flushes every open dataset before decoding, including a GeoTIFF being saved on another thread. Loading and saving now take turns behind a process-wide lock; inference still overlaps both (#52)
- Predictions are written to `<name>.partial` and renamed into place, so a crash can't leave a truncated file that `overwrite=False` skips or `overwrite=True` fails to replace. The no-data mask is always stored inside the GeoTIFF, as GDAL before 3.9 otherwise wrote it to a `.msk` sidecar file

## [1.8.0] - 2026-10-19

### Added
- `predict_from_batch` for predicting many same-sized images (e.g. dataset chips) several per batch, from arrays, tensors or file paths via a `load_func`, with optional GeoTIFF export
- `channel_norm_torch`, a batched GPU version of `channel_norm`
- GitHub Actions CI: lint and tests on Linux (Python 3.10, 3.14) and macOS (Python 3.14), plus a non-blocking Python 3.15 job
- Releases from a version tag: GitHub Actions builds, publishes to PyPI with trusted publishing, and creates a GitHub release
- `slow` pytest marker for tests that need the downloaded test scenes; CI runs `pytest -m "not slow"`
- Tests for calling `predict_from_load_func` and `predict_from_batch` from several threads
- Test for no-data masking with a NaN `no_data_value`
- Exported masks carry GDAL metadata tags: the OmniCloudMask and model versions, the percentage of valid pixels in each class and total cloud, the percentage of no-data pixels, and the class legend, so scenes can be filtered without reading the rasters (thanks @becryan, #39)
- `CLASS_NAMES`, the model's output classes
- Exported masks have a colour table and confidence maps have a band description per class, so both display without styling in GIS software
- Depends on `classcounter` for counting classes

### Changed
- Confidence maps are written with DEFLATE (level 1) and a floating-point predictor instead of LZW: 36% smaller and 19% faster to write for a full Sentinel-2 tile
- Requires Python 3.10 or newer (the code already needed 3.10; the package metadata said 3.9)
- `pairwise_argmax` and `optimized_argmax` now support any class dimension
- `compile_torch_model` accepts a `(height, width)` warm-up size
- A custom model whose class count doesn't match `pred_classes` now raises a `ValueError` explaining the mismatch, instead of an `AssertionError`
- On torch 2.10 and newer, models on MPS are no longer test-run to check whether they need the Conv2d contiguity patch, since the bug it works around only affects torch 2.5–2.9 (#45)
- The package version now comes from the git tag, via setuptools-scm, instead of `omnicloudmask/__version__.py`; builds between releases get a dev version such as `1.8.1.dev3+g1234abc`
- Model weights are stored in a directory named for the release only, so dev builds of a release share the downloaded weights
- The sdist no longer includes the example notebooks, docs, training and validation folders, which setuptools-scm would otherwise add
- Building from source needs setuptools 77 or newer, which the `license` field already required, and setuptools-scm 8 or newer

### Fixed
- `compile_models=True` was slower than not compiling on MPS: about 30× slower with `batch_size` above 8 (Dynamo's recompile limit) and slower at every size with dynamic shapes. Models now compile with static shapes for `batch_size` only, short final batches are padded to reuse the compiled graph, and warm-up is faster on every device (#43)
- Eager models are no longer reloaded when `patch_size` or `batch_size` changes, and the model cache holds more configurations
- With `compile_models=True` on MPS with torch 2.5–2.9, models are checked for the Conv2d contiguity patch before compiling; checking the compiled model ran it at another shape, which compiled it again
- `predict_from_array` no longer hangs forever when a model or patch extraction raises an error; the error is now raised to the caller
- `predict_from_load_func` now raises errors from inference and from saving predictions; previously they were printed from a background thread and the function returned paths to files that were never written
- No-data masking now works when `no_data_value` is NaN; previously no pixels were masked because NaN never compares equal
- Tiled prediction with a NaN `no_data_value` now skips patches that are entirely no data, shifts patches off no-data edges and reduces the patch size for scenes that are mostly no data, as it already did for other no-data values
- NaN pixels are now always treated as no data, whatever `no_data_value` is. Previously a single NaN with the default `no_data_value=0` made the whole patch (or the whole image in `predict_from_batch`) NaN, which came out as cloud shadow in class predictions
- Calling `predict_from_array`, `predict_from_load_func` or `predict_from_batch` from several threads on MPS no longer aborts the process; work on MPS is now serialised across threads, since PyTorch's MPS backend isn't thread-safe
- `predict_from_batch` now also serialises casting and copying MPS tensor inputs, including when inference runs on the CPU, and holds the MPS lock per batch so a `load_func` can run its own MPS prediction without deadlocking

## [1.7.1] - Mar 5, 2026

### Fixed
- Fixed MPS patch failing for custom models with non-default input channel counts
- MPS compatibility check now gracefully handles unexpected errors instead of crashing

### Added
- Tests for MPS patch across smp and fastai models with various backbones and channel counts

## [1.7.0] - Dec 16, 2025

### Added
- New v4 model weights with improved performance
- Optional legacy dependency group for backward compatibility with older models

### Changed
- Migrated default model backend from fastai to segmentation-models-pytorch (smp)
- Moved fastai to optional `legacy` dependency group for continued support of older models

## [1.6.0] - Sep 8, 2025

### Added
 - Added cache to load_model_from_weights for faster model loading

### Changed
- Modified patch compilation to use a queue to avoid overloading to cpu

### Fixed
- Fixed nodata bug in get_patch

## [1.5.0] - Aug 28, 2025

### Added
- New model weights (v3) trained on larger dataset

## [1.4.1] - Jul 11, 2025

### Fixed
- Fixed broken links in README for PyPI compatibility

## [1.4.0] - Jul 11, 2025

### Added
- GDAL style (255-0) nodata mask for geotiff exports when using `apply_no_data_mask=True`

## [1.3.1] - Jul 8, 2025

### Changed
- Default model download destination now uses `platformdirs.user_data_dir`
- Exported geotiff metadata nodata value changed from 0 to None

## [1.3.0] - Jun 23, 2025

### Added
- torch.compile model compilation support

### Changed
- New model release with improved speed and robustness across different resolutions

## [1.0.0] - Jul 9, 2024

### Added
- Initial release