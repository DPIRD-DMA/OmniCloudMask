# Changelog


## [Unreleased]

### Added
- `predict_from_batch` for predicting many same-sized images (e.g. dataset chips) several per batch, from arrays, tensors or file paths via a `load_func`, with optional GeoTIFF export
- `channel_norm_torch`, a batched GPU version of `channel_norm`
- Test for no-data masking with a NaN `no_data_value`

### Changed
- `pairwise_argmax` and `optimized_argmax` now support any class dimension
- `compile_torch_model` accepts a `(height, width)` warm-up size
- A custom model whose class count doesn't match `pred_classes` now raises a `ValueError` explaining the mismatch, instead of an `AssertionError`

### Fixed
- `predict_from_array` no longer hangs forever when a model or patch extraction raises an error; the error is now raised to the caller
- `predict_from_load_func` now raises errors from inference and from saving predictions; previously they were printed from a background thread and the function returned paths to files that were never written
- No-data masking now works when `no_data_value` is NaN; previously no pixels were masked because NaN never compares equal

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