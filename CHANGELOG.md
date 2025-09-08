# Changelog


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