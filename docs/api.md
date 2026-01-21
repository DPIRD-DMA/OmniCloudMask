# API Reference

## Prediction Functions

### predict_from_array

```python
omnicloudmask.predict_from_array(
    input_array,
    patch_size=1000,
    patch_overlap=300,
    batch_size=1,
    inference_device=None,
    mosaic_device=None,
    inference_dtype=torch.float32,
    export_confidence=False,
    softmax_output=True,
    no_data_value=0,
    apply_no_data_mask=True,
    custom_models=None,
    pred_classes=4,
    destination_model_dir=None,
    model_download_source="hugging_face",
    compile_models=False,
    compile_mode="default",
    model_version=None,
)
```

Predict cloud and cloud shadow mask from a numpy array.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_array` | `np.ndarray` | *required* | Array with shape `(3, height, width)` for Red, Green, NIR bands |
| `patch_size` | `int` | `1000` | Size of patches for inference |
| `patch_overlap` | `int` | `300` | Overlap between adjacent patches |
| `batch_size` | `int` | `1` | Number of patches per batch |
| `inference_device` | `str` or `torch.device` | `None` | Device for inference (`"cpu"`, `"cuda"`, `"mps"`). Auto-detected if `None` |
| `mosaic_device` | `str` or `torch.device` | `None` | Device for patch mosaicking. Defaults to `inference_device` |
| `inference_dtype` | `torch.dtype` or `str` | `"fp32"` | Data type. Accepts `torch.float32`, `torch.float16`, `torch.bfloat16` or strings `"fp32"`, `"fp16"`, `"bf16"` |
| `export_confidence` | `bool` | `False` | Return confidence maps instead of class predictions |
| `softmax_output` | `bool` | `True` | Apply softmax to confidence output |
| `no_data_value` | `int` or `float` | `0` | Value indicating no-data pixels in input |
| `apply_no_data_mask` | `bool` | `True` | Mask no-data regions in output |
| `custom_models` | `torch.nn.Module` or `list` | `None` | Custom model(s) instead of default |
| `pred_classes` | `int` | `4` | Number of output classes (for custom models) |
| `destination_model_dir` | `str` or `Path` | `None` | Directory to cache downloaded models |
| `model_download_source` | `str` | `"hugging_face"` | Model source: `"hugging_face"` or `"google_drive"` |
| `compile_models` | `bool` | `False` | Compile models with torch.compile for faster inference |
| `compile_mode` | `str` | `"default"` | torch.compile mode |
| `model_version` | `float` | `None` | Model version (`1.0`, `2.0`, `3.0`, `4.0`). Latest if `None` |

**Returns:**

`np.ndarray` with shape `(1, height, width)` for class predictions, or `(4, height, width)` if `export_confidence=True`.

---

### predict_from_load_func

```python
omnicloudmask.predict_from_load_func(
    scene_paths,
    load_func,
    patch_size=1000,
    patch_overlap=300,
    batch_size=1,
    inference_device=None,
    mosaic_device=None,
    inference_dtype=torch.float32,
    export_confidence=False,
    softmax_output=True,
    no_data_value=0,
    overwrite=True,
    apply_no_data_mask=True,
    output_dir=None,
    custom_models=None,
    pred_classes=4,
    destination_model_dir=None,
    model_download_source="hugging_face",
    compile_models=False,
    compile_mode="default",
    model_version=None,
)
```

Predict cloud masks for multiple scene files, saving results as GeoTIFFs.

**Parameters:**

All parameters from `predict_from_array` plus:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `scene_paths` | `list[Path]` or `list[str]` | *required* | Paths to scene files or directories |
| `load_func` | `Callable` | *required* | Function that loads scene data (see Data Loaders) |
| `overwrite` | `bool` | `True` | Overwrite existing prediction files |
| `output_dir` | `str` or `Path` | `None` | Output directory. Defaults to same directory as input |

**Returns:**

`list[Path]` of output prediction file paths.

---

## Data Loaders

### load_s2

```python
omnicloudmask.load_s2(
    input_path,
    resolution=10.0,
    required_bands=None,
)
```

Load Sentinel-2 L1C or L2A scenes from `.SAFE` directories.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | `Path` or `str` | *required* | Path to `.SAFE` directory |
| `resolution` | `float` | `10.0` | Output resolution in meters (10-50) |
| `required_bands` | `list[str]` | `["B04", "B03", "B8A"]` | Band names to load (Red, Green, NIR). B8A (20 m) is used instead of B08 (10 m) for faster loading due to smaller file size |

**Returns:**

Tuple of `(np.ndarray, rasterio.Profile)`.

---

### load_ls8

```python
omnicloudmask.load_ls8(
    input_path,
    resolution=30,
    required_bands=None,
)
```

Load Landsat 8 scenes.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | `Path` or `str` | *required* | Path to Landsat scene directory |
| `resolution` | `int` | `30` | Resolution in meters (must be 30) |
| `required_bands` | `list[str]` | `["B4", "B3", "B5"]` | Band names to load (Red, Green, NIR) |

**Returns:**

Tuple of `(np.ndarray, rasterio.Profile)`.

---

### load_multiband

```python
omnicloudmask.load_multiband(
    input_path,
    resample_res=None,
    band_order=None,
)
```

Load a multiband GeoTIFF.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | `Path` or `str` | *required* | Path to GeoTIFF file |
| `resample_res` | `float` | `None` | Target resolution for resampling |
| `band_order` | `list[int]` | `None` | 1-indexed band numbers for Red, Green, NIR. Warns and defaults to `[1, 2, 3]` if not provided |

**Returns:**

Tuple of `(np.ndarray, rasterio.Profile)`.
