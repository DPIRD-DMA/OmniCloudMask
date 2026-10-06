# Usage Guide

This page covers device selection, performance tuning, and advanced configuration options.

(spectral-channels)=
## Spectral Channels

OmniCloudMask expects a 3-band input array in the order **Red, Green, NIR** with shape `(3, height, width)`.

### What if I don't have a NIR band?

If your imagery doesn't include a near-infrared (NIR) band, you can still use OmniCloudMask — just pass an array of zeros in place of the NIR channel. The model was trained to be robust to this and will still produce good cloud mask predictions using only the Red and Green bands.

```python
import numpy as np
from omnicloudmask import predict_from_array

# red and green are 2D arrays of shape (height, width)
nir_placeholder = np.zeros_like(red)

input_array = np.stack([red, green, nir_placeholder], axis=0)
mask = predict_from_array(input_array)
```

The same approach works with `predict_from_load_func` — just return the zero-filled NIR band from your custom loader:

```python
import numpy as np
import rasterio as rio
from omnicloudmask import predict_from_load_func

def load_rgb_only(input_path):
    with rio.open(input_path) as src:
        red = src.read(1)
        green = src.read(2)
        profile = src.profile
    nir_placeholder = np.zeros_like(red)
    return np.stack([red, green, nir_placeholder], axis=0), profile

pred_paths = predict_from_load_func(scene_paths, load_rgb_only)
```

## Device Selection

OmniCloudMask automatically selects the best available device (CUDA > MPS > CPU). See the [Performance](performance.md) page for detailed benchmarks across hardware. Override this with the `inference_device` parameter:

```python
from omnicloudmask import predict_from_array

# Force CPU
mask = predict_from_array(input_array, inference_device="cpu")

# Force CUDA
mask = predict_from_array(input_array, inference_device="cuda")

# Force Apple Silicon MPS
mask = predict_from_array(input_array, inference_device="mps")
```

## GPU Performance Optimization

For NVIDIA GPUs, increase batch size and enable reduced precision (see [Performance](performance.md) for the impact of these settings):

```python
from omnicloudmask import predict_from_load_func, load_s2

pred_paths = predict_from_load_func(
    scene_paths=scene_paths,
    load_func=load_s2,
    inference_dtype="bf16",
    batch_size=4,            # increase for GPUs with more VRAM
    compile_models=True,     # faster inference for 10+ scenes
)
```

Enable `compile_models=True` when processing many scenes. This adds startup overhead but reduces per-scene inference time by 10-20%.

## Downscale for Higher Throughput

Since OmniCloudMask works with variable resolution imagery (10-50 m, and down to 5 m with recent models), you can downscale for higher throughput at the cost of some accuracy. Within the validated 10-50 m range, finer imagery produces better results for the same ground extent (see [Spatial Context](spatial-context.md) for benchmarks); below about 10 m there is little further gain. For guidance on which resolution to run at — including why running finer than 10 m rarely helps — see [Choosing a Resolution](resolution.md). For example, to process Sentinel-2 at 20 m instead of 10 m:

```python
from functools import partial
from omnicloudmask import predict_from_load_func, load_s2

loader = partial(load_s2, resolution=20)

pred_paths = predict_from_load_func(scene_paths, loader)
```

This reduces the number of pixels by 4x, significantly improving processing speed.

## Low VRAM Configuration

If you run out of GPU memory, offload mosaicking to CPU:

```python
pred_paths = predict_from_load_func(
    scene_paths=scene_paths,
    load_func=load_s2,
    inference_dtype="bf16",
    batch_size=1,
    mosaic_device="cpu",  # offload patch mosaicking to CPU
)
```

## CPU Inference

For CPU-only systems, use full precision:

```python
import torch

# Optional: set thread count for better CPU performance
torch.set_num_threads(4)

pred_paths = predict_from_load_func(
    scene_paths=scene_paths,
    load_func=load_s2,
    inference_dtype="fp32",   # important: avoid fp16/bf16 on CPU
    batch_size=1,
    inference_device="cpu",
    mosaic_device="cpu",
)
```

## Confidence Maps

Export probability scores instead of class predictions:

```python
mask = predict_from_array(
    input_array,
    export_confidence=True,
    softmax_output=True,  # normalize to probabilities
)
# Output shape: (4, height, width) - one channel per class
```

(batch-prediction)=
## Batch Prediction for Image Datasets

`predict_from_array` and `predict_from_load_func` tile one large scene into patches. If you have many small images instead, such as the chips of an existing training dataset you want to filter for clouds, use `predict_from_batch` to run several images through the model at once:

```python
import numpy as np
from omnicloudmask import predict_from_batch

# chips: (N, 3, 512, 512) array of Red, Green, NIR images
masks = predict_from_batch(chips, batch_size=16)
# Output shape: (N, 1, 512, 512)
```

`data` can be any iterable of `(3, height, width)` arrays or tensors: a 4D numpy array or torch tensor, a list, or a generator. Every image in one call must have the same shape. If your dataset has several chip sizes, call `predict_from_batch` once per size.

Each image is predicted as a single patch, so there is no overlap between images and predictions near chip edges have less spatial context than in a tiled scene. Very small chips may not give the model enough context. See [Spatial Context](spatial-context.md).

### Loading from files

Pass a `load_func` and `data` becomes a list of file paths (or anything else your loader accepts). Files are loaded in background threads while the GPU runs inference:

```python
from functools import partial
from omnicloudmask import predict_from_batch, load_multiband

loader = partial(load_multiband, band_order=[1, 2, 4])
masks = predict_from_batch(chip_paths, batch_size=16, load_func=loader)
```

Set `export_to_disk=True` to save each prediction as a GeoTIFF next to its input (or in `output_dir`) and get the file paths back instead of an array. `overwrite=False` skips chips that already have a prediction, so an interrupted run can be resumed:

```python
pred_paths = predict_from_batch(
    chip_paths,
    batch_size=16,
    load_func=loader,
    export_to_disk=True,
    output_dir="/path/to/masks",
    overwrite=False,
)
```

### Filtering a dataset by cloud cover

No-data pixels are set to `0` in the output, which is also the Clear class. Compute cloud fractions over valid pixels only, and drop chips that are mostly no data, otherwise an empty chip counts as 0% cloud:

```python
masks = predict_from_batch(chips, batch_size=16)

valid = ~np.all(chips == 0, axis=1, keepdims=True)  # use your no_data_value
valid_pixels = valid.sum(axis=(1, 2, 3))
cloudy = np.isin(masks, [1, 2, 3]) & valid  # thick cloud, thin cloud, shadow
cloud_fraction = cloudy.sum(axis=(1, 2, 3)) / np.maximum(valid_pixels, 1)
valid_fraction = valid_pixels / valid[0].size

clear_chips = chips[(cloud_fraction < 0.05) & (valid_fraction >= 0.9)]
```

See the [batch prediction notebook](https://github.com/DPIRD-DMA/OmniCloudMask/blob/main/examples/batch_prediction.ipynb) for a full example of each input and output option.

The returned array holds every prediction in memory. For very large datasets, especially with `export_confidence=True`, call `predict_from_batch` on chunks of the dataset or use `export_to_disk=True`.

## Output Directory

Save predictions to a specific directory:

```python
pred_paths = predict_from_load_func(
    scene_paths=scene_paths,
    load_func=load_s2,
    output_dir="/path/to/output",
)
```

## Skip Existing Predictions

Avoid re-processing scenes that already have predictions:

```python
pred_paths = predict_from_load_func(
    scene_paths=scene_paths,
    load_func=load_s2,
    overwrite=False,  # skip if output file exists
)
```

(custom-data-loaders)=
## Custom Data Loaders

Create a custom loader for other sensors. The loader must return a numpy array of shape `(3, height, width)` containing Red, Green, and NIR bands, plus a rasterio profile:

```python
import numpy as np
import rasterio as rio
from omnicloudmask import predict_from_load_func

def my_loader(input_path):
    with rio.open(input_path) as src:
        # Read bands in order: Red, Green, NIR
        data = src.read([4, 3, 5])  # adjust band indices for your sensor
        profile = src.profile
    return data, profile

pred_paths = predict_from_load_func(scene_paths, my_loader)
```

## Multiband GeoTIFF Loader

For generic multiband GeoTIFFs, use the built-in `load_multiband`:

```python
from functools import partial
from omnicloudmask import predict_from_load_func, load_multiband

# Specify band order: [Red, Green, NIR] (1-indexed)
loader = partial(load_multiband, band_order=[4, 3, 5])

pred_paths = predict_from_load_func(scene_paths, loader)
```

## Model Versions

Use older model versions if needed:

```python
mask = predict_from_array(input_array, model_version=3.0)
```

Available versions: `1.0`, `2.0`, `3.0`, `4.0` (default is latest). Versions 1-3 require the {ref}`legacy extra <legacy-model-support>`.

## Custom Models

Use your own trained models:

```python
import torch

my_model = torch.load("my_model.pt")

mask = predict_from_array(
    input_array,
    custom_models=my_model,
    pred_classes=4,  # number of output classes
)
```
