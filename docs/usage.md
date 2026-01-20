# Usage Guide

This page covers device selection, performance tuning, and advanced configuration options.

## Device Selection

OmniCloudMask automatically selects the best available device (CUDA > MPS > CPU). Override this with the `inference_device` parameter:

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

For NVIDIA GPUs, increase batch size and enable reduced precision:

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

Since OmniCloudMask works with variable resolution imagery (10-50 m), you can downscale for higher throughput with minimal accuracy loss. For Sentinel-2, process at 20 m instead of 10 m:

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

Available versions: `1.0`, `2.0`, `3.0`, `4.0` (default is latest). Versions 1-3 require the `legacy` extra.

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
