# Quickstart

This guide shows the two main ways to use OmniCloudMask: processing numpy arrays directly, or processing satellite scene files.

## Predict from a numpy array

Use `predict_from_array` when you already have image data loaded as a numpy array with Red, Green, and NIR bands:

```python
import numpy as np
from omnicloudmask import predict_from_array

# Load your data as (3, height, width) array: Red, Green, NIR bands
input_array = np.random.rand(3, 1024, 1024).astype(np.float32)

# Run prediction
mask = predict_from_array(input_array)

# mask shape: (1, height, width)
# Values: 0=Clear, 1=Thick Cloud, 2=Thin Cloud, 3=Cloud Shadow
```

## Predict from scene files

Use `predict_from_load_func` to process satellite scene files directly. The library includes loaders for Sentinel-2, Landsat 8, and multiband GeoTIFFs:

### Sentinel-2

```python
from pathlib import Path
from omnicloudmask import predict_from_load_func, load_s2

scene_paths = [
    Path("path/to/scene1.SAFE"),
    Path("path/to/scene2.SAFE"),
]

# Predictions saved as GeoTIFFs alongside input scenes
pred_paths = predict_from_load_func(scene_paths, load_s2)
```

### Landsat 8

```python
from pathlib import Path
from omnicloudmask import predict_from_load_func, load_ls8

scene_paths = [
    Path("path/to/LC08_scene1"),
    Path("path/to/LC08_scene2"),
]

pred_paths = predict_from_load_func(scene_paths, load_ls8)
```

### Multiband GeoTIFF

```python
from functools import partial
from pathlib import Path
from omnicloudmask import predict_from_load_func, load_multiband

scene_paths = [Path("path/to/image.tif")]

# Specify band order: [Red, Green, NIR] (1-indexed)
loader = partial(load_multiband, band_order=[4, 3, 5])

pred_paths = predict_from_load_func(scene_paths, loader)
```

## Try in Google Colab

[![Open In Colab](https://img.shields.io/badge/Try%20in%20Colab-grey?style=for-the-badge&logo=google-colab)](https://colab.research.google.com/drive/1d53lg2yiSbqhrzDWlJoS5rjHgRLRJ3WY?usp=sharing)

## Example notebooks

- [Sentinel-2](https://github.com/DPIRD-DMA/OmniCloudMask/blob/main/examples/Sentinel-2.ipynb)
- [Landsat HLS](https://github.com/DPIRD-DMA/OmniCloudMask/blob/main/examples/HLS.ipynb)
- [PlanetScope](https://github.com/DPIRD-DMA/OmniCloudMask/blob/main/examples/PlanetScope.ipynb)
- [PlanetScope Hyperspectral](https://github.com/DPIRD-DMA/OmniCloudMask/blob/main/examples/PlanetScope%20Hyperspectral.ipynb)
- [Maxar](https://github.com/DPIRD-DMA/OmniCloudMask/blob/main/examples/Maxar.ipynb)
