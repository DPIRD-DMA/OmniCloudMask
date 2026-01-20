# OmniCloudMask
[![image](https://img.shields.io/pypi/v/omnicloudmask.svg)](https://pypi.python.org/pypi/omnicloudmask)
[![image](https://static.pepy.tech/badge/omnicloudmask)](https://pepy.tech/project/omnicloudmask)
[![image](https://img.shields.io/conda/vn/conda-forge/omnicloudmask.svg)](https://anaconda.org/conda-forge/omnicloudmask)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/omnicloudmask.svg)](https://anaconda.org/conda-forge/omnicloudmask)
[![image](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://readthedocs.org/projects/omnicloudmask/badge/?version=latest)](https://omnicloudmask.readthedocs.io/)

State-of-the-art cloud and cloud shadow segmentation for high to moderate resolution satellite imagery.

Works with any imagery containing Red, Green, and NIR bands at 10-50 m resolution. Validated on Sentinel-2, Landsat 8, PlanetScope and Maxar imagery.

[Documentation](https://omnicloudmask.readthedocs.io/) | [Paper](https://www.sciencedirect.com/science/article/pii/S0034425725000987) | [Training Data Map](https://dpird-dma.github.io/OCM-training-data-map/) | [Podcast](https://www.satellite-image-deep-learning.com/p/omnicloudmask)

## Installation

```bash
pip install omnicloudmask
```

Or with [uv](https://docs.astral.sh/uv/), conda, or from source—see [installation docs](https://omnicloudmask.readthedocs.io/en/latest/installation.html).

## Quick Start

```python
import numpy as np
from omnicloudmask import predict_from_array

# Input: (3, height, width) array with Red, Green, NIR bands
input_array = np.random.rand(3, 1024, 1024).astype(np.float32)

# Output: (1, height, width) mask
# Values: 0=Clear, 1=Thick Cloud, 2=Thin Cloud, 3=Cloud Shadow
mask = predict_from_array(input_array)
```

For a Sentinel-2 scene:

```python
from pathlib import Path
from omnicloudmask import predict_from_load_func, load_s2

scene_paths = [Path("path/to/scene.SAFE")]
pred_paths = predict_from_load_func(scene_paths, load_s2)
```

See the [quickstart guide](https://omnicloudmask.readthedocs.io/en/latest/quickstart.html) for more examples.

## Try in Colab

[![Open In Colab](https://img.shields.io/badge/Try%20in%20Colab-grey?style=for-the-badge&logo=google-colab)](https://colab.research.google.com/drive/1d53lg2yiSbqhrzDWlJoS5rjHgRLRJ3WY?usp=sharing)

## How it works

[![Sensor agnostic Deep Learning with OmniCloudMask](http://img.youtube.com/vi/eoKctlbsoMs/0.jpg)](http://www.youtube.com/watch?v=eoKctlbsoMs "Sensor agnostic Deep Learning with OmniCloudMask")

## License

MIT License
