# OmniCloudMask

**State-of-the-art cloud and cloud shadow segmentation for satellite imagery**

OmniCloudMask is a Python library for cloud and cloud shadow detection in high to moderate resolution satellite imagery. It supports resolutions from 10 m to 50 m and works with imagery from Sentinel-2, Landsat, PlanetScope, Maxar, and other sensors with Red, Green, and NIR bands.

## Key Features

- Works with any imagery containing Red, Green, and NIR bands (10 m to 50 m resolution)
- Any processing level (L1C, L2A, TOA, surface reflectance, etc.)
- Validated on Sentinel-2, Landsat 8, PlanetScope, and Maxar imagery
- Supports CUDA, MPS (Apple Silicon), and CPU inference
- Optional confidence map export
- Fast inference with multi-threaded patch-based processing

## Output Classes

OmniCloudMask produces segmentation masks with four classes defined by the [CloudSEN12 dataset](https://cloudsen12.github.io/):

| Value | Class |
|-------|-------|
| 0 | Clear |
| 1 | Thick Cloud |
| 2 | Thin Cloud |
| 3 | Cloud Shadow |

## Resources

- [GitHub Repository](https://github.com/DPIRD-DMA/OmniCloudMask)
- [OmniCloudMask Paper](https://www.sciencedirect.com/science/article/pii/S0034425725000987)
- [Training Data Map](https://dpird-dma.github.io/OCM-training-data-map/)
- [Satellite Image Deep Learning Podcast](https://www.satellite-image-deep-learning.com/p/omnicloudmask)
- [Example Notebooks](https://github.com/DPIRD-DMA/OmniCloudMask/tree/main/examples)

```{toctree}
:maxdepth: 2
:caption: Contents

installation
quickstart
how-it-works
usage
api
troubleshooting
contributing
changelog
model-changelog
```
