# OmniCloudMask

OmniCloudMask is a Python library for state-of-the-art cloud and cloud shadow segmentation in high to moderate resolution satellite imagery.

As a successor to the CloudS2Mask library, OmniCloudMask offers higher accuracy while supporting a wide range of resolutions, sensors, and processing levels.

OmniCloudMask has been validated on Sentinel-2, PlanetScope and Landsat data and is also known to work well with Maxar data, it should work on any imagery with Red Green and NIR bands with a spatial resolution of 50 m or better.


[The OmniCloudMask paper is now published 🎉](https://www.sciencedirect.com/science/article/pii/S0034425725000987)


## Features

-   Process imagery resolutions from 10 m to 50 m, (higher resolutions can be down sampled to 10 m).
-   Any imagery processing level
-   Patch-based processing of large satellite images
-   Multi-threaded patch compilation and model inference
-   Option to export confidence maps
-   Only requires Red, Green and NIR bands
-   Known to work well with Sentinel-2, Landsat 8, PlanetScope and Maxar
-   Supports inference on cuda, mps and cpu

## Try in Colab

[![Colab_Button]][Link]

[Link]: https://colab.research.google.com/drive/1d53lg2yiSbqhrzDWlJoS5rjHgRLRJ3WY?usp=sharing 'Try OmniCloudMask In Colab'

[Colab_Button]: https://img.shields.io/badge/Try%20in%20Colab-grey?style=for-the-badge&logo=google-colab

## How it works
[![Sensor agnostic Deep Learning with OmniCloudMask](http://img.youtube.com/vi/eoKctlbsoMs/0.jpg)](http://www.youtube.com/watch?v=eoKctlbsoMs "Sensor agnostic Deep Learning with OmniCloudMask")

## Installation

To install the package, use one of the following command:

```bash
pip install omnicloudmask
```

```bash
pip install git+https://github.com/DPIRD-DMA/OmniCloudMask.git
```

## Docker

Alternatively you can install OmniCloudMask within a Docker container by following the [Docker instructions](docker/README.md)

## Usage

### Predict from array

To predict cloud and cloud shadow masks from a numpy array representing the Red, Green, and NIR bands, predictions are returned as a numpy array:

```python
import numpy as np
from omnicloudmask import predict_from_array

# Example input array, in practice this should be Red, Green and NIR bands
input_array = np.random.rand(3, 1024, 1024)

# Predict cloud and cloud shadow masks
pred_mask = predict_from_array(input_array)
```

### Predict from load function

To predict cloud and cloud shadow masks for a list of Sentinel-2 scenes, predictions are saved to disk along side the inputs as geotiffs, a list of prediction file paths is returned:

#### Sentinel-2
```python
from pathlib import Path
from omnicloudmask import predict_from_load_func, load_s2,

# Paths to scenes (L1C and or L2A)
scene_paths = [Path("path/to/scene1.SAFE"), Path("path/to/scene2.SAFE")]

# Predict masks for scenes
pred_paths = predict_from_load_func(scene_paths, load_s2)
```

#### Landsat
```python
from pathlib import Path
from omnicloudmask import predict_from_load_func, load_ls8,

# Paths to scenes
scene_paths = [Path("path/to/scene1"), Path("path/to/scene2")]

# Predict masks for scenes
pred_paths = predict_from_load_func(scene_paths, load_ls8)
```

## Output
- Output classes are defined by the CloudSEN12 [paper](https://www.nature.com/articles/s41597-022-01878-2) and [dataset](https://cloudsen12.github.io/) used for training.
- 0 = Clear
- 1 = Thick Cloud
- 2 = Thin Cloud
- 3 = Cloud Shadow

## Usage tips

-   If using an NVIDIA GPU make sure to increase the default 'batch_size'.
-   In most cases setting 'inference_dtype' to "bf16" should improve processing speed, if your hardware supports it.
-   If you are running out of VRAM even with a batch_size of 1 try setting the 'mosaic_device' device to 'cpu'.
-   Make sure if you are using imagery above 10 m res to downsample it before passing it to OmniCloudMask.
-   If you are processing many files try to use the 'predict_from_load_func' as it preloads data during inference, resulting in faster processing.
-   In some rare cases OmniCloudMask may fail to detect cloud if the raster data is clipped by sensor saturation or preprocessing, this results in image regions with no remaining texture to enable detection. To resolve this simply preprocess these regions and set the areas to 0, the no data value.
-   OmniCloudMask expects Red, Green and NIR bands, however if you don't have a NIR band then we have seen reasonable results passing Red Green BLUE bands into the model instead.

## Parameters

### `predict_from_load_func`

-   `scene_paths (Union[list[Path], list[str]])`: A list of paths to the scene files to be processed.
-   `load_func (Callable)`: A function to load the scene data.
-   `patch_size (int)`: Size of the patches for inference. Defaults to 1000.
-   `patch_overlap (int)`: Overlap between patches for inference. Defaults to 300.
-   `batch_size (int)`: Number of patches to process in a batch. Defaults to 1.
-   `inference_device (Union[str, torch.device])`: Device to use for inference (e.g., 'cpu', 'cuda'). Defaults to the device returned by default_device().
-   `mosaic_device (Union[str, torch.device])`: Device to use for mosaicking patches. Defaults to the device returned by default_device().
-   `inference_dtype (Union[torch.dtype, str])`: Data type for inference. Defaults to torch.float32.
-   `export_confidence (bool)`: If True, exports confidence maps instead of predicted classes. Defaults to False.
-   `softmax_output (bool)`: If True, applies a softmax to the output, only used if export_confidence = True. Defaults to True.
-   `no_data_value (int)`: Value within input scenes that specifies no data region. Defaults to 0.
-   `overwrite (bool)`: If False, skips scenes that already have a prediction file. Defaults to True.
-   `apply_no_data_mask (bool)`: If True, applies a no-data mask to the predictions. Defaults to True.
-   `output_dir (Optional[Union[Path, str]], optional)`: Directory to save the prediction files. Defaults to None. If None, the predictions will 
be saved in the same directory as the input scene.
-   `custom_models (Union[list[torch.nn.Module], torch.nn.Module], optional)`: A list or singular custom torch models to use for prediction. Defaults to [].
-   `pred_classes (int, optional)` :  Number of classes to predict. Defaults to 4, to be used with custom models. Defaults to 4.
-   `destination_model_dir (Union[str, Path, None])`: Directory to save the model weights. Defaults to None.
-   `model_download_source (str, optional)`: Source from which to download the model weights. Defaults to "google_drive", can also be "hugging_face".


### `predict_from_array`

-   `input_array (np.ndarray)`: A numpy array with shape (3, height, width) representing the Red, Green, and NIR bands.
-   `patch_size (int)`: Size of the patches for inference. Defaults to 1000.
-   `patch_overlap (int)`: Overlap between patches for inference. Defaults to 300.
-   `batch_size (int)`: Number of patches to process in a batch. Defaults to 1.
-   `inference_device (Union[str, torch.device])`: Device to use for inference (e.g., 'cpu', 'cuda'). Defaults to the device returned by default_device().
-   `mosaic_device (Union[str, torch.device])`: Device to use for mosaicking patches. Defaults to the device returned by default_device().
-   `inference_dtype (Union[torch.dtype, str])`: Data type for inference. Defaults to torch.float32.
-   `export_confidence (bool)`: If True, exports confidence maps instead of predicted classes. Defaults to False.
-   `softmax_output (bool)`: If True, applies a softmax to the output, only used if export_confidence = True. Defaults to True.
-   `no_data_value (int)`: Value within input scenes that specifies no data region. Defaults to 0.
-   `apply_no_data_mask (bool)`: If True, applies a no-data mask to the predictions. Defaults to True.
-   `custom_models (Union[list[torch.nn.Module], torch.nn.Module], optional)`: A list or singular custom torch models to use for prediction. Defaults to [].
-   `pred_classes (int, optional)` :  Number of classes to predict. Defaults to 4, to be used with custom models. Defaults to 4.
-   `destination_model_dir (Union[str, Path, None])` : Directory to save the model weights. Defaults to None.
-   `model_download_source (str, optional)`: Source from which to download the model weights. Defaults to "google_drive", can also be "hugging_face".

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss any changes.

## License

This project is licensed under the MIT License

## Acknowledgements

-   Special thanks to the [CloudSen12 project](https://cloudsen12.github.io/) for providing valuable training dataset.
