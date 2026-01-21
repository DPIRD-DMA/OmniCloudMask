# How It Works

OmniCloudMask achieves sensor-agnostic cloud and cloud shadow detection through two key innovations: **dynamic Z-score normalization** and **mixed resolution training**. These techniques enable a model trained solely on Sentinel-2 data to generalize across Landsat, PlanetScope, Maxar, and other sensors without requiring sensor-specific training data.

## The Challenge

Deep learning models for remote sensing typically only work well on the same sensor, resolution, and processing level they were trained on. This creates a problem: every new sensor requires collecting new training data and training new models. OCM solves this by making the training approach itself sensor-agnostic.

## Dynamic Z-score Normalization

Traditional deep learning normalizes inputs using statistics calculated from the entire training dataset. This works poorly across sensors because pixel value distributions vary dramatically between platforms and processing levels (L1C vs L2A, TOA vs surface reflectance).

**Dynamic Z-score normalization** instead normalizes each input patch independently:
- Calculate mean and standard deviation from each patch
- Normalize each channel to zero mean and unit standard deviation
- Apply this per-patch, per-channel normalization at both training and inference

This is analogous to applying a local histogram stretch when zooming into a scene in GIS software. The result is that imagery from any sensor gets normalized to a common representation, regardless of its original value range.

```
# Conceptually, for each patch and channel:
normalized = (patch - patch.mean()) / patch.std()
```

This approach works because for cloud detection, **image texture and spatial relationships matter more than absolute pixel values**. Clouds look like clouds regardless of whether pixel values range from 0-10000 or 0-1.

A practical benefit: **your input data can be raw values or pre-normalized**. If you've already normalized imagery for another model, OCM will still work correctly since it re-normalizes each patch independently.

## Mixed Resolution Training

To handle different sensor resolutions (10m Sentinel-2, 30m Landsat, 3m PlanetScope), OCM uses **mixed resolution training**:

- Training data is randomly resampled to resolutions between 9m and 50m during training
- This teaches the model to recognize clouds and shadows across different spatial scales
- Higher resolution imagery (like 3m PlanetScope) is resampled to 10m for inference

This approach not only enables cross-sensor generalization but actually **improves accuracy on the native training resolution** by acting as a data augmentation technique.

## Why These Techniques Matter

The techniques used in OCM are not specific to cloud masking. They represent a general approach for training sensor-agnostic remote sensing models:

1. **Dynamic Z-score normalization** decouples the model from sensor-specific pixel value distributions
2. **Mixed resolution training** decouples the model from a fixed spatial resolution

We suggest that **other remote sensing deep learning models could benefit from similar training approaches**, potentially reducing the need for sensor-specific datasets and enabling broader applicability of trained models.

## Architecture

OCM uses an ensemble of two U-Net models with soft voting to combine predictions. The ensemble improves both accuracy and generalization compared to single models.

The model architecture has evolved over time, with each version improving both accuracy and inference throughput. See the [model changelog](model-changelog.md) for details on architecture changes between versions.

For full methodology details, see the [OmniCloudMask paper](https://www.sciencedirect.com/science/article/pii/S0034425725000987).

## Video Explanation

[![Sensor agnostic Deep Learning with OmniCloudMask](http://img.youtube.com/vi/eoKctlbsoMs/0.jpg)](http://www.youtube.com/watch?v=eoKctlbsoMs "Sensor agnostic Deep Learning with OmniCloudMask")
