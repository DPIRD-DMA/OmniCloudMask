# Troubleshooting & FAQ

## Common Issues

### Out of GPU memory (CUDA out of memory)

Reduce memory usage by:

1. Setting `batch_size=1`
2. Offloading mosaicking to CPU: `mosaic_device="cpu"`

```python
pred_paths = predict_from_load_func(
    scene_paths=scene_paths,
    load_func=load_s2,
    batch_size=1,
    mosaic_device="cpu",
)
```

### Slow CPU inference

- Use `inference_dtype="fp32"` (not fp16/bf16) on CPU
- Set PyTorch thread count:

```python
import torch
torch.set_num_threads(4)  # adjust to your CPU core count
```

### Missing cloud detections

Some clouds may not be detected if pixel values are clipped by sensor saturation or preprocessing. Areas with no texture remaining cannot be classified. To resolve:

1. Identify saturated/clipped regions in your input
2. Set those regions to `0` (the default no-data value)
3. Re-run prediction

### Model download errors

If Hugging Face downloads fail, try Google Drive as an alternative:

```python
mask = predict_from_array(
    input_array,
    model_download_source="google_drive",
)
```

Or download models manually and specify the directory:

```python
mask = predict_from_array(
    input_array,
    destination_model_dir="/path/to/models",
)
```

### Input array shape errors

OmniCloudMask expects arrays with shape `(3, height, width)`:

- Dimension 0: 3 bands (Red, Green, NIR in that order)
- Dimension 1: height in pixels
- Dimension 2: width in pixels

If your data is in `(height, width, bands)` format, transpose it:

```python
input_array = np.transpose(input_array, (2, 0, 1))
```

### Minimum image size

Images must be at least 32x32 pixels. For best results, use images at least 50x50 pixels to provide adequate spatial context. Larger images generally produce better results as they give the model more context for accurate predictions.

## Performance Tips

See the [Usage Guide](usage.md) for GPU optimization, batch size tuning, downscaling strategies, and CPU inference configuration.

## FAQ

### What bands are required?

Red, Green, and NIR bands. The model was trained on these three bands from the CloudSEN12 dataset.

### How do I interpret confidence maps?

When `export_confidence=True`, the output has shape `(4, height, width)` with one channel per class:

- Channel 0: Clear probability
- Channel 1: Thick cloud probability
- Channel 2: Thin cloud probability
- Channel 3: Cloud shadow probability

Values range from ~0.001 to ~0.999 after softmax normalization.

### Which model version should I use?

Use the default (latest) version unless you have a specific reason to use an older one. Version 4.0+ uses segmentation-models-pytorch; versions 1-3 require fastai.
