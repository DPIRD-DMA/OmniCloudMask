## Model Versions

| Version | Highlights | Model Architectures | Training Datasets Added |
|---------|-------------|-----------------------|-------------------|
| V4 | Further expanded dataset coverage with smaller faster smp models. | `regnety_004`, `edgenext_small` (smp + timm) | CloudSEN12 Scribble, CloudSEN12 2k, OCM Scribble |
| V3 | Expanded dataset coverage for better generalization. | `regnety_004`, `edgenext_small` (fastai + timm) | CloudSEN12 High (Planetary Computer), CloudSEN12 High SR, KappaSet, OCM Hard Negatives |
| V2 | Introduced random interpolation for improved robustness. | `regnety_004`, `edgenext_small` (fastai + timm) | None |
| V1 | Baseline release supporting the OmniCloudMask paper. | `regnety_004`, `convnextv2_nano` (fastai + timm) | CloudSEN12 High |

Each version is also trained on all the datasets of the versions before it. See [Training Data](https://omnicloudmask.readthedocs.io/en/latest/how-it-works.html#training-data) for the full dataset-by-version table.
