## Model Versions

| Version | Highlights | Model Architectures | Training Datasets |
|---------|-------------|-----------------------|-------------------|
| V4 | Further expanded dataset coverage with smaller faster smp models. | `regnety_004`, `edgenext_small` (smp + timm) | CloudSEN12 High, KappaSet, CloudSEN12 High (Planetary Computer), CloudSEN12 High SR, CloudSEN12 Scribble, CloudSEN12 2k, OCM hard negatives, OCM Scribble |
| V3 | Expanded dataset coverage for better generalization. | `regnety_004`, `edgenext_small` (fastai + timm) | CloudSEN12 High, KappaSet, CloudSEN12 High (Planetary Computer), CloudSEN12 High SR, OCM hard negatives |
| V2 | Introduced random interpolation for improved robustness. | `regnety_004`, `edgenext_small` (fastai + timm) | CloudSEN12 High |
| V1 | Baseline release supporting the OmniCloudMask paper. | `regnety_004`, `convnextv2_nano` (fastai + timm) | CloudSEN12 High |
