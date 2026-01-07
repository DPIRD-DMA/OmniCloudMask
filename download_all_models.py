#!/usr/bin/env python3
"""
One-time script to download all available OmniCloudMask models.

Usage:
    python download_all_models.py /path/to/models
"""

from pathlib import Path

from omnicloudmask.download_models import get_models

AVAILABLE_VERSIONS = [1.0, 2.0, 3.0]

if __name__ == "__main__":


    model_path = Path("models").resolve()
    model_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading all OmniCloudMask models to: {model_path}")

    for version in AVAILABLE_VERSIONS:
        print(f"\n{'='*50}")
        print(f"Downloading models for version {version}")
        print('='*50)

        models = get_models(force_download=True, model_dir=model_path, model_version=version)

        for model in models:
            print(f"  ✓ {model['Path'].name}")

    print(f"\n{'='*50}")
    print("All models downloaded successfully!")
    print('='*50)
