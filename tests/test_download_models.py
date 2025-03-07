from pathlib import Path

import torch

from omnicloudmask.download_models import get_models
from omnicloudmask.model_utils import (
    load_model_from_weights,
)


def test_get_models():
    # Test with default model directory
    models = get_models(force_download=True)
    assert len(models) == 2
    for model in models:
        assert Path(model["Path"]).exists()
        assert Path(model["Path"]).stat().st_size > 1024 * 1024
        assert isinstance(model["timm_model_name"], str)


def test_get_models_custom_dir():
    # Test with custom model directory
    model_dir = Path.cwd() / "custom_models"
    model_dir.mkdir(exist_ok=True)
    models = get_models(force_download=True, model_dir=model_dir)
    assert len(models) == 2
    for model in models:
        # Check if the model is downloaded to the custom directory
        assert Path(model["Path"]).exists()
        assert model["Path"].parent == model_dir
        assert Path(model["Path"]).stat().st_size > 1024 * 1024
        assert isinstance(model["timm_model_name"], str)


def test_get_models_custom_dir_str():
    # Test with custom model directory
    model_dir = Path.cwd() / "custom_models"
    model_dir.mkdir(exist_ok=True)
    models = get_models(force_download=True, model_dir=str(model_dir))
    assert len(models) == 2
    for model in models:
        # Check if the model is downloaded to the custom directory
        assert Path(model["Path"]).exists()
        assert model["Path"].parent == model_dir
        assert Path(model["Path"]).stat().st_size > 1024 * 1024
        assert isinstance(model["timm_model_name"], str)


def test_load_model_from_weights_hugging_face():
    # Test loading models from weights
    models = []
    for model_details in get_models(force_download=True, source="hugging_face"):
        models.append(
            load_model_from_weights(
                model_name=model_details["timm_model_name"],
                weights_path=model_details["Path"],
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
        )
    assert len(models) == 2
    for model in models:
        assert isinstance(model, torch.nn.Module)


def test_load_model_from_weights_google_drive():
    # Test loading models from weights
    models = []
    for model_details in get_models(force_download=True, source="google_drive"):
        models.append(
            load_model_from_weights(
                model_name=model_details["timm_model_name"],
                weights_path=model_details["Path"],
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
        )
    assert len(models) == 2
    for model in models:
        assert isinstance(model, torch.nn.Module)
