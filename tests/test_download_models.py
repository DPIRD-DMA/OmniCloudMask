from pathlib import Path

import pytest
import torch

from omnicloudmask.download_models import get_models, release_version
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
        assert model["model_library"] in ["fastai", "smp"]


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
                model_library=model_details["model_library"],
            )
        )
    assert len(models) == 2
    for model in models:
        assert isinstance(model, torch.nn.Module)


# Google Drive rate-limits shared CI IP addresses
@pytest.mark.slow
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
                model_library=model_details["model_library"],
            )
        )
    assert len(models) == 2
    for model in models:
        assert isinstance(model, torch.nn.Module)


@pytest.mark.parametrize(
    "version,expected",
    [
        ("1.8.0", "1.8.0"),
        ("1.8.1.dev3+g1234abc", "1.8.1"),
        ("1.8.1.dev3+g1234abc.d20261019", "1.8.1"),
        ("2.0.0rc1", "2.0.0"),
        ("0.0.0+unknown", "0.0.0"),
    ],
)
def test_release_version(version: str, expected: str) -> None:
    """Dev builds share the model directory of their release."""
    assert release_version(version) == expected
