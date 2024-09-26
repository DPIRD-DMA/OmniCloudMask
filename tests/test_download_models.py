from pathlib import Path
import torch

from omnicloudmask.download_models import get_models
from omnicloudmask.model_utils import (
    load_model_from_weights,
)


def test_get_models():
    models = get_models(force_download=True)
    assert len(models) == 2
    for model in models:
        assert Path(model["Path"]).exists()
        assert Path(model["Path"]).stat().st_size > 1024 * 1024
        assert isinstance(model["timm_model_name"], str)


def test_load_model_from_weights():
    models = []
    for model_details in get_models():
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
