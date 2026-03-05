import warnings
from unittest.mock import patch

import pytest
import segmentation_models_pytorch as smp
import torch

from omnicloudmask.model_utils import build_fastai_model
from omnicloudmask.mps_patch import (
    _get_model_in_channels,
    patch_model_for_mps,
    patch_models_for_mps,
    requires_mps_fix,
)

CPU = torch.device("cpu")
MPS = torch.device("mps") if torch.backends.mps.is_available() else None
DTYPE = torch.float32

requires_mps = pytest.mark.skipif(MPS is None, reason="MPS device not available")


# --- _get_model_in_channels ---


class TestGetModelInChannels:
    @pytest.mark.parametrize("in_channels", [1, 3, 4, 6])
    def test_simple_conv_model(self, in_channels: int) -> None:
        model = torch.nn.Sequential(torch.nn.Conv2d(in_channels, 16, 3))
        assert _get_model_in_channels(model) == in_channels

    def test_no_conv2d_warns_and_returns_default(self) -> None:
        model = torch.nn.Sequential(torch.nn.Linear(10, 5))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _get_model_in_channels(model)
        assert result == 3
        assert len(w) == 1
        assert "No Conv2d layer found" in str(w[0].message)


# --- requires_mps_fix ---


class TestRequiresMpsFix:
    def test_healthy_model_returns_false(self) -> None:
        model = torch.nn.Sequential(torch.nn.Conv2d(3, 16, 3))
        assert requires_mps_fix(model, CPU, DTYPE) is False

    def test_non_contiguity_runtime_error_warns_and_returns_false(self) -> None:
        model = torch.nn.Sequential(torch.nn.Conv2d(3, 16, 3))
        with patch.object(
            model, "forward", side_effect=RuntimeError("some other MPS error")
        ):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = requires_mps_fix(model, CPU, DTYPE)
            assert result is False
            assert len(w) == 1
            assert "Unexpected error" in str(w[0].message)

    def test_contiguity_error_returns_true(self) -> None:
        model = torch.nn.Sequential(torch.nn.Conv2d(3, 16, 3))
        with patch.object(
            model,
            "forward",
            side_effect=RuntimeError(
                "view size is not compatible with input tensor's size and stride"
            ),
        ):
            assert requires_mps_fix(model, CPU, DTYPE) is True


# --- patch_model_for_mps ---


class TestPatchModelForMps:
    def test_no_patch_needed_on_cpu(self) -> None:
        model = torch.nn.Sequential(torch.nn.Conv2d(3, 16, 3))
        patched = patch_model_for_mps(model, CPU, DTYPE)
        # Model should be unchanged since no contiguity issue on CPU
        for module in patched.modules():
            if isinstance(module, torch.nn.Conv2d):
                assert type(module) is torch.nn.Conv2d


# --- smp models with various backbones and channel counts ---


class TestSmpModels:
    @pytest.mark.parametrize(
        "encoder_name,in_channels",
        [
            ("tu-edgenext_small", 3),
            ("tu-edgenext_small", 4),
            ("tu-edgenext_small", 1),
            ("resnet34", 3),
            ("resnet34", 6),
            ("mobilenet_v2", 2),
            ("efficientnet-b0", 5),
        ],
    )
    def test_channel_detection(self, encoder_name: str, in_channels: int) -> None:
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_channels,
            classes=1,
        )
        assert _get_model_in_channels(model) == in_channels

    @pytest.mark.parametrize(
        "encoder_name,in_channels",
        [
            ("tu-edgenext_small", 3),
            ("tu-edgenext_small", 4),
            ("tu-edgenext_small", 1),
            ("resnet34", 3),
            ("resnet34", 6),
            ("mobilenet_v2", 2),
            ("efficientnet-b0", 5),
        ],
    )
    def test_requires_mps_fix_returns_false_on_cpu(
        self, encoder_name: str, in_channels: int
    ) -> None:
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_channels,
            classes=1,
        )
        model.eval()
        result = requires_mps_fix(model, CPU, DTYPE)
        assert result is False


# --- fastai models with various backbones and channel counts ---


class TestFastaiModels:
    @pytest.mark.parametrize(
        "model_name,in_chans",
        [
            ("edgenext_small", 3),
            ("edgenext_small", 4),
            ("edgenext_small", 1),
            ("resnet34", 3),
            ("resnet34", 6),
            ("mobilenetv2_100", 2),
        ],
    )
    def test_channel_detection(self, model_name: str, in_chans: int) -> None:
        model = build_fastai_model(model_name=model_name, in_chans=in_chans, n_out=4)
        assert _get_model_in_channels(model) == in_chans

    @pytest.mark.parametrize(
        "model_name,in_chans",
        [
            ("edgenext_small", 3),
            ("edgenext_small", 4),
            ("edgenext_small", 1),
            ("resnet34", 3),
            ("resnet34", 6),
            ("mobilenetv2_100", 2),
        ],
    )
    def test_requires_mps_fix_returns_false_on_cpu(
        self, model_name: str, in_chans: int
    ) -> None:
        model = build_fastai_model(model_name=model_name, in_chans=in_chans, n_out=4)
        model.eval()
        result = requires_mps_fix(model, CPU, DTYPE)
        assert result is False


# --- MPS device tests (only run on Apple Silicon) ---


@requires_mps
class TestMpsDevice:
    """Tests that run on actual MPS hardware. Skipped on non-MPS machines."""

    @pytest.mark.parametrize(
        "encoder_name,in_channels",
        [
            ("tu-edgenext_small", 3),
            ("tu-edgenext_small", 4),
            ("resnet34", 3),
            ("resnet34", 6),
            ("mobilenet_v2", 2),
        ],
    )
    def test_smp_patch_on_mps(self, encoder_name: str, in_channels: int) -> None:
        assert MPS is not None
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_channels,
            classes=1,
        )
        model.eval().to(MPS)
        patched = patch_model_for_mps(model, MPS, DTYPE)
        test_input = torch.randn(1, in_channels, 65, 65, device=MPS, dtype=DTYPE)
        with torch.no_grad():
            output = patched(test_input)
        assert output.shape[0] == 1

    @pytest.mark.parametrize(
        "model_name,in_chans",
        [
            ("edgenext_small", 3),
            ("edgenext_small", 4),
            ("resnet34", 3),
            ("resnet34", 6),
            ("mobilenetv2_100", 2),
        ],
    )
    def test_fastai_patch_on_mps(self, model_name: str, in_chans: int) -> None:
        assert MPS is not None
        model = build_fastai_model(model_name=model_name, in_chans=in_chans, n_out=4)
        model.eval().to(MPS)
        patched = patch_model_for_mps(model, MPS, DTYPE)
        test_input = torch.randn(1, in_chans, 65, 65, device=MPS, dtype=DTYPE)
        with torch.no_grad():
            output = patched(test_input)
        assert output.shape[0] == 1

    def test_patch_models_for_mps(self) -> None:
        assert MPS is not None
        models = [
            smp.Unet(
                encoder_name="tu-edgenext_small",
                encoder_weights=None,
                in_channels=ch,
                classes=1,
            )
            for ch in [3, 4]
        ]
        for m in models:
            m.eval().to(MPS)
        patched = patch_models_for_mps(models, MPS, DTYPE)  # type: ignore
        for m, ch in zip(patched, [3, 4]):
            test_input = torch.randn(1, ch, 65, 65, device=MPS, dtype=DTYPE)
            with torch.no_grad():
                output = m(test_input)
            assert output.shape[0] == 1
