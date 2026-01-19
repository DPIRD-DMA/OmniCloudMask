"""Tests for optimization functions."""

import pytest
import torch

from omnicloudmask.optimizations import optimized_argmax, pairwise_argmax


class TestPairwiseArgmax:
    """Test suite for pairwise_argmax function."""

    @pytest.fixture
    def device(self):
        """Get CPU device for testing."""
        return torch.device("cpu")

    def test_keepdim_true(self, device):
        """Test with keepdim=True (default behavior)."""
        tensor = torch.randn(3, 100, 100, device=device)
        result = pairwise_argmax(tensor, dim=0, keepdim=True)

        assert result.shape == (1, 100, 100), (
            f"Expected (1, 100, 100), got {result.shape}"
        )

        # Should match torch.argmax
        expected = torch.argmax(tensor, dim=0, keepdim=True)
        assert torch.equal(result, expected), "Result doesn't match torch.argmax"

    def test_keepdim_false(self, device):
        """Test with keepdim=False."""
        tensor = torch.randn(3, 100, 100, device=device)
        result = pairwise_argmax(tensor, dim=0, keepdim=False)

        assert result.shape == (100, 100), f"Expected (100, 100), got {result.shape}"

        # Should match torch.argmax
        expected = torch.argmax(tensor, dim=0, keepdim=False)
        assert torch.equal(result, expected), "Result doesn't match torch.argmax"

    def test_keepdim_default(self, device):
        """Test that keepdim defaults to True."""
        tensor = torch.randn(4, 50, 50, device=device)
        result = pairwise_argmax(tensor)

        assert result.shape == (1, 50, 50), "Default keepdim should be True"

    def test_dim_zero_explicit(self, device):
        """Test with explicit dim=0."""
        tensor = torch.randn(5, 80, 80, device=device)
        result = pairwise_argmax(tensor, dim=0, keepdim=True)

        expected = torch.argmax(tensor, dim=0, keepdim=True)
        assert torch.equal(result, expected), "dim=0 doesn't match torch.argmax"

    def test_invalid_dim_raises_error(self, device):
        """Test that non-zero dim raises ValueError."""
        tensor = torch.randn(3, 100, 100, device=device)

        with pytest.raises(ValueError, match="only supports dim=0"):
            pairwise_argmax(tensor, dim=1)

        with pytest.raises(ValueError, match="only supports dim=0"):
            pairwise_argmax(tensor, dim=2)

        with pytest.raises(ValueError, match="only supports dim=0"):
            pairwise_argmax(tensor, dim=-1)

    def test_keepdim_with_different_sizes(self, device):
        """Test keepdim parameter with various tensor sizes."""
        sizes = [(2, 10, 10), (3, 50, 50), (4, 200, 200), (8, 100, 100)]

        for num_classes, h, w in sizes:
            tensor = torch.randn(num_classes, h, w, device=device)

            # Test keepdim=True
            result_keep = pairwise_argmax(tensor, keepdim=True)
            expected_keep = torch.argmax(tensor, dim=0, keepdim=True)
            assert result_keep.shape == (1, h, w), (
                f"Wrong shape with keepdim=True for {num_classes}x{h}x{w}"
            )
            assert torch.equal(result_keep, expected_keep), (
                f"Wrong result with keepdim=True for {num_classes}x{h}x{w}"
            )

            # Test keepdim=False
            result_no_keep = pairwise_argmax(tensor, keepdim=False)
            expected_no_keep = torch.argmax(tensor, dim=0, keepdim=False)
            assert result_no_keep.shape == (h, w), (
                f"Wrong shape with keepdim=False for {num_classes}x{h}x{w}"
            )
            assert torch.equal(result_no_keep, expected_no_keep), (
                f"Wrong result with keepdim=False for {num_classes}x{h}x{w}"
            )

    def test_keepdim_with_ties(self, device):
        """Test keepdim behavior with tie values."""
        # All equal values
        tensor = torch.ones(3, 10, 10, device=device)

        result_keep = pairwise_argmax(tensor, keepdim=True)
        expected_keep = torch.zeros(1, 10, 10, dtype=torch.int64, device=device)
        assert torch.equal(result_keep, expected_keep), (
            "keepdim=True tie handling incorrect"
        )

        result_no_keep = pairwise_argmax(tensor, keepdim=False)
        expected_no_keep = torch.zeros(10, 10, dtype=torch.int64, device=device)
        assert torch.equal(result_no_keep, expected_no_keep), (
            "keepdim=False tie handling incorrect"
        )

    @pytest.mark.parametrize("keepdim", [True, False])
    def test_negative_values_both_keepdim(self, device, keepdim):
        """Test with negative values for both keepdim settings."""
        tensor = torch.tensor(
            [
                [[-1.0, -2.0, -3.0], [-5.0, 0.0, 1.0]],
                [[-2.0, -1.0, -2.0], [-1.0, -1.0, -1.0]],
                [[-3.0, -3.0, -1.0], [0.0, 0.5, -2.0]],
            ],
            device=device,
        )

        result = pairwise_argmax(tensor, keepdim=keepdim)
        expected = torch.argmax(tensor, dim=0, keepdim=keepdim)

        assert torch.equal(result, expected), (
            f"Negative values failed with keepdim={keepdim}"
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_different_dtypes_keepdim_false(self, device, dtype):
        """Test keepdim=False with different dtypes."""
        tensor = torch.randn(4, 50, 50, device=device, dtype=dtype)

        result = pairwise_argmax(tensor, keepdim=False)
        expected = torch.argmax(tensor, dim=0, keepdim=False)

        assert result.shape == (50, 50), f"Wrong shape for dtype {dtype}"
        assert torch.equal(result, expected), f"Wrong result for dtype {dtype}"

    def test_large_tensor_keepdim_false(self, device):
        """Test keepdim=False with large tensor."""
        tensor = torch.randn(4, 1000, 1000, device=device)

        result = pairwise_argmax(tensor, keepdim=False)
        expected = torch.argmax(tensor, dim=0, keepdim=False)

        assert result.shape == (1000, 1000), "Wrong shape for large tensor"
        assert torch.equal(result, expected), "Wrong result for large tensor"


class TestOptimizedArgmax:
    """Test suite for optimized_argmax function."""

    def test_cpu_uses_pairwise(self):
        """Test that CPU device uses pairwise implementation."""
        tensor = torch.randn(3, 100, 100, device="cpu")

        result = optimized_argmax(tensor, dim=0, keepdim=True)
        expected = torch.argmax(tensor, dim=0, keepdim=True)

        assert torch.equal(result, expected), "CPU optimized_argmax doesn't match"

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="MPS not available"
    )
    def test_mps_uses_pairwise(self):
        """Test that MPS device uses pairwise implementation."""
        tensor = torch.randn(3, 100, 100, device="mps")

        result = optimized_argmax(tensor, dim=0, keepdim=True)
        expected = torch.argmax(tensor, dim=0, keepdim=True)

        assert torch.equal(result, expected), "MPS optimized_argmax doesn't match"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_uses_standard(self):
        """Test that CUDA device uses standard torch.argmax."""
        tensor = torch.randn(3, 100, 100, device="cuda")

        result = optimized_argmax(tensor, dim=0, keepdim=True)
        expected = torch.argmax(tensor, dim=0, keepdim=True)

        assert torch.equal(result, expected), "CUDA optimized_argmax doesn't match"

    def test_keepdim_false_cpu(self):
        """Test that keepdim=False falls back to torch.argmax on CPU."""
        tensor = torch.randn(3, 100, 100, device="cpu")

        # keepdim=False should fallback to torch.argmax
        result = optimized_argmax(tensor, dim=0, keepdim=False)
        expected = torch.argmax(tensor, dim=0, keepdim=False)

        assert result.shape == (100, 100), "Wrong shape with keepdim=False"
        assert torch.equal(result, expected), "keepdim=False doesn't match"

    def test_dim_not_zero_cpu(self):
        """Test that dim!=0 falls back to torch.argmax on CPU."""
        tensor = torch.randn(100, 3, 100, device="cpu")

        # dim=1 should fallback to torch.argmax
        result = optimized_argmax(tensor, dim=1, keepdim=True)
        expected = torch.argmax(tensor, dim=1, keepdim=True)

        assert torch.equal(result, expected), "dim=1 doesn't match"

    @pytest.mark.parametrize("dim", [0, 1, 2])
    @pytest.mark.parametrize("keepdim", [True, False])
    def test_all_combinations(self, dim, keepdim):
        """Test all dim and keepdim combinations."""
        tensor = torch.randn(10, 10, 10, device="cpu")

        result = optimized_argmax(tensor, dim=dim, keepdim=keepdim)
        expected = torch.argmax(tensor, dim=dim, keepdim=keepdim)

        assert torch.equal(result, expected), f"Failed for dim={dim}, keepdim={keepdim}"

    def test_default_parameters(self):
        """Test that default parameters work correctly."""
        tensor = torch.randn(4, 200, 200, device="cpu")

        result = optimized_argmax(tensor)
        expected = torch.argmax(tensor, dim=0, keepdim=True)

        assert torch.equal(result, expected), "Default parameters don't match"
