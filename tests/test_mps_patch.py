"""Tests for MPS-specific patches and optimizations."""

import numpy as np
import pytest
import torch

from omnicloudmask.mps_patch import fast_argmax_mps


class TestFastArgmaxMPS:
    """Test suite for fast_argmax_mps function."""

    @pytest.fixture
    def device(self):
        """Get the appropriate device for testing."""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def test_matches_torch_argmax_2_classes(self, device):
        """Test that fast_argmax_mps matches torch.argmax for 2 classes."""
        tensor = torch.randn(2, 1000, 1000, device=device, dtype=torch.bfloat16)

        result_fast = fast_argmax_mps(tensor)
        result_torch = torch.argmax(tensor, dim=0, keepdim=True).to(dtype=torch.uint8)

        assert torch.equal(result_fast, result_torch), "Results don't match for 2 classes"

    def test_matches_torch_argmax_3_classes(self, device):
        """Test that fast_argmax_mps matches torch.argmax for 3 classes."""
        tensor = torch.randn(3, 1000, 1000, device=device, dtype=torch.bfloat16)

        result_fast = fast_argmax_mps(tensor)
        result_torch = torch.argmax(tensor, dim=0, keepdim=True).to(dtype=torch.uint8)

        assert torch.equal(result_fast, result_torch), "Results don't match for 3 classes"

    def test_matches_torch_argmax_4_classes(self, device):
        """Test that fast_argmax_mps matches torch.argmax for 4 classes."""
        tensor = torch.randn(4, 1000, 1000, device=device, dtype=torch.bfloat16)

        result_fast = fast_argmax_mps(tensor)
        result_torch = torch.argmax(tensor, dim=0, keepdim=True).to(dtype=torch.uint8)

        assert torch.equal(result_fast, result_torch), "Results don't match for 4 classes"

    def test_matches_torch_argmax_8_classes(self, device):
        """Test that fast_argmax_mps matches torch.argmax for 8 classes."""
        tensor = torch.randn(8, 500, 500, device=device, dtype=torch.bfloat16)

        result_fast = fast_argmax_mps(tensor)
        result_torch = torch.argmax(tensor, dim=0, keepdim=True).to(dtype=torch.uint8)

        assert torch.equal(result_fast, result_torch), "Results don't match for 8 classes"

    def test_ties_return_lowest_index(self, device):
        """Test that ties return the lowest index (matching argmax behavior)."""
        # Create tensor where all values are equal
        tensor = torch.ones(3, 10, 10, device=device, dtype=torch.bfloat16)

        result = fast_argmax_mps(tensor)

        # All indices should be 0 (first class)
        expected = torch.zeros(1, 10, 10, device=device, dtype=torch.uint8)
        assert torch.equal(result, expected), "Ties should return index 0"

    def test_ties_mixed_values(self, device):
        """Test tie-breaking with mixed values."""
        tensor = torch.tensor([
            [[1.0, 1.0, 2.0], [1.0, 1.0, 1.0]],
            [[1.0, 2.0, 1.0], [1.0, 1.0, 1.0]],
            [[2.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        ], device=device, dtype=torch.bfloat16)

        result_fast = fast_argmax_mps(tensor)
        result_torch = torch.argmax(tensor, dim=0, keepdim=True).to(dtype=torch.uint8)

        assert torch.equal(result_fast, result_torch), "Tie-breaking doesn't match torch.argmax"

    def test_negative_values(self, device):
        """Test with negative values."""
        tensor = torch.tensor([
            [[-1.0, -2.0, -3.0], [-5.0, 0.0, 1.0]],
            [[-2.0, -1.0, -2.0], [-1.0, -1.0, -1.0]],
            [[-3.0, -3.0, -1.0], [0.0, 0.5, -2.0]]
        ], device=device, dtype=torch.bfloat16)

        result_fast = fast_argmax_mps(tensor)
        result_torch = torch.argmax(tensor, dim=0, keepdim=True).to(dtype=torch.uint8)

        assert torch.equal(result_fast, result_torch), "Negative values handled incorrectly"

    def test_zeros(self, device):
        """Test with zero values."""
        tensor = torch.tensor([
            [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        ], device=device, dtype=torch.bfloat16)

        result_fast = fast_argmax_mps(tensor)
        result_torch = torch.argmax(tensor, dim=0, keepdim=True).to(dtype=torch.uint8)

        assert torch.equal(result_fast, result_torch), "Zero values handled incorrectly"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_different_dtypes(self, device, dtype):
        """Test with different floating point dtypes."""
        tensor = torch.randn(4, 500, 500, device=device, dtype=dtype)

        result_fast = fast_argmax_mps(tensor)
        result_torch = torch.argmax(tensor, dim=0, keepdim=True).to(dtype=torch.uint8)

        assert torch.equal(result_fast, result_torch), f"Results don't match for dtype {dtype}"

    def test_output_shape(self, device):
        """Test that output has correct shape."""
        tensor = torch.randn(4, 100, 200, device=device, dtype=torch.bfloat16)

        result = fast_argmax_mps(tensor)

        assert result.shape == (1, 100, 200), f"Expected shape (1, 100, 200), got {result.shape}"

    def test_output_dtype(self, device):
        """Test that output has correct dtype (int64 to match torch.argmax)."""
        tensor = torch.randn(4, 100, 100, device=device, dtype=torch.bfloat16)

        result = fast_argmax_mps(tensor)

        assert result.dtype == torch.int64, f"Expected dtype int64, got {result.dtype}"

    def test_large_tensor(self, device):
        """Test with production-size tensor."""
        # Simulate real production tensor size (4 classes, 10000x5000)
        tensor = torch.randn(4, 10000, 5000, device=device, dtype=torch.bfloat16)

        result_fast = fast_argmax_mps(tensor)
        result_torch = torch.argmax(tensor, dim=0, keepdim=True).to(dtype=torch.uint8)

        # For large tensors, compare as numpy arrays
        result_fast_np = result_fast.cpu().numpy()
        result_torch_np = result_torch.cpu().numpy()

        assert np.array_equal(result_fast_np, result_torch_np), \
            "Results don't match for large production-size tensor"

    def test_deterministic(self, device):
        """Test that results are deterministic."""
        tensor = torch.randn(4, 500, 500, device=device, dtype=torch.bfloat16)

        result1 = fast_argmax_mps(tensor)
        result2 = fast_argmax_mps(tensor)

        assert torch.equal(result1, result2), "Results are not deterministic"

    @pytest.mark.parametrize("num_classes", [2, 3, 4, 5, 8, 16])
    def test_various_class_counts(self, device, num_classes):
        """Test with various numbers of classes."""
        tensor = torch.randn(num_classes, 100, 100, device=device, dtype=torch.bfloat16)

        result_fast = fast_argmax_mps(tensor)
        result_torch = torch.argmax(tensor, dim=0, keepdim=True).to(dtype=torch.uint8)

        assert torch.equal(result_fast, result_torch), \
            f"Results don't match for {num_classes} classes"

    def test_extreme_values(self, device):
        """Test with extreme values."""
        tensor = torch.tensor([
            [[1e6, -1e6, 0.0], [1e-6, -1e-6, 0.0]],
            [[0.0, 1e6, -1e6], [0.0, 1e-6, -1e-6]],
            [[-1e6, 0.0, 1e6], [-1e-6, 0.0, 1e-6]]
        ], device=device, dtype=torch.float32)  # Use float32 for extreme values

        result_fast = fast_argmax_mps(tensor)
        result_torch = torch.argmax(tensor, dim=0, keepdim=True).to(dtype=torch.uint8)

        assert torch.equal(result_fast, result_torch), "Extreme values handled incorrectly"
