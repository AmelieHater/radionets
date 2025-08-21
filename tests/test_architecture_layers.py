import torch
import pytest

from radionets.architecture.layers import LocallyConnected2d

torch.manual_seed(1)


class TestLocallyConnected2d:
    """Test suite for LocallyConnected2d layer."""

    def test_initialization_without_bias(self):
        """Test layer initialization without bias."""
        layer = LocallyConnected2d(
            in_channels=3,
            out_channels=16,
            output_size=(8, 8),
            kernel_size=3,
            stride=1,
            bias=False
        )

        # Check weight shape
        expected_weight_shape = (1, 16, 3, 8, 8, 9)  # kernel_size^2 = 9
        assert layer.weight.shape == expected_weight_shape

        # Check bias is None
        assert layer.bias is None

        # Check kernel_size and stride are converted to tuples
        assert layer.kernel_size == (3, 3)
        assert layer.stride == (1, 1)

    def test_initialization_with_bias(self):
        """Test layer initialization with bias."""
        layer = LocallyConnected2d(
            in_channels=3,
            out_channels=16,
            output_size=(8, 8),
            kernel_size=3,
            stride=1,
            bias=True
        )

        # Check bias shape
        expected_bias_shape = (1, 16, 8, 8)
        assert layer.bias.shape == expected_bias_shape
        assert isinstance(layer.bias, torch.nn.Parameter)

    def test_forward_pass_basic(self):
        """Test basic forward pass functionality."""
        layer = LocallyConnected2d(
            in_channels=3,
            out_channels=16,
            output_size=(8, 8),
            kernel_size=3,
            stride=1,
            bias=False
        )

        # Input that should produce 8x8 output with kernel_size=3, stride=1
        # Input size: 10x10 -> (10-3+1)/1 = 8x8
        input_tensor = torch.randn(2, 3, 10, 10)
        output = layer(input_tensor)

        expected_output_shape = (2, 16, 8, 8)
        assert output.shape == expected_output_shape
        assert output.dtype == torch.float32

    def test_forward_pass_with_bias(self):
        """Test forward pass with bias."""
        layer = LocallyConnected2d(
            in_channels=3,
            out_channels=16,
            output_size=(8, 8),
            kernel_size=3,
            stride=1,
            bias=True
        )

        input_tensor = torch.randn(2, 3, 10, 10)
        output = layer(input_tensor)

        expected_output_shape = (2, 16, 8, 8)
        assert output.shape == expected_output_shape

    def test_different_stride(self):
        """Test with different stride values."""
        layer = LocallyConnected2d(
            in_channels=3,
            out_channels=8,
            output_size=(4, 4),
            kernel_size=3,
            stride=2,
            bias=False
        )

        # Input size: 10x10 -> (10-3+1)/2 = 4x4
        input_tensor = torch.randn(1, 3, 10, 10)
        output = layer(input_tensor)

        expected_output_shape = (1, 8, 4, 4)
        assert output.shape == expected_output_shape

    def test_single_channel_input_output(self):
        """Test with single input and output channels."""
        layer = LocallyConnected2d(
            in_channels=1,
            out_channels=1,
            output_size=(6, 6),
            kernel_size=2,
            stride=1,
            bias=False
        )

        # Input size: 7x7 -> (7-2+1)/1 = 6x6
        input_tensor = torch.randn(1, 1, 7, 7)
        output = layer(input_tensor)

        expected_output_shape = (1, 1, 6, 6)
        assert output.shape == expected_output_shape

    def test_gradient_flow(self):
        """Test that gradients flow through the layer correctly."""
        layer = LocallyConnected2d(
            in_channels=2,
            out_channels=4,
            output_size=(5, 5),
            kernel_size=2,
            stride=1,
            bias=True
        )

        input_tensor = torch.randn(1, 2, 6, 6, requires_grad=True)
        output = layer(input_tensor)

        # Create a simple loss
        loss = output.sum()
        loss.backward()

        # Check that gradients exist
        assert input_tensor.grad is not None
        assert layer.weight.grad is not None
        assert layer.bias.grad is not None

        # Check gradient shapes
        assert input_tensor.grad.shape == input_tensor.shape
        assert layer.weight.grad.shape == layer.weight.shape
        assert layer.bias.grad.shape == layer.bias.shape

    def test_batch_processing(self):
        """Test that the layer handles different batch sizes correctly."""
        layer = LocallyConnected2d(
            in_channels=3,
            out_channels=8,
            output_size=(3, 3),
            kernel_size=3,
            stride=1,
            bias=False
        )

        # Test different batch sizes
        for batch_size in [1, 4, 8, 16]:
            input_tensor = torch.randn(batch_size, 3, 5, 5)
            output = layer(input_tensor)
            expected_shape = (batch_size, 8, 3, 3)
            assert output.shape == expected_shape

    def test_deterministic_output(self):
        """Test that the same input produces the same output (deterministic)."""
        layer = LocallyConnected2d(
            in_channels=2,
            out_channels=4,
            output_size=(3, 3),
            kernel_size=2,
            stride=1,
            bias=True
        )

        input_tensor = torch.randn(1, 2, 4, 4)

        # Run forward pass twice
        output1 = layer(input_tensor)
        output2 = layer(input_tensor)

        # Outputs should be identical
        assert torch.allclose(output1, output2)

    def test_weight_initialization_range(self):
        """Test that weights are initialized with reasonable values."""
        layer = LocallyConnected2d(
            in_channels=3,
            out_channels=16,
            output_size=(8, 8),
            kernel_size=3,
            stride=1,
            bias=True
        )

        # Check that weights are not all zeros or all the same
        assert not torch.allclose(layer.weight, torch.zeros_like(layer.weight))
        assert not torch.allclose(layer.bias, torch.zeros_like(layer.bias))

        # Check that weights have reasonable variance (not too large or too small)
        weight_std = layer.weight.std().item()
        bias_std = layer.bias.std().item()

        assert 0.1 < weight_std < 10.0  # Reasonable range for standard deviation
        assert 0.1 < bias_std < 10.0

    def test_large_kernel_size(self):
        """Test with larger kernel size."""
        layer = LocallyConnected2d(
            in_channels=1,
            out_channels=2,
            output_size=(1, 1),
            kernel_size=5,
            stride=1,
            bias=False
        )

        # Input size: 5x5 -> (5-5+1)/1 = 1x1
        input_tensor = torch.randn(1, 1, 5, 5)
        output = layer(input_tensor)

        expected_output_shape = (1, 2, 1, 1)
        assert output.shape == expected_output_shape

        # Check weight shape includes kernel_size^2 = 25
        expected_weight_shape = (1, 2, 1, 1, 1, 25)
        assert layer.weight.shape == expected_weight_shape

    @pytest.mark.parametrize("in_channels,out_channels,kernel_size,stride", [
        (1, 1, 2, 1),
        (3, 8, 3, 1),
        (16, 32, 2, 2),
        (8, 16, 4, 2),
    ])
    def test_parametrized_configurations(self, in_channels, out_channels, kernel_size, stride):
        """Test various parameter combinations."""
        # Calculate expected output size for a 10x10 input
        output_h = (10 - kernel_size) // stride + 1
        output_w = (10 - kernel_size) // stride + 1
        output_size = (output_h, output_w)

        layer = LocallyConnected2d(
            in_channels=in_channels,
            out_channels=out_channels,
            output_size=output_size,
            kernel_size=kernel_size,
            stride=stride,
            bias=True
        )

        input_tensor = torch.randn(2, in_channels, 10, 10)
        output = layer(input_tensor)

        expected_shape = (2, out_channels, output_h, output_w)
        assert output.shape == expected_shape


def test_edge_cases():
    """Test edge cases and potential error conditions."""

    def test_minimum_input_size():
        """Test with minimum possible input size."""
        layer = LocallyConnected2d(
            in_channels=1,
            out_channels=1,
            output_size=(1, 1),
            kernel_size=3,
            stride=1,
            bias=False
        )

        # Minimum input size for 3x3 kernel to produce 1x1 output
        input_tensor = torch.randn(1, 1, 3, 3)
        output = layer(input_tensor)

        assert output.shape == (1, 1, 1, 1)
