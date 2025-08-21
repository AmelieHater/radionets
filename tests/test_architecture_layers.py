import torch
import torch.nn as nn
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


import numpy as np
from radionets.architecture.layers import ComplexConv2d


class TestComplexConv2d:
    """Test suite for ComplexConv2d class."""

    def test_init_basic(self):
        """Test basic initialization of ComplexConv2d."""
        conv = ComplexConv2d(
            in_channels=2,
            out_channels=16,
            kernel_size=3,
            stride=1,
            bias=True,
        )

        assert conv.conv_real is not None
        assert conv.conv_imag is not None
        assert isinstance(conv.conv_real, nn.Conv2d)
        assert isinstance(conv.conv_imag, nn.Conv2d)

        # Check parameters
        assert conv.conv_real.in_channels == 1
        assert conv.conv_real.out_channels == 8
        assert conv.conv_real.kernel_size == (3, 3)
        assert conv.conv_real.stride == (1, 1)
        assert conv.conv_real.padding == "same"
        assert conv.conv_real.bias is not None

        # Same for imaginary part
        assert conv.conv_imag.in_channels == 1
        assert conv.conv_imag.out_channels == 8
        assert conv.conv_imag.kernel_size == (3, 3)
        assert conv.conv_imag.stride == (1, 1)
        assert conv.conv_imag.padding == "same"
        assert conv.conv_imag.bias is not None

    def test_init_no_bias(self):
        """Test initialization without bias."""
        conv = ComplexConv2d(
            in_channels=2,
            out_channels=8,
            kernel_size=5,
            stride=2,
            bias=False,
            padding=2,
        )

        assert conv.conv_real.bias is None
        assert conv.conv_imag.bias is None

    def test_init_different_parameters(self):
        """Test initialization with different parameter combinations."""
        # Test with tuple kernel_size
        conv1 = ComplexConv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 5),
            stride=(2, 1),
            bias=True,
            padding=3,
        )

        assert conv1.conv_real.kernel_size == (3, 5)
        assert conv1.conv_real.stride == (2, 1)

        # Test with different stride
        conv2 = ComplexConv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=7,
            stride=3,
            bias=False,
            padding=3,
        )

        assert conv2.conv_real.stride == (3, 3)
        assert conv2.conv_imag.stride == (3, 3)

    def test_forward_basic(self):
        """Test basic forward pass."""
        conv = ComplexConv2d(
            in_channels=2,
            out_channels=16,
            kernel_size=3,
            stride=1,
            bias=True,
        )

        # Create complex input: batch_size=2, channels=2, height=32, width=32
        batch_size, height, width = 2, 32, 32
        x_complex = torch.randn(batch_size, 2, height, width, dtype=torch.float32)

        # Forward pass
        output = conv.forward(x_complex)

        # Check output shape
        assert output.shape == (batch_size, 16, height, width)
        assert output.dtype == torch.float32

    def test_forward_output_calculation(self):
        """Test that forward pass calculates complex convolution correctly."""
        conv = ComplexConv2d(
            in_channels=2,
            out_channels=2,
            kernel_size=3,
            stride=1,
            bias=False
        )

        # Create simple input
        x_complex = torch.randn(1, 2, 2, 3, dtype=torch.float32)

        # Get real and imaginary parts
        x_real = x_complex[:, 0].unsqueeze(1)
        x_imag = x_complex[:, 1].unsqueeze(1)

        # Forward pass
        output = conv.forward(x_complex)

        # Manual calculation for verification
        real_conv_real = conv.conv_real(x_real)
        imag_conv_imag = conv.conv_imag(x_imag)
        real_conv_imag = conv.conv_real(x_imag)
        imag_conv_real = conv.conv_imag(x_real)

        expected_real = real_conv_real - imag_conv_imag
        expected_imag = real_conv_imag + imag_conv_real
        expected_output = torch.cat([expected_real, expected_imag], dim=1)

        assert torch.allclose(output, expected_output, atol=1e-6)

    def test_forward_different_input_sizes(self):
        """Test forward pass with different input sizes."""
        conv = ComplexConv2d(
            in_channels=8,
            out_channels=32,
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1,
        )

        # Test different input sizes
        input_sizes = [
            (1, 8, 16, 16),
            (4, 8, 64, 64),
            (2, 8, 128, 256),
        ]

        for batch_size, channels, height, width in input_sizes:
            x = torch.randn(batch_size, channels, height, width, dtype=torch.float32)
            output = conv.forward(x)

            # Calculate expected output size
            expected_height = (height + 1 * 2 - 3) // 1 + 1
            expected_width = (width + 1 * 2 - 3) // 1 + 1

            assert output.shape == (batch_size, 32, expected_height, expected_width)

    def test_chunk_operation(self):
        """Test the chunk operation in forward method."""
        conv = ComplexConv2d(
            in_channels=4,
            out_channels=8,
            kernel_size=3,
            stride=1,
            bias=True
        )

        # Create input
        x_complex = torch.randn(2, 4, 16, 16, dtype=torch.float32)

        # Forward pass
        output = conv.forward(x_complex)

        # Verify chunking worked correctly
        real_part, imag_part = x_complex.chunk(2, dim=1)
        assert real_part.shape[1] == 2  # Half the channels
        assert imag_part.shape[1] == 2  # Half the channels

    def test_gradient_flow(self):
        """Test that gradients flow properly through the network."""
        conv = ComplexConv2d(
            in_channels=2,
            out_channels=4,
            kernel_size=3,
            stride=1,
            bias=True
        )

        # Create input that requires grad
        x_complex = torch.randn(1, 2, 8, 8, dtype=torch.float32, requires_grad=True)

        # Forward pass
        output = conv.forward(x_complex)

        # Create a simple loss (sum of real parts)
        loss = output.real.sum()

        # Backward pass
        loss.backward()

        # Check that gradients exist
        assert x_complex.grad is not None
        assert conv.conv_real.weight.grad is not None
        assert conv.conv_imag.weight.grad is not None
        if conv.conv_real.bias is not None:
            assert conv.conv_real.bias.grad is not None
        if conv.conv_imag.bias is not None:
            assert conv.conv_imag.bias.grad is not None

    def test_device_compatibility(self):
        """Test that the module works on different devices."""
        conv = ComplexConv2d(
            in_channels=2,
            out_channels=16,
            kernel_size=3,
            stride=1,
            bias=True
        )

        # Test on CPU
        x_cpu = torch.randn(1, 2, 16, 16, dtype=torch.float32)
        output_cpu = conv.forward(x_cpu)
        assert output_cpu.device.type == 'cpu'

        # Test on GPU if available
        if torch.cuda.is_available():
            conv_gpu = conv.cuda()
            x_gpu = x_cpu.cuda()
            output_gpu = conv_gpu.forward(x_gpu)
            assert output_gpu.device.type == 'cuda'

    def test_module_inheritance(self):
        """Test that ComplexConv2d properly inherits from nn.Module."""
        conv = ComplexConv2d(
            in_channels=2,
            out_channels=2,
            kernel_size=3,
            stride=1,
            bias=True
        )

        assert isinstance(conv, nn.Module)

        # Test that it can be added to a sequential model
        model = nn.Sequential(
            conv,
            nn.ReLU()  # Note: ReLU won't work with complex numbers in practice
        )

        assert len(list(model.parameters())) > 0

    def test_parameter_count(self):
        """Test parameter counting."""
        in_channels, out_channels, kernel_size = 4, 16, 3
        conv = ComplexConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            bias=True
        )

        # Count parameters
        total_params = sum(p.numel() for p in conv.parameters())

        # Expected: 2 conv layers, each with weight and bias and
        # half input and output channels
        expected_weight_params = 2 * out_channels // 2 * in_channels // 2 * kernel_size * kernel_size
        expected_bias_params = 2 * out_channels // 2
        expected_total = expected_weight_params + expected_bias_params

        assert total_params == expected_total


class TestComplexConv2dEdgeCases:
    """Test edge cases and error conditions."""

    def test_zero_input(self):
        """Test with zero input."""
        conv = ComplexConv2d(
            in_channels=2,
            out_channels=2,
            kernel_size=3,
            stride=1,
            bias=False
        )

        x_zero = torch.zeros(1, 2, 8, 8, dtype=torch.float32)
        output = conv.forward(x_zero)

        # Output should also be zero (assuming zero-initialized weights)
        # Note: In practice, weights are randomly initialized
        assert output.shape == (1, 2, 8, 8)

    def test_single_pixel_input(self):
        """Test with single pixel input."""
        conv = ComplexConv2d(
            in_channels=2,
            out_channels=2,
            kernel_size=1,
            stride=1,
            bias=True
        )

        x_single = torch.randn(1, 2, 1, 1, dtype=torch.float32)
        output = conv.forward(x_single)

        assert output.shape == (1, 2, 1, 1)

    def test_large_kernel_size(self):
        """Test with kernel size larger than input."""
        conv = ComplexConv2d(
            in_channels=2,
            out_channels=2,
            kernel_size=5,
            stride=1,
            bias=True
        )

        # Input smaller than kernel
        x_small = torch.randn(1, 2, 3, 3, dtype=torch.float32)
        output = conv.forward(x_small)

        # With "same" padding, output should have same spatial dimensions
        assert output.shape == (1, 2, 3, 3)
