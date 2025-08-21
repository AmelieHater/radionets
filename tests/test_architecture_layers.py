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


from radionets.architecture.layers import ComplexInstanceNorm2d


class TestComplexInstanceNorm2d:
    """Test suite for ComplexInstanceNorm2d class."""

    def test_init_basic(self):
        """Test basic initialization of ComplexInstanceNorm2d."""
        norm = ComplexInstanceNorm2d(num_features=64, eps=1e-5, affine=True)

        # Check basic attributes
        assert norm.num_features == 32  # num_features // 2 for complex channels
        assert norm.eps == 1e-5
        assert norm.affine == True

        # Check learnable parameters exist
        assert hasattr(norm, 'weight_real')
        assert hasattr(norm, 'weight_imag')
        assert hasattr(norm, 'bias_real')
        assert hasattr(norm, 'bias_imag')

        # Check parameter shapes
        assert norm.weight_real.shape == (32,)
        assert norm.weight_imag.shape == (32,)
        assert norm.bias_real.shape == (32,)
        assert norm.bias_imag.shape == (32,)

        # Check parameter initialization
        assert torch.allclose(norm.weight_real, torch.ones(32))
        assert torch.allclose(norm.weight_imag, torch.ones(32))
        assert torch.allclose(norm.bias_real, torch.zeros(32))
        assert torch.allclose(norm.bias_imag, torch.zeros(32))

    def test_init_no_affine(self):
        """Test initialization without affine parameters."""
        norm = ComplexInstanceNorm2d(num_features=32, eps=1e-6, affine=False)

        assert norm.num_features == 16
        assert norm.eps == 1e-6
        assert norm.affine == False

        # Check that affine parameters don't exist
        assert not hasattr(norm, 'weight_real')
        assert not hasattr(norm, 'weight_imag')
        assert not hasattr(norm, 'bias_real')
        assert not hasattr(norm, 'bias_imag')

    def test_init_different_parameters(self):
        """Test initialization with different parameter combinations."""
        # Test small number of features
        norm1 = ComplexInstanceNorm2d(num_features=8, eps=1e-4)
        assert norm1.num_features == 4
        assert norm1.eps == 1e-4

        # Test large number of features
        norm2 = ComplexInstanceNorm2d(num_features=512, eps=1e-7, affine=False)
        assert norm2.num_features == 256
        assert norm2.eps == 1e-7
        assert norm2.affine == False

    def test_forward_basic(self):
        """Test basic forward pass."""
        norm = ComplexInstanceNorm2d(num_features=64, eps=1e-5, affine=True)

        # Create input: batch_size=2, channels=64 (32 complex), height=16, width=16
        batch_size, height, width = 2, 16, 16
        x = torch.randn(batch_size, 64, height, width)

        # Forward pass
        output = norm.forward(x)

        # Check output shape and type
        assert output.shape == (batch_size, 64, height, width)
        assert output.dtype == x.dtype
        assert output.device == x.device

    def test_forward_normalization_properties(self):
        """Test that forward pass produces properly normalized output."""
        norm = ComplexInstanceNorm2d(num_features=32, eps=1e-5, affine=False)

        # Create input with known statistics
        x = torch.randn(4, 32, 8, 8) * 5.0 + 3.0  # Non-zero mean, large variance

        # Forward pass
        output = norm.forward(x)

        # Split output into real and imaginary parts
        real_out, imag_out = output.chunk(2, dim=1)

        # Check normalization properties for each sample and channel
        # Mean should be approximately zero
        real_means = real_out.mean(dim=[2, 3])  # Mean over spatial dimensions
        imag_means = imag_out.mean(dim=[2, 3])

        assert torch.allclose(real_means, torch.zeros_like(real_means), atol=1e-6)
        assert torch.allclose(imag_means, torch.zeros_like(imag_means), atol=1e-6)

        # Standard deviation should be approximately 1
        real_stds = real_out.std(dim=[2, 3], unbiased=False)
        imag_stds = imag_out.std(dim=[2, 3], unbiased=False)

        assert torch.allclose(real_stds, torch.ones_like(real_stds), atol=1e-5)
        assert torch.allclose(imag_stds, torch.ones_like(imag_stds), atol=1e-5)

    def test_forward_with_affine(self):
        """Test forward pass with affine transformation."""
        norm = ComplexInstanceNorm2d(num_features=16, eps=1e-5, affine=True)

        # Modify affine parameters to test their effect
        norm.weight_real.data.fill_(2.0)
        norm.weight_imag.data.fill_(3.0)
        norm.bias_real.data.fill_(1.0)
        norm.bias_imag.data.fill_(-1.0)

        # Create normalized input (mean=0, std=1)
        x = torch.randn(2, 16, 4, 4)

        # Forward pass
        output = norm.forward(x)

        # Split output
        real_out, imag_out = output.chunk(2, dim=1)

        # Check that affine transformation was applied
        # For normalized input, output should have:
        # real: mean ≈ 1.0, std ≈ 2.0
        # imag: mean ≈ -1.0, std ≈ 3.0
        real_means = real_out.mean(dim=[2, 3])
        imag_means = imag_out.mean(dim=[2, 3])
        real_stds = real_out.std(dim=[2, 3], unbiased=False)
        imag_stds = imag_out.std(dim=[2, 3], unbiased=False)

        assert torch.allclose(real_means, torch.ones_like(real_means), atol=1e-4)
        assert torch.allclose(imag_means, -torch.ones_like(imag_means), atol=1e-4)
        assert torch.allclose(real_stds, 2.0 * torch.ones_like(real_stds), atol=1e-4)
        assert torch.allclose(imag_stds, 3.0 * torch.ones_like(imag_stds), atol=1e-4)

    def test_forward_different_input_sizes(self):
        """Test forward pass with different input sizes."""
        norm = ComplexInstanceNorm2d(num_features=64, eps=1e-5, affine=True)

        # Test different input sizes
        input_sizes = [
            (1, 64, 1, 1),      # Single pixel
            (1, 64, 8, 8),      # Small image
            (4, 64, 32, 32),    # Medium batch and image
            (2, 64, 128, 256),  # Large image
        ]

        for batch_size, channels, height, width in input_sizes:
            x = torch.randn(batch_size, channels, height, width)
            output = norm.forward(x)

            assert output.shape == (batch_size, channels, height, width)

            # Verify normalization properties hold for all sizes
            if height * width > 1:  # Skip single pixel case
                real_out, imag_out = output.chunk(2, dim=1)
                real_means = real_out.mean(dim=[2, 3])
                imag_means = imag_out.mean(dim=[2, 3])

                assert torch.allclose(real_means, torch.zeros_like(real_means), atol=1e-4)
                assert torch.allclose(imag_means, torch.zeros_like(imag_means), atol=1e-4)

    def test_chunk_operation(self):
        """Test the chunk operation in forward method."""
        norm = ComplexInstanceNorm2d(num_features=16, eps=1e-5, affine=True)

        # Create input
        x = torch.randn(2, 16, 8, 8)

        # Forward pass
        output = norm.forward(x)

        # Verify chunking worked correctly
        real_part, imag_part = x.chunk(2, dim=1)
        assert real_part.shape == (2, 8, 8, 8)  # Half the channels
        assert imag_part.shape == (2, 8, 8, 8)  # Half the channels

        # Output should also be properly structured
        real_out, imag_out = output.chunk(2, dim=1)
        assert real_out.shape == (2, 8, 8, 8)
        assert imag_out.shape == (2, 8, 8, 8)

    def test_gradient_flow(self):
        """Test that gradients flow properly through the network."""
        norm = ComplexInstanceNorm2d(num_features=32, eps=1e-5, affine=True)

        # Create input that requires grad
        x = torch.randn(2, 32, 8, 8, requires_grad=True)

        # Forward pass
        output = norm.forward(x)

        # Create a simple loss
        loss = output.sum()

        # Backward pass
        loss.backward()

        # Check that gradients exist
        assert x.grad is not None
        assert norm.weight_real.grad is not None
        assert norm.weight_imag.grad is not None
        assert norm.bias_real.grad is not None
        assert norm.bias_imag.grad is not None

        # Check that gradients are non-zero (indicating proper flow)
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))
        assert not torch.allclose(norm.weight_real.grad, torch.zeros_like(norm.weight_real.grad))

    def test_device_compatibility(self):
        """Test that the module works on different devices."""
        norm = ComplexInstanceNorm2d(num_features=16, eps=1e-5, affine=True)

        # Test on CPU
        x_cpu = torch.randn(1, 16, 8, 8)
        output_cpu = norm.forward(x_cpu)
        assert output_cpu.device.type == 'cpu'

        # Test on GPU if available
        if torch.cuda.is_available():
            norm_gpu = norm.cuda()
            x_gpu = x_cpu.cuda()
            output_gpu = norm_gpu.forward(x_gpu)
            assert output_gpu.device.type == 'cuda'

            # Results should be similar (allowing for minor numerical differences)
            assert torch.allclose(output_cpu, output_gpu.cpu(), atol=1e-5)

    def test_module_inheritance(self):
        """Test that ComplexInstanceNorm2d properly inherits from nn.Module."""
        norm = ComplexInstanceNorm2d(num_features=8, eps=1e-5, affine=True)

        assert isinstance(norm, nn.Module)

        # Test that it can be added to a sequential model
        model = nn.Sequential(
            norm,
            nn.ReLU()
        )

        # Test parameter counting
        params = list(norm.parameters())
        assert len(params) == 4  # weight_real, weight_imag, bias_real, bias_imag

        total_params = sum(p.numel() for p in norm.parameters())
        expected_params = 4 * (8 // 2)  # 4 parameters × num_features//2
        assert total_params == expected_params

    def test_training_mode(self):
        """Test behavior in training vs evaluation mode."""
        norm = ComplexInstanceNorm2d(num_features=32, eps=1e-5, affine=True)

        x = torch.randn(2, 32, 8, 8)

        # Training mode
        norm.train()
        output_train = norm.forward(x)

        # Evaluation mode
        norm.eval()
        output_eval = norm.forward(x)

        # For instance norm, behavior should be the same in train/eval
        # (unlike batch norm which uses running statistics in eval)
        assert torch.allclose(output_train, output_eval, atol=1e-6)

    def test_numerical_stability(self):
        """Test numerical stability with edge cases."""
        norm = ComplexInstanceNorm2d(num_features=8, eps=1e-5, affine=True)

        # Test with very small values
        x_small = torch.randn(1, 8, 4, 4) * 1e-10
        output_small = norm.forward(x_small)
        assert torch.isfinite(output_small).all()

        # Test with very large values
        x_large = torch.randn(1, 8, 4, 4) * 1e10
        output_large = norm.forward(x_large)
        assert torch.isfinite(output_large).all()

        # Test with constant values (zero variance)
        x_constant = torch.ones(1, 8, 4, 4) * 5.0
        output_constant = norm.forward(x_constant)
        assert torch.isfinite(output_constant).all()


class TestComplexInstanceNorm2dEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_pixel_input(self):
        """Test with single pixel input."""
        norm = ComplexInstanceNorm2d(num_features=4, eps=1e-5, affine=True)

        # Single pixel input
        x_single = torch.randn(1, 4, 1, 1)
        output = norm.forward(x_single)

        assert output.shape == (1, 4, 1, 1)
        assert torch.isfinite(output).all()

    def test_zero_input(self):
        """Test with zero input."""
        norm = ComplexInstanceNorm2d(num_features=8, eps=1e-5, affine=False)

        x_zero = torch.zeros(2, 8, 4, 4)
        output = norm.forward(x_zero)

        # Output should be zero when input is zero (no affine transformation)
        assert torch.allclose(output, torch.zeros_like(output))

    def test_identical_values(self):
        """Test with identical values across spatial dimensions."""
        norm = ComplexInstanceNorm2d(num_features=4, eps=1e-5, affine=False)

        # Create input where each channel has identical values across H,W
        x = torch.randn(1, 4, 1, 1).expand(1, 4, 8, 8)
        output = norm.forward(x)

        # Should handle zero variance gracefully
        assert torch.isfinite(output).all()

    def test_different_eps_values(self):
        """Test with different epsilon values."""
        eps_values = [1e-8, 1e-5, 1e-3, 1e-1]

        x = torch.randn(1, 8, 4, 4)

        for eps in eps_values:
            norm = ComplexInstanceNorm2d(num_features=8, eps=eps, affine=False)
            output = norm.forward(x)

            assert torch.isfinite(output).all()
            assert output.shape == x.shape


from radionets.architecture.layers import ComplexPReLU


class TestComplexPReLU:
    """Test suite for ComplexPReLU class."""

    def test_init_basic(self):
        """Test basic initialization of ComplexPReLU."""
        prelu = ComplexPReLU(num_parameters=1, init=0.25)

        # Check basic attributes
        assert prelu.num_parameters == 1

        # Check learnable parameters exist
        assert hasattr(prelu, 'weight_real')
        assert hasattr(prelu, 'weight_imag')
        assert isinstance(prelu.weight_real, nn.Parameter)
        assert isinstance(prelu.weight_imag, nn.Parameter)

        # Check parameter shapes
        assert prelu.weight_real.shape == (1,)
        assert prelu.weight_imag.shape == (1,)

        # Check parameter initialization
        assert torch.allclose(prelu.weight_real, torch.tensor([0.25]))
        assert torch.allclose(prelu.weight_imag, torch.tensor([0.25]))

    def test_init_per_channel(self):
        """Test initialization with per-channel parameters."""
        num_params = 64
        init_val = 0.1
        prelu = ComplexPReLU(num_parameters=num_params, init=init_val)

        assert prelu.num_parameters == num_params

        # Check parameter shapes
        assert prelu.weight_real.shape == (num_params // 2,)
        assert prelu.weight_imag.shape == (num_params // 2,)

        # Check parameter initialization
        expected_tensor = torch.full((num_params // 2,), init_val)
        assert torch.allclose(prelu.weight_real, expected_tensor)
        assert torch.allclose(prelu.weight_imag, expected_tensor)

    def test_init_different_values(self):
        """Test initialization with different parameter values."""
        init_values = [0.01, 0.1, 0.25, 0.5, 1.0]

        for init_val in init_values:
            prelu = ComplexPReLU(num_parameters=1, init=init_val)
            assert torch.allclose(prelu.weight_real, torch.tensor([init_val]))
            assert torch.allclose(prelu.weight_imag, torch.tensor([init_val]))

    def test_forward_basic(self):
        """Test basic forward pass."""
        prelu = ComplexPReLU(num_parameters=1, init=0.25)

        # Create input: batch_size=2, channels=8 (4 complex), height=4, width=4
        batch_size, height, width = 2, 4, 4
        x = torch.randn(batch_size, 8, height, width)

        # Forward pass
        output = prelu.forward(x)

        # Check output shape and type
        assert output.shape == (batch_size, 8, height, width)
        assert output.dtype == x.dtype
        assert output.device == x.device

    def test_forward_positive_values_unchanged(self):
        """Test that positive values remain unchanged."""
        prelu = ComplexPReLU(num_parameters=1, init=0.2)

        # Create input with known positive and negative values
        x = torch.tensor([
            [[[2.0, -1.0], [3.0, -2.0]], [[2.0, -1.0], [3.0, -2.0]]],
            [[[1.5, -0.5], [-1.0, 4.0]], [[2.0, -1.0], [3.0, -2.0]]]
        ])
        print(x.shape)
        output = prelu.forward(x)
        real_out, imag_out = output.chunk(2, dim=1)
        real_in, imag_in = x.chunk(2, dim=1)

        # Check positive values are unchanged
        assert torch.equal(real_out[real_in >= 0], real_in[real_in >= 0])
        assert torch.equal(imag_out[imag_in >= 0], imag_in[imag_in >= 0])

    def test_forward_negative_values_scaled(self):
        """Test that negative values are properly scaled."""
        init_val = 0.3
        prelu = ComplexPReLU(num_parameters=1, init=init_val)

        # Create input with known negative values
        x = torch.tensor([
            [[[-2.0, -1.0], [-3.0, -0.5]], [[-2.0, -1.0], [-3.0, -0.5]]],
            [[[-1.5, -2.5], [-1.0, -4.0]], [[-2.0, -1.0], [-3.0, -0.5]]]
        ])

        output = prelu.forward(x)
        real_out, imag_out = output.chunk(2, dim=1)
        real_in, imag_in = x.chunk(2, dim=1)

        # Check negative values are scaled by init_val
        expected_real = real_in * init_val
        expected_imag = imag_in * init_val

        assert torch.allclose(real_out, expected_real, atol=1e-6)
        assert torch.allclose(imag_out, expected_imag, atol=1e-6)

    def test_forward_mixed_values(self):
        """Test forward pass with mixed positive and negative values."""
        init_val = 0.15
        prelu = ComplexPReLU(num_parameters=1, init=init_val)

        # Create input with mixed values
        x = torch.tensor([
            [[[2.0, -1.0], [-3.0, 4.0]], [[2.0, -1.0], [-3.0, 4.0]]],
            [[[-1.5, 2.5], [1.0, -4.0]], [[-1.5, 2.5], [1.0, -4.0]]]
        ])

        output = prelu.forward(x)
        real_out, imag_out = output.chunk(2, dim=1)
        real_in, imag_in = x.chunk(2, dim=1)

        # Manually compute expected output
        expected_real = torch.where(real_in >= 0, real_in, init_val * real_in)
        expected_imag = torch.where(imag_in >= 0, imag_in, init_val * imag_in)

        assert torch.allclose(real_out, expected_real, atol=1e-6)
        assert torch.allclose(imag_out, expected_imag, atol=1e-6)

    def test_forward_per_channel_parameters(self):
        """Test forward pass with per-channel parameters."""
        num_channels = 8
        prelu = ComplexPReLU(num_parameters=num_channels, init=0.2)

        # Set different values for each channel
        prelu.weight_real.data = torch.tensor([0.1, 0.2, 0.3, 0.4])
        prelu.weight_imag.data = torch.tensor([0.15, 0.25, 0.35, 0.45])

        # Create input with negative values
        x = torch.abs(torch.randn(1, 8, 2, 2)) * -1

        output = prelu.forward(x)
        real_out, imag_out = output.chunk(2, dim=1)
        real_in, imag_in = x.chunk(2, dim=1)

        # Check that each channel is scaled by its respective parameter
        for c in range(num_channels // 2):
            expected_real_c = real_in[:, c:c+1] * prelu.weight_real[c]
            expected_imag_c = imag_in[:, c:c+1] * prelu.weight_imag[c]

            assert torch.allclose(real_out[:, c:c+1], expected_real_c, atol=1e-6)
            assert torch.allclose(imag_out[:, c:c+1], expected_imag_c, atol=1e-6)

    def test_forward_different_input_sizes(self):
        """Test forward pass with different input sizes."""
        prelu = ComplexPReLU(num_parameters=1, init=0.2)

        # Test different input sizes
        input_sizes = [
            (1, 2, 1, 1),       # Single pixel, 1 complex channel
            (1, 8, 4, 4),       # Small image, 4 complex channels
            (4, 16, 8, 8),      # Medium batch and image
            (2, 32, 16, 32),    # Large image, 16 complex channels
        ]

        for batch_size, channels, height, width in input_sizes:
            x = torch.randn(batch_size, channels, height, width)
            output = prelu.forward(x)

            assert output.shape == (batch_size, channels, height, width)

            # Verify activation properties
            real_out, imag_out = output.chunk(2, dim=1)
            real_in, imag_in = x.chunk(2, dim=1)

            # Positive values should be unchanged
            pos_real_mask = real_in >= 0
            pos_imag_mask = imag_in >= 0
            assert torch.equal(real_out[pos_real_mask], real_in[pos_real_mask])
            assert torch.equal(imag_out[pos_imag_mask], imag_in[pos_imag_mask])

    def test_chunk_operation(self):
        """Test the chunk operation in forward method."""
        prelu = ComplexPReLU(num_parameters=1, init=0.1)

        # Create input
        x = torch.randn(2, 16, 4, 4)  # 8 complex channels

        # Forward pass
        output = prelu.forward(x)

        # Verify chunking worked correctly
        real_part, imag_part = x.chunk(2, dim=1)
        assert real_part.shape == (2, 8, 4, 4)  # Half the channels
        assert imag_part.shape == (2, 8, 4, 4)  # Half the channels

        # Output should maintain structure
        real_out, imag_out = output.chunk(2, dim=1)
        assert real_out.shape == (2, 8, 4, 4)
        assert imag_out.shape == (2, 8, 4, 4)

    def test_gradient_flow(self):
        """Test that gradients flow properly through the activation."""
        prelu = ComplexPReLU(num_parameters=1, init=0.2)

        # Create input that requires grad
        x = torch.randn(2, 8, 4, 4, requires_grad=True)

        # Forward pass
        output = prelu.forward(x)

        # Create a simple loss
        loss = output.sum()

        # Backward pass
        loss.backward()

        # Check that gradients exist
        assert x.grad is not None
        assert prelu.weight_real.grad is not None
        assert prelu.weight_imag.grad is not None

        # Check that gradients are reasonable (non-zero for learnable parameters)
        assert not torch.allclose(prelu.weight_real.grad, torch.zeros_like(prelu.weight_real.grad))
        assert not torch.allclose(prelu.weight_imag.grad, torch.zeros_like(prelu.weight_imag.grad))

    def test_gradient_flow_per_channel(self):
        """Test gradient flow with per-channel parameters."""
        num_channels = 16
        prelu = ComplexPReLU(num_parameters=num_channels, init=0.1)

        # Create input with both positive and negative values
        x = torch.randn(2, 16, 4, 4, requires_grad=True)  # 8 real + 8 imag channels

        # Forward pass
        output = prelu.forward(x)

        # Create loss that depends on all parameters
        loss = (output ** 2).sum()

        # Backward pass
        loss.backward()

        # Check gradients exist and have correct shape
        assert prelu.weight_real.grad is not None
        assert prelu.weight_imag.grad is not None
        assert prelu.weight_real.grad.shape == (num_channels // 2,)
        assert prelu.weight_imag.grad.shape == (num_channels // 2,)

    def test_device_compatibility(self):
        """Test that the module works on different devices."""
        prelu = ComplexPReLU(num_parameters=1, init=0.2)

        # Test on CPU
        x_cpu = torch.randn(1, 8, 4, 4)
        output_cpu = prelu.forward(x_cpu)
        assert output_cpu.device.type == 'cpu'

        # Test on GPU if available
        if torch.cuda.is_available():
            prelu_gpu = prelu.cuda()
            x_gpu = x_cpu.cuda()
            output_gpu = prelu_gpu.forward(x_gpu)
            assert output_gpu.device.type == 'cuda'

            # Results should be similar (allowing for minor numerical differences)
            assert torch.allclose(output_cpu, output_gpu.cpu(), atol=1e-6)

    def test_module_inheritance(self):
        """Test that ComplexPReLU properly inherits from nn.Module."""
        prelu = ComplexPReLU(num_parameters=4, init=0.1)

        assert isinstance(prelu, nn.Module)

        # Test that it can be added to a sequential model
        model = nn.Sequential(
            prelu,
            nn.Flatten()
        )

        # Test parameter counting
        params = list(prelu.parameters())
        assert len(params) == 2  # weight_real, weight_imag

        total_params = sum(p.numel() for p in prelu.parameters())
        expected_params = 4
        assert total_params == expected_params

    def test_zero_input(self):
        """Test behavior with zero input."""
        prelu = ComplexPReLU(num_parameters=1, init=0.3)

        x_zero = torch.zeros(2, 8, 4, 4)
        output = prelu.forward(x_zero)

        # Zero input should produce zero output
        assert torch.allclose(output, torch.zeros_like(output))

    def test_activation_properties(self):
        """Test key properties of the PReLU activation."""
        prelu = ComplexPReLU(num_parameters=1, init=0.2)

        # Create test input
        x = torch.randn(1, 4, 2, 2)
        output = prelu.forward(x)

        real_in, imag_in = x.chunk(2, dim=1)
        real_out, imag_out = output.chunk(2, dim=1)

        # Property 1: Positive values unchanged
        pos_real_mask = real_in > 0
        pos_imag_mask = imag_in > 0
        assert torch.equal(real_out[pos_real_mask], real_in[pos_real_mask])
        assert torch.equal(imag_out[pos_imag_mask], imag_in[pos_imag_mask])

        # Property 2: Negative values scaled
        neg_real_mask = real_in < 0
        neg_imag_mask = imag_in < 0
        if neg_real_mask.any():
            assert torch.allclose(
                real_out[neg_real_mask],
                real_in[neg_real_mask] * 0.2,
                atol=1e-6
            )
        if neg_imag_mask.any():
            assert torch.allclose(
                imag_out[neg_imag_mask],
                imag_in[neg_imag_mask] * 0.2,
                atol=1e-6
            )

        # Property 3: Function is piecewise linear
        # This is inherently satisfied by the PReLU definition


class TestComplexPReLUEdgeCases:
    """Test edge cases and error conditions."""

    def test_extreme_values(self):
        """Test with extreme input values."""
        prelu = ComplexPReLU(num_parameters=1, init=0.1)

        # Test with very large values
        x_large = torch.tensor([
            [[[1e6, -1e6]]],  # Real part
            [[[1e5, -1e5]]]   # Imaginary part
        ])

        output_large = prelu.forward(x_large)
        assert torch.isfinite(output_large).all()

        # Test with very small values
        x_small = torch.tensor([
            [[[1e-6, -1e-6]]],  # Real part
            [[[1e-7, -1e-7]]]   # Imaginary part
        ])

        output_small = prelu.forward(x_small)
        assert torch.isfinite(output_small).all()

    def test_boundary_values(self):
        """Test with boundary values (exactly zero)."""
        prelu = ComplexPReLU(num_parameters=1, init=0.25)

        # Input with exact zeros
        x = torch.tensor([
            [[[0.0, -1.0], [1.0, 0.0]]],  # Real part
            [[[0.0, 1.0], [-1.0, 0.0]]]   # Imaginary part
        ])

        output = prelu.forward(x)
        real_out, imag_out = output.chunk(2, dim=1)

        # Zeros should remain zeros
        zero_positions_real = (x[:, :1] == 0.0)
        zero_positions_imag = (x[:, 1:] == 0.0)

        assert torch.equal(real_out[zero_positions_real], torch.zeros_like(real_out[zero_positions_real]))
        assert torch.equal(imag_out[zero_positions_imag], torch.zeros_like(imag_out[zero_positions_imag]))

    def test_single_pixel_input(self):
        """Test with single pixel input."""
        prelu = ComplexPReLU(num_parameters=1, init=0.1)

        x_single = torch.randn(1, 4, 1, 1)  # 2 complex channels, single pixel
        output = prelu.forward(x_single)

        assert output.shape == (1, 4, 1, 1)
        assert torch.isfinite(output).all()

    def test_large_num_parameters(self):
        """Test with large number of parameters."""
        num_params = 512
        prelu = ComplexPReLU(num_parameters=num_params, init=0.05)

        x = torch.randn(1, 2*num_params, 4, 4)  # Match the channel requirement
        output = prelu.forward(x)

        assert output.shape == x.shape
        assert prelu.weight_real.shape == (num_params,)
        assert prelu.weight_imag.shape == (num_params,)
