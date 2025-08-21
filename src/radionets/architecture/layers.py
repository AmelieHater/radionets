import torch
from torch import nn
from torch.nn.modules.utils import _pair


class LocallyConnected2d(nn.Module):
    """
    A 2D locally connected layer implementation.

    Unlike convolutional layers that share weights across spatial locations,
    locally connected layers use different weights for each spatial position.
    This allows the layer to learn location-specific features while maintaining
    the sliding window approach of convolutions.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    output_size : tuple of int
        Expected output spatial dimensions as (height, width).
    kernel_size : int
        Size of the sliding window (assumes square kernel).
    stride : int
        Stride of the sliding window (assumes same stride for both dimensions).
    bias : bool, optional
        If True, adds a learnable bias parameter. Default is False.

    Attributes
    ----------
    weight : nn.Parameter
        Learnable weights with shape
        (1, out_channels, in_channels, output_height, output_width, kernel_size²).
    bias : nn.Parameter or None
        Learnable bias with shape
        (1, out_channels, output_height, output_width) if bias=True, else None.
    kernel_size : tuple of int
        Kernel size as (height, width).
    stride : tuple of int
        Stride as (height, width).
    """

    def __init__(
        self, in_channels, out_channels, output_size, kernel_size, stride, bias=False
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(
                1,
                out_channels,
                in_channels,
                output_size[0],
                output_size[1],
                kernel_size**2,
            )
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size[0], output_size[1])
            )
        else:
            self.register_parameter("bias", None)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)

    def forward(self, x):
        _, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out


class ComplexConv2d(nn.Module):
    """
    2D convolution layer for complex-valued tensors.

    This layer performs 2D convolution on complex-valued inputs by decomposing
    the operation into separate real and imaginary components. It implements
    the mathematical formula for complex multiplication:
    (a + bi) * (c + di) = (ac - bd) + (ad + bc)i

    Parameters
    ----------
    in_channels : int
        Number of channels in the input tensor.
    out_channels : int
        Number of channels produced by the convolution.
    kernel_size : int or tuple of int
        Size of the convolving kernel. If int, the same value is used for
        both height and width dimensions.
    stride : int or tuple of int, optional
        Stride of the convolution. If int, the same value is used for both
        height and width dimensions. Default is 1.
    bias : bool, optional
        If True, adds a learnable bias to the output. Default is True.

    Attributes
    ----------
    conv_real : torch.nn.Conv2d
        Convolution layer for processing real components of the complex
        multiplication formula.
    conv_imag : torch.nn.Conv2d
        Convolution layer for processing imaginary components of the complex
        multiplication formula.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding="same",
        bias=True,
    ):
        """
        Initialize the ComplexConv2d layer.

        Parameters
        ----------
        in_channels : int
            Number of channels in the input tensor.
        out_channels : int
            Number of channels produced by the convolution.
        kernel_size : int or tuple of int
            Size of the convolving kernel.
        stride : int or tuple of int, optional
            Stride of the convolution. Default is 1.
        bias : bool, optional
            If True, adds a learnable bias to the output. Default is True.
        """
        super().__init__()

        # Initialize real component convolution layer
        self.conv_real = nn.Conv2d(
            in_channels // 2,
            out_channels // 2,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

        # Initialize imaginary component convolution layer
        self.conv_imag = nn.Conv2d(
            in_channels // 2,
            out_channels // 2,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x):
        """
        Forward pass of the complex convolution layer.

        Performs complex-valued 2D convolution by applying separate
        convolutions for real and imaginary values, and combining
        results according to complex multiplication rules.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width) with
            dtype (torch.float32 or torch.float64). Expected channels are equally
            split into real and imag channels, e.g., num channels is 2 for first
            network layer.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, out_channels, out_height, out_width)
            with the same complex dtype as input.
        """
        real, imag = x.chunk(2, dim=1)

        real_out = self.conv_real(real) - self.conv_imag(imag)
        imag_out = self.conv_real(imag) + self.conv_imag(real)

        return torch.cat([real_out, imag_out], dim=1)
