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
