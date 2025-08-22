from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.utils import _pair


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class GeneralRelu(nn.Module):
    def __init__(self, leak=None, sub=None, maxv=None):
        super().__init__()
        self.leak = leak
        self.sub = sub
        self.maxv = maxv

    def forward(self, x):
        x = F.leaky_relu(x, self.leak) if self.leak is not None else F.relu(x)
        if self.sub is not None:
            x = x - self.sub
        if self.maxv is not None:
            x.clamp_max_(self.maxv)
        return x


class GeneralELU(nn.Module):
    def __init__(self, add=None, maxv=None):
        super().__init__()
        self.add = add
        self.maxv = maxv

    def forward(self, x):
        x = F.elu(x)
        if self.add is not None:
            x = x + self.add
        if self.maxv is not None:
            x.clamp_max_(self.maxv)
        return x


def init_cnn_(m, f):
    if isinstance(m, nn.Conv2d):
        f(m.weight, a=0.1)
        if getattr(m, "bias", None) is not None:
            m.bias.data.zero_()
    for c in m.children():
        init_cnn_(c, f)


def init_cnn(m, uniform=False):
    f = nn.init.kaiming_uniform_ if uniform else nn.init.kaiming_normal_
    init_cnn_(m, f)


def conv(ni, nc, ks, stride, padding):
    conv = (nn.Conv2d(ni, nc, ks, stride, padding),)
    bn = (nn.BatchNorm2d(nc),)
    act = GeneralRelu(leak=0.1, sub=0.4)  # nn.ReLU()
    layers = [*conv, *bn, act]
    return layers


def load_pre_model(learn, pre_path, visualize=False, plot_loss=False):
    """
    :param learn:       object of type learner
    :param pre_path:    string wich contains the path of the model
    :param lr_find:     bool which is True if lr_find is used
    """
    name_pretrained = Path(pre_path).stem
    print(f"\nLoad pretrained model: {name_pretrained}\n")
    if torch.cuda.is_available() and not plot_loss:
        checkpoint = torch.load(pre_path, weights_only=False)
    else:
        checkpoint = torch.load(pre_path, map_location=torch.device("cpu"))

    if visualize:
        learn.load_state_dict(checkpoint["model"])
        return checkpoint["norm_dict"]
    elif plot_loss:
        learn.avg_loss.loss_train = checkpoint["train_loss"]
        learn.avg_loss.loss_valid = checkpoint["valid_loss"]
        learn.avg_loss.lrs = checkpoint["lrs"]
    else:
        learn.model.load_state_dict(checkpoint["model"])
        learn.opt.load_state_dict(checkpoint["opt"])
        learn.epoch = checkpoint["epoch"]
        learn.avg_loss.loss_train = checkpoint["train_loss"]
        learn.avg_loss.loss_valid = checkpoint["valid_loss"]
        learn.avg_loss.lrs = checkpoint["lrs"]
        learn.recorder.iters = checkpoint["iters"]
        learn.recorder.values = checkpoint["vals"]


def save_model(learn, model_path):
    if hasattr(learn, "normalize"):
        if learn.normalize.mode == "mean":
            norm_dict = {
                "mean_real": learn.normalize.mean_real,
                "mean_imag": learn.normalize.mean_imag,
                "std_real": learn.normalize.std_real,
                "std_imag": learn.normalize.std_imag,
            }
        elif learn.normalize.mode == "max":
            norm_dict = {"max_scaling": 0}
        elif learn.normalize.mode == "all":
            norm_dict = {"all": 0}
        else:
            raise ValueError(f"Undefined mode {learn.normalize.mode}, check for typos")
    else:
        norm_dict = {}

    torch.save(
        {
            "model": learn.model.state_dict(),
            "opt": learn.opt.state_dict(),
            "epoch": learn.epoch,
            "iters": learn.recorder.iters,
            "vals": learn.recorder.values,
            "train_loss": learn.avg_loss.loss_train,
            "valid_loss": learn.avg_loss.loss_valid,
            "lrs": learn.avg_loss.lrs,
            "norm_dict": norm_dict,
        },
        model_path,
    )


class LocallyConnected2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, output_size, kernel_size, stride, bias=False
    ):
        super(LocallyConnected2d, self).__init__()
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
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias):
        super().__init__()
        self.conv_real = nn.Conv2d(
            in_channels // 2,
            out_channels // 2,
            kernel_size,
            stride=stride,
            padding="same",
            bias=bias,
        )
        self.conv_imag = nn.Conv2d(
            in_channels // 2,
            out_channels // 2,
            kernel_size,
            stride=stride,
            padding="same",
            bias=bias,
        )

    def forward(self, x):
        real, imag = x.chunk(2, dim=1)  # Split real and imaginary parts
        # Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        real_out = self.conv_real(real) - self.conv_imag(imag)
        imag_out = self.conv_real(imag) + self.conv_imag(real)
        return torch.cat([real_out, imag_out], dim=1)


class ComplexInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(ComplexInstanceNorm2d, self).__init__()
        self.num_features = num_features // 2
        self.eps = eps
        self.affine = affine

        if self.affine:
            # Separate parameters for real and imaginary parts
            self.weight_real = nn.Parameter(torch.ones(self.num_features))
            self.weight_imag = nn.Parameter(torch.ones(self.num_features))
            self.bias_real = nn.Parameter(torch.zeros(self.num_features))
            self.bias_imag = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x):
        """
        x: Complex tensor of shape [batch, channels, height, width]
        First half of channels = real part, second half = imaginary part
        """
        real, imag = x.chunk(2, dim=1)

        # Instance normalization for each part separately
        # Calculate mean and variance across spatial dimensions (H, W)
        real_mean = real.mean(dim=[2, 3], keepdim=True)
        imag_mean = imag.mean(dim=[2, 3], keepdim=True)

        real_var = real.var(dim=[2, 3], keepdim=True, unbiased=False)
        imag_var = imag.var(dim=[2, 3], keepdim=True, unbiased=False)

        # Normalize
        real_norm = (real - real_mean) / torch.sqrt(real_var + self.eps)
        imag_norm = (imag - imag_mean) / torch.sqrt(imag_var + self.eps)

        if self.affine:
            # Apply learnable parameters
            real_norm = real_norm * self.weight_real.view(
                1, -1, 1, 1
            ) + self.bias_real.view(1, -1, 1, 1)
            imag_norm = imag_norm * self.weight_imag.view(
                1, -1, 1, 1
            ) + self.bias_imag.view(1, -1, 1, 1)

        return torch.cat([real_norm, imag_norm], dim=1)


class ComplexPReLU(nn.Module):
    def __init__(self, num_parameters=1, init=0.25):
        """
        Complex PReLU with separate parameters for real and imaginary parts

        Args:
            num_parameters: Number of parameters (1 for shared,
            or num_channels for per-channel)
            init: Initial value for the negative slope parameter
        """
        super(ComplexPReLU, self).__init__()
        self.num_parameters = num_parameters

        # Separate learnable parameters for real and imaginary parts
        self.weight_real = nn.Parameter(torch.full((num_parameters,), init))
        self.weight_imag = nn.Parameter(torch.full((num_parameters,), init))

    def forward(self, x):
        """
        x: Complex tensor of shape [batch, 2*channels, height, width]
        First half = real, second half = imaginary
        """
        real, imag = x.chunk(2, dim=1)

        # Apply PReLU to each component
        if self.num_parameters == 1:
            # Shared parameter across all channels
            real_out = torch.where(real >= 0, real, self.weight_real * real)
            imag_out = torch.where(imag >= 0, imag, self.weight_imag * imag)
        else:
            # Per-channel parameters
            weight_real = self.weight_real.view(1, -1, 1, 1)
            weight_imag = self.weight_imag.view(1, -1, 1, 1)

            real_out = torch.where(real >= 0, real, weight_real * real)
            imag_out = torch.where(imag >= 0, imag, weight_imag * imag)

        return torch.cat([real_out, imag_out], dim=1)


class SRBlock(nn.Module):
    def __init__(self, ni, nf, stride=1):
        super().__init__()
        self.convs = self._conv_block(ni, nf, stride)
        self.idconv = nn.Identity() if ni == nf else nn.Conv2d(ni, nf, 1)
        self.pool = (
            nn.Identity() if stride == 1 else nn.AvgPool2d(2, ceil_mode=True)
        )  # nn.AvgPool2d(8, 2, ceil_mode=True)

    def forward(self, x):
        return self.convs(x) + self.idconv(self.pool(x))

    def _conv_block(self, ni, nf, stride):
        # return nn.Sequential(
        #     ComplexConv2d(ni, nf, 3, stride=stride, bias=False),
        #     ComplexInstanceNorm2d(nf),
        #     ComplexPReLU(),
        #     ComplexConv2d(nf, nf, 3, stride=1, bias=False),
        #     ComplexInstanceNorm2d(nf),
        # )
        return nn.Sequential(
            nn.Conv2d(ni, nf, 3, stride=stride, bias=False, padding=1),
            nn.InstanceNorm2d(nf),
            nn.ReLU(),
            nn.Conv2d(nf, nf, 3, stride=1, bias=False, padding=1),
            nn.InstanceNorm2d(nf),
        )


def symmetry(x):
    if x.shape[-1] % 2 != 0:
        raise ValueError("The symmetry function only works for even image sizes.")
    upper_half = x[:, :, 0 : x.shape[2] // 2, :].clone()
    upper_left = upper_half[:, :, :, 0 : upper_half.shape[3] // 2].clone()
    upper_right = upper_half[:, :, :, upper_half.shape[3] // 2 :].clone()
    a = torch.flip(upper_left, dims=[-2, -1])
    b = torch.flip(upper_right, dims=[-2, -1])

    upper_half[:, :, :, 0 : upper_half.shape[3] // 2] = b
    upper_half[:, :, :, upper_half.shape[3] // 2 :] = a

    x[:, 0, x.shape[2] // 2 :, :] = upper_half[:, 0]
    x[:, 1, x.shape[2] // 2 :, :] = -upper_half[:, 1]
    return x
