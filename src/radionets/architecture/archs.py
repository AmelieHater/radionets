from math import pi

import torch
from torch import nn

from radionets.architecture.activation import GeneralReLU
from radionets.architecture.blocks import ComplexSRBlock, SRBlock
from radionets.architecture.layers import (
    ComplexConv2d,
    ComplexInstanceNorm2d,
    ComplexPReLU,
)

__all__ = [
    "SRResNet",
    "SRResNetComplex",
    "SRResNet18",
    "SRResNet18Complex",
    "SRResNet18AmpPhase",
    "SRResNet18Amp",
    "SRResNet18Phase",
    "SRResNet34",
    "SRResNet34AmpPhase",
    "SRResNet34_unc",
    "SRResNet34_unc_no_grad",
]


class SRResNet(nn.Module):
    def __init__(self, channels=2, groups=2):
        super().__init__()

        self.channels = 64

        self.preBlock = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=self.channels,
                kernel_size=9,
                stride=1,
                padding=4,
                groups=groups,
            ),
            nn.PReLU(),
        )

        self.postBlock = nn.Sequential(
            nn.Conv2d(
                in_channels=self.channels,
                out_channels=self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm2d(self.channels),
        )

        self.final = nn.Sequential(
            nn.Conv2d(
                in_channels=self.channels,
                out_channels=channels,
                kernel_size=9,
                stride=1,
                padding=4,
                groups=groups,
            ),
        )

    def _create_blocks(self, n_blocks, **kwargs):
        blocks = []
        for _ in range(n_blocks):
            blocks.append(SRBlock(64, 64, **kwargs))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        x = self.preBlock(input)
        x = x + self.postBlock(self.blocks(x))
        x = self.final(x)

        return {"pred": x}


class SRResNetComplex(nn.Module):
    def __init__(self):
        super().__init__()

        self.channels = 128

        self.preBlock = nn.Sequential(
            ComplexConv2d(
                in_channels=2,
                out_channels=self.channels,
                kernel_size=3,
                stride=1,
            ),
            ComplexPReLU(num_parameters=2),
        )

        self.postBlock = nn.Sequential(
            ComplexConv2d(
                in_channels=self.channels,
                out_channels=self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            ComplexInstanceNorm2d(self.channels),
        )

        self.final = nn.Sequential(
            ComplexConv2d(
                in_channels=self.channels,
                out_channels=2,
                kernel_size=9,
                stride=1,
                padding=4,
            ),
        )

    def _create_blocks(self, n_blocks):
        blocks = []
        for _ in range(n_blocks):
            blocks.append(ComplexSRBlock(64, 64))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        x = self.preBlock(input)
        x = x + self.postBlock(self.blocks(x))
        x = self.final(x)

        return {"pred": x}


class SRResNet18(SRResNet):
    def __init__(self):
        super().__init__()

        # Create 8 ResBlocks to build a SRResNet18
        self._create_blocks(8)


class SRResNet18Complex(SRResNetComplex):
    def __init__(self):
        super().__init__()

        # Create 8 ResBlocks to build a SRResNet18
        self._create_blocks(8)


class SRResNet18AmpPhase(SRResNet):
    def __init__(self):
        super().__init__()

        # Create 8 ResBlocks to build a SRResNet18
        self._create_blocks(8, groups=1)

        self.hardtanh = nn.Hardtanh(-pi, pi)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = super().forward(x)["pred"]

        x_amp = self.relu(x[:, 0].unsqueeze(1))
        x_phase = self.hardtanh(x[:, 1].unsqueeze(1))

        return {"pred": torch.cat([x_amp, x_phase], dim=1)}


###########################################Meine lustigen Ideen:


class SRResNet18Amp(SRResNet):
    def __init__(self, channels=1, groups=1):
        super().__init__(channels=1, groups=1)

        # Create 8 ResBlocks to build a SRResNet18
        self._create_blocks(8, groups=1)

        self.relu = nn.ReLU()

    def forward(self, input):
        x_amp = input[:, 0].unsqueeze(1)
        x_amp = self.preBlock(x_amp)
        x_amp = x_amp + self.postBlock(self.blocks(x_amp))
        x_amp = self.final(x_amp)

        x_amp = self.relu(x_amp)

        x_phase = input[:, 1].unsqueeze(1)  # Die unveränderte Phase wird wieder vereint
        # mit der trainierten Amplitude durch torch.cat

        return {"pred": torch.cat([x_amp, x_phase], dim=1)}

    # pred = x["pred"]
    # inp_amp = pred[:, 0, :]
    # inp_phase = pred[:, 1, :]


class SRResNet18Phase(SRResNet):
    def __init__(self, channels=1, groups=1):
        super().__init__(channels=1, groups=1)

        # Create 8 ResBlocks to build a SRResNet18
        self._create_blocks(8, groups=1)

        self.hardtanh = nn.Hardtanh(-pi, pi)

    def forward(self, input):
        x_phase = input[:, 1].unsqueeze(1)
        x_phase = self.preBlock(x_phase)
        x_phase = x_phase + self.postBlock(self.blocks(x_phase))
        x_phase = self.final(x_phase)

        x_phase = self.hardtanh(x_phase)

        x_amp = input[:, 0].unsqueeze(1)
        # Die unveränderte Amplitude wird wieder vereint mit
        # der trainierten Amplitude durch torch.cat

        return {"pred": torch.cat([x_amp, x_phase], dim=1)}


##################################################################


class SRResNet34(SRResNet):
    def __init__(self):
        super().__init__()

        # Create 16 ResBlocks to build a SRResNet34
        self._create_blocks(16)


class SRResNet34AmpPhase(SRResNet):
    def __init__(self):
        super().__init__()

        # Create 16 ResBlocks to build a SRResNet34
        self._create_blocks(16)

        self.hardtanh = nn.Hardtanh(-pi, pi)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = super().forward(x)

        x_amp = self.relu(x[:, 0].unsqueeze(1))
        x_phase = self.hardtanh(x[:, 1].unsqueeze(1))

        return {"pred": torch.cat([x_amp, x_phase], dim=1)}


class SRResNet34_unc(SRResNet):
    def __init__(self):
        super().__init__()

        self._create_blocks(16)

        self.postBlock = nn.Sequential(
            nn.Conv2d(
                in_channels=self.channels,
                out_channels=self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm2d(self.channels),
        )

        self.elu = GeneralReLU(sub=-1e-10)

    def forward(self, x):
        s = x.shape[-1]

        x = self.preBlock(x)

        x = x + self.postBlock(self.blocks(x))

        x = self.final(x)

        x0 = x[:, 0].reshape(-1, 1, s // 2 + 1, s)
        x1 = x[:, 1].reshape(-1, 1, s // 2 + 1, s)
        x3 = x[:, 2].reshape(-1, 1, s // 2 + 1, s)
        x3 = self.elu(x3)
        x4 = x[:, 3].reshape(-1, 1, s // 2 + 1, s)
        x4 = self.elu(x4)

        return {"pred": torch.cat([x0, x3, x1, x4], dim=1)}


class SRResNet34_unc_no_grad(SRResNet34_unc):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        s = x.shape[-1]

        x = self.preBlock(x)

        x = x + self.postBlock(self.blocks(x))

        x = self.final(x)

        x0 = x[:, 0].reshape(-1, 1, s // 2 + 1, s)
        x1 = x[:, 1].reshape(-1, 1, s // 2 + 1, s)
        x3 = x[:, 2].reshape(-1, 1, s // 2 + 1, s)
        with torch.no_grad():
            x3 = self.elu(x3)

        x4 = x[:, 3].reshape(-1, 1, s // 2 + 1, s)
        with torch.no_grad():
            x4 = self.elu(x4)

        return {"pred": torch.cat([x0, x3, x1, x4], dim=1)}
