from math import pi

import torch
from torch import nn

from radionets.dl_framework.model import (  # ComplexInstanceNorm2d,; ComplexPReLU,
    ComplexConv2d,
    GeneralRelu,
    SRBlock,
)


class SRResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.preBlock = nn.Sequential(ComplexConv2d(2, 64, 3, 1, bias=True), nn.PReLU())

        # ResBlock 8
        self.blocks = nn.Sequential(
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
        )

        self.postBlock = nn.Sequential(
            ComplexConv2d(64, 64, 3, 1, bias=False), nn.BatchNorm2d(64)
        )

        self.final = nn.Sequential(
            ComplexConv2d(64, 2, 3, stride=1, bias=True),
        )
        self.map = nn.Hardtanh(
            min_val=-500,
            max_val=500,
        )

    def forward(self, x):
        s = x.shape[-1]
        x_start = x.clone()

        x = self.preBlock(x)

        x = x + self.postBlock(self.blocks(x))

        x = self.map(self.final(x) + x_start)

        x0 = x[:, 0].reshape(-1, 1, s // 2 + 5, s)
        x1 = x[:, 1].reshape(-1, 1, s // 2 + 5, s)

        return torch.cat([x0, x1], dim=1)


class SRResNet_16(nn.Module):
    def __init__(self):
        super().__init__()

        self.preBlock = nn.Sequential(
            # ComplexConv2d(2, 64, 3, 1, bias=True), ComplexPReLU()
            nn.Conv2d(2, 128, 3, 1, bias=True, padding=1, groups=2),
            nn.InstanceNorm2d(128),
            nn.PReLU(),
        )

        # ResBlock 16
        self.blocks_amp = nn.Sequential(
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
        )
        self.blocks_phase = nn.Sequential(
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
        )

        self.postBlock = nn.Sequential(
            # ComplexConv2d(64, 64, 3, 1, bias=False),
            # ComplexInstanceNorm2d(64),
            nn.Conv2d(128, 128, 3, 1, bias=False, padding=1, groups=2),
            nn.InstanceNorm2d(128),
        )

        self.final = nn.Sequential(
            # ComplexConv2d(64, 2, 3, stride=1, bias=True),
            nn.Conv2d(128, 2, 3, stride=1, bias=True, padding=1, groups=2),
        )
        self.map_amp = nn.LeakyReLU(0.1)

        # nn.Hardtanh(
        #     min_val=0,
        #     max_val=2,
        # )
        self.map_phase = nn.Hardtanh(
            min_val=-pi,
            max_val=pi,
        )

    def forward(self, x):
        # s = x.shape[-1]
        # mask = x == 0

        # x_start = x.clone()

        x = self.preBlock(x)

        x = x + self.postBlock(
            torch.cat([self.blocks_amp(x[:, :64]), self.blocks_phase(x[:, 64:])], dim=1)
        )

        x = self.final(x)  # + x_start

        x0 = self.map_amp(x[:, 0])[:, None]
        x1 = self.map_phase(x[:, 1])[:, None]

        return torch.cat([x0, x1], dim=1)


class SRResNet_16_unc(nn.Module):
    def __init__(self):
        super().__init__()

        self.preBlock = nn.Sequential(
            nn.Conv2d(2, 64, 9, stride=1, padding=4, groups=2), nn.PReLU()
        )

        # ResBlock 16
        self.blocks = nn.Sequential(
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
        )

        self.postBlock = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(64),
        )

        self.final = nn.Sequential(nn.Conv2d(64, 4, 9, stride=1, padding=4, groups=2))

        self.hardtanh = nn.Hardtanh(-pi, pi)
        self.relu = nn.ReLU()
        self.elu = GeneralRelu(sub=-1e-10)

    def forward(self, x):
        s = x.shape[-1]

        x = self.preBlock(x)

        x = x + self.postBlock(self.blocks(x))

        x = self.final(x)

        x0 = x[:, 0].reshape(-1, 1, s // 2 + 1, s)
        # x0 = self.relu(x0)
        x1 = x[:, 1].reshape(-1, 1, s // 2 + 1, s)
        # x1 = self.hardtanh(x1)
        x3 = x[:, 2].reshape(-1, 1, s // 2 + 1, s)
        x3 = self.elu(x3)
        x4 = x[:, 3].reshape(-1, 1, s // 2 + 1, s)
        x4 = self.elu(x4)

        return torch.cat([x0, x3, x1, x4], dim=1)


class SRResNet_16_unc_no_grad(nn.Module):
    def __init__(self):
        super().__init__()

        self.preBlock = nn.Sequential(
            nn.Conv2d(2, 64, 9, stride=1, padding=4, groups=2), nn.PReLU()
        )

        # ResBlock 16
        self.blocks = nn.Sequential(
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
            SRBlock(64, 64),
        )

        self.postBlock = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(64),
        )

        self.final = nn.Sequential(nn.Conv2d(64, 4, 9, stride=1, padding=4, groups=2))

        self.hardtanh = nn.Hardtanh(-pi, pi)
        self.relu = nn.ReLU()
        self.elu = GeneralRelu(sub=-1e-10)

    def forward(self, x):
        s = x.shape[-1]

        x = self.preBlock(x)

        x = x + self.postBlock(self.blocks(x))

        x = self.final(x)

        x0 = x[:, 0].reshape(-1, 1, s // 2 + 1, s)
        # x0 = self.relu(x0)
        x1 = x[:, 1].reshape(-1, 1, s // 2 + 1, s)
        # x1 = self.hardtanh(x1)
        x3 = x[:, 2].reshape(-1, 1, s // 2 + 1, s)
        with torch.no_grad():
            x3 = self.elu(x3)

        x4 = x[:, 3].reshape(-1, 1, s // 2 + 1, s)
        with torch.no_grad():
            x4 = self.elu(x4)

        return torch.cat([x0, x3, x1, x4], dim=1)
