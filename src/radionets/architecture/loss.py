import numpy as np
import torch
from torch import Tensor, nn


class SplittedL1Loss:
    def __init__(self, reduction: str = "mean") -> None:
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Runs the forward pass.
        """
        inp_amp = pred[:, 0, :]
        inp_phase = pred[:, 1, :]

        tar_amp = target[:, 0, :]
        tar_phase = target[:, 1, :]

        l1 = nn.L1Loss(self.reduction)
        loss_amp = l1(inp_amp, tar_amp)
        loss_phase = l1(inp_phase, tar_phase)
        loss = loss_amp + loss_phase

        return loss


class MaskedSplittedL1Loss(SplittedL1Loss):
    def __init__(
        self,
        reduction: str = "mean",
        width: int = 256,
        height: int = 256,
        center: list | tuple = None,
        radius: int = 50,
    ) -> None:
        super().__init__(reduction)

        self.width = width
        self.height = height
        self.center = center
        self.radius = radius

    def create_circular_mask(
        self,
        w: int,
        h: int,
        center: list | tuple = None,
        radius: int = None,
        bs: int = 64,
    ) -> np.ndarray:
        if center is None:
            center = (int(w / 2), int(h / 2))

        if radius is None:
            radius = min(center[0], center[1], w - center[0], h - center[1])

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

        mask = dist_from_center <= radius

        return np.repeat([mask], bs, axis=0)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        inp_amp = pred[:, 0, :]
        inp_phase = pred[:, 1, :]

        tar_amp = target[:, 0, :]
        tar_phase = target[:, 1, :]

        mask = torch.from_numpy(
            self.create_circular_mask(
                w=self.width,
                h=self.height,
                center=self.center,
                radius=self.radius,
                bs=target.shape[0],
            )
        )

        inp_amp[~mask] *= 0.3
        inp_phase[~mask] *= 0.3
        tar_amp[~mask] *= 0.3
        tar_phase[~mask] *= 0.3

        l1 = nn.L1Loss()
        loss_amp = l1(inp_amp, tar_amp)
        loss_phase = l1(inp_phase, tar_phase)
        loss = loss_amp + loss_phase

        return loss
