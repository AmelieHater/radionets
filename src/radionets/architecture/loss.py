import numpy as np
import torch
from torch import Tensor, nn


class SplittedL1Loss(nn.Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
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


class MaskedSplittedL1Loss(nn.Module):
    def __init__(
        self,
        size_average: bool = None,
        reduce: bool = None,
        reduction: str = "mean",
        width: int = 256,
        height: int = 256,
        center: list | tuple = None,
        radius: int = 50,
    ) -> None:
        super().__init__()

        self.reduction = reduction
        self.width = width
        self.height = height
        self.center = center
        self.radius = radius

        # Assign mask so it can be cached during forward call;
        # None at first, then torch.Tensor after caching
        self._mask: torch.Tensor | None = None

    def _create_circular_mask(
        self,
        w: int,
        h: int,
        center: list | tuple = None,
        radius: int = None,
        device: torch.device = None,
    ) -> np.ndarray:
        if center is None:
            center = (int(w / 2), int(h / 2))

        if radius is None:
            radius = min(center[0], center[1], w - center[0], h - center[1])

        x = torch.arange(w, device=device).view(1, -1)
        y = torch.arange(h, device=device).view(-1, 1)
        dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

        mask = dist_from_center <= radius

        return mask

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        inp_amp = input[:, 0, :]
        inp_phase = input[:, 1, :]

        tar_amp = target[:, 0, :]
        tar_phase = target[:, 1, :]

        if self._mask is None or self._mask.device != input.device:
            self._mask = self._create_circular_mask(
                w=self.width,
                h=self.height,
                center=self.center,
                radius=self.radius,
                device=input.device,
            )

        weight = torch.where(self._mask, 1.0, 0.3)

        inp_amp *= weight
        inp_phase *= weight
        tar_amp *= weight
        tar_phase *= weight

        l1 = nn.L1Loss(reduction=self.reduction)
        loss_amp = l1(inp_amp, tar_amp)
        loss_phase = l1(inp_phase, tar_phase)
        loss = loss_amp + loss_phase

        return loss
