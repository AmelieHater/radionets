import numpy as np
import torch
from torch import nn

__all__ = [
    "masked_patch_loss",
    "weighted_masked_patch_loss",
    "beta_nll_loss",
    "create_circular_mask",
    "jet_seg",
    "l1",
    "mse",
    "splitted_L1",
    "splitted_L1_masked",
]


def masked_patch_loss(x, y):
    pred, mask = x["pred"], x["mask"]
    target = y

    mse = nn.MSELoss()
    loss = mse(pred[mask.bool()], target[mask.bool()])
    return loss


def weighted_masked_patch_loss(x, y):
    pred, mask = x["pred"], x["mask"]
    target = y

    loss_reco = masked_patch_loss(x, y)
    mse = nn.MSELoss()
    loss_input = mse(pred[~mask.bool()], target[~mask.bool()])

    return 0.6 * loss_reco + 0.4 * loss_input


def l1(x, y):
    pred = x["pred"]

    l1 = nn.L1Loss()
    loss = l1(pred, y)

    return loss


def create_circular_mask(h, w, center=None, radius=None, bs=64):
    if center is None:
        center = (int(w / 2), int(h / 2))

    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius

    return np.repeat([mask], bs, axis=0)


def splitted_L1_masked(x, y):
    pred = x["pred"]
    inp_amp = pred[:, 0, :]
    inp_phase = pred[:, 1, :]

    tar_amp = y[:, 0, :]
    tar_phase = y[:, 1, :]

    mask = torch.tensor(create_circular_mask(256, 256, radius=50, bs=y.shape[0]))

    inp_amp[~mask] *= 0.3
    inp_phase[~mask] *= 0.3
    tar_amp[~mask] *= 0.3
    tar_phase[~mask] *= 0.3

    l1 = nn.L1Loss()
    loss_amp = l1(inp_amp, tar_amp)
    loss_phase = l1(inp_phase, tar_phase)
    loss = loss_amp + loss_phase

    return loss


def splitted_L1(x, y):
    pred = x["pred"]
    inp_amp = pred[:, 0, :]
    inp_phase = pred[:, 1, :]

    tar_amp = y[:, 0, :]
    tar_phase = y[:, 1, :]

    l1 = nn.L1Loss()
    loss_amp = l1(inp_amp, tar_amp)
    loss_phase = l1(inp_phase, tar_phase)
    loss = loss_amp + loss_phase

    return loss


def beta_nll_loss(x: torch.tensor, y: torch.tensor, beta: float = 0.5):
    """Compute beta-NLL loss

    Parameters
    ----------
    x : :func:`torch.tensor`
        Prediction of the model.
    y : :func:`torch.tensor`
        Ground truth.
    beta : float
        Parameter from range [0, 1] controlling relative
        weighting between data points, where "0" corresponds to
        high weight on low error points and "1" to an equal weighting.

    Returns
    -------
    float : Loss per batch element of shape B
    """
    pred = x["pred"]
    pred_amp = pred[:, 0, :]
    pred_phase = pred[:, 2, :]
    mean = torch.stack([pred_amp, pred_phase], axis=1)

    unc_amp = pred[:, 1, :]
    unc_phase = pred[:, 3, :]
    variance = torch.stack([unc_amp, unc_phase], axis=1)

    tar_amp = y[:, 0, :]
    tar_phase = y[:, 1, :]
    target = torch.stack([tar_amp, tar_phase], axis=1)

    loss = 0.5 * ((target - mean) ** 2 / variance + variance.log())

    if beta > 0:
        loss = loss * variance.detach() ** beta

    return loss.mean()


def mse(x, y):
    pred = x["pred"]
    mse = nn.MSELoss()
    loss = mse(pred, y)

    return loss


def jet_seg(x, y):
    pred = x["pred"]

    # weight components farer outside more
    loss_l1_weighted = 0
    for i in range(pred.shape[1]):
        loss_l1_weighted += l1(pred[:, i], y[:, i]) * (i + 1)

    return loss_l1_weighted
