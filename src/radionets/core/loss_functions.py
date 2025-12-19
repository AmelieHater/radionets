import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from radionets.evaluation.utils import apply_symmetry

__all__ = [
    "beta_nll_loss",
    "create_circular_mask",
    "splitted_L1_masked_amp",
    "InverseFouriertransformation",
    "L1_loss_max_brightness_amp",
    "jet_seg",
    "l1",
    "mse",
    "splitted_L1",
    "splitted_L1_amp",
    "splitted_L1_phase",
    "splitted_L1_masked",
]


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


########################für Fouriertransformierte Bilder Amplitude######################


def InverseFouriertransformation(amp, phase):
    amp_symm = apply_symmetry({"amp": amp})["amp"]
    phase_symm = apply_symmetry({"phase": phase})["phase"]
    comp_fourier = amp_symm * (torch.exp(1j * phase_symm))
    real_pic = torch.fft.fftshift(torch.fft.ifft2(torch.fft.fftshift(comp_fourier)))
    return real_pic


def L1_loss_max_brightness_amp(x, y):
    pred = x["pred"]
    inp_amp = pred[:, 0, :]
    # inp_phase = pred[:, 1, :]

    tar_amp = y[:, 0, :]
    tar_phase = y[:, 1, :]

    real_inp_pic = InverseFouriertransformation(inp_amp, tar_phase)
    real_tar_pic = InverseFouriertransformation(tar_amp, tar_phase)

    max_fluss_diff = torch.abs(
        torch.max(real_inp_pic.real) - torch.max(real_tar_pic.real)
    )

    sum_fluss_diff = torch.abs(
        torch.sum(real_inp_pic.real) - torch.sum(real_tar_pic.real)
    )

    l1 = nn.L1Loss()
    loss_amp = l1(inp_amp, tar_phase)

    loss = max_fluss_diff + loss_amp + sum_fluss_diff

    return loss


#####################für Amplitude############################################
def splitted_L1_masked_amp(x, y):
    pred = x["pred"]
    inp_amp = pred[:, 0, :]
    # inp_phase = pred[:, 1, :]

    tar_amp = y[:, 0, :]
    tar_phase = y[:, 1, :]

    inp_amp_symm = apply_symmetry({"inp_amp": inp_amp})["inp_amp"][0]
    tar_amp_symm = apply_symmetry({"tar_amp": tar_amp})["tar_amp"][0]
    # tar_amp_symm_original = tar_amp_symm.clone()

    tar_phase_sym = apply_symmetry({"tar_phase": tar_phase})["tar_phase"][0]

    mask = torch.tensor(create_circular_mask(512, 512, radius=20, bs=y.shape[0]))

    inp_amp_symm[mask] *= 100.1
    # inp_amp_symm[~mask] *= 0
    # inp_phase[~mask] *= 0.3
    tar_amp_symm[mask] *= 100.1
    # tar_amp_symm[~mask] *= 0
    # tar_phase[~mask] *= 0.3

    l1 = nn.L1Loss()
    loss_amp_symm = l1(inp_amp_symm, tar_amp_symm)
    # loss_phase = l1(inp_phase, tar_phase)
    loss = loss_amp_symm  # + loss_phase

    # tar_complex_mask = tar_amp_symm * (
    # torch.sin(tar_phase_sym) + 1j * torch.cos(tar_phase_sym)
    # )
    #
    # tar_complex = tar_amp_symm * (
    # torch.sin(tar_amp_symm_original) + 1j * torch.cos(tar_phase_sym)
    # )

    # plt.imshow(
    # torch.fft.fftshift(torch.fft.ifft2(torch.fft.fftshift(tar_complex_mask)))
    # .detach()
    # .real.cpu()
    # .numpy()[0]
    # )
    # plt.colorbar()
    # plt.imshow(inp_amp_symm[0].detach().cpu().numpy())
    # plt.savefig("plot_dirty_image_tar_masked.pdf")
    # plt.clf()
    #
    # plt.imshow(
    # torch.fft.fftshift(torch.fft.ifft2(torch.fft.fftshift(tar_complex)))
    # .detach()
    # .real.cpu()
    # .numpy()[0]
    # )
    # plt.colorbar()
    # plt.savefig("plot_dirty_image_tar_non_masked.pdf")
    # plt.clf()
    #
    # plt.imshow(tar_amp_symm.detach().cpu().numpy()[0], norm="log")
    # plt.colorbar()
    # plt.savefig("plot_amp_symm_tar_masked.pdf")
    # plt.clf()
    #
    # plt.imshow(tar_amp_symm_original.detach().cpu().numpy()[0], norm="log")
    # plt.colorbar()
    # plt.savefig("plot_amp_symm_tar_non_masked.pdf")
    #
    # raise TypeError
    return loss


# def create_ring_mask(h, w, inner_radius, outer_radius, center=None, bs=64):
#    """
#    Erstellt eine Ringmaske durch Subtraktion.
#
#    Parameters
#    ----------
#    h : int
#        Höhe
#    w : int
#        Breite
#    inner_radius : float
#        Innerer Radius
#    outer_radius : float
#        Äußerer Radius
#    center : tuple, optional
#        Zentrum (x, y)
#    bs : int
#        Batch size
#
#    Returns
#    -------
#    ring_mask : np.ndarray
#        Ring-Maske mit shape (bs, h, w)
#    """
#    # Große Maske erstellen
#    outer_mask = create_circular_mask(h, w, center=center, radius=outer_radius, bs=bs)
#
#    # Kleine Maske erstellen
#    inner_mask = create_circular_mask(h, w, center=center, radius=inner_radius, bs=bs)
#
#    # Ring = Große Maske - Kleine Maske
#    ring_mask = outer_mask - inner_mask
#    # ring_mask = outer_mask & ~inner_mask  # Boolean Operation
#    # oder: ring_mask = outer_mask ^ inner_mask  # XOR
#    # oder: ring_mask = outer_mask - inner_mask  # Bei float masks
#
#    return ring_mask

# def splitted_L1_masked_amp_doughnut(x, y):
#    pred = x["pred"]
#    inp_amp = pred[:, 0, :]
#    # inp_phase = pred[:, 1, :]
#
#    tar_amp = y[:, 0, :]
#    # tar_phase = y[:, 1, :]
#
#    mask = torch.tensor(create_ring_mask(261, 512, 10, 50, bs=y.shape[0]))
#
#    inp_amp[~mask] *= 6.6
#    # inp_phase[~mask] *= 0.3
#    tar_amp[~mask] *= 6.6
#    # tar_phase[~mask] *= 0.3
#
#    l1 = nn.L1Loss()
#    loss_amp = l1(inp_amp, tar_amp)
#    # loss_phase = l1(inp_phase, tar_phase)
#    loss = loss_amp  # + loss_phase
#
#    return loss

#################################################################


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


#######################################Meine Sachen################


def splitted_L1_amp(x, y):
    pred = x["pred"]
    inp_amp = pred[:, 0, :]

    tar_amp = y[:, 0, :]

    l1 = nn.L1Loss()
    loss_amp = l1(torch.sqrt(torch.abs(inp_amp)), torch.sqrt(torch.abs(tar_amp)))

    return loss_amp


def splitted_L1_phase(x, y):
    pred = x["pred"]
    inp_phase = pred[:, 1, :]

    tar_phase = y[:, 1, :]

    l1 = nn.L1Loss()
    loss_phase = l1(inp_phase, tar_phase)

    return loss_phase


##################################################################


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
