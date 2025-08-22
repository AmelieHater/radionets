import numpy as np
import torch
from torch import nn


def l1(x, y):
    l1 = nn.L1Loss()
    loss = l1(x, y)
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
    inp_amp = x[:, 0, :]
    inp_phase = x[:, 1, :]

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
    inp_amp = x[:, 0, :]
    inp_phase = x[:, 1, :]

    tar_amp = y[:, 0, :]
    tar_phase = y[:, 1, :]

    l1 = nn.L1Loss(reduction="sum")
    loss_amp = l1(inp_amp, tar_amp)
    loss_phase = l1(inp_phase, tar_phase)
    loss = loss_amp + loss_phase
    return loss


def splitted_L1_mk(x, y):
    inp_amp = x[:, 0, :]
    inp_phase = x[:, 1, :]

    tar_amp = y[:, 0, :]
    tar_phase = y[:, 1, :]

    mask1 = create_circular_mask(261, 512, center=(256, 256), radius=50, bs=y.shape[0])
    mask2 = create_circular_mask(261, 512, center=(256, 256), radius=150, bs=y.shape[0])
    mask3 = create_circular_mask(261, 512, center=(256, 256), radius=250, bs=y.shape[0])
    mask11 = torch.from_numpy(mask1).bool()
    mask22 = torch.from_numpy(~mask1 & mask2).bool()
    mask33 = torch.from_numpy(~mask1 & ~mask2 & mask3).bool()
    mask44 = torch.from_numpy(~mask1 & ~mask2 & ~mask3).bool()

    l1 = nn.L1Loss(reduction="sum")
    loss_amp = l1(inp_amp[mask44], tar_amp[mask44]) * 0.5
    loss_amp_out = l1(inp_amp[mask33], tar_amp[mask33]) * 0.8
    loss_amp_mid = l1(inp_amp[mask22], tar_amp[mask22]) * 1.0
    loss_amp_inner = l1(inp_amp[mask11], tar_amp[mask11]) * 1.5
    loss_phase = l1(inp_phase[mask44], tar_phase[mask44]) * 0.5
    loss_phase_out = l1(inp_phase[mask33], tar_phase[mask33]) * 0.8
    loss_phase_mid = l1(inp_phase[mask22], tar_phase[mask22]) * 1.0
    loss_phase_inner = l1(inp_phase[mask11], tar_phase[mask11]) * 1.5
    loss = (
        loss_amp
        + loss_phase
        + loss_amp_out
        + loss_phase_out
        + loss_amp_mid
        + loss_phase_mid
        + loss_amp_inner
        + loss_phase_inner
    )
    return loss


def phase_loss(
    pred_fourier,
    target_fourier,
    linear_threshold=0.1,
    magnitude_threshold=1e-4,
    epsilon=1e-8,
):
    """
    Symlog phase loss with magnitude-based weighting
    """
    pred_fourier = pred_fourier[:, :2]
    pred_real, pred_imag = pred_fourier.chunk(2, dim=1)
    target_real, target_imag = target_fourier.chunk(2, dim=1)

    # Calculate magnitudes for weighting
    target_mag = torch.sqrt(target_real**2 + target_imag**2 + epsilon)

    # Calculate phases
    pred_phase = torch.atan2(pred_imag, pred_real + epsilon)
    target_phase = torch.atan2(target_imag, target_real + epsilon)

    # Phase difference with wrapping
    phase_diff = torch.angle(torch.exp(1j * (pred_phase - target_phase)))

    # Symlog transformation
    def adaptive_symlog(phase_diff, magnitude, threshold=linear_threshold):
        """
        Adaptive symlog based on signal magnitude
        Stronger signals get more sensitive phase treatment
        """
        # Adapt threshold based on magnitude
        adaptive_threshold = threshold * torch.exp(-magnitude / magnitude_threshold)
        adaptive_threshold = torch.clamp(adaptive_threshold, 0.01, threshold)

        sign = torch.sign(phase_diff)
        abs_diff = torch.abs(phase_diff)

        linear_region = abs_diff < adaptive_threshold
        linear_part = phase_diff

        log_part = sign * (
            adaptive_threshold * torch.log(abs_diff / adaptive_threshold + 1)
            + adaptive_threshold
        )

        return torch.where(linear_region, linear_part, log_part)

    # Apply adaptive symlog
    transformed_phase_diff = adaptive_symlog(phase_diff, target_mag)

    # Weight by magnitude importance
    magnitude_mask = (target_mag > magnitude_threshold).float()
    magnitude_weights = target_mag / (target_mag.max() + epsilon)

    # Final loss
    weighted_loss = (
        magnitude_mask * magnitude_weights * transformed_phase_diff**2
    ).sum()
    normalizer = (magnitude_mask * magnitude_weights).sum() + epsilon

    return weighted_loss / normalizer


def mag_loss(
    pred_fourier,
    target_fourier,
    magnitude_threshold=1e-4,
    epsilon=1e-8,
):
    """
    Weight magnitude loss by signal strength - similar to phase weighting
    """
    pred_fourier = pred_fourier[:, :2]
    pred_real, pred_imag = pred_fourier.chunk(2, dim=1)
    target_real, target_imag = target_fourier.chunk(2, dim=1)

    pred_mag = torch.sqrt(pred_real**2 + pred_imag**2 + epsilon)
    target_mag = torch.sqrt(target_real**2 + target_imag**2 + epsilon)

    # Create magnitude-based weights
    # Higher weight for stronger signals (more important to get right)
    magnitude_weights = torch.tanh(target_mag / magnitude_threshold)

    # Only consider pixels above noise threshold
    significant_mask = (target_mag > magnitude_threshold).float()

    # Magnitude loss with weighting
    mag_diff = (pred_mag - target_mag) ** 2
    weighted_mag_loss = (significant_mask * magnitude_weights * mag_diff).sum()
    normalizer = (significant_mask * magnitude_weights).sum() + epsilon

    return weighted_mag_loss / normalizer


class AdaptiveLossBalancer(nn.Module):
    def __init__(
        self,
        initial_mag_weight=1.0,
        initial_phase_weight=1.0,
        adaptation_rate=0.1,
        target_ratio=1.0,
    ):
        super().__init__()

        # Learnable weights (in log space for stability)
        self.log_mag_weight = nn.Parameter(
            torch.log(torch.tensor(initial_mag_weight))
        ).cuda()
        self.log_phase_weight = nn.Parameter(
            torch.log(torch.tensor(initial_phase_weight))
        ).cuda()

        self.adaptation_rate = adaptation_rate
        self.target_ratio = target_ratio

        # Running averages for loss magnitudes
        self.register_buffer("mag_loss_ema", torch.tensor(1.0))
        self.register_buffer("phase_loss_ema", torch.tensor(1.0))

    def forward(self, mag_loss, phase_loss):
        # Update exponential moving averages
        self.mag_loss_ema = (
            1 - self.adaptation_rate
        ) * self.mag_loss_ema + self.adaptation_rate * mag_loss.detach()
        self.phase_loss_ema = (
            1 - self.adaptation_rate
        ) * self.phase_loss_ema + self.adaptation_rate * phase_loss.detach()

        # Current weights
        mag_weight = torch.exp(self.log_mag_weight)
        phase_weight = torch.exp(self.log_phase_weight)

        # Calculate current ratio
        current_ratio = (mag_weight * self.mag_loss_ema) / (
            phase_weight * self.phase_loss_ema + 1e-8
        )

        # Adaptive adjustment (optional - can also let optimizer handle it)
        if self.training:
            ratio_error = current_ratio - self.target_ratio
            # Adjust weights to bring ratio closer to target
            adjustment = self.adaptation_rate * ratio_error
            with torch.no_grad():
                self.log_mag_weight -= adjustment * 0.5
                self.log_phase_weight += adjustment * 0.5

        return mag_weight * mag_loss + phase_weight * phase_loss, {
            "mag_weight": mag_weight.item(),
            "phase_weight": phase_weight.item(),
            "ratio": current_ratio.item(),
        }


def cab(x, y, max_ratio=10.0):
    """
    Adaptive balancing with ratio constraints
    """
    balancer = AdaptiveLossBalancer()
    ml = mag_loss(x, y)
    pl = phase_loss(x, y)
    balanced_loss, info = balancer(ml, pl)

    # Ensure ratio doesn't exceed max_ratio
    current_ratio = info["ratio"]
    if current_ratio < max_ratio:
        # Cap the dominant loss
        adjustment = max_ratio / current_ratio
        if info["mag_weight"] * ml > info["phase_weight"] * pl:
            mag_weight = info["mag_weight"] / adjustment
            phase_weight = info["phase_weight"]
        else:
            mag_weight = info["mag_weight"]
            phase_weight = info["phase_weight"] / adjustment

        balanced_loss = mag_weight * ml + phase_weight * pl

    return balanced_loss


def splitted_L1_dsa(x, y):
    inp_amp = x[:, 0, :]
    inp_phase = x[:, 1, :]

    tar_amp = y[:, 0, :]
    tar_phase = y[:, 1, :]

    mask1 = create_circular_mask(517, 1024, center=(512, 512), radius=50, bs=y.shape[0])
    mask2 = create_circular_mask(
        517, 1024, center=(512, 512), radius=100, bs=y.shape[0]
    )
    mask3 = create_circular_mask(
        517, 1024, center=(512, 512), radius=200, bs=y.shape[0]
    )
    mask4 = create_circular_mask(
        517, 1024, center=(512, 512), radius=400, bs=y.shape[0]
    )

    mask11 = torch.from_numpy(mask1).bool()
    mask22 = torch.from_numpy(~mask1 & mask2).bool()
    mask33 = torch.from_numpy(~mask1 & ~mask2 & mask3).bool()
    mask44 = torch.from_numpy(~mask1 & ~mask2 & ~mask3 & mask4).bool()
    mask55 = torch.from_numpy(~mask1 & ~mask2 & ~mask3 & ~mask4).bool()

    l1 = nn.L1Loss(reduction="sum")
    loss_amp = l1(inp_amp[mask55], tar_amp[mask55])
    loss_amp_1 = l1(inp_amp[mask44], tar_amp[mask44])
    loss_amp_2 = l1(inp_amp[mask33], tar_amp[mask33])
    loss_amp_3 = l1(inp_amp[mask22], tar_amp[mask22])
    loss_amp_4 = l1(inp_amp[mask11], tar_amp[mask11])
    loss_phase = l1(inp_phase[mask55], tar_phase[mask55])
    loss_phase_1 = l1(inp_phase[mask44], tar_phase[mask44])
    loss_phase_2 = l1(inp_phase[mask33], tar_phase[mask33])
    loss_phase_3 = l1(inp_phase[mask22], tar_phase[mask22])
    loss_phase_4 = l1(inp_phase[mask11], tar_phase[mask11])
    loss = (
        loss_amp * 1
        + loss_phase * 1
        + loss_amp_1 * 0.9
        + loss_phase_1 * 0.9
        + loss_amp_2 * 0.8
        + loss_phase_2 * 0.8
        + loss_amp_3 * 0.5
        + loss_phase_3 * 0.5
        + loss_amp_4 * 0.25
        + loss_phase_4 * 0.25
    )
    return loss


def beta_nll_loss(x, y, beta=0.5):
    """Compute beta-NLL loss

    :param mean: Predicted mean of shape B x D
    :param variance: Predicted variance of shape B x D
    :param target: Target of shape B x D
    :param beta: Parameter from range [0, 1] controlling relative
    weighting between data points, where "0" corresponds to
    high weight on low error points and "1" to an equal weighting.
    :returns: Loss per batch element of shape B
    """
    pred_amp = x[:, 0, :]
    pred_phase = x[:, 2, :]
    mean = torch.stack([pred_amp, pred_phase], axis=1)

    unc_amp = x[:, 1, :]
    unc_phase = x[:, 3, :]
    variance = torch.stack([unc_amp, unc_phase], axis=1)

    tar_amp = y[:, 0, :]
    tar_phase = y[:, 1, :]
    target = torch.stack([tar_amp, tar_phase], axis=1)

    loss = 0.5 * ((target - mean) ** 2 / variance + variance.log())

    if beta > 0:
        loss = loss * variance.detach() ** beta

    return loss.mean()


def mse(x, y):
    mse = nn.MSELoss()
    loss = mse(x, y)
    return loss


def jet_seg(x, y):
    # weight components farer outside more
    loss_l1_weighted = 0
    for i in range(x.shape[1]):
        loss_l1_weighted += l1(x[:, i], y[:, i]) * (i + 1)

    return loss_l1_weighted
