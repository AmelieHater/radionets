from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

_BOX_FACTORS = np.array([0.3, 0.22, 0.16])


def get_boxsize(num_corners: int, num_pixel: int = 63) -> int:
    """
    Compute corner box size based on number of corners used.

    Parameters
    ----------
    num_corners : int
        Number of corners to use (2, 3, or 4).
    num_pixel : int, optional
        Image size in pixels. Default: 63

    Returns
    -------
    int
        Box size in pixels.
    """
    return int(num_pixel * _BOX_FACTORS[num_corners - 2])


def select_box(rms: NDArray, sensitivity: float = 1e-6) -> NDArray:
    """
    Select valid corner boxes based on RMS threshold.

    Parameters
    ----------
    rms : :func:`~numpy.ndarray`, shape (4, B)
        RMS values for each corner.
    sensitivity : float, optional
        Threshold below which corners are considered valid.
        Default: 1e-6.

    Returns
    -------
    :func:`numpy.ndarray`, shape (B,)
        Number of valid corners per sample.
    """
    valid_corners = rms <= sensitivity
    return valid_corners.sum(axis=0)


def compute_rms(batch: ArrayLike, size: int) -> NDArray:
    """
    Compute RMS in all four corner boxes.

    Parameters
    ----------
    batch : :func:`~numpy.ndarray`
        Batch of images, shape (B, H, W).
    size : int
        Corner box size in pixels.

    Returns
    -------
    :func:`numpy.ndarray`
        RMS values for each corner, shape (4, B).
    """
    corners = np.stack(
        [
            batch[:, :size, :size],  # top left
            batch[:, :size, -size:],  # top right
            batch[:, -size:, :size],  # bottom left
            batch[:, -size:, -size:],  # bottom right
        ]
    )

    return np.sqrt((corners.reshape(4, len(batch), size * size) ** 2).mean(axis=2))


def get_rms(
    ifft_truth: NDArray,
    ifft_pred: NDArray,
    sensitivity: float = 1e-6,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Compute RMS values for ground truth and prediction.

    Parameters
    ----------
    ifft_truth : :func:`numpy.ndarray`, shape (B, H, W)
        Ground truth images.
    ifft_pred : :func:`numpy.ndarray`, shape (B, H, W)
        Predicted images.
    sensitivity : float, optional
        Threshold below which corners are considered valid.
        Default: 1e-6.

    Returns
    -------
    rms_truth : :func:`~numpy.ndarray`, shape (B,)
        Averaged RMS for ground truth.
    rms_pred : :func:`~numpy.ndarray`, shape (B,)
        Averaged RMS for predictions.
    rms_boxes : :func:`~numpy.ndarray`, shape (B,)
        Number of valid corners per sample.
    corners : :func:`~numpy.ndarray`, shape (B, 4)
        Corner validity mask.
    """
    _rms_truth_boxes = {}
    _rms_pred_boxes = {}

    for num_corners in [4, 3, 2]:
        size = get_boxsize(num_corners)
        _rms_truth_boxes[num_corners] = compute_rms(ifft_truth, size)
        _rms_pred_boxes[num_corners] = compute_rms(ifft_pred, size)

    rms_boxes = select_box(_rms_truth_boxes[4], sensitivity=sensitivity)
    current_batch_size = len(ifft_pred)

    corners = (_rms_truth_boxes[4] <= sensitivity).T.astype(np.float64)

    for num_corners in [3, 2]:
        invalid_mask = _rms_truth_boxes[num_corners] > sensitivity
        _rms_pred_boxes[4][invalid_mask] = 0

    rms_truth = np.zeros(current_batch_size)
    rms_pred = np.zeros(current_batch_size)

    for num_corners in [4, 3, 2]:
        mask = rms_boxes == num_corners

        if not mask.any():
            continue

        rms_truth[mask] = (
            np.abs(_rms_truth_boxes[num_corners][:, mask]).sum(axis=0) / num_corners
        )
        rms_pred[mask] = (
            np.abs(_rms_pred_boxes[num_corners][:, mask]).sum(axis=0) / num_corners
        )

    return rms_truth, rms_pred, rms_boxes, corners


def calc_dr(ifft_truth, ifft_pred):
    """
    Calculate dynamic range for ground truth and predicted images.

    The dynamic range is the peak value divided by RMS
    noise in corner (off-)regions (i.e., where no signal is expected).

    Parameters
    ----------
    ifft_truth : :func:`~numpy.ndarray`
        Ground truth inverse FFT images (image space), shape (B, H, W).
    ifft_pred : :func:`~numpy.ndarray`
        Predicted inverse FFT images (image space), shape (B, H, W).

    Returns
    -------
    dr_truth : :func:`~numpy.ndarray`
        Dynamic range for truth.
    dr_pred : :func:`~numpy.ndarray`
        Dynamic range for predictions.
    rms_boxes : np. ndarray
        Number of valid corners per sample.
    corners : :func:`~numpy.ndarray`
        Corner validity mask.
    """
    rms_truth, rms_pred, rms_boxes, corners = get_rms(ifft_truth, ifft_pred)

    peak_truth = ifft_truth.reshape(len(ifft_truth), -1).max(axis=1)
    peak_pred = ifft_pred.reshape(len(ifft_pred), -1).max(axis=1)

    valid_truth = rms_truth != 0
    valid_pred = rms_pred != 0
    dr_truth = peak_truth[valid_truth] / rms_truth[valid_truth]
    dr_pred = peak_pred[valid_pred] / rms_pred[valid_pred]

    return dr_truth, dr_pred, rms_boxes, corners
