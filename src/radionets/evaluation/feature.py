"""Feature detection submodule."""

from __future__ import annotations

from math import sqrt
from typing import TYPE_CHECKING

import numpy as np
import torch
from skimage.feature import blob_log

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


def calc_blobs(
    ifft_pred: ArrayLike,
    ifft_truth: ArrayLike,
) -> tuple[NDArray, NDArray]:
    """Detect blobs using Laplacian of Gaussian in prediction
    and truth images.

    Parameters
    ----------
    ifft_pred : :class:`~torch.Tensor` or :func:`numpy.ndarray`
        Predicted image (inverse FFT result), shape (N, 3).
    ifft_truth : :class:`~torch.Tensor` or :func:`numpy.ndarray`
        Ground truth image (inverse FFT result), shape (N, 3).

    Returns
    -------
    blobs_log_pred : :func:`~numpy.ndarray`
        Detected blobs in prediction, shape (N, 3) with columns [y, x, radius].
    blobs_log_truth : :func:`~numpy.ndarray`
        Detected blobs in ground truth, shape (N, 3) with columns [y, x, radius].
    """
    if isinstance(ifft_pred, torch.Tensor):
        ifft_pred = ifft_pred.detach().cpu().numpy()

    if isinstance(ifft_truth, torch.Tensor):
        ifft_truth = ifft_truth.detach().cpu().numpy()

    treshold = ifft_truth.max() * 0.1
    kwargs = {
        "min_sigma": 1,
        "max_sigma": 10,
        "num_sigma": 100,
        "threshold": treshold,
        "overlap": 0.9,
    }

    blobs_log_pred = blob_log(ifft_pred, **kwargs)
    blobs_log_truth = blob_log(ifft_truth, **kwargs)

    # Compute radii in the 3rd column.
    blobs_log_pred[:, 2] = blobs_log_pred[:, 2] * sqrt(2)
    blobs_log_truth[:, 2] = blobs_log_truth[:, 2] * sqrt(2)

    return blobs_log_pred, blobs_log_truth


def crop_first_component(
    pred: ArrayLike,
    truth: ArrayLike,
    blob_truth: list | tuple,
) -> tuple[NDArray, NDArray]:
    """Return cropped images around the first component of the true image.

    Parameters
    ----------
    pred : :func:`~numpy.ndarray`
        Predicted source image.
    truth : :func:`~numpy.ndarray`
        True source image.
    blob_truth : list or tuple
        Coordinates (y, x, r) for the first component.

    Returns
    -------
    flux_pred : :func:`~numpy.ndarray`
        Cropped prediction image.
    flux_truth : :func:`~numpy.ndarray`
        Cropped truth image.
    """
    y, x, r = blob_truth
    x_coord, y_coord = _corners(y, x, r)

    flux_truth = truth[x_coord[0] : x_coord[1], y_coord[0] : y_coord[1]]
    flux_pred = pred[x_coord[0] : x_coord[1], y_coord[0] : y_coord[1]]

    return flux_pred, flux_truth


def _corners(
    x: int | float,
    y: int | float,
    r: int | float,
) -> tuple[list[int], list[int]]:
    """Generate coordinate ranges for cropping the first component.

    Parameters
    ----------
    x : int or float
        X coordinate of the component center.
    y : int or float
        Y coordinate of the component center.
    r : int or float
        Radius of the first component.

    Returns
    -------
    x_coord : list of int
        Start and end indices for x-axis cropping.
    y_coord : list of int
        Start and end indices for y-axis cropping.
    """
    r = int(np.round(r))
    x = int(x)
    y = int(y)

    x_coord = [x - r, x + r + 1]
    y_coord = [y - r, y + r + 1]

    return x_coord, y_coord
