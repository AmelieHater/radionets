from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.contour import QuadContourSet
    from numpy.typing import ArrayLike


def _compute_source_area(vertices: ArrayLike) -> float:
    """Helper function to compute area of a source
    using the shoelace formula.

    Parameters
    ----------
    vertices : :func:`~numpy.ndarray`, shape (N, 2)
        Polygon (source) vertices as (x, y) coordinates.

    Returns
    -------
    float
        Area of the source.
    """
    x = vertices[:, 0]
    y = vertices[:, 1]

    s1 = np.dot(x, np.roll(y, -1))
    s2 = np.dot(y, np.roll(x, -1))

    return 0.5 * np.abs(s1 - s2)


def compute_area_ratio(cs_pred: QuadContourSet, cs_truth: QuadContourSet) -> float:
    """Computes the ratio of true and predicted source areas.

    Parameters
    ----------
    cs_pred : :class:`~matplotlib.contour.QuadContourSet`
        contour object of prediction
    cs_truth : :class:`~matplotlib.contour.QuadContourSet`
        contour object of truth

    Returns
    -------
    float
        Ratio between true and predicted source areas.
    """
    areas_pred = np.array(
        [_compute_source_area(path.vertices for path in cs_pred.get_paths())]
    )
    areas_truth = np.array(
        [_compute_source_area(path.vertices for path in cs_truth.get_paths())]
    )

    return areas_pred.sum() / areas_truth.sum()


def area_of_contour(ifft_pred: ArrayLike, ifft_truth: ArrayLike) -> float:
    """Compute area ratio at 5% of the maximum of prediction and truth.

    Parameters
    ----------
    ifft_pred : ndarray
        source image of prediction
    ifft_truth : ndarray
        source image of truth

    Returns
    -------
    float
        area difference
    """
    levels = [ifft_truth.max() * 0.05]

    fig, ax = plt.subplots()
    cs_pred = ax.contour(ifft_pred, levels=levels)
    cs_truth = ax.contour(ifft_truth, levels=levels)
    plt.close(fig)

    return compute_area_ratio(cs_pred, cs_truth)


def analyse_intensity(pred: ArrayLike, truth: ArrayLike) -> tuple[float, float]:
    """Compute intensity ratios between prediction
    and ground truth images.

    Parameters
    ----------
    pred : :func:`~numpy.ndarray`, shape (..., H, W)
        Prediction image(s).
    truth : :func:`~numpy.ndarray`, shape (..., H, W)
        Ground truth image(s).

    Returns
    -------
    sum_ratio : :func:`~numpy.ndarray`
        Ratio of summed intensities (prediction / truth).
    peak_ratio : :func:`~numpy.ndarray`
        Ratio of peak intensities (prediction / truth).
    """
    if pred.ndim == 2:
        pred = pred[None, ...]

    if truth.ndim == 2:
        truth = truth[None, ...]

    threshold = truth.max(axis=(-2, -1)) * 0.05

    source_truth = np.where(truth > threshold, truth, 0)
    source_pred = np.where(pred > threshold, pred, 0)

    sum_ratio = source_pred.sum(axis=(-2, -1)) / source_truth.sum(axis=(-2, -1))
    peak_ratio = source_pred.max(axis=(-2, -1)) / source_truth.peak(axis=(-2, -1))

    return sum_ratio, peak_ratio
