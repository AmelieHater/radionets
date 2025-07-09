from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def histogram_jet_angles(dif, out_path, plot_format="png"):
    mean = np.mean(dif)
    std = np.std(dif, ddof=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
    ax1.hist(dif, 51, color="darkorange", linewidth=3, histtype="step", alpha=0.75)
    ax1.set_xlabel("Offset / deg")
    ax1.set_ylabel("Number of sources")

    extra_1 = Rectangle(
        (0, 0), 1, 1, fc="w", fill=False, edgecolor="darkorange", linewidth=1
    )
    extra_2 = Rectangle(
        (0, 0), 1, 1, fc="w", fill=False, edgecolor="darkorange", linewidth=1
    )
    ax1.legend([extra_1, extra_2], (f"Mean: {mean:.2f}", f"Std: {std:.2f}"))

    ax2.hist(
        dif[(dif > -10) & (dif < 10)],
        25,
        color="darkorange",
        linewidth=3,
        histtype="step",
        alpha=0.75,
    )
    ax2.set_xticks([-10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10])
    ax2.set_xlabel("Offset / deg")
    ax2.set_ylabel("Number of sources")

    fig.tight_layout()

    outpath = str(out_path) + f"/jet_offsets.{plot_format}"
    plt.savefig(outpath, bbox_inches="tight", pad_inches=0.01, dpi=150)


def histogram_dynamic_ranges(dr_truth, dr_pred, out_path, plot_format="png"):
    # dif = dr_pred - dr_truth

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 12))
    ax1.set_title("True Images")
    ax1.hist(dr_truth, 51, color="darkorange", linewidth=3, histtype="step", alpha=0.75)
    ax1.set_xlabel("Dynamic range")
    ax1.set_ylabel("Number of sources")

    ax2.set_title("Predictions")
    ax2.hist(dr_pred, 25, color="darkorange", linewidth=3, histtype="step", alpha=0.75)
    ax2.set_xlabel("Dynamic range")
    ax2.set_ylabel("Number of sources")

    fig.tight_layout()

    outpath = str(out_path) + f"/dynamic_ranges.{plot_format}"
    plt.savefig(outpath, bbox_inches="tight", pad_inches=0.01, dpi=150)


def histogram_ms_ssim(msssim, out_path, bins=30, plot_format="png"):
    mean = np.mean(msssim)
    std = np.std(msssim, ddof=1)
    fig, (ax1) = plt.subplots(1, figsize=(6, 4))
    ax1.hist(
        msssim,
        bins=bins,
        color="darkorange",
        linewidth=3,
        histtype="step",
        alpha=0.75,
    )
    ax1.set_xlabel("ms ssim")
    ax1.set_ylabel("Number of sources")

    ax1.text(
        0.1,
        0.8,
        f"Mean: {mean:.2f}\nStd: {std:.2f}",
        horizontalalignment="left",
        verticalalignment="center",
        transform=ax1.transAxes,
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            edgecolor="lightgray",
            alpha=0.8,
        ),
    )
    fig.tight_layout()

    outpath = str(out_path) + f"/ms_ssim.{plot_format}"
    plt.savefig(outpath, bbox_inches="tight", pad_inches=0.01, dpi=150)


def histogram_sum_intensity(ratios_sum, out_path, bins=30, plot_format="png"):
    fig, (ax1) = plt.subplots(1, figsize=(6, 4))
    mean = np.mean(ratios_sum)
    std = np.std(ratios_sum, ddof=1)
    ax1.hist(
        ratios_sum,
        bins=bins,
        color="darkorange",
        linewidth=3,
        histtype="step",
        alpha=0.75,
    )
    ax1.axvline(1, color="red", linestyle="dashed")
    ax1.set_xlabel("Ratio of integrated flux densities")
    ax1.set_ylabel("Number of sources")

    ax1.text(
        0.1,
        0.8,
        f"Mean: {mean:.2f}\nStd: {std:.2f}",
        horizontalalignment="left",
        verticalalignment="center",
        transform=ax1.transAxes,
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            edgecolor="lightgray",
            alpha=0.8,
        ),
    )

    fig.tight_layout()

    outpath = str(out_path) + f"/intensity_sum.{plot_format}"
    plt.savefig(outpath, bbox_inches="tight", pad_inches=0.01, dpi=150)


def histogram_peak_intensity(ratios_peak, out_path, bins=30, plot_format="png"):
    fig, (ax1) = plt.subplots(1, figsize=(6, 4))
    mean = np.mean(ratios_peak)
    std = np.std(ratios_peak, ddof=1)
    ax1.hist(
        ratios_peak,
        bins=bins,
        color="darkorange",
        linewidth=3,
        histtype="step",
        alpha=0.75,
    )
    ax1.axvline(1, color="red", linestyle="dashed")
    ax1.set_xlabel("Ratio of peak flux densities")
    ax1.set_ylabel("Number of sources")

    ax1.text(
        0.1,
        0.8,
        f"Mean: {mean:.2f}\nStd: {std:.2f}",
        horizontalalignment="left",
        verticalalignment="center",
        transform=ax1.transAxes,
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            edgecolor="lightgray",
            alpha=0.8,
        ),
    )

    fig.tight_layout()

    outpath = str(out_path) + f"/intensity_peak.{plot_format}"
    plt.savefig(outpath, bbox_inches="tight", pad_inches=0.01, dpi=150)


def histogram_mean_diff(vals, out_path, plot_format="png"):
    vals = vals.numpy()
    mean = np.mean(vals)
    std = np.std(vals, ddof=1)
    fig, (ax1) = plt.subplots(1, figsize=(6, 4))
    ax1.hist(vals, 51, color="darkorange", linewidth=3, histtype="step", alpha=0.75)
    ax1.set_xlabel("Mean flux deviation / %")
    ax1.set_ylabel("Number of sources")
    extra_1 = Rectangle(
        (0, 0), 1, 1, fc="w", fill=False, edgecolor="darkorange", linewidth=1
    )
    extra_2 = Rectangle(
        (0, 0), 1, 1, fc="w", fill=False, edgecolor="darkorange", linewidth=1
    )
    ax1.legend([extra_1, extra_2], (f"Mean: {mean:.2f}", f"Std: {std:.2f}"))

    fig.tight_layout()

    outpath = str(out_path) + f"/mean_diff.{plot_format}"
    plt.savefig(outpath, bbox_inches="tight", pad_inches=0.01, dpi=150)


def histogram_area(vals, out_path, bins=30, plot_format="png"):
    vals = vals.numpy()
    mean = np.mean(vals)
    std = np.std(vals, ddof=1)
    fig, (ax1) = plt.subplots(1, figsize=(6, 4))
    ax1.hist(
        vals, bins=bins, color="darkorange", linewidth=3, histtype="step", alpha=0.75
    )
    ax1.axvline(1, color="red", linestyle="dashed")
    ax1.set_xlabel("Ratio of areas")
    ax1.set_ylabel("Number of sources")

    ax1.text(
        0.1,
        0.8,
        f"Mean: {mean:.2f}\nStd: {std:.2f}",
        horizontalalignment="left",
        verticalalignment="center",
        transform=ax1.transAxes,
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            edgecolor="lightgray",
            alpha=0.8,
        ),
    )

    fig.tight_layout()

    outpath = str(out_path) + f"/hist_area.{plot_format}"
    plt.savefig(outpath, bbox_inches="tight", pad_inches=0.01, dpi=150)


def hist_point(vals, mask, out_path, plot_format="png"):
    binwidth = 5
    min_all = vals.min()
    bins = np.arange(min_all, 100 + binwidth, binwidth)

    mean_point = np.mean(vals[mask])
    std_point = np.std(vals[mask], ddof=1)
    mean_extent = np.mean(vals[~mask])
    std_extent = np.std(vals[~mask], ddof=1)
    fig, (ax1) = plt.subplots(1, figsize=(6, 4))
    ax1.hist(
        vals[mask],
        bins=bins,
        color="darkorange",
        linewidth=2,
        histtype="step",
        alpha=0.75,
    )
    ax1.hist(
        vals[~mask],
        bins=bins,
        color="#1f77b4",
        linewidth=2,
        histtype="step",
        alpha=0.75,
    )
    ax1.axvline(0, linestyle="dotted", color="red")
    ax1.set_ylabel("Number of sources")
    ax1.set_xlabel("Mean specific intensity deviation")

    extra_1 = Rectangle(
        (0, 0), 1, 1, fc="w", fill=False, edgecolor="darkorange", linewidth=2
    )
    extra_2 = Rectangle(
        (0, 0), 1, 1, fc="w", fill=False, edgecolor="#1f77b4", linewidth=2
    )
    ax1.legend(
        [extra_1, extra_2],
        [
            rf"Point: $({mean_point:.2f}\pm{std_point:.2f})\,\%$",
            rf"Extended: $({mean_extent:.2f}\pm{std_extent:.2f})\,\%$",
        ],
    )
    outpath = str(out_path) + f"/hist_point.{plot_format}"
    plt.savefig(outpath, bbox_inches="tight", pad_inches=0.01, dpi=150)


def hist_jet_gaussian_distance(dist, path, save=False, plot_format="pdf"):
    """
    Plotting the distances between predicted and true component of several images.
    Parameters
    ----------
    dist: 2d array
        array of shape (n, 2), where n is the number of distances
    """
    ran = [0, 50]

    fig, ax = plt.subplots()

    for i in range(10):
        ax.hist(
            dist[dist[:, 0] == i][:, 1],
            bins=20,
            range=ran,
            alpha=0.7,
            label=f"Component {i}",
        )

    ax.set(xlabel="Distance", ylabel="Counts")
    ax.legend()

    if save:
        Path(path).mkdir(parents=True, exist_ok=True)
        outpath = str(path) + f"/hist_jet_gaussian_distance.{plot_format}"
        plt.savefig(outpath, bbox_inches="tight", pad_inches=0.01)

    plt.close()


def histogram_gan_sources(
    ratio, num_zero, above_zero, below_zero, num_images, out_path, plot_format="png"
):
    fig, ax1 = plt.subplots(1)
    bins = np.arange(0, ratio.max() + 0.1, 0.1)
    ax1.hist(
        ratio,
        bins=bins,
        histtype="step",
        label=f"mean: {ratio.mean():.2f}, max: {ratio.max():.2f}",
    )
    ax1.set_xlabel(r"Maximum difference to maximum true flux ratio")
    ax1.set_ylabel(r"Number of sources")
    ax1.legend(loc="best")

    fig.tight_layout()

    outpath = str(out_path) + f"/ratio.{plot_format}"
    plt.savefig(outpath, bbox_inches="tight", pad_inches=0.01, dpi=150)

    plt.clf()

    bins = np.arange(0, 102, 2)
    num_zero = num_zero.reshape(4, num_images)
    for i, label in enumerate(["1e-4", "1e-3", "1e-2", "1e-1"]):
        plt.hist(num_zero[i], bins=bins, histtype="step", label=label)
    plt.xlabel(r"Proportion of pixels close to 0 / %")
    plt.ylabel(r"Number of sources")
    plt.legend(loc="upper center")

    plt.tight_layout()

    outpath = str(out_path) + f"/num_zeros.{plot_format}"
    plt.savefig(outpath, bbox_inches="tight", pad_inches=0.01, dpi=150)

    plt.clf()

    bins = np.arange(0, 102, 2)
    plt.hist(
        above_zero,
        bins=bins,
        histtype="step",
        label=f"Above, mean: {above_zero.mean():.2f}%, max: {above_zero.max():.2f}%",
    )
    plt.hist(
        below_zero,
        bins=bins,
        histtype="step",
        label=f"Below, mean: {below_zero.mean():.2f}%, max: {below_zero.max():.2f}%",
    )
    plt.xlabel(r"Proportion of pixels below or above 0%")
    plt.ylabel(r"Number of sources")
    plt.legend(loc="upper center")
    plt.tight_layout()

    outpath = str(out_path) + f"/above_below.{plot_format}"
    plt.savefig(outpath, bbox_inches="tight", pad_inches=0.01, dpi=150)


def histogram_unc(vals, out_path, plot_format="png"):
    mean = np.mean(vals)
    std = np.std(vals, ddof=1)
    bins = np.arange(0, 105, 5)
    fig, (ax1) = plt.subplots(1, figsize=(6, 4))
    ax1.hist(
        vals, bins=bins, color="darkorange", linewidth=3, histtype="step", alpha=0.75
    )
    ax1.set_xlabel("Percentage of matching pixels")
    ax1.set_ylabel("Number of sources")

    ax1.text(
        0.1,
        0.8,
        f"Mean: {mean:.2f}\nStd: {std:.2f}",
        horizontalalignment="left",
        verticalalignment="center",
        transform=ax1.transAxes,
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            edgecolor="lightgray",
            alpha=0.8,
        ),
    )

    fig.tight_layout()

    outpath = str(out_path) + f"/hist_unc.{plot_format}"
    plt.savefig(outpath, bbox_inches="tight", pad_inches=0.01, dpi=150)
