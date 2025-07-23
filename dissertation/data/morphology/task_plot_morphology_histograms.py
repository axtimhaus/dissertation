from pytask import task
from dissertation.config import FIGSIZE_INCH, ROOT_DIR, image_produces, in_build_dir
from dissertation.data.morphology.batches import BATCHES, BATCHES_DIR
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from scipy.stats import Mixture, make_distribution, norm, beta, weibull_min, uniform
from scipy.optimize import least_squares

DENSITY_COLOR = "C0"
CUMULATIVE_COLOR = "C1"

for b in BATCHES:

    @task(id=b)
    def task_plot_morphology_histograms_circular(
        fits_file=in_build_dir(BATCHES_DIR / "fits_circular" / f"{b}.csv"),
        produces=image_produces(in_build_dir(BATCHES_DIR / "hist_circular" / f"{b}.png")),
    ):
        df = pd.read_csv(fits_file, header=0, index_col=0)

        fig, ax = plt.subplots()
        twin = ax.twinx()

        r0_dist, r0_dist_text = fit_weibull_mix2(df["r0"], 0, bins=100)
        pdf(ax, df["r0"], r0_dist, bins=100)
        cdf(twin, df["r0"], r0_dist, bins=100)
        twin.annotate(r0_dist_text, (0.6, 0.3), xycoords="axes fraction")

        ax.set_xlabel(r"$\Radius_0$ in \unit{\micro\meter}")

        ax.tick_params(axis="y", colors=DENSITY_COLOR)
        ax.yaxis.label.set_color(DENSITY_COLOR)
        ax.set_ylabel(r"probability density")

        ax.tick_params(axis="y", colors=CUMULATIVE_COLOR)
        ax.yaxis.label.set_color(CUMULATIVE_COLOR)
        ax.set_ylabel(r"cumulative density")

        for p in produces:
            fig.savefig(p)

    @task(id=b)
    def task_plot_morphology_histograms_oval(
        fits_file=in_build_dir(BATCHES_DIR / "fits_oval" / f"{b}.csv"),
        produces=image_produces(in_build_dir(BATCHES_DIR / "hist_oval" / f"{b}.png")),
    ):
        df = pd.read_csv(fits_file, header=0, index_col=0)

        fig, axs = plt.subplots(2, 1)
        twins = [ax.twinx() for ax in axs]

        r0_dist, r0_dist_text = fit_weibull_mix2(df["r0"], 0, bins=100)
        pdf(axs[0], df["r0"], r0_dist, bins=100)
        cdf(twins[0], df["r0"], r0_dist, bins=100)
        twins[0].annotate(r0_dist_text, (0.6, 0.3), xycoords="axes fraction")

        o_dist, o_dist_text = fit_weibull(df["o"], 1)
        pdf(axs[1], df["o"], o_dist, lower=1)
        cdf(twins[1], df["o"], o_dist, lower=1)
        twins[1].annotate(o_dist_text, (0.5, 0.3), xycoords="axes fraction")

        axs[0].set_xlabel(r"$\Radius_0$ in \unit{\micro\meter}")
        axs[1].set_xlabel(r"$\Ovality$")

        for ax in axs:
            ax.tick_params(axis="y", colors=DENSITY_COLOR)
            ax.yaxis.label.set_color(DENSITY_COLOR)
            ax.set_ylabel(r"probability density")

        for ax in twins:
            ax.tick_params(axis="y", colors=CUMULATIVE_COLOR)
            ax.yaxis.label.set_color(CUMULATIVE_COLOR)
            ax.set_ylabel(r"cumulative density")

        for p in produces:
            fig.savefig(p)

    @task(id=b)
    def task_plot_morphology_histograms_shape(
        fits_file=in_build_dir(BATCHES_DIR / "fits_shape" / f"{b}.csv"),
        produces=image_produces(in_build_dir(BATCHES_DIR / "hist_shape" / f"{b}.png")),
    ):
        df = pd.read_csv(fits_file, header=0, index_col=0)

        fig = plt.figure(figsize=(FIGSIZE_INCH[0], 1.5 * FIGSIZE_INCH[1]))
        gs = GridSpec(3, 2, figure=fig)
        axs = [
            fig.add_subplot(gs[0, :]),
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[1, 1]),
            fig.add_subplot(gs[2, 0]),
            fig.add_subplot(gs[2, 1]),
        ]
        twins = [ax.twinx() for ax in axs]

        r0_dist, r0_dist_text = fit_weibull_mix2(df["r0"], 0, bins=100)
        pdf(axs[0], df["r0"], r0_dist, bins=100)
        cdf(twins[0], df["r0"], r0_dist, bins=100)
        twins[0].annotate(r0_dist_text, (0.6, 0.3), xycoords="axes fraction")

        o_dist, o_dist_text = fit_weibull(df["o"], 1)
        pdf(axs[1], df["o"], o_dist, lower=1)
        cdf(twins[1], df["o"], o_dist, lower=1)
        twins[1].annotate(o_dist_text, (0.5, 0.3), xycoords="axes fraction")

        h_dist, h_dist_text = fit_beta(df["h"])
        pdf(axs[2], df["h"], h_dist)
        cdf(twins[2], df["h"], h_dist)
        twins[2].annotate(h_dist_text, (0.5, 0.5), xycoords="axes fraction")

        p_dist = uniform(0, 0.5)
        pdf(axs[3], df["p"], p_dist)
        cdf(twins[3], df["p"], p_dist)
        twins[3].annotate("uniform", (0.5, 0.1), xycoords="axes fraction")

        pdf_int(axs[4], df["n"])
        cdf_int(twins[4], df["n"])

        axs[0].set_xlabel(r"$\Radius_0$ in \unit{\micro\meter}")
        axs[1].set_xlabel(r"$\Ovality$")
        axs[2].set_xlabel(r"$\WaveHeight$")
        axs[3].set_xlabel(r"$\WaveShift$")
        axs[4].set_xlabel(r"$\WaveCount$")

        for ax in axs:
            ax.tick_params(axis="y", colors=DENSITY_COLOR)
            ax.yaxis.label.set_color(DENSITY_COLOR)
            ax.set_ylabel(r"probability density")

        for ax in twins:
            ax.tick_params(axis="y", colors=CUMULATIVE_COLOR)
            ax.yaxis.label.set_color(CUMULATIVE_COLOR)
            ax.set_ylabel(r"cumulative density")

        axs[-1].set_ylabel(r"relative frequency")
        twins[-1].set_ylabel(r"cumulative frequency")

        for p in produces:
            fig.savefig(p)


def fit_weibull(data, lower: float, bins=20):
    hist, edges = np.histogram(data, bins=bins, range=(lower, np.max(data)), density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    cum_hist = np.cumsum(hist) * np.diff(edges)

    fit = least_squares(
        lambda p: cum_hist - weibull_min.cdf(centers, c=p[0], scale=p[1], loc=lower),
        x0=[1, 1],
    )
    return weibull_min(
        c=fit.x[0], scale=fit.x[1], loc=lower
    ), f"$k = \\num{{{fit.x[0]:.6f}}}$\n$S = \\num{{{fit.x[1]:.6f}}}$\n$L = \\num{{1}}$"


def fit_beta(data, bins=20):
    hist, edges = np.histogram(data, bins=bins, range=(0, np.max(data)), density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    cum_hist = np.cumsum(hist) * np.diff(edges)

    fit = least_squares(
        lambda p: cum_hist - beta.cdf(centers, a=p[0], b=p[1]),
        x0=[1, 1],
    )

    return beta(a=fit.x[0], b=fit.x[1]), f"$\\alpha = \\num{{{fit.x[0]:.6f}}}$\n$\\beta = \\num{{{fit.x[1]:.6f}}}$"


def fit_weibull_mix2(data, lower: float, bins=100):
    hist, edges = np.histogram(data, bins=bins, range=(lower, np.max(data)), density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    cum_hist = np.cumsum(hist) * np.diff(edges)

    Weibull = make_distribution(weibull_min)

    def weibull_mix2(c1, s1, c2, s2, w):
        return Mixture([Weibull(c=c1) * s1 + lower, Weibull(c=c2) * s2 + lower], weights=[w, 1 - w])

    fit = least_squares(
        lambda p: cum_hist - weibull_mix2(*p).cdf(centers),
        x0=[1, 500, 1, 1500, 0.8],
        bounds=np.transpose([(0, np.inf)] * 4 + [(0, 1)]),
    )

    return (
        weibull_mix2(*fit.x),
        f"$k_1 = \\num{{{fit.x[0]:.6f}}}$\n$S_1 = \\num{{{fit.x[1]:.6f}}}$\n$k_2 = \\num{{{fit.x[2]:.6f}}}$\n$S_2 = \\num{{{fit.x[3]:.6f}}}$\n$w = \\num{{{fit.x[4]:.6f}}}$\n$L_1 = L_2 = \\num{{0}}$",
    )


def pdf(ax, data, dist, bins=20, lower=0):
    ax.hist(data, bins=bins, density=True, alpha=0.5, color=DENSITY_COLOR)
    if dist:
        x = np.linspace(lower, data.max(), 201)
        ax.plot(x, dist.pdf(x), c=DENSITY_COLOR)


def cdf(ax, data, dist, bins=20, lower=0):
    ax.hist(data, bins=bins, density=True, alpha=0.5, color=CUMULATIVE_COLOR, cumulative=True)
    if dist:
        x = np.linspace(lower, data.max(), 201)
        ax.plot(x, dist.cdf(x), c=CUMULATIVE_COLOR)


def pdf_int(ax, data):
    locs, heights = np.unique_counts(data)
    heights = heights / np.sum(heights)
    ax.bar(locs, heights, alpha=0.5, color=DENSITY_COLOR)


def cdf_int(ax, data):
    locs, heights = np.unique_counts(data)
    heights = np.cumsum(heights) / np.sum(heights)
    ax.bar(locs, heights, alpha=0.5, color=CUMULATIVE_COLOR)
