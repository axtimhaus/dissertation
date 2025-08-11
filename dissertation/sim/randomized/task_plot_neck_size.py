import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from matplotlib.gridspec import GridSpec
from pytask import mark, task

from dissertation.config import FIGSIZE_INCH, image_produces
from dissertation.sim.randomized.cases import CASES
from dissertation.sim.randomized.input import REFERENCE_PARTICLE, TIME_NORM_SURFACE

NECK_SIZE_LIMITS = (1e-3, 1e-1)
CUTS = [1e-6, 1e-4]
CUT_COLORS = ["C3", "C4"]

for case in CASES:

    @task(id=case.key)
    @mark.plot
    def task_plot_neck_size_randomized(
        produces=image_produces(case.dir() / "neck_size"),
        results_files=[case.dir(i) / "output.parquet" for i, _ in enumerate(case.samples)],
        case=case,
    ):
        data_frames = (pq.read_table(f).flatten().flatten() for f in results_files)

        fig = plt.figure(figsize=(FIGSIZE_INCH[0], FIGSIZE_INCH[1] * 1.5))
        gs = GridSpec(3, 2, figure=fig)
        axs = [
            fig.add_subplot(gs[0:2, :]),
            fig.add_subplot(gs[2, 0]),
            fig.add_subplot(gs[2, 1]),
        ]
        axs[0].set_xscale("log")
        axs[0].set_yscale("log")
        axs[0].set_xlim(1e-7, 1e-3)
        axs[0].set_ylim(*NECK_SIZE_LIMITS)
        axs[0].grid(True, "both")

        cuts = {t: np.zeros_like(results_files) for t in CUTS}

        for i, df in enumerate(data_frames):
            times, values = get_neck_sizes(df)
            axs[0].plot(times, values, **case.LINE_STYLE, alpha=0.3)

            for t, arr in cuts.items():
                arr[i] = np.interp(t, times, values)

        for i, (t, arr) in enumerate(cuts.items()):
            axs[0].axvline(t, color=CUT_COLORS[i])
            axs[i + 1].set_title(f"$\\Time / \\TimeNorm_{{\\Surface}} = \\num[print-unity-mantissa=false]{{{t:.0e}}}$")
            axs[i + 1].hist(arr, bins=np.geomspace(*NECK_SIZE_LIMITS, 51), density=True, color=CUT_COLORS[i], alpha=0.5)
            axs[i + 1].axvline(np.mean(arr), color=CUT_COLORS[i])

        axs[0].set_xlabel("Normalized Time $\\Time / \\TimeNorm_{\\Surface}$")
        axs[0].set_ylabel("Average Relative Neck Size $\\Radius_{\\Neck} / \\Radius_0$")

        for ax in axs[1:]:
            ax.set_xlabel("Average Relative Neck Size $\\Radius_{\\Neck} / \\Radius_0$")
            ax.set_ylabel("Probability Density")
            ax.set_xscale("log")

        for p in produces:
            fig.savefig(p)

        plt.close(fig)


def get_neck_sizes(df: pa.Table):
    grain_boundary: pd.DataFrame = (
        df.filter(pc.field("Node.Type") == 1)
        .group_by(["State.Id"])
        .aggregate(
            [
                ("State.Time", "one"),
                ("Node.SurfaceDistance.ToUpper", "mean"),
                ("Node.SurfaceDistance.ToLower", "mean"),
            ]
        )
        .sort_by("State.Time_one")
        .to_pandas()
    )

    times = grain_boundary["State.Time_one"].to_numpy() / TIME_NORM_SURFACE
    mask = np.diff(times, prepend=[0]) > 0

    neck_sizes = (
        (
            grain_boundary["Node.SurfaceDistance.ToUpper_mean"][mask]
            + grain_boundary["Node.SurfaceDistance.ToLower_mean"][mask]
        )
        / REFERENCE_PARTICLE.radius
        / 2
    )

    return times[mask], neck_sizes
