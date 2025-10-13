import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from matplotlib.gridspec import GridSpec
from pytask import mark, task

from dissertation.config import DEFAULT_FIGSIZE, image_produces
from dissertation.sim.randomized.cases import CASES, MEAN_LINE_STYLE, NOMINAL
from dissertation.sim.randomized.helper import read_parquet_output_files
from dissertation.sim.randomized.input import REFERENCE_PARTICLE, TIME_NORM_SURFACE

NECK_SIZE_LIMITS = (1e-1, 1e-0)
CUTS = [1e-6, 1e-4]
CUT_COLORS = ["C2", "C4"]

for case in CASES:
    for count in [None, 1, 2, 10, 100]:

        @task(id=f"{case.key}{count or ''}")
        @mark.plot
        def task_plot_neck_size_randomized(
            produces=image_produces(case.dir() / f"neck_size{count or ''}"),
            results_files=[case.dir(i) / "output.parquet" for i, _ in enumerate(case.samples)][:count],
            nominal_result=NOMINAL.dir() / "output.parquet",
            case=case,
            alpha=0.7 if count else 0.1,
        ):
            nominal_df = pq.read_table(nominal_result).flatten().flatten()
            data_frames = read_parquet_output_files(results_files)

            fig = plt.figure(figsize=tuple(DEFAULT_FIGSIZE * [1, 2]))
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
            mean_supports = {t: np.zeros_like(results_files) for t in np.geomspace(1e-7, 1e-3, 100)}
            sample_label = "individual samples"

            for i, df in data_frames:
                times, values = get_neck_sizes(df)
                axs[0].plot(times, values, **case.LINE_STYLE, alpha=alpha, label=sample_label)
                sample_label = None

                for t, arr in cuts.items():
                    arr[i] = np.interp(t, times, values)

                for t, arr in mean_supports.items():
                    arr[i] = np.interp(t, times, values)

            for i, (t, arr) in enumerate(cuts.items()):
                axs[0].axvline(t, color=CUT_COLORS[i])
                axs[i + 1].set_title(
                    f"$\\Time / \\TimeNorm_{{\\Surface}} = \\num[print-unity-mantissa=false]{{{t:.0e}}}$"
                )
                axs[i + 1].hist(
                    arr, bins=np.geomspace(*NECK_SIZE_LIMITS, 51), density=True, color=CUT_COLORS[i], alpha=0.5
                )
                mean = np.mean(arr)
                std = np.std(arr)
                axs[i + 1].axvline(
                    mean,
                    color=CUT_COLORS[i],
                    label=f"$\\Estimated\\Expectation = \\qty{{{mean * 100:.3f}}}{{\\percent}}$",
                )
                axs[i + 1].axvline(
                    mean + std,
                    color=CUT_COLORS[i],
                    ls="--",
                    label=f"$\\Estimated\\StandardDeviation = \\qty{{{std * 100:.3f}}}{{\\percent}}$",
                )
                axs[i + 1].axvline(mean - std, color=CUT_COLORS[i], ls="--")

            times, values = get_neck_sizes(nominal_df)
            axs[0].plot(times, values, **NOMINAL.LINE_STYLE, label="nominal")

            for i, v in enumerate(np.interp(t, times, values) for t in CUTS):
                axs[i + 1].axvline(
                    v, **NOMINAL.LINE_STYLE, label=f"$\\text{{nom.}} = \\qty{{{v * 100:.3f}}}{{\\percent}}$"
                )

            times, values = np.asarray(list(mean_supports.keys())), np.mean(list(mean_supports.values()), axis=-1)
            axs[0].plot(times, values, **MEAN_LINE_STYLE, label="mean")

            axs[0].set_xlabel("Normalized Time $\\Time / \\TimeNorm_{\\Surface}$")
            axs[0].set_ylabel("Average Relative Neck Size $\\Radius_{\\Neck} / \\Radius_0$")
            axs[0].legend()

            for ax in axs[1:]:
                ax.set_xlabel("Average Relative Neck Size $\\Radius_{\\Neck} / \\Radius_0$")
                ax.set_ylabel("Probability Density")
                ax.set_xscale("log")
                ax.legend()

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
