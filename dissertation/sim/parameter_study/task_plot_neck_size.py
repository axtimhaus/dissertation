from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pytask import task

from dissertation.config import image_produces, integer_log_space
from dissertation.sim.parameter_study.studies import PARTICLE1_ID, STUDIES

THIS_DIR = Path(__file__).parent
RESAMPLE_COUNT = 500

for study in STUDIES:

    @task(id=f"{study}")
    def task_plot_neck_size(
        study=study,
        produces=image_produces(study.dir("plots") / "neck_size"),
        results_files={value: study.dir(value) / "output.parquet" for value in study.parameter_values},
    ):
        data_frames = load_data(results_files)

        fig: plt.Figure = plt.figure(dpi=600)
        ax: plt.Axes = fig.subplots()
        ax.set_xscale("log")
        ax.set_yscale("log")

        for key, df in data_frames.items():
            times, neck_sizes = get_neck_sizes(key, df)
            ax.plot(times, neck_sizes, label=f"{key:.2f}")

        ax.legend(title=study.display_tex, ncols=2)
        ax.set_xlabel("Normalized Time $\\Time / \\TimeNorm_{\\Surface}$")
        ax.set_ylabel(r"Relative Neck Size $\Radius_{\Neck} / \Radius_0$")

        for p in produces:
            fig.savefig(p)


    @task(id=f"{study}")
    def task_plot_neck_size_map(
            study=study,
            produces=image_produces(study.dir("plots") / "neck_size_map"),
            results_files={value: study.dir(value) / "output.parquet" for value in study.parameter_values}
    ):
        data_frames = load_data(results_files)

        fig: plt.Figure = plt.figure(dpi=600)
        ax: plt.Axes = fig.subplots()
        ax.set_xscale("log")
        ax.set_yscale("log")

        params = np.array(list(data_frames.keys()))
        times_neck_size = [get_neck_sizes(key, df) for key, df in data_frames.items()]

        start_time = max([t[0] for t, _ in times_neck_size])
        end_time = max([t[-1] for t, _ in times_neck_size])

        times = np.geomspace(start_time, end_time, RESAMPLE_COUNT)
        neck_sizes = np.array([np.interp(times, t, s) for t, s in times_neck_size])

        grid_x, grid_y = np.meshgrid(times, params)

        locs = integer_log_space(1, -1, 1, 0)

        cs = ax.contour(grid_x, grid_y, neck_sizes, levels=locs, norm="log", cmap="copper")
        ax.clabel(cs, fmt=lambda level: f"{level:g}")

        ax.set_xlabel("Normalized Time $\\Time / \\TimeNorm_{\\Surface}$")
        ax.set_ylabel(study.display_tex)

        for p in produces:
            fig.savefig(p)


def load_data(results_files):
    return {k: pq.read_table(f).flatten().flatten() for k, f in results_files.items()}


def get_neck_sizes(param, df: pa.Table):
    grain_boundary: pd.DataFrame = (
        df.filter((pc.field("Particle.Id") == PARTICLE1_ID.bytes) & (pc.field("Node.Type") == 1))
        .group_by(["State.Id"])
        .aggregate(
            [
                ("State.Time", "one"),
                ("Node.SurfaceDistance.ToUpper", "one"),
                ("Node.SurfaceDistance.ToLower", "one"),
            ]
        )
        .sort_by("State.Time_one")
        .to_pandas()
    )

    input = study.input_for(param)
    mask = (grain_boundary["State.Time_one"] > 1) & (np.diff(grain_boundary["State.Time_one"], prepend=[0]) > 0)
    times = grain_boundary["State.Time_one"][mask] / input.time_norm_surface
    neck_sizes = (
            grain_boundary["Node.SurfaceDistance.ToUpper_one"][mask]
            + grain_boundary["Node.SurfaceDistance.ToLower_one"][mask] / input.particle1.radius
    )

    return times.array, neck_sizes.array
