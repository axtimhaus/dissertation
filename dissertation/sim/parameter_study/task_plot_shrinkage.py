from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow as pa
import pyarrow.parquet as pq
from matplotlib import ticker
from pytask import task, mark

from dissertation.config import image_produces, integer_log_space
from dissertation.sim.parameter_study.studies import PARTICLE1_ID, PARTICLE2_ID, STUDIES, ParameterStudy

THIS_DIR = Path(__file__).parent
RESAMPLE_COUNT = 500

for study in STUDIES:

    @task(id=f"{study}")
    @mark.plot
    @mark.parameter_study
    def task_plot_shrinkage(
        study=study,
        produces=image_produces(study.dir("plots") / "shrinkage"),
        results_files={value: study.dir(value) / "output.parquet" for value in study.parameter_values},
    ):
        data_frames = load_data(results_files)

        fig: plt.Figure = plt.figure(dpi=600)
        ax: plt.Axes = fig.subplots()
        ax.set_xscale("log")
        ax.set_yscale("log")

        for key, df in data_frames.items():
            times, shrinkages = get_shrinkages(key, df, study)
            ax.plot(times, shrinkages, label=f"{key:.2f}")

        ax.legend(title=study.display_tex, ncols=2)
        ax.set_xlabel("Normalized Time $\\Time / \\TimeNorm_{\\Surface}$")
        ax.set_ylabel("Shrinkage")

        for p in produces:
            fig.savefig(p)


    @task(id=f"{study}")
    @mark.plot
    @mark.parameter_study
    def task_plot_shrinkage_map(
            study=study,
            produces=image_produces(study.dir("plots") / "shrinkage_map"),
            results_files={value: study.dir(value) / "output.parquet" for value in study.parameter_values}
    ):
        data_frames = load_data(results_files)

        fig: plt.Figure = plt.figure(dpi=600)
        ax: plt.Axes = fig.subplots()
        ax.set_xscale("log")
        ax.set_yscale("log")

        params = np.array(list(data_frames.keys()))
        times_shrinkages = [get_shrinkages(key, df, study) for key, df in data_frames.items()]

        start_time = max([t[0] for t, _ in times_shrinkages])
        end_time = max([t[-1] for t, _ in times_shrinkages])

        times = np.geomspace(start_time, end_time, RESAMPLE_COUNT)
        shrinkages = np.array([np.interp(times, t, s) for t, s in times_shrinkages])

        grid_x, grid_y = np.meshgrid(times, params)

        formatter = ticker.LogFormatterSciNotation()
        locs = integer_log_space(1, -3, 3, -1)

        cs = ax.contour(grid_x, grid_y, shrinkages, levels=locs, norm="log", cmap="copper")
        ax.clabel(cs, fmt=lambda level: formatter(level))

        ax.set_xlabel("Normalized Time $\\Time / \\TimeNorm_{\\Surface}$")
        ax.set_ylabel(study.display_tex)

        for p in produces:
            fig.savefig(p)


def load_data(results_files):
    return {k: pq.read_table(f).flatten().flatten() for k, f in results_files.items()}


def distance(particle1_x, particle1_y, particle2_x, particle2_y):
    return np.sqrt((particle2_x - particle1_x) ** 2 + (particle2_y - particle1_y) ** 2)


def get_shrinkages(param, df: pa.Table, study: ParameterStudy):
    particle1: pd.DataFrame = df.filter(pc.field("Particle.Id") == PARTICLE1_ID.bytes).group_by(["State.Id"]).aggregate(
        [("State.Time", "one"), ("Particle.Coordinates.X", "one"), ("Particle.Coordinates.Y", "one")]).sort_by(
        "State.Time_one").to_pandas()
    particle2: pd.DataFrame = df.filter(pc.field("Particle.Id") == PARTICLE2_ID.bytes).group_by(["State.Id"]).aggregate(
        [("State.Time", "one"), ("Particle.Coordinates.X", "one"), ("Particle.Coordinates.Y", "one")]).sort_by(
        "State.Time_one").to_pandas()

    distances = distance(particle1["Particle.Coordinates.X_one"], particle1["Particle.Coordinates.Y_one"],
                         particle2["Particle.Coordinates.X_one"], particle2["Particle.Coordinates.Y_one"])
    distance0 = distances[0]

    mask = (particle1["State.Time_one"] > 1) & (np.diff(particle1["State.Time_one"], prepend=[0]) > 0)
    times = particle1["State.Time_one"][mask] / study.input_for(param).time_norm_surface
    shrinkages = (distance0 - distances[mask]) / distance0

    return times.array, shrinkages.array
