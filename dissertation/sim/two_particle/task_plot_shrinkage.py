import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow as pa
import pyarrow.parquet as pq
from pytask import mark, task
from matplotlib import ticker

from dissertation.config import image_produces, integer_log_space
from dissertation.sim.two_particle.studies import PARTICLE1_ID, PARTICLE2_ID, STUDIES, StudyBase, DimlessParameterStudy

RESAMPLE_COUNT = 500

for t in STUDIES:

    @task(id=f"{t.KEY}")
    @mark.plot
    def task_plot_shrinkage(
        produces=image_produces(t.DIR / "shrinkage"),
        study_type: type[StudyBase] = t,
        studies: dict[str, StudyBase] = {str(study): study for study in t.INSTANCES},
        results_files={str(study): study.dir / "output.parquet" for study in t.INSTANCES},
    ):
        data_frames = ((k, pq.read_table(f).flatten().flatten()) for k, f in results_files.items())

        fig = plt.figure(dpi=600)
        ax = fig.subplots()
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True)

        for key, df in data_frames:
            study = studies[key]
            times, values = get_shrinkages(study, df)
            ax.plot(times, values, label=study.display, **study.line_style)

        ax.legend(title=study_type.TITLE)
        ax.set_xlabel("Normalized Time $\\Time / \\TimeNorm_{\\Surface}$")
        ax.set_ylabel("Shrinkage")
        fig.tight_layout()

        for p in produces:
            fig.savefig(p)

    if issubclass(t, DimlessParameterStudy):

        @task(id=f"{t.KEY}")
        @mark.plot
        def task_plot_shrinkage_map(
            produces=image_produces(t.DIR / "shrinkage_map"),
            study_type: type[DimlessParameterStudy] = t,
            studies: dict[str, DimlessParameterStudy] = {str(study): study for study in t.INSTANCES},  # type: ignore
            results_files={str(study): study.dir / "output.parquet" for study in t.INSTANCES},
        ):
            data_frames = ((k, pq.read_table(f).flatten().flatten()) for k, f in results_files.items())

            fig = plt.figure(dpi=600)
            ax = fig.subplots()
            ax.set_xscale("log")
            ax.set_yscale("log" if study_type.SCALE == "geom" else study_type.SCALE)

            params = np.array([s.real_value for s in studies.values()])
            times_shrinkages = [get_shrinkages(studies[key], df) for key, df in data_frames]

            start_time = max([t[0] for t, _ in times_shrinkages])
            end_time = max([t[-1] for t, _ in times_shrinkages])

            times = np.geomspace(start_time, end_time, RESAMPLE_COUNT)
            shrinkages = np.array([np.interp(times, t, s) for t, s in times_shrinkages])

            grid_x, grid_y = np.meshgrid(times, params)

            locs = integer_log_space(1, -3, 3, -1)
            formatter = ticker.LogFormatterSciNotation()

            cs = ax.contour(grid_x, grid_y, shrinkages, levels=locs, norm="log", cmap=study_type.CMAP)
            ax.clabel(cs, fmt=lambda level: formatter(level))

            ax.set_xlabel("Normalized Time $\\Time / \\TimeNorm_{\\Surface}$")
            ax.set_ylabel(study_type.TITLE)

            for p in produces:
                fig.savefig(p)


def distance(particle1_x, particle1_y, particle2_x, particle2_y):
    return np.sqrt((particle2_x - particle1_x) ** 2 + (particle2_y - particle1_y) ** 2)


def get_shrinkages(study, df: pa.Table):
    particle1: pd.DataFrame = (
        df.filter(pc.field("Particle.Id") == PARTICLE1_ID.bytes)
        .group_by(["State.Id"])
        .aggregate([("State.Time", "one"), ("Particle.Coordinates.X", "one"), ("Particle.Coordinates.Y", "one")])
        .sort_by("State.Time_one")
        .to_pandas()
    )
    particle2: pd.DataFrame = (
        df.filter(pc.field("Particle.Id") == PARTICLE2_ID.bytes)
        .group_by(["State.Id"])
        .aggregate([("State.Time", "one"), ("Particle.Coordinates.X", "one"), ("Particle.Coordinates.Y", "one")])
        .sort_by("State.Time_one")
        .to_pandas()
    )

    distances = distance(
        particle1["Particle.Coordinates.X_one"],
        particle1["Particle.Coordinates.Y_one"],
        particle2["Particle.Coordinates.X_one"],
        particle2["Particle.Coordinates.Y_one"],
    )
    distance0 = distances[0]

    mask = (particle1["State.Time_one"] > 1) & (np.diff(particle1["State.Time_one"], prepend=[0]) > 0)
    times = particle1["State.Time_one"][mask] / study.input.time_norm_surface
    shrinkages = (distance0 - distances[mask]) / distance0

    return times.array, shrinkages.array
