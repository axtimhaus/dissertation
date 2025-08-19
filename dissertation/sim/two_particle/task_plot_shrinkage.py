import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from matplotlib import ticker
from pytask import mark, task

from dissertation.config import image_produces, integer_log_space125
from dissertation.sim.two_particle.helper import ashby_grid
from dissertation.sim.two_particle.studies import PARTICLE1_ID, PARTICLE2_ID, STUDIES, DimlessParameterStudy, StudyBase

RESAMPLE_COUNT = 100
SHRINKAGE_LIMITS = (1e-3, 1)

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
        ax.grid(True, "both")
        upper_mag = -6

        for key, df in data_frames:
            study = studies[key]
            times, values = get_shrinkages(study, df)
            ax.plot(times, values, label=study.display, **study.line_style)
            upper_mag = max(upper_mag, int(np.floor(np.log10(np.max(times)))))

        ax.legend(title=study_type.TITLE, ncols=3)
        ax.set_xlabel("Normalized Time $\\Time / \\TimeNorm_{\\Surface}$")
        ax.set_ylabel("Shrinkage $\\Shrinkage$")
        ax.set_xlim(1e-6, 10**upper_mag)
        ax.set_ylim(*SHRINKAGE_LIMITS)
        ax.set_ylim(auto=True)

        for p in produces:
            fig.savefig(p)

        plt.close(fig)

    if issubclass(t, DimlessParameterStudy):

        @task(id=f"{t.KEY}")
        @mark.plot
        def task_plot_shrinkage_map(
            produces=image_produces(t.DIR / "shrinkage_map"),
            study_type: type[DimlessParameterStudy] = t,
            studies: dict[str, DimlessParameterStudy] = {str(study): study for study in t.INSTANCES},  # type: ignore
            results_files={str(study): study.dir / "output.parquet" for study in t.INSTANCES},
        ):
            data_frames = [(k, pq.read_table(f).flatten().flatten()) for k, f in results_files.items()]

            fig = plt.figure(dpi=600)
            ax = fig.subplots()
            ax.set_xscale(study_type.axis_scale)
            ax.set_yscale("log")
            ax.grid(True, "both")

            study_params = np.array([s.real_value for s in studies.values()])
            params = (np.linspace if study_type.axis_scale == "linear" else np.geomspace)(
                study_params.min(), study_params.max(), RESAMPLE_COUNT
            )
            shrinkages = np.geomspace(*SHRINKAGE_LIMITS, RESAMPLE_COUNT)
            shrinkage_curves = [get_shrinkages(studies[key], df) for key, df in data_frames]
            grid_x, grid_y, times = ashby_grid(study_params, shrinkage_curves, params, shrinkages)

            lower_mag = -6
            upper_mag = int(np.floor(np.log10(np.max(times))))
            locs = integer_log_space125(lower_mag, upper_mag)
            formatter = ticker.LogFormatterSciNotation()

            cs = ax.contour(
                grid_x,
                grid_y,
                times,
                levels=locs,
                norm="log",
                cmap=study_type.CMAP,
            )
            # ax.clabel(cs, fmt=lambda level: formatter(level))
            fig.colorbar(cs, format=formatter, label="Normalized Time $\\Time / \\TimeNorm_{\\Surface}$")

            ax.set_xlabel(study_type.TITLE)
            ax.set_ylabel("Shrinkage $\\Shrinkage$")

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

    times = particle1["State.Time_one"] / study.input.time_norm_surface
    mask = np.diff(times, prepend=[0]) > 0
    shrinkages = (distance0 - distances[mask]) / distance0

    return times[mask].to_numpy(), shrinkages.to_numpy()
