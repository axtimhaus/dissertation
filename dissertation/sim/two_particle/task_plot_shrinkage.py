import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow as pa
import pyarrow.parquet as pq
from pytask import mark

from dissertation.config import image_produces
from dissertation.sim.two_particle.studies import PARTICLE1_ID, PARTICLE2_ID, STUDIES

for t in STUDIES:

    @mark.plot
    def task_plot_shrinkage(
        produces=image_produces(t.DIR / "shrinkage"),
        studies={str(study): study for study in t.INSTANCES},
        results_files={str(study): study.dir / "output.parquet" for study in t.INSTANCES},
    ):
        data_frames = ((k, pq.read_table(f).flatten().flatten()) for k, f in results_files.items())

        fig = plt.figure(dpi=600)
        ax = fig.subplots()
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True)

        for key, df in data_frames:
            times, shrinkages = get_shrinkages(studies[key], df)
            ax.plot(times, shrinkages, label=key, lw=1)

        ax.legend(title="Maximum Displacement Angle")
        ax.set_xlabel("Normalized Time $\\Time / \\TimeNorm_{\\Surface}$")
        ax.set_ylabel("Shrinkage")
        fig.tight_layout()

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
