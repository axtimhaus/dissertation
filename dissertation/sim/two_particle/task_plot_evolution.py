import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pytask import mark, task

from dissertation.config import image_produces
from dissertation.sim.two_particle.studies import PARTICLE1_ID, PARTICLE2_ID, STUDIES, StudyBase

for t in STUDIES:
    for study in t.INSTANCES:

        @task(id=study.key)
        @mark.plot
        @mark.time_step_study
        def task_plot_evolution(
            study: StudyBase = study,
            results_file=study.dir / "output.parquet",
            produces=image_produces(study.dir / "evolution"),
        ):
            df = pq.read_table(results_file).flatten().flatten()

            fig = plt.figure(dpi=600)
            ax = fig.subplots()
            ax.set_aspect("equal")
            viridis = mpl.colormaps["viridis"]

            times, particle1_x, particle1_y, particle2_x, particle2_y = get_states(df, study)

            num = 10

            times_to_plot = np.geomspace(times[0], times[-1], 10)

            for j, t in enumerate(times_to_plot):
                color = viridis(float(j) / num)
                i = np.searchsorted(times, t)
                ax.fill(particle1_x[i], particle1_y[i], label=f"{times[i]:.2f}", edgecolor=color, fill=False, lw=0.5)
                ax.fill(particle2_x[i], particle2_y[i], edgecolor=color, fill=False, lw=0.5)

            ax.set_xlabel("$x$ in \\unit{\\micro\\meter}")
            ax.set_ylabel("$y$ in \\unit{\\micro\\meter}")
            fig.tight_layout()

            for p in produces:
                fig.savefig(p)


def get_states(df: pa.Table, study):
    particle1: pd.DataFrame = (
        df.filter(pc.field("Particle.Id") == PARTICLE1_ID.bytes)
        .group_by(["State.Id"])
        .aggregate([("State.Time", "one"), ("Node.Coordinates.X", "list"), ("Node.Coordinates.Y", "list")])
        .sort_by("State.Time_one")
        .to_pandas()
    )
    particle2: pd.DataFrame = (
        df.filter(pc.field("Particle.Id") == PARTICLE2_ID.bytes)
        .group_by(["State.Id"])
        .aggregate([("State.Time", "one"), ("Node.Coordinates.X", "list"), ("Node.Coordinates.Y", "list")])
        .sort_by("State.Time_one")
        .to_pandas()
    )

    mask = (particle1["State.Time_one"] > 1) & (np.diff(particle1["State.Time_one"], prepend=[0]) > 0)
    times = particle1["State.Time_one"][mask] / study.input.time_norm_surface
    particle1_x = particle1["Node.Coordinates.X_list"] * 1e6
    particle1_y = particle1["Node.Coordinates.Y_list"] * 1e6
    particle2_x = particle2["Node.Coordinates.X_list"] * 1e6
    particle2_y = particle2["Node.Coordinates.Y_list"] * 1e6

    return times.array, particle1_x.array, particle1_y.array, particle2_x.array, particle2_y.array
