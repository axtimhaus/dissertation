import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pytask import mark, task

from dissertation.config import FIGSIZE_INCH, image_produces
from dissertation.sim.two_particle.studies import PARTICLE1_ID, PARTICLE2_ID, STUDIES, StudyBase

for t in STUDIES:
    for study in t.INSTANCES:

        @task(id=study.key)
        @mark.plot
        def task_plot_evolution(
            study: StudyBase = study,
            results_file=study.dir / "output.parquet",
            produces=image_produces(study.dir / "evolution"),
        ):
            df = pq.read_table(results_file).flatten().flatten()

            fig = plt.figure(figsize=(FIGSIZE_INCH[0], 4))
            ax = fig.subplots()
            ax.set_aspect("equal", adjustable="datalim")
            viridis = mpl.colormaps["viridis"]

            times, particle1_x, particle1_y, particle2_x, particle2_y = get_states(df, study)

            times_to_plot = np.geomspace(times[0], times[-1], 10)
            log_time_min = np.log(1 / study.input.time_norm_surface)
            log_time_max = np.log(study.input.duration / study.input.time_norm_surface)

            for t in times_to_plot:
                color = viridis((np.log(t) - log_time_min) / (log_time_max - log_time_min))
                i = np.searchsorted(times, t)
                ax.fill(particle1_x[i], particle1_y[i], label=f"{times[i]:.2f}", edgecolor=color, fill=False, lw=1)
                ax.fill(particle2_x[i], particle2_y[i], edgecolor=color, fill=False, lw=1)

            # ax.set_title(f"{study.TITLE}\n{study.display}")
            ax.axhline(0, color="red", lw=0.5)
            ax.set_xlabel("$x$ in \\unit{\\micro\\meter}")
            ax.set_ylabel("$y$ in \\unit{\\micro\\meter}")

            min_x = np.max(particle1_x[0]) - 100
            max_x = min_x + 200
            max_y = np.max(particle2_y[0][particle2_x[0] < max_x])

            ax.set_xlim(0, max_x)
            ax.set_ylim(-5, max_y)
            ax.grid(True)

            for p in produces:
                fig.savefig(p)

            plt.close(fig)


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
