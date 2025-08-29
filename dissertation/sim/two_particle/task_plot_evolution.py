import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from matplotlib.contour import ContourSet
from matplotlib.ticker import LogLocator
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

            times, particle1_x, particle1_y, particle2_x, particle2_y = get_states(df, study)

            times_to_plot = np.geomspace(1e-6, times[-1], 10)
            time_indices = [np.searchsorted(times, t) for t in times_to_plot]

            cs = ContourSet(
                ax,
                times_to_plot,
                [
                    [
                        np.pad(np.transpose([particle1_x[i], particle1_y[i]]), [(0, 1), (0, 0)], mode="wrap"),
                        np.pad(np.transpose([particle2_x[i], particle2_y[i]]), [(0, 1), (0, 0)], mode="wrap"),
                    ]
                    for i, t in zip(time_indices, times_to_plot, strict=True)
                ],
                cmap="viridis",
                norm="log",
                linewidths=1,
            )
            cb = fig.colorbar(
                cs,
                orientation="horizontal",
                ticks=LogLocator(),
                label="Normalized Time $\\Time / \\TimeNorm_{\\Surface}$",
                aspect=50,
            )
            cb.minorticks_on()
            range = np.log10(times_to_plot[-1]) - np.log10(times_to_plot[0])
            cb.ax.set_xlim(
                10 ** (np.log10(times_to_plot[0]) - 0.01 * range),
                10 ** (np.log10(times_to_plot[-1]) + 0.01 * range),
                auto=False,
            )

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
