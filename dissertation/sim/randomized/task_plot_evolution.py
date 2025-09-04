import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from matplotlib.contour import ContourSet
from matplotlib.ticker import LogLocator
from pytask import mark, task

from dissertation.config import image_produces
from dissertation.sim.randomized.cases import CASES, Case
from dissertation.sim.randomized.input import TIME_NORM_SURFACE, Input

for case in CASES:
    for i, sample in enumerate(case.samples):

        @task(id=f"{case.key}/{i}")
        @mark.plot
        def task_plot_evolution_randomized(
            case: Case = case,
            sample: Input = sample,
            results_file=case.dir(i) / "output.parquet",
            produces=image_produces(case.dir(i) / "evolution"),
        ):
            df = pq.read_table(results_file).flatten().flatten()

            fig = plt.figure()
            ax = fig.subplots()
            ax.set_aspect("equal", adjustable="datalim")

            particles = [get_states(df, sample, i) for i in range(len(sample.particles))]
            times = particles[0][0]

            times_to_plot = np.geomspace(1e-6, times[-1], 10)
            time_indices = [np.searchsorted(times, t) for t in times_to_plot]

            cs = ContourSet(
                ax,
                times_to_plot,
                [
                    [np.pad(np.transpose([p[1][i], p[2][i]]), [(0, 1), (0, 0)], mode="wrap") for p in particles]
                    for i, t in zip(time_indices, times_to_plot, strict=True)
                ],
                cmap="viridis",
                norm="log",
            )
            cb = fig.colorbar(
                cs,
                orientation="vertical",
                ticks=LogLocator(),
                label="Normalized Time $\\Time / \\TimeNorm_{\\Surface}$",
                aspect=30,
            )
            cb.minorticks_on()
            t_range = np.log10(times_to_plot[-1]) - np.log10(times_to_plot[0])
            cb.ax.set_ylim(
                10 ** (np.log10(times_to_plot[0]) - 0.01 * t_range),
                10 ** (np.log10(times_to_plot[-1]) + 0.01 * t_range),
                auto=False,
            )

            ax.set_xlabel("$x$ in \\unit{\\micro\\meter}")
            ax.set_ylabel("$y$ in \\unit{\\micro\\meter}")

            for p in produces:
                fig.savefig(p)

            plt.close(fig)

    def get_states(df: pa.Table, sample: Input, particle_index: int):
        particle: pd.DataFrame = (
            df.filter(pc.field("Particle.Id") == sample.particles[particle_index].id.bytes)
            .group_by(["State.Id"])
            .aggregate([("State.Time", "one"), ("Node.Coordinates.X", "list"), ("Node.Coordinates.Y", "list")])
            .sort_by("State.Time_one")
            .to_pandas()
        )

        mask = (particle["State.Time_one"] > 1) & (np.diff(particle["State.Time_one"], prepend=[0]) > 0)
        times = np.asarray(particle["State.Time_one"][mask] / TIME_NORM_SURFACE)
        particle_x = np.asarray(particle["Node.Coordinates.X_list"] * 1e6)
        particle_y = np.asarray(particle["Node.Coordinates.Y_list"] * 1e6)

        return times, particle_x, particle_y
