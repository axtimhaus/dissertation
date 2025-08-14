import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pytask import mark, task

from dissertation.config import FIGSIZE_INCH, image_produces
from dissertation.sim.packings.cases import CASES, Case

for case in CASES:

    @task(id=case.key)
    @mark.plot
    def task_plot_evolution_packings(
        case: Case = case,
        results_file=case.dir / "output.parquet",
        produces=image_produces(case.dir / "evolution"),
    ):
        df = pq.read_table(results_file).flatten().flatten()

        fig = plt.figure(figsize=(FIGSIZE_INCH[0], 4))
        ax = fig.subplots()
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True)
        viridis = mpl.colormaps["viridis"]

        particles = [get_states(df, case, i) for i in range(len(case.input.particles))]
        times = particles[0][0]

        num = 10

        times_to_plot = np.geomspace(times[0], times[-1], 10)

        for j, t in enumerate(times_to_plot):
            color = viridis(float(j) / num)
            i = np.searchsorted(times, t)
            label = f"{i}/{len(times)}"

            for p in particles:
                ax.fill(p[1][i], p[2][i], label=label, edgecolor=color, fill=False, lw=0.5)
                label = None

        # ax.set_title(f"{case.display}")
        ax.set_xlabel("$x$ in \\unit{\\micro\\meter}")
        ax.set_ylabel("$y$ in \\unit{\\micro\\meter}")

        min_x = min(p.x for p in case.input.particles) * 1e6
        max_x = max(p.x for p in case.input.particles) * 1e6
        ax.set_xlim(min_x, max_x)
        min_y = min(p.y for p in case.input.particles) * 1e6
        max_y = max(p.y for p in case.input.particles) * 1e6
        ax.set_ylim(min_y, max_y if not np.isclose(max_y, min_y) else None)

        fig.tight_layout()

        for p in produces:
            fig.savefig(p)

        plt.close(fig)


def get_states(df: pa.Table, case: Case, particle_index: int):
    particle: pd.DataFrame = (
        df.filter(pc.field("Particle.Id") == case.input.particles[particle_index].id.bytes)
        .group_by(["State.Id"])
        .aggregate([("State.Time", "one"), ("Node.Coordinates.X", "list"), ("Node.Coordinates.Y", "list")])
        .sort_by("State.Time_one")
        .to_pandas()
    )

    mask = (particle["State.Time_one"] > 1) & (np.diff(particle["State.Time_one"], prepend=[0]) > 0)
    times = particle["State.Time_one"][mask] / case.input.time_norm_surface
    particle_x = particle["Node.Coordinates.X_list"] * 1e6
    particle_y = particle["Node.Coordinates.Y_list"] * 1e6

    return times.array, particle_x.array, particle_y.array
