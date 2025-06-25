import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from matplotlib import ticker
from pytask import mark, task

from dissertation.config import image_produces, in_build_dir, integer_log_space
from dissertation.sim.randomized.input import Input
from dissertation.sim.randomized.samples import SAMPLES, THIS_DIR, dir
from dissertation.sim.two_particle.task_plot_shrinkage import distance


@mark.plot
def task_plot_shrinkage_randomized(
    produces=image_produces(in_build_dir(THIS_DIR) / "shrinkage"),
    results_files=[dir(i) / "output.parquet" for i in range(len(SAMPLES))],
):
    data_frames = [pq.read_table(f).flatten().flatten() for f in results_files]

    fig = plt.figure(dpi=600)
    ax = fig.subplots()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True)

    for i, df in enumerate(data_frames):
        case = SAMPLES[i]
        times, values = get_shrinkages_shoelace(case, df)
        ax.plot(times, values, alpha=0.5, color="C0")

    ax.legend()
    ax.set_xlabel("Normalized Time $\\Time / \\TimeNorm_{\\Surface}$")
    ax.set_ylabel("Shrinkage")
    fig.tight_layout()

    for p in produces:
        fig.savefig(p)

    plt.close(fig)


def shoelace(x: np.ndarray, y: np.ndarray):
    x_diffs = x - np.roll(x, -1)
    y_sums = y + np.roll(y, -1)
    return np.sum(x_diffs * y_sums) / 2


def get_shrinkages_shoelace(input: Input, df: pa.Table):
    particles: list[pd.DataFrame] = [
        (
            df.filter(pc.field("Particle.Id") == p.id.bytes)
            .group_by(["State.Id"])
            .aggregate([("State.Time", "one"), ("Particle.Coordinates.X", "one"), ("Particle.Coordinates.Y", "one")])
            .sort_by("State.Time_one")
            .to_pandas()
        )
        for p in input.particles
    ]
    times = particles[0]["State.Time_one"].to_numpy()
    mask = (times > 1) & (np.diff(times, prepend=[0]) > 0)
    times = times[mask] / input.time_norm_surface

    if len(particles) > 2:
        x = np.array([p["Particle.Coordinates.X_one"].to_numpy() for p in particles]).T
        y = np.array([p["Particle.Coordinates.Y_one"].to_numpy() for p in particles]).T
        volumes = np.array([shoelace(x_, y_) for x_, y_ in zip(x, y)])
        volume0 = volumes[0]
        shrinkages = (volume0 - volumes[mask]) / volume0
    else:
        distances = distance(
            particles[0]["Particle.Coordinates.X_one"].to_numpy(),
            particles[0]["Particle.Coordinates.Y_one"].to_numpy(),
            particles[1]["Particle.Coordinates.X_one"].to_numpy(),
            particles[1]["Particle.Coordinates.Y_one"].to_numpy(),
        )
        distance0 = distances[0]
        shrinkages = (distance0 - distances[mask]) / distance0

    return times, shrinkages
