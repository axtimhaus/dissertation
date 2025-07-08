import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from matplotlib.lines import Line2D
from pytask import mark

from dissertation.config import image_produces, in_build_dir
from dissertation.sim.packings.cases import CASES, THIS_DIR, Case
from dissertation.sim.two_particle.task_plot_shrinkage import distance

RESAMPLE_COUNT = 500


@mark.plot
def task_plot_shrinkage_packings(
    produces=image_produces(in_build_dir(THIS_DIR) / "shrinkage"),
    results_files={c.key: c.dir / "output.parquet" for c in CASES},
    cases={c.key: c for c in CASES},
):
    data_frames = ((k, pq.read_table(f).flatten().flatten()) for k, f in results_files.items())

    fig = plt.figure(dpi=600)
    ax = fig.subplots()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, "both")

    for key, df in data_frames:
        case = cases[key]
        times, shrinkages = get_shrinkages(case, df)
        ax.plot(times, shrinkages, **case.line_style, label=case.display)

    ax.legend()
    ax.set_xlabel("Normalized Time $\\Time / \\TimeNorm_{\\Surface}$")
    ax.set_ylabel("Shrinkage $\\Shrinkage$")
    fig.tight_layout()

    for p in produces:
        fig.savefig(p)

    plt.close(fig)


def shoelace(x: np.ndarray, y: np.ndarray):
    x_diffs = x - np.roll(x, -1)
    y_sums = y + np.roll(y, -1)
    return np.sum(x_diffs * y_sums) / 2


def get_shrinkages(case: Case, df: pa.Table):
    particles: list[pd.DataFrame] = [
        (
            df.filter(pc.field("Particle.Id") == p.id.bytes)
            .group_by(["State.Id"])
            .aggregate([("State.Time", "one"), ("Particle.Coordinates.X", "one"), ("Particle.Coordinates.Y", "one")])
            .sort_by("State.Time_one")
            .to_pandas()
        )
        for p in case.input.particles
    ]
    times = particles[0]["State.Time_one"].to_numpy()
    mask = (times > 1) & (np.diff(times, prepend=[0]) > 0)
    times = times[mask] / case.input.time_norm_surface

    if len(particles) > 2:
        x = np.array([p["Particle.Coordinates.X_one"].to_numpy() for p in particles]).T
        y = np.array([p["Particle.Coordinates.Y_one"].to_numpy() for p in particles]).T
        volumes = np.array([shoelace(x_, y_) for x_, y_ in zip(x, y, strict=False)])
        sqrt_volume0 = np.sqrt(volumes[0])
        shrinkages = (sqrt_volume0 - np.sqrt(volumes[mask])) / sqrt_volume0
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
