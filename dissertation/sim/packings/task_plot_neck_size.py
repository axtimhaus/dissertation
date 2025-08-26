import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pytask import mark

from dissertation.config import image_produces, in_build_dir
from dissertation.sim.packings.cases import CASES, THIS_DIR, Case

TIME_MIN = 1e-6
NECK_SIZE_MIN = 2e-1


@mark.plot
def task_plot_neck_size_packings(
    produces=image_produces(in_build_dir(THIS_DIR) / "neck_size"),
    results_files={c.key: c.dir / "output.parquet" for c in CASES},
    cases={c.key: c for c in CASES},
):
    data_frames = ((k, pq.read_table(f).flatten().flatten()) for k, f in results_files.items())

    fig = plt.figure(dpi=600)
    ax = fig.subplots()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, "both")
    max_time = TIME_MIN
    max_neck_size = NECK_SIZE_MIN

    for key, df in data_frames:
        case = cases[key]
        times, values = get_neck_sizes(case, df)
        ax.plot(times, values, label=case.display, **case.line_style)
        max_time = max(max_time, np.max(times))
        max_neck_size = max(max_neck_size, np.max(values))

    ax.legend()
    ax.set_xlabel("Normalized Time $\\Time / \\TimeNorm_{\\Surface}$")
    ax.set_ylabel("Average Relative Neck Size $\\Radius_{\\Neck} / \\Radius_0$")
    ax.set_xlim(TIME_MIN, max_time)
    ax.set_ylim(NECK_SIZE_MIN, min(10 ** np.ceil(np.log10(max_neck_size)), 1.3))

    for p in produces:
        fig.savefig(p)

    plt.close(fig)


def get_neck_sizes(case: Case, df: pa.Table):
    grain_boundary: pd.DataFrame = (
        df.filter(pc.field("Node.Type") == 1)
        .group_by(["State.Id"])
        .aggregate(
            [
                ("State.Time", "one"),
                ("Node.SurfaceDistance.ToUpper", "mean"),
                ("Node.SurfaceDistance.ToLower", "mean"),
            ]
        )
        .sort_by("State.Time_one")
        .to_pandas()
    )

    times = grain_boundary["State.Time_one"].to_numpy()
    mask = np.diff(times, prepend=[0]) > 0
    times = times[mask] / case.input.time_norm_surface

    neck_sizes = (
        (
            grain_boundary["Node.SurfaceDistance.ToUpper_mean"][mask]
            + grain_boundary["Node.SurfaceDistance.ToLower_mean"][mask]
        )
        / case.input.particles[0].radius
        / 2
    )

    return times, neck_sizes
