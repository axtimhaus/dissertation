import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pytask import mark

from dissertation.config import image_produces, in_build_dir
from dissertation.sim.packings.cases import CASES, THIS_DIR, Case

RESAMPLE_COUNT = 500


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

    for key, df in data_frames:
        case = cases[key]
        times, values = get_neck_sizes(case, df)
        ax.plot(times, values, label=case.display, **case.line_style)

    ax.legend()
    ax.set_xlabel("Normalized Time $\\Time / \\TimeNorm_{\\Surface}$")
    ax.set_ylabel("Average Relative Neck Size $\\Radius_{\\Neck} / \\Radius_0$")
    fig.tight_layout()

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
    mask = (times > 1) & (np.diff(times, prepend=[0]) > 0)
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
