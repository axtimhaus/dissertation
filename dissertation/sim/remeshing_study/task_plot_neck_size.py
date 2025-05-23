from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pytask import mark

from dissertation.config import image_produces
from dissertation.sim.remeshing_study.studies import PARTICLE1_ID, STUDIES, LINE_LEGEND

THIS_DIR = Path(__file__).parent
RESAMPLE_COUNT = 500

PLOTS_DIR = Path(__file__).parent / "plots"


@mark.plot
@mark.remeshing_study
def task_plot_neck_size(
    produces=image_produces(PLOTS_DIR / "neck_size"),
    studies={str(study): study for study in STUDIES},
    results_files={str(study): study.dir() / "output.parquet" for study in STUDIES},
):
    data_frames = load_data(results_files)

    fig = plt.figure(dpi=600)
    ax = fig.subplots()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True)

    for key, df in data_frames:
        times, neck_sizes = get_neck_sizes(studies[key], df)
        ax.plot(times, neck_sizes, lw=1, **studies[key].line_style)

    ax.legend(**LINE_LEGEND)
    ax.set_xlabel("Normalized Time $\\Time / \\TimeNorm_{\\Surface}$")
    ax.set_ylabel(r"Relative Neck Size $\Radius_{\Neck} / \Radius_0$")
    fig.tight_layout()

    for p in produces:
        fig.savefig(p)


def load_data(results_files):
    return ((k, pq.read_table(f).flatten().flatten()) for k, f in results_files.items())


def get_neck_sizes(study, df: pa.Table):
    grain_boundary: pd.DataFrame = (
        df.filter((pc.field("Particle.Id") == PARTICLE1_ID.bytes) & (pc.field("Node.Type") == 1))
        .group_by(["State.Id"])
        .aggregate(
            [
                ("State.Time", "one"),
                ("Node.SurfaceDistance.ToUpper", "one"),
                ("Node.SurfaceDistance.ToLower", "one"),
            ]
        )
        .sort_by("State.Time_one")
        .to_pandas()
    )

    mask = (grain_boundary["State.Time_one"] > 1) & (np.diff(grain_boundary["State.Time_one"], prepend=[0]) > 0)
    times = grain_boundary["State.Time_one"][mask] / study.input.time_norm_surface
    neck_sizes = (
        grain_boundary["Node.SurfaceDistance.ToUpper_one"][mask]
        + grain_boundary["Node.SurfaceDistance.ToLower_one"][mask] / study.input.particle1.radius
    )

    return times.array, neck_sizes.array
