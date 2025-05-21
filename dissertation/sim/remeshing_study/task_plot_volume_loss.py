from pathlib import Path
from uuid import UUID

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pytask import mark

from dissertation.config import image_produces
from dissertation.sim.remeshing_study.studies import (
    LIMIT_COLORS,
    NODE_COUNT_STYLES,
    PARTICLE1_ID,
    PARTICLE2_ID,
    STUDIES,
    NODE_COUNTS,
    LIMITS,
)

THIS_DIR = Path(__file__).parent
RESAMPLE_COUNT = 500

PLOTS_DIR = Path(__file__).parent / "plots"


@mark.plot
@mark.remeshing_study
def task_plot_volume_loss(
    produces=image_produces(PLOTS_DIR / "volume_loss"),
    studies={str(study): study for study in STUDIES},
    results_files={str(study): study.dir() / "output.parquet" for study in STUDIES},
):
    data_frames = load_data(results_files)

    fig: plt.Figure = plt.figure(dpi=600)
    ax: plt.Axes = fig.subplots()
    ax.set_xscale("log")
    ax.set_yscale("asinh")
    ax.grid(True)

    for key, df in data_frames:
        times1, volume_losses1 = get_volume_losses(studies[key], df, PARTICLE1_ID)
        plot1 = ax.plot(
            times1,
            volume_losses1,
            color=LIMIT_COLORS[studies[key].surface_remesher_limit],
            linestyle=NODE_COUNT_STYLES[studies[key].node_count],
            lw=1,
        )[0]
        times2, volume_losses2 = get_volume_losses(studies[key], df, PARTICLE2_ID)
        ax.plot(times2, volume_losses2, color=plot1.get_color(), ls=plot1.get_linestyle(), lw=1)

    handles = [Line2D([], [], color="k", linestyle=NODE_COUNT_STYLES[node_count]) for node_count in NODE_COUNTS] + [
        Line2D([], [], color=LIMIT_COLORS[limit]) for limit in LIMITS
    ]
    labels = [f"n = {node_count}" for node_count in NODE_COUNTS] + [f"limit = {limit}" for limit in LIMITS]
    ax.legend(handles, labels)
    ax.set_xlabel("Normalized Time $\\Time / \\TimeNorm_{\\Surface}$")
    ax.set_ylabel(r"Relative Volume Loss")
    fig.tight_layout()

    for p in produces:
        fig.savefig(p)


def load_data(results_files):
    return ((k, pq.read_table(f).flatten().flatten()) for k, f in results_files.items())


def get_volume_losses(study, df: pa.Table, particle_id: UUID):
    states: pd.DataFrame = (
        df.filter(pc.field("Particle.Id") == particle_id.bytes)
        .group_by(["State.Id"])
        .aggregate(
            [
                ("State.Time", "one"),
                ("Node.Volume.ToUpper", "sum"),
            ]
        )
        .sort_by("State.Time_one")
        .to_pandas()
    )

    initial_volume = states["Node.Volume.ToUpper_sum"].iloc[0]
    mask = (states["State.Time_one"] > 1) & (np.diff(states["State.Time_one"], prepend=[0]) > 0)
    times = states["State.Time_one"][mask] / study.input.time_norm_surface
    volumes = states["Node.Volume.ToUpper_sum"][mask]
    volume_losses = (volumes - initial_volume) / initial_volume

    return times.array, volume_losses.array
