from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow as pa
import pyarrow.parquet as pq
from pytask import mark

from dissertation.config import image_produces
from dissertation.sim.time_step_study.studies import PARTICLE1_ID, PARTICLE2_ID, STUDIES

THIS_DIR = Path(__file__).parent
RESAMPLE_COUNT = 500

PLOTS_DIR = Path(__file__).parent / "plots"
BAR_WIDTH = 0.3


@mark.plot
@mark.time_step_study
def task_plot_step_count_and_durations(
    produces=image_produces(PLOTS_DIR / "step_count"),
    studies={str(study): study for study in STUDIES},
    results_files={str(study): study.dir() / "output.parquet" for study in STUDIES},
    time_files={str(study): study.dir() / "time.txt" for study in STUDIES},
):
    data_frames = load_data(results_files)

    fig = plt.figure(dpi=600)
    ax = fig.subplots()
    ax2 = ax.twinx()

    angle_limits = [str(s.angle_limit) for s in studies.values()]
    step_counts = [get_step_count(df) for key, df in data_frames]

    ax.bar(angle_limits, step_counts, label="Step Count", color="C0", width=BAR_WIDTH, align="edge")

    times = [float(f.read_text()) for key, f in time_files.items()]
    ax2.bar(angle_limits, times, label="Simulation Duration", color="C1", width=-BAR_WIDTH, align="edge")

    ax.set_xlabel("Maximum Displacement Angle")
    ax.set_ylabel("Step Count", color="C0")
    ax.tick_params(axis="y", labelcolor="C0")
    ax2.set_ylabel(r"Simulation Duration in $\unit{\second}$", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")
    ax2.grid(True)

    for p in produces:
        fig.savefig(p)


def load_data(results_files):
    return ((k, pq.read_table(f).flatten().flatten()) for k, f in results_files.items())


def distance(particle1_x, particle1_y, particle2_x, particle2_y):
    return np.sqrt((particle2_x - particle1_x) ** 2 + (particle2_y - particle1_y) ** 2)


def get_step_count(df: pa.Table):
    states: pd.DataFrame = (
        df.group_by(["State.Id"]).aggregate([("State.Time", "one")]).sort_by("State.Time_one").to_pandas()
    )

    mask = np.diff(states["State.Time_one"], prepend=[0]) > 0
    return len(states[mask])
