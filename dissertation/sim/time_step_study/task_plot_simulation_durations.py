from pathlib import Path
from uuid import UUID

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pytask import mark

from dissertation.config import image_produces
from dissertation.sim.time_step_study.studies import PARTICLE1_ID, STUDIES, PARTICLE2_ID

THIS_DIR = Path(__file__).parent
RESAMPLE_COUNT = 500

PLOTS_DIR = Path(__file__).parent / "plots"


@mark.plot
@mark.time_step_study
def task_plot_volume_loss(
    produces=image_produces(PLOTS_DIR / "simulation_durations"),
    studies={str(study): study for study in STUDIES},
    results_files={str(study): study.dir() / "time.txt" for study in STUDIES},
):
    fig: plt.Figure = plt.figure(dpi=600)
    ax: plt.Axes = fig.subplots()
    ax.grid(True)

    labels = [str(s.angle_limit) for s in studies.values()]
    times = [float(f.read_text()) for key, f in results_files.items()]

    ax.bar(labels, times)

    ax.legend()
    ax.set_ylabel("Simulation Duration in $\\unit{\second}$")
    fig.tight_layout()

    for p in produces:
        fig.savefig(p)
