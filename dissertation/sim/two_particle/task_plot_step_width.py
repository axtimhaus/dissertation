from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pytask import mark

from dissertation.config import image_produces
from dissertation.sim.two_particle.studies import STUDIES

for t in STUDIES:

    @mark.plot
    @mark.time_step_study
    def task_plot_time_step_width(
        produces=image_produces(t.DIR / "time_step_width"),
        studies={str(study): study for study in t.INSTANCES},
        results_files={str(study): study.dir / "output.parquet" for study in t.INSTANCES},
    ):
        data_frames = ((k, pq.read_table(f).flatten().flatten()) for k, f in results_files.items())

        fig = plt.figure(dpi=600)
        ax = fig.subplots()
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True)

        for key, df in data_frames:
            times, steps = get_time_steps(studies[key], df)
            p = ax.plot(times, steps, label=key, lw=1)[0]

        ax.legend(title="Maximum Displacement Angle")
        ax.set_xlabel("Normalized Time $\\Time / \\TimeNorm_{\\Surface}$")
        ax.set_ylabel("Time Step Width $\\Diff\\Time / \\TimeNorm_{\\Surface}$")
        fig.tight_layout()

        for p in produces:
            fig.savefig(p)


def get_time_steps(study, df: pa.Table):
    states = df.group_by(["State.Id"]).aggregate([("State.Time", "one")]).sort_by("State.Time_one").to_pandas()
    times = states["State.Time_one"] / study.input.time_norm_surface
    diffs = np.diff(times, append=[0])
    mask = diffs > 0

    return times[mask].array, diffs[mask]
