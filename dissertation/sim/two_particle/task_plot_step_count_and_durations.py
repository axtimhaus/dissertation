import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pytask import mark, task

from dissertation.config import image_produces
from dissertation.sim.two_particle.studies import STUDIES, StudyBase

BAR_WIDTH = 0.3

for t in STUDIES:

    @task(id=f"{t.KEY}")
    @mark.plot
    def task_plot_step_count_and_durations(
        produces=image_produces(t.DIR / "step_count"),
        study_type: type[StudyBase] = t,
        studies: dict[str, StudyBase] = {str(study): study for study in t.INSTANCES},
        results_files={str(study): study.dir / "output.parquet" for study in t.INSTANCES},
        time_files={str(study): study.dir / "time.txt" for study in t.INSTANCES},
    ):
        data_frames = ((k, pq.read_table(f).flatten().flatten()) for k, f in results_files.items())

        fig = plt.figure(dpi=600)
        ax = fig.subplots()
        ax2 = ax.twinx()

        categories = [s.display.replace("/", "\n") for s in studies.values()]

        step_counts = [get_step_count(df) for key, df in data_frames]
        ax.bar(categories, step_counts, label="Step Count", color="C0", width=BAR_WIDTH, align="edge")

        times = [float(f.read_text()) for key, f in time_files.items()]
        ax2.bar(categories, times, label="Simulation Duration", color="C1", width=-BAR_WIDTH, align="edge")

        ax.set_xlabel(study_type.TITLE)
        ax.set_ylabel("Step Count", color="C0")
        ax.tick_params(axis="y", labelcolor="C0")
        ax2.set_ylabel(r"Simulation Duration in $\unit{\second}$", color="C1")
        ax2.tick_params(axis="y", labelcolor="C1")
        ax2.grid(True)
        ax.set_yscale("log")
        ax2.set_yscale("log")

        for p in produces:
            fig.savefig(p)

        plt.close(fig)


def distance(particle1_x, particle1_y, particle2_x, particle2_y):
    return np.sqrt((particle2_x - particle1_x) ** 2 + (particle2_y - particle1_y) ** 2)


def get_step_count(df: pa.Table):
    states: pd.DataFrame = (
        df.group_by(["State.Id"]).aggregate([("State.Time", "one")]).sort_by("State.Time_one").to_pandas()
    )

    mask = np.diff(states["State.Time_one"], prepend=[0]) > 0
    return len(states[mask])
