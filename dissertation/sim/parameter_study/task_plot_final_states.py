from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pytask import task
import matplotlib.pyplot as plt

from dissertation.config import image_produces
from dissertation.sim.parameter_study.studies import STUDIES, PARTICLE1_ID, PARTICLE2_ID

THIS_DIR = Path(__file__).parent

for study in STUDIES:
    @task(id=f"{study}")
    def task_plot_final_states(
            study=study,
            produces=image_produces(study.dir("plots") / "final_states"),
            results_files={value: study.dir(value) / "output.parquet" for value in study.parameter_values}
    ):
        data_frames = {k: pq.read_table(f).flatten().flatten() for k, f in results_files.items()}

        def distance(particle1_x, particle1_y, particle2_x, particle2_y):
            return np.sqrt((particle2_x - particle1_x) ** 2 + (particle2_y - particle2_y) ** 2)

        fig: plt.Figure = plt.figure(dpi=600)
        ax: plt.Axes = fig.subplots()
        ax.set_xscale("log")
        ax.set_yscale("log")

        for key, df in data_frames.items():
            particle1: pd.DataFrame = df.filter(pc.field("Particle.Id") == PARTICLE1_ID.bytes).group_by(["State.Id"]).aggregate([("State.Time", "one"), ("Particle.Coordinates.X", "one"), ("Particle.Coordinates.Y", "one"), ("Node.Coordinates.R", "list"), ("Node.Coordinates.Phi", "list")]).sort_by("State.Time_one").to_pandas()
            particle2: pd.DataFrame = df.filter(pc.field("Particle.Id") == PARTICLE2_ID.bytes).group_by(["State.Id"]).aggregate([("State.Time", "one"), ("Particle.Coordinates.X", "one"), ("Particle.Coordinates.Y", "one")]).sort_by("State.Time_one").to_pandas()

            distances = distance(particle1["Particle.Coordinates.X_one"], particle1["Particle.Coordinates.Y_one"], particle2["Particle.Coordinates.X_one"], particle2["Particle.Coordinates.Y_one"])
            distance0 = distances[0]

            mask = particle1["State.Time_one"] > 1
            times = particle1["State.Time_one"][mask] / study.input_for(key).time_norm_surface
            shrinkages = (distance0 - distances[mask]) / distance0

            ax.plot(times , shrinkages, label=f"{key}")

        ax.legend()
        ax.set_xlabel("Normalized Time $\\Time / \\TimeNorm_{\\Surface}$")
        ax.set_ylabel("Shrinkage")

        for p in produces:
            fig.savefig(p)
