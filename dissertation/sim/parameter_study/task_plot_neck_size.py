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
        def task_plot_neck_size(
                study=study,
                produces=image_produces(study.dir("plots") / "neck_size"),
                results_files={value: study.dir(value) / "output.parquet" for value in study.parameter_values}
        ):
            data_frames = {k: pq.read_table(f).flatten().flatten() for k, f in results_files.items()}

            fig: plt.Figure = plt.figure(dpi=600)
            ax: plt.Axes = fig.subplots()
            ax.set_xscale("log")
            ax.set_yscale("log")

            for key, df in data_frames.items():
                grain_boundary: pd.DataFrame = df.filter((pc.field("Particle.Id") == PARTICLE1_ID.bytes) & (pc.field("Node.Type") == 1)).group_by(["State.Id"]).aggregate([("State.Time", "one"), ("Node.SurfaceDistance.ToUpper", "one"), ("Node.SurfaceDistance.ToLower", "one")]).sort_by("State.Time_one").to_pandas()

                input = study.input_for(key)
                mask = grain_boundary["State.Time_one"] > 1
                times = grain_boundary["State.Time_one"][mask] / input.time_norm_surface
                neck_sizes = grain_boundary["Node.SurfaceDistance.ToUpper_one"][mask] + grain_boundary["Node.SurfaceDistance.ToLower_one"][mask] / input.particle1.radius

                ax.plot(times , neck_sizes, label=f"{key:.2f}")

            ax.legend()
            ax.set_xlabel("Normalized Time $\\Time / \\TimeNorm_{\\Surface}$")
            ax.set_ylabel("Relative Neck Size")

            for p in produces:
                fig.savefig(p)
