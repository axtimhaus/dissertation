from cProfile import label
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pyarrow as pa
from pytask import task
import matplotlib.pyplot as plt
import scipy.interpolate as sip
import scipy.optimize as sop
import scipy.signal as ssi

from dissertation.config import image_produces
from dissertation.sim.parameter_study.studies import STUDIES, PARTICLE1_ID, PARTICLE2_ID

THIS_DIR = Path(__file__).parent

for study in STUDIES:
        @task(id=f"{study}")
        def task_plot_shrinkage_map(
                study=study,
                produces=image_produces(study.dir("plots") / "shrinkage_map"),
                results_files={value: study.dir(value) / "output.parquet" for value in study.parameter_values}
        ):
            data_frames = {k: pq.read_table(f).flatten().flatten() for k, f in results_files.items()}

            fig: plt.Figure = plt.figure(dpi=600)
            ax: plt.Axes = fig.subplots()
            ax.set_xscale("log")
            # ax.set_yscale("log")

            params = np.array(list(data_frames.keys()))
            approximations = [get_shrinkage_approx(key, df) for key, df in data_frames.items()]

            # times = np.logspace(-8, -1, 100)
            # for ip in approximations:
            #     ax.plot(times, ip(times))

            for shrinkage in np.logspace(-2, -1, 5, endpoint=True):
                log_shrinkage = np.log(shrinkage)
                times = np.array([sop.root_scalar(lambda t: np.log(ip(t)) - log_shrinkage, bracket=[0, 10]).root for ip in approximations])
                mask = times > 0
                ax.plot(times[mask], params[mask], label=f"{shrinkage:.2e}")

            ax.legend()
            ax.set_xlabel("Normalized Time $\\Time / \\TimeNorm_{\\Surface}$")
            ax.set_ylabel("Shrinkage")

            for p in produces:
                fig.savefig(p)


def distance(particle1_x, particle1_y, particle2_x, particle2_y):
    return np.sqrt((particle2_x - particle1_x) ** 2 + (particle2_y - particle1_y) ** 2)


def get_shrinkage_approx(param, df: pa.Table):
    particle1: pd.DataFrame = df.filter(pc.field("Particle.Id") == PARTICLE1_ID.bytes).group_by(["State.Id"]).aggregate(
        [("State.Time", "one"), ("Particle.Coordinates.X", "one"), ("Particle.Coordinates.Y", "one")]).sort_by(
        "State.Time_one").to_pandas()
    particle2: pd.DataFrame = df.filter(pc.field("Particle.Id") == PARTICLE2_ID.bytes).group_by(["State.Id"]).aggregate(
        [("State.Time", "one"), ("Particle.Coordinates.X", "one"), ("Particle.Coordinates.Y", "one")]).sort_by(
        "State.Time_one").to_pandas()

    distances = distance(particle1["Particle.Coordinates.X_one"], particle1["Particle.Coordinates.Y_one"],
                         particle2["Particle.Coordinates.X_one"], particle2["Particle.Coordinates.Y_one"])
    distance0 = distances[0]

    mask = (particle1["State.Time_one"] > 1) & (np.diff(particle1["State.Time_one"], prepend=[0]) > 0)
    times = particle1["State.Time_one"][mask] / study.input_for(param).time_norm_surface
    shrinkages = (distance0 - distances[mask]) / distance0

    log_shrinkages = np.log(shrinkages)

    fit = sop.least_squares(lambda params: log_linear_approx(times, params) - log_shrinkages, [1, 1])

    return lambda t: np.exp(log_linear_approx(t, fit.x))

def log_linear_approx(t, params):
    return params[0] + params[1] * np.log(t)