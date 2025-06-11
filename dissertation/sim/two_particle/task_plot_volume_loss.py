from uuid import UUID

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pytask import mark, task

from dissertation.config import image_produces
from dissertation.sim.two_particle.studies import PARTICLE1_ID, PARTICLE2_ID, STUDIES, StudyBase

for t in STUDIES:

    @task(id=f"{t.KEY}")
    @mark.plot
    def task_plot_volume_loss(
        produces=image_produces(t.DIR / "volume_loss"),
        study_type: type[StudyBase] = t,
        studies: dict[str, StudyBase] = {str(study): study for study in t.INSTANCES},
        results_files={str(study): study.dir / "output.parquet" for study in t.INSTANCES},
    ):
        data_frames = ((k, pq.read_table(f).flatten().flatten()) for k, f in results_files.items())

        fig = plt.figure(dpi=600)
        ax = fig.subplots()
        ax.set_xscale("log")
        ax.set_yscale("asinh")
        ax.grid(True)

        for key, df in data_frames:
            study = studies[key]
            times1, volume_losses1 = get_volume_losses(study, df, PARTICLE1_ID)
            ax.plot(times1, volume_losses1, label=study.display, **study.line_style)[0]
            times2, volume_losses2 = get_volume_losses(study, df, PARTICLE2_ID)
            ax.plot(times2, volume_losses2, **study.line_style)

        ax.legend(title=study_type.TITLE, ncols=3)
        ax.set_xlabel(r"Normalized Time $\Time / \TimeNorm_{\Surface}$")
        ax.set_ylabel(r"Relative Volume Loss $(\Volume - \Volume_0) / \Volume_0$")
        ax.set_ylim(0, 1e-3)
        fig.tight_layout()

        for p in produces:
            fig.savefig(p)

        plt.close(fig)


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
    mask = (states["State.Time_one"] > 0) & (np.diff(states["State.Time_one"], prepend=[0]) > 0)
    times = states["State.Time_one"][mask] / study.input.time_norm_surface
    volumes = states["Node.Volume.ToUpper_sum"][mask]
    volume_losses = (volumes - initial_volume) / initial_volume

    return times.array, volume_losses.array
