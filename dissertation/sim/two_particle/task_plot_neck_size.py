import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pytask import mark, task

from dissertation.config import image_produces, integer_log_space
from dissertation.sim.two_particle.studies import PARTICLE1_ID, STUDIES, DimlessParameterStudy, StudyBase

RESAMPLE_COUNT = 500

for t in STUDIES:

    @task(id=f"{t.KEY}")
    @mark.plot
    def task_plot_neck_size(
        produces=image_produces(t.DIR / "neck_size"),
        study_type: type[StudyBase] = t,
        studies: dict[str, StudyBase] = {str(study): study for study in t.INSTANCES},
        results_files={str(study): study.dir / "output.parquet" for study in t.INSTANCES},
    ):
        data_frames = ((k, pq.read_table(f).flatten().flatten()) for k, f in results_files.items())

        fig = plt.figure(dpi=600)
        ax = fig.subplots()
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True)

        for key, df in data_frames:
            study = studies[key]
            times, values = get_neck_sizes(study, df)
            ax.plot(times, values, label=study.display, **study.line_style)

        ax.legend(title=study_type.TITLE, ncols=3)
        ax.set_xlabel("Normalized Time $\\Time / \\TimeNorm_{\\Surface}$")
        ax.set_ylabel(r"Relative Neck Size $\Radius_{\Neck} / \Radius_0$")
        fig.tight_layout()

        for p in produces:
            fig.savefig(p)

        plt.close(fig)

    if issubclass(t, DimlessParameterStudy):

        @task(id=f"{t.KEY}")
        @mark.plot
        def task_plot_neck_size_map(
            produces=image_produces(t.DIR / "neck_size_map"),
            study_type: type[DimlessParameterStudy] = t,
            studies: dict[str, DimlessParameterStudy] = {str(study): study for study in t.INSTANCES},  # type: ignore
            results_files={str(study): study.dir / "output.parquet" for study in t.INSTANCES},
        ):
            data_frames = ((k, pq.read_table(f).flatten().flatten()) for k, f in results_files.items())

            fig = plt.figure(dpi=600)
            ax = fig.subplots()
            ax.set_xscale("log")
            ax.set_yscale(study_type.axis_scale)

            params = np.array([s.real_value for s in studies.values()])
            times_neck_size = [get_neck_sizes(studies[key], df) for key, df in data_frames]

            start_time = max([t[0] for t, _ in times_neck_size])
            end_time = max([t[-1] for t, _ in times_neck_size])

            times = np.geomspace(start_time, end_time, RESAMPLE_COUNT)
            neck_sizes = np.array([np.interp(times, t, s) for t, s in times_neck_size])

            grid_x, grid_y = np.meshgrid(times, params)

            locs = integer_log_space(1, -1, 1, 0)

            cs = ax.contour(grid_x, grid_y, neck_sizes, levels=locs, norm="log", cmap=study_type.CMAP)
            ax.clabel(cs, fmt=lambda level: f"{level:g}")

            ax.set_xlabel("Normalized Time $\\Time / \\TimeNorm_{\\Surface}$")
            ax.set_ylabel(study_type.TITLE)

            for p in produces:
                fig.savefig(p)

            plt.close(fig)


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

    times = grain_boundary["State.Time_one"] / study.input.time_norm_surface
    mask = (times > 1e-6) & (np.diff(times, prepend=[0]) > 0)
    neck_sizes = (
        grain_boundary["Node.SurfaceDistance.ToUpper_one"][mask]
        + grain_boundary["Node.SurfaceDistance.ToLower_one"][mask] / study.input.particle1.radius
    )

    return times[mask].to_numpy(), neck_sizes.to_numpy()
