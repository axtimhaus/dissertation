import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from matplotlib import ticker
from pytask import mark, task

from dissertation.config import image_produces, integer_log_space125
from dissertation.sim.two_particle.studies import PARTICLE1_ID, STUDIES, DimlessParameterStudy, StudyBase
from dissertation.sim.two_particle.helper import ashby_grid

RESAMPLE_COUNT = 100
NECK_SIZE_LIMITS = (2e-1, 8e-1)

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
        upper_mag = -6

        for key, df in data_frames:
            study = studies[key]
            times, values = get_neck_sizes(study, df)
            ax.plot(times, values, label=study.display, **study.line_style)
            upper_mag = max(upper_mag, int(np.floor(np.log10(np.max(times)))))

        ax.legend(title=study_type.TITLE, ncols=3)
        ax.set_xlabel("Normalized Time $\\Time / \\TimeNorm_{\\Surface}$")
        ax.set_ylabel(r"Relative Neck Size $\Radius_{\Neck} / \Radius_0$")
        ax.set_xlim(1e-6, 10**upper_mag)
        ax.set_ylim(*NECK_SIZE_LIMITS)
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
            ax.set_xscale(study_type.axis_scale)

            study_params = np.array([s.real_value for s in studies.values()])
            params = (np.linspace if study_type.axis_scale == "linear" else np.geomspace)(
                study_params.min(), study_params.max(), RESAMPLE_COUNT
            )
            neck_sizes = np.geomspace(*NECK_SIZE_LIMITS, RESAMPLE_COUNT)
            neck_size_curves = [get_neck_sizes(studies[key], df) for key, df in data_frames]
            grid_x, grid_y, times = ashby_grid(study_params, neck_size_curves, params, neck_sizes)

            lower_mag = -6
            upper_mag = int(np.floor(np.log10(np.max(times))))
            locs = integer_log_space125(lower_mag, upper_mag)
            formatter = ticker.LogFormatterSciNotation()

            cs = ax.contour(grid_x, grid_y, times, levels=locs, norm="log", cmap=study_type.CMAP)
            ax.clabel(cs, fmt=lambda level: formatter(level))

            ax.set_xlabel(study_type.TITLE)
            ax.set_ylabel(r"Relative Neck Size $\Radius_{\Neck} / \Radius_0$")

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
    mask = np.diff(times, prepend=[0]) > 0
    neck_sizes = (
        grain_boundary["Node.SurfaceDistance.ToUpper_one"][mask]
        + grain_boundary["Node.SurfaceDistance.ToLower_one"][mask] / study.input.particle1.radius
    )

    return times[mask].to_numpy(), neck_sizes.to_numpy()
