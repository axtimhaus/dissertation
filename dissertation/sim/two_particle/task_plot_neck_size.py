import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from matplotlib import contour, ticker
from pytask import mark, task

from dissertation.config import image_produces, integer_log_space125
from dissertation.sim.two_particle.helper import ashby_grid
from dissertation.sim.two_particle.studies import PARTICLE1_ID, STUDIES, DimlessParameterStudy, StudyBase

RESAMPLE_COUNT = 100
TIME_MIN = 1e-6
NECK_SIZE_MIN = 2e-1

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
        ax.grid(True, "both")
        max_time = TIME_MIN
        max_neck_size = NECK_SIZE_MIN

        for key, df in data_frames:
            study = studies[key]
            times, values = get_neck_sizes(study, df)
            ax.plot(times, values, label=study.display, **study.line_style)
            max_time = max(max_time, np.max(times))
            max_neck_size = max(max_neck_size, np.max(values))

        if issubclass(study_type, DimlessParameterStudy):
            cb = fig.colorbar(
                contour.ContourSet(
                    ax,
                    [s.value for s in studies.values()],
                    [[[(0, 0)]]] * len(studies),
                    norm=study_type.axis_scale,
                    cmap=study_type.CMAP,
                ),
                label=study_type.TITLE,
                orientation="horizontal",
            )
            cb.minorticks_on()
        else:
            ax.legend(title=study_type.TITLE, ncols=3)

        ax.set_xlabel("Normalized Time $\\Time / \\TimeNorm_{\\Surface}$")
        ax.set_ylabel(r"Relative Neck Size $\Radius_{\Neck} / \Radius_0$")
        ax.set_xlim(TIME_MIN, max_time)
        ax.set_ylim(NECK_SIZE_MIN, min(10 ** np.ceil(np.log10(max_neck_size)), 1.3))

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
            ax.set_yscale("log")
            ax.grid(True, "both")

            study_params = np.array([s.real_value for s in studies.values()])
            params = (np.linspace if study_type.axis_scale == "linear" else np.geomspace)(
                study_params.min(), study_params.max(), RESAMPLE_COUNT
            )
            neck_size_curves = [get_neck_sizes(studies[key], df) for key, df in data_frames]
            max_neck_size = np.max([np.max(n) for _, n in neck_size_curves])
            neck_sizes = np.geomspace(NECK_SIZE_MIN, min(10 ** np.ceil(np.log10(max_neck_size)), 1.3), RESAMPLE_COUNT)
            grid_x, grid_y, times = ashby_grid(study_params, neck_size_curves, params, neck_sizes)

            lower_mag = -6
            upper_mag = int(np.floor(np.log10(np.max(times))))
            locs = integer_log_space125(lower_mag, upper_mag)
            formatter = ticker.LogFormatterSciNotation()

            cs = ax.contour(
                grid_x,
                grid_y,
                times,
                levels=locs,
                norm="log",
                cmap=study_type.CMAP,
            )
            cb = fig.colorbar(
                cs,
                format=formatter,
                label="Normalized Time $\\Time / \\TimeNorm_{\\Surface}$",
                orientation="horizontal",
            )
            cb.minorticks_on()

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
        (
            grain_boundary["Node.SurfaceDistance.ToUpper_one"][mask]
            + grain_boundary["Node.SurfaceDistance.ToLower_one"][mask]
        )
        / study.input.particle1.radius
        / 2
    )

    return times[mask].to_numpy(), neck_sizes.to_numpy()
