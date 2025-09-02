import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytask import mark, task
from scipy.optimize import least_squares
from shapely import Polygon
from shapely.affinity import translate

from dissertation.config import DEFAULT_FIGSIZE, in_build_dir
from dissertation.data.morphology.batches import BATCHES, BATCHES_DIR
from dissertation.data.morphology.shape_function import particle_shape_function


def create_model_geom(phi, n, r0, o, h, p, rotation):
    x, y = get_model_x_y(phi, n, r0, o, h, p, rotation)
    return Polygon(np.stack([x, y], axis=1))


def get_model_x_y(phi, n, r0, o, h, p, rotation):
    r = r0 * particle_shape_function(phi - rotation, o, n, h, p)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return x, y


for b, files in BATCHES.items():

    @task(id=b)
    @mark.persist
    def task_fit_morphology_circular(files=files, produces=in_build_dir(BATCHES_DIR / "fits_circular" / f"{b}.csv")):
        files = list(files)
        df = pd.DataFrame(
            columns=pd.Index(["r0"]), index=pd.Index([f"{f.parent.name}/{f.stem}" for f in files], dtype=str)
        )

        for i, file in zip(df.index, files, strict=True):
            data = pd.read_csv(
                file,
                skiprows=4,
                names=["x", "y"],
                sep=r"\s+",
            )
            orig_geom = Polygon(np.stack([data.x, data.y], axis=1))
            centralized_geom = translate(orig_geom, -orig_geom.centroid.x, -orig_geom.centroid.y)
            cx, cy = np.asarray(centralized_geom.exterior.xy)

            trajectories = np.atan2(centralized_geom.exterior.xy[1], centralized_geom.exterior.xy[0])
            r = np.sqrt(
                np.asarray(centralized_geom.exterior.xy[0]) ** 2 + np.asarray(centralized_geom.exterior.xy[1]) ** 2
            )
            r_max = np.max(r)

            def objective_function(p, trajectories=trajectories, cx=cx, cy=cy):
                mx, my = get_model_x_y(trajectories, 0, p[0], 1, 0, 0, 0)
                return np.sqrt((mx - cx) ** 2 + (my - cy) ** 2)

            best_fit = least_squares(
                objective_function,
                x0=[r_max / 2],
                bounds=np.array([0, r_max]),
            )

            model_geom = create_model_geom(np.linspace(0, 2 * np.pi, 401), 0, best_fit.x[0], 1, 0, 0, 0)
            plot_fit(
                produces.parent / produces.stem / f"{file.parent.name}_{file.stem}.pdf",
                centralized_geom,
                model_geom,
                f"$\\Radius_0 = \\qty{{{best_fit.x[0]:.3f}}}{{\\micro\\meter}}$",
            )

            df.at[i, "r0"] = best_fit.x[0]

        df.to_csv(produces)

    @task(id=b)
    @mark.persist
    def task_fit_morphology_oval(files=files, produces=in_build_dir(BATCHES_DIR / "fits_oval" / f"{b}.csv")):
        files = list(files)
        df = pd.DataFrame(
            columns=pd.Index(["r0", "o"]), index=pd.Index([f"{f.parent.name}/{f.stem}" for f in files], dtype=str)
        )

        for i, file in zip(df.index, files, strict=True):
            data = pd.read_csv(
                file,
                skiprows=4,
                names=["x", "y"],
                sep=r"\s+",
            )
            orig_geom = Polygon(np.stack([data.x, data.y], axis=1))
            centralized_geom = translate(orig_geom, -orig_geom.centroid.x, -orig_geom.centroid.y)
            cx, cy = np.asarray(centralized_geom.exterior.xy)

            trajectories = np.atan2(centralized_geom.exterior.xy[1], centralized_geom.exterior.xy[0])
            r = np.sqrt(
                np.asarray(centralized_geom.exterior.xy[0]) ** 2 + np.asarray(centralized_geom.exterior.xy[1]) ** 2
            )
            r_max = np.max(r)

            def objective_function(p, trajectories=trajectories, cx=cx, cy=cy):
                mx, my = get_model_x_y(trajectories, 0, p[0], p[1], 0, 0, p[2])
                return np.sqrt((mx - cx) ** 2 + (my - cy) ** 2)

            rot_delims = np.linspace(0, 2 * np.pi, 10, endpoint=False)
            fits = [
                least_squares(
                    objective_function,
                    x0=[
                        r_max / 2,
                        2,
                        (rot_bounds[0] + rot_bounds[1]) / 2,
                    ],
                    bounds=np.array(
                        [
                            (0, r_max),
                            (1, 5),
                            rot_bounds,
                        ]
                    ).T,
                )
                for rot_bounds in zip(rot_delims[:-1], rot_delims[1:], strict=True)
            ]
            best_fit = min(fits, key=lambda f: f.cost)

            model_geom = create_model_geom(
                np.linspace(0, 2 * np.pi, 401), 0, best_fit.x[0], best_fit.x[1], 0, 0, best_fit.x[2]
            )
            plot_fit(
                produces.parent / produces.stem / f"{file.parent.name}_{file.stem}.pdf",
                centralized_geom,
                model_geom,
                f"$\\Radius_0 = \\qty{{{best_fit.x[0]:.3f}}}{{\\micro\\meter}}$\n$\\Ovality = \\num{{{best_fit.x[1]:.3f}}}$\n$\\RotationAngle = \\num{{{best_fit.x[2]:.3f}}}$",
            )

            df.at[i, "r0"] = best_fit.x[0]
            df.at[i, "o"] = best_fit.x[1]

        df.to_csv(produces)

    @task(id=b)
    @mark.persist
    def task_fit_morphology_shape(files=files, produces=in_build_dir(BATCHES_DIR / "fits_shape" / f"{b}.csv")):
        files = list(files)
        df = pd.DataFrame(
            columns=pd.Index(["r0", "o", "h", "p", "n"]),
            index=pd.Index([f"{f.parent.name}/{f.stem}" for f in files], dtype=str),
        )

        for i, file in zip(df.index, files, strict=True):
            data = pd.read_csv(
                file,
                skiprows=4,
                names=["x", "y"],
                sep=r"\s+",
            )
            orig_geom = Polygon(np.stack([data.x, data.y], axis=1))
            centralized_geom = translate(orig_geom, -orig_geom.centroid.x, -orig_geom.centroid.y)
            cx, cy = np.asarray(centralized_geom.exterior.xy)

            trajectories = np.atan2(centralized_geom.exterior.xy[1], centralized_geom.exterior.xy[0])
            r = np.sqrt(
                np.asarray(centralized_geom.exterior.xy[0]) ** 2 + np.asarray(centralized_geom.exterior.xy[1]) ** 2
            )
            r_max = np.max(r)

            def objective_function(p, n, trajectories=trajectories, cx=cx, cy=cy):
                mx, my = get_model_x_y(trajectories, n, *p)
                return np.sqrt((mx - cx) ** 2 + (my - cy) ** 2)

            rot_delims = np.linspace(0, 2 * np.pi, 10, endpoint=False)
            fits = [
                (
                    n,
                    least_squares(
                        objective_function,
                        x0=[
                            r_max / 2,
                            2,
                            0.2,
                            0,
                            (rot_bounds[0] + rot_bounds[1]) / 2,
                        ],
                        bounds=np.array(
                            [
                                (0, r_max),
                                (1, 5),
                                (0, 0.5),
                                (0, 0.499),
                                rot_bounds,
                            ]
                        ).T,
                        args=[n],
                    ),
                )
                for n in range(3, 10)
                for rot_bounds in zip(rot_delims[:-1], rot_delims[1:], strict=True)
            ]
            best_n, best_fit = min(fits, key=lambda f: f[1].cost)

            model_geom = create_model_geom(np.linspace(0, 2 * np.pi, 401), best_n, *best_fit.x)
            plot_fit(
                produces.parent / produces.stem / f"{file.parent.name}_{file.stem}.pdf",
                centralized_geom,
                model_geom,
                f"$\\Radius_0 = \\qty{{{best_fit.x[0]:.3f}}}{{\\micro\\meter}}$\n$\\Ovality = \\num{{{best_fit.x[1]:.3f}}}$\n$\\WaveHeight = \\num{{{best_fit.x[2]:.3f}}}$\n$\\WaveShift = \\num{{{best_fit.x[3]:.3f}}}$\n$\\WaveCount = \\num{{{best_n}}}$\n$\\RotationAngle = \\num{{{best_fit.x[4]:.3f}}}$",
            )

            df.at[i, "r0"] = best_fit.x[0]
            df.at[i, "o"] = best_fit.x[1]
            df.at[i, "h"] = best_fit.x[2]
            df.at[i, "p"] = best_fit.x[3]
            df.at[i, "n"] = best_n

        df.to_csv(produces)


def plot_fit(file, orig_geom, model_geom, param_text):
    fig = plt.figure(figsize=tuple(DEFAULT_FIGSIZE / [2, 1]))
    ax = fig.subplots()
    ax.set_aspect("equal", adjustable="datalim")
    ax.plot(orig_geom.exterior.xy[0], orig_geom.exterior.xy[1])
    ax.plot(model_geom.exterior.xy[0], model_geom.exterior.xy[1])
    ax.text(0, 0, param_text, horizontalalignment="center", verticalalignment="center")
    file.parent.mkdir(exist_ok=True, parents=True)
    ax.grid(True)
    ax.set_xlabel("$x$ in \\unit{\\micro\\meter}")
    ax.set_ylabel("$y$ in \\unit{\\micro\\meter}")
    fig.savefig(file)
    plt.close(fig)
