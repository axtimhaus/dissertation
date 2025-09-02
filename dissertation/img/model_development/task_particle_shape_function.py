from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytask
from matplotlib import colormaps

from dissertation.config import image_produces, in_build_dir

THIS_DIR = Path(__file__).parent

O = [1, 1.5, 2, 3]
H = [0, 0.1, 0.2, 0.5]
N = [0, 3, 5, 9]
P = [0, 0.2, 0.3, 0.5, 0.8]


def particle(angles: np.ndarray, o: float, n: int, h: float, p: float):
    return np.sqrt(o / (o**2 * np.sin(angles) ** 2 + np.cos(angles) ** 2)) * (
        h * np.cos(n * angles + 2 * np.pi * p) + 1
    )


for key, values in [("o", O), ("h", H), ("n", N), ("p", P)]:

    @pytask.task(id=key)
    def task_plot_particle_shape_function(
        key=key, values=values, produces=image_produces(in_build_dir(THIS_DIR / f"particle_shape_function_{key}"))
    ):
        fig = plt.figure()
        ax = fig.subplots()

        angles = np.linspace(0, 2 * np.pi, endpoint=True, num=200)
        cmap = colormaps["viridis"]
        cmap_max = max(values)
        cmap_min = min(values)

        ax.set_aspect("equal", adjustable="box")

        for v in values:
            r = particle(angles, **(dict(o=1.5, h=0.2, n=0 if key == "o" else 5, p=0) | {key: v}))
            ax.plot(
                r * np.cos(angles),
                r * np.sin(angles),
                label=f"\\num{{{v}}}",
                c=cmap((v - cmap_min) / (cmap_max - cmap_min)),
            )
            ax.legend(title=f"${key}_1$")

        ax.set_xlabel(r"$\X / \Radius_0$")
        ax.set_ylabel(r"$\Y / \Radius_0$")
        ax.grid(True)

        fig.subplots_adjust(hspace=0, wspace=0)

        for f in produces:
            fig.savefig(f)
