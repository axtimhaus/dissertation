from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pytask

from dissertation.config import IMAGE_FILE_FORMATS, in_build_dir

THIS_DIR = Path(__file__).parent

AU = 1
AL = 1


def aup(ds, delta):
    return np.sqrt(AU ** 2 + ds ** 2 - 2 * AU * ds * np.cos(delta))


def alp(ds, delta):
    return np.sqrt(AL ** 2 + ds ** 2 + 2 * AL * ds * np.cos(delta))


def dg(ds, delta, gamma_u, gamma_l):
    return gamma_u * (aup(ds, delta) - AU) + gamma_l * (alp(ds, delta) - AL)


def dg_lin(ds, delta, gamma_u, gamma_l):
    return -(gamma_u - gamma_l) * np.cos(delta) * ds


def task_plot_tangential_potential(
        produces: list[Path] = [in_build_dir(THIS_DIR / f"plot_tangential_potential.{e}") for e in IMAGE_FILE_FORMATS]
):
    fig: plt.Figure = plt.figure(figsize=(6, 4), dpi=600)
    ax: plt.Axes = fig.subplots()

    ds = np.linspace(-0.1, 0.1, 50)

    for i, (gamma_u, gamma_l, delta) in enumerate([
        (1, 1, 60),
        (2, 1, 60),
        (1, 2, 60),
        # (2, 1, 120),
    ]):
        ax.plot(
            ds,
            dg(ds, np.deg2rad(delta), gamma_u, gamma_l),
            c=f"C{i}",
            label=rf"$\InterfaceEnergy_\Upper=\qty{{{gamma_u}}}{{\joule\per\square\meter}}, \InterfaceEnergy_\Lower=\qty{{{gamma_l}}}{{\joule\per\square\meter}}, \SurfaceVectorAngle_\Tangential=\qty{{{delta}}}{{\degree}}$"
        )
        ax.plot(
            ds,
            dg_lin(ds, np.deg2rad(delta), gamma_u, gamma_l),
            ls="--",
            c=f"C{i}",
        )

    ax.set_xlabel(r"$\Step\Shift_\Tangential$ in \unit{\meter}")
    ax.set_ylabel(r"$\Step\GibbsEnergy_\Tangential$ in \unit{\joule\per\meter}")
    ax.grid(True)
    ax.set_ylim(bottom=1.5 * ax.get_ylim()[0])

    ax.legend(loc="lower left")

    for f in produces:
        fig.savefig(f)
