from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pytask

from config import IMAGE_FILE_FORMATS, in_build_dir

THIS_DIR = Path(__file__).parent

AU = 1
AL = 1


def aup(ds, delta):
    return np.sqrt(AU ** 2 + ds ** 2 - 2 * AU * ds * np.cos(delta))


def alp(ds, delta):
    return np.sqrt(AL ** 2 + ds ** 2 - 2 * AL * ds * np.cos(delta))


def dg(ds, delta, gamma_u, gamma_l):
    return gamma_u * (aup(ds, delta) - AU) + gamma_l * (alp(ds, delta) - AL)


def dg_lin(ds, delta, gamma_u, gamma_l):
    return -(gamma_u + gamma_l) * np.cos(delta) * ds


@pytask.mark.produces([in_build_dir(THIS_DIR / f"plot_normal_potential.{e}") for e in IMAGE_FILE_FORMATS])
def task_plot_normal_potential(produces: dict[..., Path]):
    fig: plt.Figure = plt.figure(figsize=(6, 4), dpi=600)
    ax: plt.Axes = fig.subplots()

    ds = np.linspace(-0.1, 0.1, 50)

    for i, (gamma_u, gamma_l, delta) in enumerate([
        (1, 1, 60),
        (2, 1, 60),
        (1, 1, 120),
        (1, 1, 90),
    ]):
        ax.plot(
            ds,
            dg(ds, np.deg2rad(delta), gamma_u, gamma_l),
            c=f"C{i}",
            label=rf"$\Upper\InterfaceEnergy=\qty{{{gamma_u}}}{{\joule\per\square\meter}}, \Lower\InterfaceEnergy=\qty{{{gamma_l}}}{{\joule\per\square\meter}}, \Normal\SurfaceRadiusAngle=\qty{{{delta}}}{{\degree}}$"
        )
        ax.plot(
            ds,
            dg_lin(ds, np.deg2rad(delta), gamma_u, gamma_l),
            ls="--",
            c=f"C{i}",
        )

    ax.set_xlabel(r"$\Normal\ShiftStep$ in \unit{\meter}")
    ax.set_ylabel(r"$\Normal\GibbsEnergyStep$ in \unit{\joule\per\meter}")
    ax.grid(True)
    ax.set_ylim(bottom=1.5 * ax.get_ylim()[0])

    ax.legend(loc="lower left")

    for f in produces.values():
        fig.savefig(f)
