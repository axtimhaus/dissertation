from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pytask
import matplotlib.lines as mlines

from config import IMAGE_FILE_FORMATS

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


@pytask.mark.produces([f"plot_tangential_potential.{e}" for e in IMAGE_FILE_FORMATS])
def task_plot_tangential_potential(produces: dict[..., Path]):
    fig: plt.Figure = plt.figure(dpi=600)
    ax: plt.Axes = fig.subplots()

    ds = np.linspace(-0.1, 0.1, 50)

    handles = []

    for i, (gamma_u, gamma_l, delta) in enumerate([
        (1, 1, 60),
        (2, 1, 60),
        (1, 2, 60),
        # (2, 1, 120),
    ]):
        handles += ax.plot(
            ds,
            dg(ds, np.deg2rad(delta), gamma_u, gamma_l),
            c=f"C{i}",
            label=rf"$\Upper\InterfaceEnergy=\qty{{{gamma_u}}}{{\joule\per\square\meter}}, \Lower\InterfaceEnergy=\qty{{{gamma_l}}}{{\joule\per\square\meter}}, \Tangential\SurfaceRadiusAngle=\qty{{{delta}}}{{\degree}}$"
        )
        ax.plot(
            ds,
            dg_lin(ds, np.deg2rad(delta), gamma_u, gamma_l),
            ls="--",
            c=f"C{i}",
        )

    handles+=[
        mlines.Line2D([],[], color="k", ls="-", label="Exakt Solution"),
        mlines.Line2D([], [], color="k", ls="--", label=r"Tangent in $\Tangential\ShiftStep=0$"),
    ]

    ax.set_xlabel(r"$\Tangential\ShiftStep$ in \unit{\meter}")
    ax.set_ylabel(r"$\Tangential\GibbsEnergyStep$ in \unit{\joule\per\meter}")
    ax.grid(True)
    ax.set_ylim(bottom=1.5 * ax.get_ylim()[0])

    ax.legend(handles=handles, loc="lower left")

    for f in produces.values():
        fig.savefig(f)
