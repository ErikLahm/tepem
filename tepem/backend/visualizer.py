from typing import Tuple

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import numpy.typing as npt


def solution_visualizer(  # type: ignore
    solution: npt.NDArray[np.float64],
    num_vel_dof: int,
    velo_coords: npt.NDArray[np.float64],
    pres_coords: npt.NDArray[np.float64],
    pres_shape: Tuple[int, int],
    info: dict[str, str]
    # num_slabs: int,
):
    u_x = solution[:num_vel_dof]
    u_y = solution[num_vel_dof : 2 * num_vel_dof]
    p = solution[2 * num_vel_dof :]
    arrow_lengths = np.sqrt(u_x**2 + u_y**2)
    fig, ax = plt.subplots(2, figsize=(15, 10))  # type: ignore
    pc = ax[0].quiver(  # type: ignore
        velo_coords[:, 0],
        velo_coords[:, 1],
        u_x,
        u_y,
        arrow_lengths,
        scale=0.0025,
        scale_units="width",
        units="width",
        width=0.0025,
        headlength=3,
        headaxislength=3,
        # cmap="Greys",
    )
    ax[0].grid(True)  # type: ignore

    x_pressure = pres_coords[0 :: pres_shape[1] + 1, 0]
    # y_pressure = pres_coords[: pres_shape[1] + 1, 1]
    # xv, yv = np.meshgrid(x_pressure, y_pressure)
    linestyles = ["solid", "solid", "dashed", "dashdot"]
    labels = ["lower boundary", "middle of pipe", "upper boundary"]
    colors = ["red", "grey", "yellow"]
    for i in range(pres_shape[1] + 1):
        ax[1].plot(
            x_pressure,
            p[i :: pres_shape[1] + 1],
            label=f"layer {i} ({labels[i]})",
            linestyle=linestyles[i],
            color=colors[i],
        )
    ax[1].set_ylabel("pressure")
    ax[0].set_ylabel("radius in y-direction")
    ax[1].set_xlabel("legth in x-direction")
    ax[0].set_xlabel("legth in x-direction")
    ax[1].grid(True)
    ax[1].legend()
    text = (
        f'velocity shape function: $\\mathbb{{Q}}_{{{info["velo_sf"][0]},{info["velo_sf"][1]}}}$\n'
        f'pressure shape function: $\\mathbb{{Q}}_{{{info["pres_sf"][0]},{info["pres_sf"][1]}}}$\n'
        f'number of slabs: $n={info["num_slabs"]}$\n'
        f'opening angle domain: $\\alpha={info["angle"]}\\degree$\n'
    )
    plt.figtext(0.8, 0.9, text)  # type: ignore
    return fig, ax  # type: ignore


def plot_velo_pres_sol(
    p_sol: npt.NDArray[np.float64],
    x_p: npt.NDArray[np.float64],
    y_p: npt.NDArray[np.float64],
    u_x: npt.NDArray[np.float64],
    u_y: npt.NDArray[np.float64],
    x_u: npt.NDArray[np.float64],
    y_u: npt.NDArray[np.float64],
):
    fig, ax = plt.subplots()  # type: ignore
    triang = tri.Triangulation(x_p, y_p)
    # ax.triplot(triang, "bo-", lw=0.2)
    # plot only triangles with sidelength smaller some max_radius
    max_radius = 0.1
    triangles = triang.triangles

    # Mask off unwanted triangles.
    xtri = x_p[triangles] - np.roll(x_p[triangles], 1, axis=1)
    ytri = y_p[triangles] - np.roll(y_p[triangles], 1, axis=1)
    maxi = np.max(np.sqrt(xtri**2 + ytri**2), axis=1)
    triang.set_mask(maxi > max_radius)

    # ax.triplot(triang, color="indigo", lw=2.6)
    ax.tricontour(triang, p_sol, levels=25, linewidths=0.5, colors="k")  # type: ignore
    c_data = ax.tricontourf(triang, p_sol, levels=50, cmap="RdBu_r")  # type: ignore
    fig.colorbar(c_data)  # type: ignore
    # ax.plot(x_p, y_p, "ko")
    arrow_lengths = np.sqrt(u_x**2 + u_y**2)
    pc = ax.quiver(  # type: ignore
        x_u,
        y_u,
        u_x,
        u_y,
        arrow_lengths,
        # scale=0.0025,
        # scale_units="width",
        # units="width",
        # width=0.0025,
        # headlength=3,
        # headaxislength=3,
        cmap="Greys",
    )
    return fig, ax
