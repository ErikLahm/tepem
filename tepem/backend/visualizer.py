from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def solution_visualizer(
    solution: npt.NDArray[np.float64],
    num_vel_dof: int,
    velo_coords: npt.NDArray[np.float64],
    pres_coords: npt.NDArray[np.float64],
    pres_shape: Tuple[int, int],
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
        width=0.01,
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
    return fig, ax
