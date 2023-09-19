import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def solution_visualizer(
    solution: npt.NDArray[np.float64],
    num_vel_dof: int,
    velo_coords: npt.NDArray[np.float64],
    pres_coords: npt.NDArray[np.float64],
):
    u_x = solution[:num_vel_dof]
    u_y = solution[num_vel_dof : 2 * num_vel_dof]
    # p = solution[2 * num_vel_dof :]
    arrow_lengths = np.sqrt(u_x**2 + u_y**2)
    fig, ax = plt.subplots()  # type: ignore
    pc = ax.quiver(  # type: ignore
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
    ax.grid(True)  # type: ignore
    return fig, ax
