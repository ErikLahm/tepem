import matplotlib.pyplot as plt
import numpy as np
from backend.assembling import add_inital_penalty, assemble_g, assemble_n, assemble_rhs
from backend.domain import Domain
from backend.ltg_generator import generate_ltg
from backend.mapping import Mapping
from backend.shape_functions.q_1_2_sf import Q12_GRAD_SF_LIST, Q12_SF_LIST
from backend.shape_functions.q_1_3_sf import Q13_GRAD_SF_LIST
from backend.shape_functions.q_1_4_velo_sf import Q14_GRAD_SF_LIST
from backend.shape_functions.q_1_5_velo_sf import Q15_GRAD_SF_LIST
from backend.shape_functions.q_1_6_velo_sf import Q16_GRAD_SF_LIST
from backend.shape_functions.q_1_7_velo_sf import Q17_GRAD_SF_LIST
from backend.shape_functions.q_2_sf import Q2_SF_LIST
from backend.visualizer import solution_visualizer

# from backend.shape_functions.q_1_4_sf import Q14_GRAD_SF_LIST


NU = 100
PRESSURE_GRAD = -2.5
C_CONST = 1 / 2 * 1 / NU * (-PRESSURE_GRAD)
RADIUS = 1
LENGTH = 10


def g_1(x_1: float, x_2: float) -> float:
    if x_1 < 1e-14:
        return C_CONST * (RADIUS - x_2) * (x_2 + RADIUS)
    else:
        return 0


SHALL_SOL = [
    0.009375,
    0.0125,
    0.009375,
    0.009375,
    0.0125,
    0.009375,
    0.009375,
    0.0125,
    0.009375,
    0.009375,
    0.0125,
    0.009375,
    0.009375,
    0.0125,
    0.009375,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    10,
    10,
    10,
    7.5,
    7.5,
    7.5,
    5,
    5,
    5,
    2.5,
    2.5,
    2.5,
    0,
    0,
    0,
]

EX_SOL = np.array(SHALL_SOL).reshape((45, 1))
VELO_SHAPE = (1, 4)
PRESSURE_SHAPE = (1, 2)
VELO_PARTIAL = True


def main() -> None:
    num_slabs = 4
    # dom = Domain(
    #     LENGTH, upper_bdn=lambda x: x**2 / 10 + 0.2, lower_bdn=lambda x: -0.2
    # )
    dom = Domain(LENGTH, upper_bdn=lambda x: RADIUS, lower_bdn=lambda x: -RADIUS)
    dom_coords, dom_ltg = dom.slice_domain(num_slabs)
    phys_coords = dom.get_phy_dof_coords(
        num_slabs, sf_shape=VELO_SHAPE, velo_sf=VELO_PARTIAL
    )
    phys_pre_coords = dom.get_phy_dof_coords(num_slabs, sf_shape=PRESSURE_SHAPE)
    velo_ltg = generate_ltg(
        num_slabs=num_slabs, fe_order=VELO_SHAPE, velocity_ltg=VELO_PARTIAL
    )
    pres_ltg = generate_ltg(num_slabs=num_slabs, fe_order=PRESSURE_SHAPE)
    n_matrix = assemble_n(
        ref_ltg=velo_ltg,
        grad_sfs=Q14_GRAD_SF_LIST,
        domain_coords=dom_coords,
        domain_ltg=dom_ltg,
    )
    n_matrix = add_inital_penalty(
        num_slabs=num_slabs,
        n_matrix=n_matrix,
        velo_sf_shape=VELO_SHAPE,
        velo_pt=VELO_PARTIAL,
    )
    g_matrix = assemble_g(
        velo_ltg=velo_ltg,
        press_ltg=pres_ltg,
        press_sfs=Q12_SF_LIST,
        grad_vel_sfs=Q14_GRAD_SF_LIST,
        domain_coords=dom_coords,
        domain_ltg=dom_ltg,
    )
    rhs = assemble_rhs(
        num_slabs=num_slabs,
        velo_sf_shape=VELO_SHAPE,
        press_sf_shape=PRESSURE_SHAPE,
        phys_velo_dof_coords=phys_coords,
        boundary_func=g_1,
        velo_pt=VELO_PARTIAL,
    )
    d_matrix = np.transpose(g_matrix)
    zero_block = np.zeros(shape=(d_matrix.shape[0], g_matrix.shape[1]))
    upper = np.hstack((n_matrix, g_matrix))
    lower = np.hstack((d_matrix, zero_block))
    s_matrix = np.vstack((upper, lower))
    solution = np.linalg.solve(s_matrix, rhs)
    num_velo_dof = (1 * num_slabs + 1) * (VELO_SHAPE[1] + 1)
    if VELO_PARTIAL:
        num_velo_dof = (1 * num_slabs + 1) * (VELO_SHAPE[1] - 1)
    _, ax_sol = solution_visualizer(
        solution=solution,
        num_vel_dof=num_velo_dof,
        velo_coords=phys_coords,
        pres_coords=phys_pre_coords,
    )
    # ----------------------------------------
    # Troubleshooting
    # ex_rhs = s_matrix.dot(EX_SOL)
    # _, ax_ex_sol = solution_visualizer(
    #     solution=EX_SOL,
    #     num_vel_dof=num_velo_dof,
    #     velo_coords=phys_coords,
    #     pres_coords=phys_pre_coords,
    # )
    # residue = rhs - ex_rhs
    schur = np.dot(np.dot(d_matrix, np.linalg.inv(n_matrix)), g_matrix)
    s, v, dh = np.linalg.svd(schur)
    rank_schur = np.linalg.matrix_rank(schur)
    # -----------------------------------------
    map_k = Mapping(slab_coord=dom_coords[dom_ltg[3]])
    ref_x, ref_y = 0.55, 0.75
    p_x, p_y = map_k.slab_map(ref_x, ref_y)
    _, ax = dom.visualise_domain(coords=dom_coords, ltg=dom_ltg)  # type: ignore
    ax.scatter(p_x, p_y, c="red", marker="+", label="test mapping")  # type: ignore
    ax.scatter(phys_coords[:, 0], phys_coords[:, 1], c="green", label="velocity dof")
    ax.scatter(
        phys_pre_coords[:, 0],
        phys_pre_coords[:, 1],
        facecolors="none",
        edgecolors="b",
        label="pressure dof",
    )
    for i, coords in enumerate(phys_coords):
        ax.annotate(str(i), (coords[0], coords[1]))
    ax.legend()
    # plt.show()  # type: ignore


if __name__ == "__main__":
    main()
