import datetime
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from backend.assembling import add_inital_penalty, assemble_g, assemble_n, assemble_rhs
from backend.curve_shapes.boundary_shapes import linear_boundary, sudden_linear
from backend.curve_shapes.curves import (
    cos_curve,
    cos_curve_normal,
    curved_upward,
    curved_upward_normal,
    straight_curve,
    straight_curve_normal,
)
from backend.curve_shapes.sudden_change_dom import SuddenChangeDomain
from backend.curve_shapes.wave_domain import WaveDomain
from backend.domain import Domain
from backend.expected_solutions import EX_SOL, EX_SOL_q16
from backend.high_res_mapping import get_high_res_phys_coords, get_high_res_solution
from backend.ltg_generator import generate_ltg
from backend.mapping import Mapping
from backend.shape_functions.q_1_2_sf import Q12_GRAD_SF_LIST, Q12_SF_LIST
from backend.shape_functions.q_1_3_sf import Q13_GRAD_SF_LIST, Q13_SF_LIST
from backend.shape_functions.q_1_4_velo_sf import Q14_GRAD_SF_LIST
from backend.shape_functions.q_1_5_velo_sf import Q15_GRAD_SF_LIST
from backend.shape_functions.q_1_6_velo_sf import Q16_GRAD_SF_LIST
from backend.shape_functions.q_1_7_velo_sf import Q17_GRAD_SF_LIST
from backend.shape_functions.q_2_4_velo_sf import Q24_GRAD_SF_LIST
from backend.shape_functions.q_2_4_velo_sf_cheb import Q24_GRAD_SF_CHEB_LIST
from backend.shape_functions.q_2_5_velo_sf import Q25_GRAD_SF_LIST, Q25_SF_LIST
from backend.shape_functions.q_2_5_velo_sf_cheb import (
    Q25_GRAD_SF_CHEB_LIST,
    Q25_SF_CHEB_LIST,
)
from backend.shape_functions.q_2_6_velo_cheb import (
    Q26_GRAD_SF_CHEB_LIST,
    Q26_SF_CHEB_LIST,
)
from backend.shape_functions.q_2_6_velo_sf import Q26_GRAD_SF_LIST, Q26_SF_LIST
from backend.shape_functions.q_2_sf import Q2_SF_LIST
from backend.shape_functions.q_3_4_velo_sf import Q34_GRAD_SF_LIST
from backend.visualizer import plot_velo_pres_sol, solution_visualizer

# from backend.shape_functions.q_1_4_sf import Q14_GRAD_SF_LIST


NU = 8.1e-2  # [Pa * s]
# PRESSURE_GRAD = -2.5
# C_CONST = 1 / 2 * 1 / NU * (-PRESSURE_GRAD)
RADIUS = 0.0125  # [m]
LENGTH = 0.22  # [m]
LENGTH_STR_INLET = 0.01  # [m]
START_STR_OUTLET = 0.21
ANGLE = -5  # [°]
CURVE_ANGLE = 0
VOLUME_FLUX = 5e-3  # [m^3/(m^2*s) = m/s]


def get_constant(mass_flow: float) -> float:
    return mass_flow / (
        (RADIUS**2 * (2 * RADIUS) - 1 / 3 * (RADIUS**3 - (-RADIUS) ** 3))
    )


C_CONST = get_constant(VOLUME_FLUX)


def straight_upper(s: float) -> Tuple[float, float]:
    x = (
        straight_curve(s, CURVE_ANGLE)[0]
        + linear_boundary(s) * straight_curve_normal(s, CURVE_ANGLE)[0]
    )
    y = (
        straight_curve(s, CURVE_ANGLE)[1]
        + linear_boundary(s) * straight_curve_normal(s, CURVE_ANGLE)[1]
    )
    return x, y


def straight_lower(s: float) -> Tuple[float, float]:
    x = (
        straight_curve(s, CURVE_ANGLE)[0]
        - linear_boundary(s) * straight_curve_normal(s, CURVE_ANGLE)[0]
    )
    y = (
        straight_curve(s, CURVE_ANGLE)[1]
        - linear_boundary(s) * straight_curve_normal(s, CURVE_ANGLE)[1]
    )
    return x, y


def straight_conv_upper(s: float) -> Tuple[float, float]:
    if s <= LENGTH_STR_INLET:
        x = (
            straight_curve(s, CURVE_ANGLE)[0]
            + linear_boundary(s, n=RADIUS) * straight_curve_normal(s, CURVE_ANGLE)[0]
        )
        y = (
            straight_curve(s, CURVE_ANGLE)[1]
            + linear_boundary(s, n=RADIUS) * straight_curve_normal(s, CURVE_ANGLE)[1]
        )
    elif s > LENGTH_STR_INLET and s <= START_STR_OUTLET:
        x = (
            straight_curve(s, CURVE_ANGLE)[0]
            + (
                linear_boundary(s, m=angle_to_gradient(ANGLE), n=RADIUS)
                - LENGTH_STR_INLET * angle_to_gradient(ANGLE)
            )
            * straight_curve_normal(s, CURVE_ANGLE)[0]
        )
        y = (
            straight_curve(s, CURVE_ANGLE)[1]
            + (
                linear_boundary(s, m=angle_to_gradient(ANGLE), n=RADIUS)
                - LENGTH_STR_INLET * angle_to_gradient(ANGLE)
            )
            * straight_curve_normal(s, CURVE_ANGLE)[1]
        )
    else:
        n_decl = RADIUS - angle_to_gradient(ANGLE) * LENGTH_STR_INLET
        n_out = angle_to_gradient(ANGLE) * START_STR_OUTLET + n_decl
        x = (
            straight_curve(s, CURVE_ANGLE)[0]
            + linear_boundary(s, n=n_out) * straight_curve_normal(s, CURVE_ANGLE)[0]
        )
        y = (
            straight_curve(s, CURVE_ANGLE)[1]
            + linear_boundary(s, n=n_out) * straight_curve_normal(s, CURVE_ANGLE)[1]
        )
    return x, y


def straight_conv_lower(s: float) -> Tuple[float, float]:
    if s <= LENGTH_STR_INLET:
        x = (
            straight_curve(s, CURVE_ANGLE)[0]
            - linear_boundary(s, n=RADIUS) * straight_curve_normal(s, CURVE_ANGLE)[0]
        )
        y = (
            straight_curve(s, CURVE_ANGLE)[1]
            - linear_boundary(s, n=RADIUS) * straight_curve_normal(s, CURVE_ANGLE)[1]
        )
    elif s > LENGTH_STR_INLET and s <= START_STR_OUTLET:
        x = (
            straight_curve(s, CURVE_ANGLE)[0]
            - (
                linear_boundary(s, m=angle_to_gradient(ANGLE), n=RADIUS)
                - LENGTH_STR_INLET * angle_to_gradient(ANGLE)
            )
            * straight_curve_normal(s, CURVE_ANGLE)[0]
        )
        y = (
            straight_curve(s, CURVE_ANGLE)[1]
            - (
                linear_boundary(s, m=angle_to_gradient(ANGLE), n=RADIUS)
                - LENGTH_STR_INLET * angle_to_gradient(ANGLE)
            )
            * straight_curve_normal(s, CURVE_ANGLE)[1]
        )
    else:
        n_decl = RADIUS - angle_to_gradient(ANGLE) * LENGTH_STR_INLET
        n_out = angle_to_gradient(ANGLE) * START_STR_OUTLET + n_decl
        x = (
            straight_curve(s, CURVE_ANGLE)[0]
            - linear_boundary(s, n=n_out) * straight_curve_normal(s, CURVE_ANGLE)[0]
        )
        y = (
            straight_curve(s, CURVE_ANGLE)[1]
            - linear_boundary(s, n=n_out) * straight_curve_normal(s, CURVE_ANGLE)[1]
        )
    return x, y


def cos_upper(s: float) -> Tuple[float, float]:
    if s <= LENGTH_STR_INLET:
        x = (
            straight_curve(s, CURVE_ANGLE)[0]
            + linear_boundary(s, n=RADIUS) * straight_curve_normal(s, CURVE_ANGLE)[0]
        )
        y = (
            straight_curve(s, CURVE_ANGLE)[1]
            + linear_boundary(s, n=RADIUS) * straight_curve_normal(s, CURVE_ANGLE)[1]
        )
    elif s <= START_STR_OUTLET:
        x = (
            cos_curve(s, LENGTH_STR_INLET)[0]
            + linear_boundary(
                s, n=RADIUS + (0.00625 / 0.21) * 0.01, m=-(0.00625 / 0.21)
            )
            * cos_curve_normal(s, LENGTH_STR_INLET)[0]
        )
        y = (
            cos_curve(s, LENGTH_STR_INLET)[1]
            + linear_boundary(
                s, n=RADIUS + (0.00625 / 0.21) * 0.01, m=-(0.00625 / 0.21)
            )
            * cos_curve_normal(s, LENGTH_STR_INLET)[1]
        )
    else:
        x = (
            straight_curve(s, CURVE_ANGLE)[0]
            + linear_boundary(s) * straight_curve_normal(s, CURVE_ANGLE)[0]
        )
        y = (
            straight_curve(s, CURVE_ANGLE)[1]
            + linear_boundary(s) * straight_curve_normal(s, CURVE_ANGLE)[1]
        )
    return x, y


def cos_lower(s: float) -> Tuple[float, float]:
    if s <= LENGTH_STR_INLET:
        x = (
            straight_curve(s, CURVE_ANGLE)[0]
            - linear_boundary(s, n=RADIUS) * straight_curve_normal(s, CURVE_ANGLE)[0]
        )
        y = (
            straight_curve(s, CURVE_ANGLE)[1]
            - linear_boundary(s, n=RADIUS) * straight_curve_normal(s, CURVE_ANGLE)[1]
        )
    elif s <= START_STR_OUTLET:
        x = (
            cos_curve(s, LENGTH_STR_INLET)[0]
            - linear_boundary(
                s, n=RADIUS + (0.00625 / 0.21) * 0.01, m=-(0.00625 / 0.22)
            )
            * cos_curve_normal(s, LENGTH_STR_INLET)[0]
        )
        y = (
            cos_curve(s, LENGTH_STR_INLET)[1]
            - linear_boundary(
                s, n=RADIUS + (0.00625 / 0.21) * 0.01, m=-(0.00625 / 0.22)
            )
            * cos_curve_normal(s, LENGTH_STR_INLET)[1]
        )
    else:
        x = (
            straight_curve(s, CURVE_ANGLE)[0]
            - linear_boundary(s) * straight_curve_normal(s, CURVE_ANGLE)[0]
        )
        y = (
            straight_curve(s, CURVE_ANGLE)[1]
            - linear_boundary(s) * straight_curve_normal(s, CURVE_ANGLE)[1]
        )
    return x, y


def sudden_bdn_upper(s: float) -> Tuple[float, float]:
    x = (
        straight_curve(s, CURVE_ANGLE)[0]
        + linear_boundary(s, n=RADIUS) * straight_curve_normal(s, CURVE_ANGLE)[0]
    )
    y = (
        straight_curve(s, CURVE_ANGLE)[1]
        + linear_boundary(s, n=RADIUS) * straight_curve_normal(s, CURVE_ANGLE)[1]
    )
    return x, y


def sudden_bdn_lower(s: float) -> Tuple[float, float]:
    x = (
        straight_curve(s, CURVE_ANGLE)[0]
        - sudden_linear(s) * straight_curve_normal(s, CURVE_ANGLE)[0]
    )
    y = (
        straight_curve(s, CURVE_ANGLE)[1]
        - sudden_linear(s) * straight_curve_normal(s, CURVE_ANGLE)[1]
    )
    return x, y


# def sudden_bdn_lower(s: float) -> Tuple[float, float]:
#     if s <= 0.035 or s > 0.085:
#         x = (
#             straight_curve(s, CURVE_ANGLE)[0]
#             - linear_boundary(s, n=RADIUS) * straight_curve_normal(s, CURVE_ANGLE)[0]
#         )
#         y = (
#             straight_curve(s, CURVE_ANGLE)[1]
#             - linear_boundary(s, n=RADIUS) * straight_curve_normal(s, CURVE_ANGLE)[1]
#         )
#     else:
#         x = (
#             straight_curve(s, CURVE_ANGLE)[0]
#             - linear_boundary(s, n=0.04) * straight_curve_normal(s, CURVE_ANGLE)[0]
#         )
#         y = (
#             straight_curve(s, CURVE_ANGLE)[1]
#             - linear_boundary(s, n=0.04) * straight_curve_normal(s, CURVE_ANGLE)[1]
#         )
#     return x, y


# def curved_upper(s: float) -> Tuple[float, float]:
#     x = curved_upward(s)[0] + linear_boundary(s) * curved_upward_normal(s)[0]
#     y = curved_upward(s)[1] + linear_boundary(s) * curved_upward_normal(s)[1]
#     return x, y


# def curved_lower(s: float) -> Tuple[float, float]:
#     x = curved_upward(s)[0] - linear_boundary(s) * curved_upward_normal(s)[0]
#     y = curved_upward(s)[1] - linear_boundary(s) * curved_upward_normal(s)[1]
#     return x, y


def g_1(x_1: float, x_2: float) -> float:
    if x_1 < 1e-14:
        return C_CONST * (RADIUS - x_2) * (x_2 + RADIUS)
    else:
        return 0


# def upper_joint_out_bdn(x: float) -> float:
#     if x < 1:
#         return RADIUS
#     if x >= 1 and x <= 5:
#         return -1 / 50 * ((x - 3) ** 2) + 4 * 1 / 50 + RADIUS
#     return RADIUS


# def upper_joint_in_bdn(x: float) -> float:
#     if x < 1:
#         return RADIUS
#     if x >= 1 and x <= 5:
#         return 1 / 30 * ((x - 3) ** 2) - 4 * 1 / 30 + RADIUS
#     return RADIUS


# def lower_joint_out_bdn(x: float) -> float:
#     if x < 1:
#         return -RADIUS
#     if x >= 1 and x <= 5:
#         return 1 / 50 * ((x - 3) ** 2) - 4 * 1 / 50 - RADIUS
#     return -RADIUS


def angle_to_gradient(angle: float) -> float:
    rad = angle_to_radian(angle=angle)
    return np.tan(rad)


def angle_to_radian(angle: float) -> float:
    return angle / 180 * np.pi


VELO_SHAPE = (2, 6)
PRESSURE_SHAPE = (1, 3)
VELO_PARTIAL = True


def main() -> None:
    num_slabs = 10
    wave_bdn = WaveDomain(
        length_str_inlet=LENGTH_STR_INLET,
        start_str_outlet=START_STR_OUTLET,
        init_radius=RADIUS,
        gradient=0,
        period=0.20 / 2,
    )
    bulge_bdn = SuddenChangeDomain(
        dom_length=LENGTH, max_height=0.04, init_radius=RADIUS
    )
    dom = Domain(
        LENGTH,
        # upper_bdn=straight_upper,
        # lower_bdn=straight_lower,
        # upper_bdn=straight_conv_upper,
        # lower_bdn=straight_conv_lower,
        # upper_bdn=bulge_bdn.sudden_bdn_upper,
        # lower_bdn=bulge_bdn.sudden_bdn_lower,
        upper_bdn=wave_bdn.cos_upper,
        lower_bdn=wave_bdn.cos_lower,
        # upper_bdn=sudden_bdn_upper,
        # lower_bdn=sudden_bdn_lower,
        # upper_bdn=curved_upper,
        # lower_bdn=curved_lower,
        # upper_bdn=lambda x: angle_to_gradient(ANGLE) * x + RADIUS,
        # lower_bdn=lambda x: -angle_to_gradient(ANGLE) * x - RADIUS,
        # upper_bdn=upper_joint_out_bdn,
        # lower_bdn=lower_joint_out_bdn,
        # upper_bdn=upper_joint_in_bdn,
    )
    dom_coords, dom_ltg = dom.slice_domain(num_slabs)
    phys_coords = dom.get_phy_dof_coords(
        num_slabs, sf_shape=VELO_SHAPE, velo_sf=VELO_PARTIAL, cheby_points=True
    )
    phys_pre_coords = dom.get_phy_dof_coords(
        num_slabs, sf_shape=PRESSURE_SHAPE, cheby_points=True
    )
    velo_ltg = generate_ltg(
        num_slabs=num_slabs, fe_order=VELO_SHAPE, velocity_ltg=VELO_PARTIAL
    )
    pres_ltg = generate_ltg(num_slabs=num_slabs, fe_order=PRESSURE_SHAPE)
    n_matrix = assemble_n(
        ref_ltg=velo_ltg,
        grad_sfs=Q26_GRAD_SF_CHEB_LIST,
        domain_coords=dom_coords,
        domain_ltg=dom_ltg,
        nu=NU,
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
        press_sfs=Q13_SF_LIST,
        grad_vel_sfs=Q26_GRAD_SF_CHEB_LIST,
        domain_coords=dom_coords,
        domain_ltg=dom_ltg,
    )
    rhs = assemble_rhs(
        num_slabs=num_slabs,
        velo_sf_shape=VELO_SHAPE,
        press_sf_shape=PRESSURE_SHAPE,
        phys_velo_dof_coords=phys_coords,
        const=C_CONST,
        radius=RADIUS,
        velo_pt=VELO_PARTIAL,
    )
    d_matrix = np.transpose(g_matrix)
    zero_block = np.zeros(shape=(d_matrix.shape[0], g_matrix.shape[1]))
    upper = np.hstack((n_matrix, g_matrix))
    lower = np.hstack((d_matrix, zero_block))
    s_matrix = np.vstack((upper, lower))
    # np.savetxt(
    #     f"tepem/exports/full_matrix_q{VELO_SHAPE[0]}{VELO_SHAPE[1]}_q{PRESSURE_SHAPE[0]}{PRESSURE_SHAPE[1]}_slabs{num_slabs}.txt",
    #     s_matrix,
    #     delimiter=",",
    # )
    # np.savetxt(
    #     f"tepem/exports/rhs_q{VELO_SHAPE[0]}{VELO_SHAPE[1]}_q{PRESSURE_SHAPE[0]}{PRESSURE_SHAPE[1]}_slabs{num_slabs}.txt",
    #     rhs,
    #     delimiter=",",
    # )
    solution = np.linalg.solve(s_matrix, rhs)
    num_velo_dof = (VELO_SHAPE[0] * num_slabs + 1) * (VELO_SHAPE[1] + 1)
    if VELO_PARTIAL:
        num_velo_dof = (VELO_SHAPE[0] * num_slabs + 1) * (VELO_SHAPE[1] - 1)
    high_res_u, xv, yv = get_high_res_solution(
        solution_coeff=solution[:num_velo_dof],
        coef_ltg=velo_ltg,
        x_res=2,
        y_res=40,
        shape_funcs=Q26_SF_CHEB_LIST,
    )
    high_res_v, _, _ = get_high_res_solution(
        solution_coeff=solution[num_velo_dof : 2 * num_velo_dof],
        coef_ltg=velo_ltg,
        x_res=2,
        y_res=40,
        shape_funcs=Q26_SF_CHEB_LIST,
    )
    high_res_p, x_p, y_p = get_high_res_solution(
        solution_coeff=solution[2 * num_velo_dof :],
        coef_ltg=pres_ltg,
        x_res=40,
        y_res=1000,
        shape_funcs=Q13_SF_LIST,
    )
    x_hr_p, y_hr_p = get_high_res_phys_coords(x_p, y_p, dom_coords, dom_ltg)
    x_hr_u, y_hr_u = get_high_res_phys_coords(xv, yv, dom_coords, dom_ltg)
    plot_velo_pres_sol(
        high_res_p, x_hr_p, y_hr_p, high_res_u, high_res_v, x_hr_u, y_hr_u
    )
    info = {
        "velo_sf": VELO_SHAPE,
        "pres_sf": PRESSURE_SHAPE,
        "num_slabs": num_slabs,
        "angle": ANGLE,
    }
    fig_sol, ax_sol = solution_visualizer(
        solution=solution,
        num_vel_dof=num_velo_dof,
        velo_coords=phys_coords,
        pres_coords=phys_pre_coords,
        pres_shape=PRESSURE_SHAPE,
        info=info,
    )
    # fig_sol.savefig(
    #     f"tepem/exports/solution_q{VELO_SHAPE[0]}{VELO_SHAPE[1]}_q{PRESSURE_SHAPE[0]}{PRESSURE_SHAPE[1]}_{ANGLE}deg_n{num_slabs}_{datetime.datetime.now()}.png",
    #     dpi=300,
    # )
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
    # diff_solution = abs(solution - EX_SOL)
    # schur = np.dot(np.dot(d_matrix, np.linalg.inv(n_matrix)), g_matrix)
    # s, v, dh = np.linalg.svd(schur)
    # rank_schur = np.linalg.matrix_rank(schur)
    # -----------------------------------------
    # map_k = Mapping(slab_coord=dom_coords[dom_ltg[0]])
    # ref_x, ref_y = 0, 0
    # p_x, p_y = map_k.slab_map(ref_x, ref_y)
    fig, ax = dom.visualise_domain(coords=dom_coords, ltg=dom_ltg)  # type: ignore
    # ax.scatter(p_x, p_y, c="red", marker="+", label=f"test mapping in slab 4:\n $x_{{ref}}=${ref_x}, $y_{{ref}}=${ref_y}")  # type: ignore
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
    ax.set_ylabel("radius in y-direction")
    ax.set_xlabel("legth in x-direction")
    ax.legend()
    # fig.savefig(
    #     f"tepem/exports/domain_q{VELO_SHAPE[0]}{VELO_SHAPE[1]}_q{PRESSURE_SHAPE[0]}{PRESSURE_SHAPE[1]}.png",
    #     dpi=300,
    # )
    plt.show()  # type: ignore


if __name__ == "__main__":
    main()
