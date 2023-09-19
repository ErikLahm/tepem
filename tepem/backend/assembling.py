from typing import Callable, List, Tuple

import numpy as np
import numpy.typing as npt
from backend.integrals import integral_a, integral_b
from backend.mapping import Mapping

EPSILON = 1e-6


def assemble_n(
    ref_ltg: npt.NDArray[np.int64],
    grad_sfs: List[Callable[[float, float], npt.NDArray[np.float64]]],
    domain_coords: npt.NDArray[np.float64],
    domain_ltg: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:
    assert (
        len(grad_sfs) == ref_ltg.shape[1]
    ), "Number of shape functions given does not correspond to num_loc_dof in ltg map."
    num_vel_dof = ref_ltg.max() + 1
    num_slabs = ref_ltg.shape[0]
    num_loc_dof = ref_ltg.shape[1]
    n_matrix = np.zeros(shape=(num_vel_dof, num_vel_dof))
    for j in range(num_slabs):
        aff_map = Mapping(domain_coords[domain_ltg[j]])
        for k in range(num_loc_dof):
            row = ref_ltg[j, k]
            for l in range(num_loc_dof):
                col = ref_ltg[j, l]
                integral = integral_a(
                    jacobian=aff_map.jacobian,
                    grad_sf_k=grad_sfs[k],
                    grad_sf_j=grad_sfs[l],
                )
                n_matrix[row][col] += integral
    zero_buffer = np.zeros(shape=(num_vel_dof, num_vel_dof))
    upper_half = np.vstack((n_matrix, zero_buffer))
    lower_half = np.vstack((zero_buffer, n_matrix))
    n_matrix = np.hstack((upper_half, lower_half))
    # assert np.linalg.matrix_rank(n_matrix) == min(n_matrix.shape), (
    #     f"Laplacian matrix is not full rank: rank(N)={np.linalg.matrix_rank(n_matrix)} "
    #     f"but should be rank(N)=min(m,m)={min(n_matrix.shape)}."
    # )
    return n_matrix


def add_inital_penalty(
    num_slabs: int,
    n_matrix: npt.NDArray[np.float64],
    velo_sf_shape: Tuple[int, int],
    velo_pt: bool = True,
) -> npt.NDArray[np.float64]:
    num_velo_dof = (velo_sf_shape[0] * num_slabs + 1) * (velo_sf_shape[1] - 1)
    num_velo_dof_bnd = velo_sf_shape[1] - 1
    if not velo_pt:
        num_velo_dof = (velo_sf_shape[0] * num_slabs + 1) * (velo_sf_shape[1] + 1)
        num_velo_dof_bnd = velo_sf_shape[1] + 1
    for diag in range(num_velo_dof_bnd):
        n_matrix[diag][diag] += 1 / EPSILON
        n_matrix[diag + num_velo_dof][diag + num_velo_dof] += 1 / EPSILON
    return n_matrix


def assemble_g(
    velo_ltg: npt.NDArray[np.int64],
    press_ltg: npt.NDArray[np.int64],
    press_sfs: List[Callable[[float, float], float]],
    grad_vel_sfs: List[Callable[[float, float], npt.NDArray[np.float64]]],
    domain_coords: npt.NDArray[np.float64],
    domain_ltg: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:
    assert (
        len(press_sfs) == press_ltg.shape[1]
    ), "Number of pressure ref.-sf. does not coincide with number of local dof."
    assert (
        len(grad_vel_sfs) == velo_ltg.shape[1]
    ), "Number of velo. grad. ref.-sf. does not coincide with number of local dof."
    assert (
        velo_ltg.shape[0] == press_ltg.shape[0]
    ), "Different numbers of slabs for velocity and pressure ltg."
    num_vel_dof = velo_ltg.max() + 1
    num_pres_dof = press_ltg.max() + 1
    num_slabs = velo_ltg.shape[0]
    num_loc_velo_dof = velo_ltg.shape[1]
    num_loc_pres_dof = press_ltg.shape[1]
    g_matrix = np.zeros(shape=(num_vel_dof, num_pres_dof))
    g_matrix_halfs: List[npt.NDArray[np.float64]] = []
    for component in range(2):
        for j in range(num_slabs):
            aff_map = Mapping(domain_coords[domain_ltg[j]])
            for k in range(num_loc_velo_dof):
                row = velo_ltg[j][k]
                for l in range(num_loc_pres_dof):
                    col = press_ltg[j][l]
                    integral = integral_b(
                        jacobian=aff_map.jacobian,
                        sf_k=press_sfs[l],
                        grad_sf_j=grad_vel_sfs[k],
                        component=component,
                    )
                    g_matrix[row][col] += integral
        g_matrix_halfs.append(g_matrix)
        g_matrix = np.zeros(shape=(num_vel_dof, num_pres_dof))
    g_matrix = np.vstack((g_matrix_halfs[0], g_matrix_halfs[1]))
    # assert np.linalg.matrix_rank(g_matrix) == min(g_matrix.shape), (
    #     f"Gradient matrix is not full rank: rank(G)={np.linalg.matrix_rank(g_matrix)} "
    #     f"but should be rank(G)=min(m,n)={min(g_matrix.shape)}."
    # )
    return g_matrix


def assemble_rhs(
    num_slabs: int,
    velo_sf_shape: Tuple[int, int],
    press_sf_shape: Tuple[int, int],
    phys_velo_dof_coords: npt.NDArray[np.float64],
    boundary_func: Callable[[float, float], float],
    velo_pt: bool = True,
) -> npt.NDArray[np.float64]:
    num_pres_dof = (press_sf_shape[0] * num_slabs + 1) * (press_sf_shape[1] + 1)
    rhs = np.zeros(shape=(2 * phys_velo_dof_coords.shape[0] + num_pres_dof, 1))
    num_velo_bnd = velo_sf_shape[1] - 1
    if not velo_pt:
        num_velo_bnd = velo_sf_shape[1] + 1
    for i in range(num_velo_bnd):
        rhs[i] = (
            boundary_func(phys_velo_dof_coords[i][0], phys_velo_dof_coords[i][1])
            * 1
            / EPSILON
        )
    return rhs
