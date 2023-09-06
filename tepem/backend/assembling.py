from typing import Callable, List

import numpy as np
import numpy.typing as npt
from backend.integrals import integral_a, integral_b
from backend.mapping import Mapping


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
    d_matrix = np.zeros(shape=(num_vel_dof, num_pres_dof))
    d_matrix_halfs: List[npt.NDArray[np.float64]] = []
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
                    d_matrix[row][col] += integral
        d_matrix_halfs.append(d_matrix)
    d_matrix = np.vstack((d_matrix_halfs[0], d_matrix_halfs[1]))
    return d_matrix
