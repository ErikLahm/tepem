from typing import Callable, Tuple

import numpy as np
import numpy.typing as npt
from backend.mapping import Mapping


def get_high_res_solution(
    x_res: int,
    y_res: int,
    solution_coeff: npt.NDArray[np.float64],
    shape_funcs: list[Callable[[float, float], float]],
    coef_ltg: npt.NDArray[np.int64],
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Creates a higher resolution of the solution by interpolating the solution based on
    the corresponding shape function. THe solution is given by the solution coefficients
    and the reference shape functions (defined on unit square)

    Parameters
    ----------
    x_res: int,
        resolution per slab in x-direction
    y_res: int,
        resolution per slab in y-direction
    solution_coeff: npt.NDArray[np.float64],
        coefficients of the calculated solution
    shape_funcs: list[Callable[[float, float], float]],
        shape functions defined in the reference element (unit square) which correspond to
        the solution coffencients
    coef_ltg: npt.NDArray[np.int64],
        local to global map indicating which coefficient corresponds to which local dof in
        which element

    Returns
    -------
    Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]
        1: solution of higher resolution (solution at more physical points)
        2: x values of all physical coordinates in the domain
        3: y values of all physical coordinates in the domain
    """

    xv, yv = get_mesh_unit_square(x_res=x_res, y_res=y_res)
    high_res_sol = []
    for i, elem_coeff_idx in enumerate(coef_ltg):
        calc_high_res_sol_one_element(
            elem_coeffs=solution_coeff[elem_coeff_idx],
            shape_funcs=shape_funcs,
            high_res_x=xv.flatten(),
            high_res_y=yv.flatten(),
            high_res_sol=high_res_sol,
            slab_number=i,
        )
    return np.array(high_res_sol), xv, yv


def get_high_res_phys_coords(
    x_ref: npt.NDArray[np.float64],
    y_ref: npt.NDArray[np.float64],
    map_coords: npt.NDArray[np.float64],
    ltg: npt.NDArray[np.int64],
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Creates the coordinates of the physical space based on a meshgrid on the unit
    square

    Parameters
    ----------
    x_ref: npt.NDArray[np.float64],
        x coordinates of the meshgrid
    y_ref: npt.NDArray[np.float64],
        y coordinates of the meshgrid
    map_coords: npt.NDArray[np.float64],
        all domain coordinates that set up the mapping
    ltg: npt.NDArray[np.float64],
        local to global map of the domain coordinates

    Returns
    -------
    Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        Tuple of x and y coordinates of all coordinates across the whole domain
    """

    x_phys, y_phys = [], []
    for slab_num, slab in enumerate(ltg):
        mapping = Mapping(map_coords[slab])
        for ref_coord in list(zip(x_ref.flatten(), y_ref.flatten())):
            if slab_num != 0 and ref_coord[0] == 0:
                continue
            x, y = mapping.slab_map(ref_coord[0], ref_coord[1])
            x_phys.append(x)  # type: ignore
            y_phys.append(y)  # type: ignore
    return np.array(x_phys), np.array(y_phys)


def get_mesh_unit_square(
    x_res: int, y_res: int
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Creates a meshgrid on the unit square [0,1]x[0,1] based on the resolution given in
    x and y direction.

    Parameter
    ---------
    x_res: int
        resolution of mesh in x-direction
    y_res: int
        resolution of mesh in y-direction

    Returns
    -------
    Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        Tuple of meshgrid coordinates in the form of np.meshgrid() output
    """

    x_vals = np.linspace(0, 1, x_res + 1)
    y_vals = np.linspace(0, 1, y_res + 1)
    xv, yv = np.meshgrid(x_vals, y_vals)
    return xv, yv


def calc_high_res_sol_one_element(
    elem_coeffs: npt.NDArray[np.float64],
    shape_funcs: list[Callable[[float, float], float]],
    high_res_x: npt.NDArray[np.float64],
    high_res_y: npt.NDArray[np.float64],
    high_res_sol: list[float],
    slab_number: int,
) -> list[float]:
    assert len(elem_coeffs) == len(
        shape_funcs
    ), f"Number of coefficients and shape functions does not coincode"
    for coords in list(zip(high_res_x, high_res_y)):
        if slab_number != 0 and coords[0] == 0:
            continue
        sol = sum(
            [
                float(elem_coeffs[i]) * shape_funcs[i](coords[0], coords[1])
                for i, _ in enumerate(elem_coeffs)
            ]
        )
        high_res_sol.append(sol)
    return high_res_sol
