import os
from typing import Callable, List, Tuple

import numpy as np
import numpy.typing as npt
from backend.assembling import add_inital_penalty, assemble_g, assemble_n, assemble_rhs
from backend.curve_shapes.nozzle_dom import NozzleDomain
from backend.curve_shapes.sudden_change_dom import SuddenChangeDomain
from backend.curve_shapes.wave_domain import WaveDomain
from backend.domain import Domain
from backend.high_res_mapping import get_high_res_phys_coords, get_high_res_solution
from backend.ltg_generator import generate_ltg
from backend.shape_functions.q_1_3_sf_cheb import Q13_SF_CHEB_LIST
from backend.shape_functions.q_2_6_velo_cheb import (
    Q26_GRAD_SF_CHEB_LIST,
    Q26_SF_CHEB_LIST,
)

# NU = 8.1e-2  # [Pa * s]
# RADIUS = 0.0125  # [m]
# LENGTH = 0.12  # [m]
# LENGTH_STR_INLET = 0.01  # 0.01  # [m]
# START_STR_OUTLET = 0.11  # 0.11 # [m]
# ANGLE = -5  # [°]
# CURVE_ANGLE = 0
# VOLUME_FLUX = 5e-3  # 5e-4  # [m^3/(m^2*s) = m/s]

# velo_shape = (2, 6)
# pressure_shape = (1, 3)
VELO_PARTIAL = True


def angle_to_gradient(angle: float) -> float:
    rad = angle_to_radian(angle=angle)
    return np.tan(rad)


def angle_to_radian(angle: float) -> float:
    return angle / 180 * np.pi


def get_constant(mass_flow: float, radius: float) -> float:
    return mass_flow / (
        (radius**2 * (2 * radius) - 1 / 3 * (radius**3 - (-radius) ** 3))
    )


def assemble_system(
    num_slabs: int,
    dom: Domain,
    v_sf: list[Callable[[float, float], npt.NDArray[np.float64]]],
    p_sf: list[Callable[[float, float], float]],
    velo_shape: Tuple[int, int],
    pressure_shape: Tuple[int, int],
    bdn_const: float,
    radius: float,
    nu: float,
):
    dom_coords, dom_ltg = dom.slice_domain(num_slabs)
    phys_coords = dom.get_phy_dof_coords(
        num_slabs, sf_shape=velo_shape, velo_sf=VELO_PARTIAL, cheby_points=True
    )
    velo_ltg = generate_ltg(
        num_slabs=num_slabs, fe_order=velo_shape, velocity_ltg=VELO_PARTIAL
    )
    pres_ltg = generate_ltg(num_slabs=num_slabs, fe_order=pressure_shape)
    n_matrix = assemble_n(
        ref_ltg=velo_ltg,
        grad_sfs=v_sf,
        domain_coords=dom_coords,
        domain_ltg=dom_ltg,
        nu=nu,
    )
    n_matrix = add_inital_penalty(
        num_slabs=num_slabs,
        n_matrix=n_matrix,
        velo_sf_shape=velo_shape,
        velo_pt=VELO_PARTIAL,
    )
    g_matrix = assemble_g(
        velo_ltg=velo_ltg,
        press_ltg=pres_ltg,
        press_sfs=p_sf,
        grad_vel_sfs=v_sf,
        domain_coords=dom_coords,
        domain_ltg=dom_ltg,
    )
    rhs = assemble_rhs(
        num_slabs=num_slabs,
        velo_sf_shape=velo_shape,
        press_sf_shape=pressure_shape,
        phys_velo_dof_coords=phys_coords,
        const=bdn_const,
        radius=radius,
        velo_pt=VELO_PARTIAL,
    )
    d_matrix = np.transpose(g_matrix)
    zero_block = np.zeros(shape=(d_matrix.shape[0], g_matrix.shape[1]))
    upper = np.hstack((n_matrix, g_matrix))
    lower = np.hstack((d_matrix, zero_block))
    s_matrix = np.vstack((upper, lower))
    return s_matrix, rhs


def interpolate_high_res_sol(
    low_res_sol: npt.NDArray[np.float64],
    ltg: npt.NDArray[np.int64],
    dom_coords: npt.NDArray[np.float64],
    dom_ltg: npt.NDArray[np.int64],
    x_res: int,
    y_res: int,
    shape_funcs: list[Callable[[float, float], float]],
):
    high_res_u, x_u, y_u = get_high_res_solution(
        solution_coeff=low_res_sol,
        coef_ltg=ltg,
        x_res=x_res,
        y_res=y_res,
        shape_funcs=shape_funcs,
    )
    x_hr_u, y_hr_u = get_high_res_phys_coords(x_u, y_u, dom_coords, dom_ltg)
    return high_res_u, x_hr_u, y_hr_u


def simulate_nozzles(
    num_slabs: int,
    angles: list[float],
    fluxes: dict[str, float],
    v_sf: list[Callable[[float, float], float]],
    v_sf_grad: list[Callable[[float, float], npt.NDArray[np.float64]]],
    p_sf: list[Callable[[float, float], float]],
    v_shape: Tuple[int, int],
    p_shape: Tuple[int, int],
    length: float,
    init_radius: float,
    length_str_inlet: float,
    start_str_outlet: float,
    nu: float,
):
    for speed, flux in fluxes.items():
        folder_name = f"nozzles_n{num_slabs}_q{speed}"
        bdn_constant = get_constant(mass_flow=flux, radius=init_radius)
        for angle in angles:
            nozzle_bdn = NozzleDomain(
                length_str_inlet=length_str_inlet,
                start_str_outlet=start_str_outlet,
                init_radius=init_radius,
                angle=angle,
            )
            dom = Domain(
                length=length,
                upper_bdn=nozzle_bdn.straight_conv_upper,
                lower_bdn=nozzle_bdn.straight_conv_lower,
            )
            s_matrix, rhs = assemble_system(
                num_slabs=num_slabs,
                dom=dom,
                v_sf=v_sf_grad,
                p_sf=p_sf,
                velo_shape=v_shape,
                pressure_shape=p_shape,
                bdn_const=bdn_constant,
                radius=init_radius,
                nu=nu,
            )
            solution = np.linalg.solve(s_matrix, rhs)
            num_velo_dof = (v_shape[0] * num_slabs + 1) * (v_shape[1] + 1)
            if VELO_PARTIAL:
                num_velo_dof = (v_shape[0] * num_slabs + 1) * (v_shape[1] - 1)

            # get original solution
            u = solution[:num_velo_dof]
            v = solution[num_velo_dof : 2 * num_velo_dof]
            p = solution[2 * num_velo_dof :]
            phys_coords = dom.get_phy_dof_coords(
                num_slabs, sf_shape=v_shape, velo_sf=VELO_PARTIAL, cheby_points=True
            )
            phys_pre_coords = dom.get_phy_dof_coords(
                num_slabs, sf_shape=p_shape, cheby_points=True
            )
            velocity_sol = np.hstack((u, v))
            velocity_lr = np.hstack((phys_coords, velocity_sol))
            pressure_lr = np.hstack((phys_pre_coords, p))

            # get high resolution solution
            dom_coords, dom_ltg = dom.slice_domain(num_slabs)
            velo_ltg = generate_ltg(
                num_slabs=num_slabs, fe_order=v_shape, velocity_ltg=VELO_PARTIAL
            )
            pres_ltg = generate_ltg(num_slabs=num_slabs, fe_order=p_shape)
            u_hr, x_hr_u, y_hr_u = interpolate_high_res_sol(
                low_res_sol=u,
                ltg=velo_ltg,
                dom_coords=dom_coords,
                dom_ltg=dom_ltg,
                x_res=2,
                y_res=40,
                shape_funcs=v_sf,
            )
            v_hr, _, _ = interpolate_high_res_sol(
                low_res_sol=v,
                ltg=velo_ltg,
                dom_coords=dom_coords,
                dom_ltg=dom_ltg,
                x_res=2,
                y_res=40,
                shape_funcs=v_sf,
            )
            p_hr, x_hr_p, y_hr_p = interpolate_high_res_sol(
                low_res_sol=p,
                ltg=pres_ltg,
                dom_coords=dom_coords,
                dom_ltg=dom_ltg,
                x_res=40,
                y_res=120,
                shape_funcs=Q13_SF_CHEB_LIST,
            )
            velocity_sol = np.vstack((u_hr, v_hr))
            velocity_coords_hr = np.vstack((x_hr_u, y_hr_u))
            velocity_hr = np.vstack((velocity_coords_hr, velocity_sol)).T
            pressure_coords_hr = np.vstack((x_hr_p, y_hr_p))
            pressure_hr = np.vstack((pressure_coords_hr, p_hr)).T
            save_arrays_as_csv(
                arrays=[velocity_lr, pressure_lr, velocity_hr, pressure_hr],
                directory_path="/Users/erikweilandt/Documents/university/master_thesis/tepem/results",
                folder_name=folder_name,
                sub_folder_name=f"H{abs(angle)}",
            )
            save_output_file(
                directory_path=os.path.join(
                    "/Users/erikweilandt/Documents/university/master_thesis/tepem/results",
                    folder_name,
                ),
                rho=980,
                mu=nu,
                q=flux,
                radius=init_radius,
                length=length,
                num_slabs=num_slabs,
            )


def simulate_wave(
    num_slabs: int,
    fluxes: dict[str, float],
    v_sf: list[Callable[[float, float], float]],
    v_sf_grad: list[Callable[[float, float], npt.NDArray[np.float64]]],
    p_sf: list[Callable[[float, float], float]],
    v_shape: Tuple[int, int],
    p_shape: Tuple[int, int],
    length: float,
    init_radius: float,
    gradient: float,
    period: float,
    length_str_inlet: float,
    start_str_outlet: float,
    nu: float,
):
    folder_name = f"wave_n{num_slabs}"
    for speed, flux in fluxes.items():
        bdn_constant = get_constant(mass_flow=flux, radius=init_radius)
        wave_bdn = WaveDomain(
            length_str_inlet=length_str_inlet,
            start_str_outlet=start_str_outlet,
            init_radius=init_radius,
            gradient=gradient,
            period=period,
        )
        dom = Domain(
            length=length,
            upper_bdn=wave_bdn.cos_upper,
            lower_bdn=wave_bdn.cos_lower,
        )
        s_matrix, rhs = assemble_system(
            num_slabs=num_slabs,
            dom=dom,
            v_sf=v_sf_grad,
            p_sf=p_sf,
            velo_shape=v_shape,
            pressure_shape=p_shape,
            bdn_const=bdn_constant,
            radius=init_radius,
            nu=nu,
        )
        solution = np.linalg.solve(s_matrix, rhs)
        num_velo_dof = (v_shape[0] * num_slabs + 1) * (v_shape[1] + 1)
        if VELO_PARTIAL:
            num_velo_dof = (v_shape[0] * num_slabs + 1) * (v_shape[1] - 1)

        # get original solution
        u = solution[:num_velo_dof]
        v = solution[num_velo_dof : 2 * num_velo_dof]
        p = solution[2 * num_velo_dof :]
        phys_coords = dom.get_phy_dof_coords(
            num_slabs, sf_shape=v_shape, velo_sf=VELO_PARTIAL, cheby_points=True
        )
        phys_pre_coords = dom.get_phy_dof_coords(
            num_slabs, sf_shape=p_shape, cheby_points=True
        )
        velocity_sol = np.hstack((u, v))
        velocity_lr = np.hstack((phys_coords, velocity_sol))
        pressure_lr = np.hstack((phys_pre_coords, p))

        # get high resolution solution
        dom_coords, dom_ltg = dom.slice_domain(num_slabs)
        velo_ltg = generate_ltg(
            num_slabs=num_slabs, fe_order=v_shape, velocity_ltg=VELO_PARTIAL
        )
        pres_ltg = generate_ltg(num_slabs=num_slabs, fe_order=p_shape)
        u_hr, x_hr_u, y_hr_u = interpolate_high_res_sol(
            low_res_sol=u,
            ltg=velo_ltg,
            dom_coords=dom_coords,
            dom_ltg=dom_ltg,
            x_res=2,
            y_res=40,
            shape_funcs=v_sf,
        )
        v_hr, _, _ = interpolate_high_res_sol(
            low_res_sol=v,
            ltg=velo_ltg,
            dom_coords=dom_coords,
            dom_ltg=dom_ltg,
            x_res=2,
            y_res=40,
            shape_funcs=v_sf,
        )
        p_hr, x_hr_p, y_hr_p = interpolate_high_res_sol(
            low_res_sol=p,
            ltg=pres_ltg,
            dom_coords=dom_coords,
            dom_ltg=dom_ltg,
            x_res=40,
            y_res=120,
            shape_funcs=Q13_SF_CHEB_LIST,
        )
        velocity_sol = np.vstack((u_hr, v_hr))
        velocity_coords_hr = np.vstack((x_hr_u, y_hr_u))
        velocity_hr = np.vstack((velocity_coords_hr, velocity_sol)).T
        pressure_coords_hr = np.vstack((x_hr_p, y_hr_p))
        pressure_hr = np.vstack((pressure_coords_hr, p_hr)).T
        save_arrays_as_csv(
            arrays=[velocity_lr, pressure_lr, velocity_hr, pressure_hr],
            directory_path="/Users/erikweilandt/Documents/university/master_thesis/tepem/results",
            folder_name=folder_name,
            sub_folder_name=f"W{speed}",
        )
        save_output_file(
            directory_path=os.path.join(
                "/Users/erikweilandt/Documents/university/master_thesis/tepem/results",
                folder_name,
            ),
            rho=980,
            mu=nu,
            q=flux,
            radius=init_radius,
            length=length,
            num_slabs=num_slabs,
        )


def simulate_bulge(
    num_slabs: int,
    fluxes: dict[str, float],
    v_sf: list[Callable[[float, float], float]],
    v_sf_grad: list[Callable[[float, float], npt.NDArray[np.float64]]],
    p_sf: list[Callable[[float, float], float]],
    v_shape: Tuple[int, int],
    p_shape: Tuple[int, int],
    length: float,
    init_radius: float,
    max_heights: list[float],
    nu: float,
):
    for speed, flux in fluxes.items():
        folder_name = f"bulge_n{num_slabs}_q{speed}"
        bdn_constant = get_constant(mass_flow=flux, radius=init_radius)
        for height in max_heights:
            bulge_bdn = SuddenChangeDomain(
                dom_length=length,
                max_height=height,
                init_radius=init_radius,
            )
            dom = Domain(
                length=length,
                upper_bdn=bulge_bdn.sudden_bdn_upper,
                lower_bdn=bulge_bdn.sudden_bdn_lower,
            )
            s_matrix, rhs = assemble_system(
                num_slabs=num_slabs,
                dom=dom,
                v_sf=v_sf_grad,
                p_sf=p_sf,
                velo_shape=v_shape,
                pressure_shape=p_shape,
                bdn_const=bdn_constant,
                radius=init_radius,
                nu=nu,
            )
            solution = np.linalg.solve(s_matrix, rhs)
            num_velo_dof = (v_shape[0] * num_slabs + 1) * (v_shape[1] + 1)
            if VELO_PARTIAL:
                num_velo_dof = (v_shape[0] * num_slabs + 1) * (v_shape[1] - 1)

            # get original solution
            u = solution[:num_velo_dof]
            v = solution[num_velo_dof : 2 * num_velo_dof]
            p = solution[2 * num_velo_dof :]
            phys_coords = dom.get_phy_dof_coords(
                num_slabs, sf_shape=v_shape, velo_sf=VELO_PARTIAL, cheby_points=True
            )
            phys_pre_coords = dom.get_phy_dof_coords(
                num_slabs, sf_shape=p_shape, cheby_points=True
            )
            velocity_sol = np.hstack((u, v))
            velocity_lr = np.hstack((phys_coords, velocity_sol))
            pressure_lr = np.hstack((phys_pre_coords, p))

            # get high resolution solution
            dom_coords, dom_ltg = dom.slice_domain(num_slabs)
            velo_ltg = generate_ltg(
                num_slabs=num_slabs, fe_order=v_shape, velocity_ltg=VELO_PARTIAL
            )
            pres_ltg = generate_ltg(num_slabs=num_slabs, fe_order=p_shape)
            u_hr, x_hr_u, y_hr_u = interpolate_high_res_sol(
                low_res_sol=u,
                ltg=velo_ltg,
                dom_coords=dom_coords,
                dom_ltg=dom_ltg,
                x_res=2,
                y_res=40,
                shape_funcs=v_sf,
            )
            v_hr, _, _ = interpolate_high_res_sol(
                low_res_sol=v,
                ltg=velo_ltg,
                dom_coords=dom_coords,
                dom_ltg=dom_ltg,
                x_res=2,
                y_res=40,
                shape_funcs=v_sf,
            )
            p_hr, x_hr_p, y_hr_p = interpolate_high_res_sol(
                low_res_sol=p,
                ltg=pres_ltg,
                dom_coords=dom_coords,
                dom_ltg=dom_ltg,
                x_res=40,
                y_res=120,
                shape_funcs=Q13_SF_CHEB_LIST,
            )
            velocity_sol = np.vstack((u_hr, v_hr))
            velocity_coords_hr = np.vstack((x_hr_u, y_hr_u))
            velocity_hr = np.vstack((velocity_coords_hr, velocity_sol)).T
            pressure_coords_hr = np.vstack((x_hr_p, y_hr_p))
            pressure_hr = np.vstack((pressure_coords_hr, p_hr)).T
            save_arrays_as_csv(
                arrays=[velocity_lr, pressure_lr, velocity_hr, pressure_hr],
                directory_path="/Users/erikweilandt/Documents/university/master_thesis/tepem/results",
                folder_name=folder_name,
                sub_folder_name=f"B{str(height).replace('.','')}",
            )
            save_output_file(
                directory_path=os.path.join(
                    "/Users/erikweilandt/Documents/university/master_thesis/tepem/results",
                    folder_name,
                ),
                rho=980,
                mu=nu,
                q=flux,
                radius=init_radius,
                length=length,
                num_slabs=num_slabs,
            )


def save_output_file(
    directory_path: str,
    rho: float,
    mu: float,
    q: float,
    radius: float,
    length: float,
    num_slabs: int,
):
    f = open(os.path.join(directory_path, "output.txt"), "w")
    output = (
        f"density [kg/m^3]: {rho}\n"
        f"viscosity [Pa s]: {mu}\n"
        f"flow flux [m/s]: {q}\n"
        f"radius [m]: {radius}\n"
        f"length [m]: {length}\n"
        f"number of slabs [1]: {num_slabs}\n"
        f"Reynolds number [1]: {(rho * q)/mu}"
    )
    f.write(output)


def save_arrays_as_csv(
    arrays: List[npt.NDArray[np.float64]],
    directory_path: str,
    folder_name: str,
    sub_folder_name: str,
) -> None:
    solution = {
        "u_lr": arrays[0],
        "p_lr": arrays[1],
        "u_hr": arrays[2],
        "p_hr": arrays[3],
    }

    folder_path = os.path.join(directory_path, folder_name)
    sub_folder_path = os.path.join(folder_path, sub_folder_name)
    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(sub_folder_path)

    for name, array in solution.items():
        file_name = f"{name}.csv"
        file_path = os.path.join(sub_folder_path, file_name)
        np.savetxt(file_path, array, delimiter=",")


fluxes = {
    "slow": 5e-4,
    "normal": 5e-3,
    "fast": 5e-2,
    "ffast": 5e-1,
    "fffast": 5,
    "ffffast": 50,
}


def main():
    simulate_nozzles(
        num_slabs=10,
        angles=[0, -3, -5],
        fluxes=fluxes,
        v_sf=Q26_SF_CHEB_LIST,
        v_sf_grad=Q26_GRAD_SF_CHEB_LIST,
        p_sf=Q13_SF_CHEB_LIST,
        v_shape=(2, 6),
        p_shape=(1, 3),
        init_radius=0.0125,
        length_str_inlet=0.01,
        start_str_outlet=0.11,
        length=0.12,
        nu=8.1e-2,
    )
    # simulate_wave(
    #     num_slabs=10,
    #     fluxes=fluxes,
    #     v_sf=Q26_SF_CHEB_LIST,
    #     v_sf_grad=Q26_GRAD_SF_CHEB_LIST,
    #     p_sf=Q13_SF_CHEB_LIST,
    #     v_shape=(2, 6),
    #     p_shape=(1, 3),
    #     init_radius=0.0125,
    #     length=0.22,
    #     gradient=-(0.00625 / 0.21),
    #     period=0.21,
    #     length_str_inlet=0.01,
    #     start_str_outlet=0.22,
    #     nu=8.1e-2,
    # )
    # simulate_bulge(
    #     num_slabs=10,
    #     fluxes={"normal": 5e-3},
    #     v_sf=Q26_SF_CHEB_LIST,
    #     v_sf_grad=Q26_GRAD_SF_CHEB_LIST,
    #     p_sf=Q13_SF_CHEB_LIST,
    #     v_shape=(2, 6),
    #     p_shape=(1, 3),
    #     length=0.22,
    #     init_radius=0.0125,
    #     max_heights=[0.02, 0.03, 0.04, 0.05],
    #     nu=8.1e-2,
    # )
    print("success")


if __name__ == "__main__":
    main()
