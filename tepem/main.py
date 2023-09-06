import matplotlib.pyplot as plt
from backend.assembling import assemble_g, assemble_n
from backend.domain import Domain
from backend.ltg_generator import generate_ltg
from backend.mapping import Mapping
from backend.q_1_4_velo_sf import Q14_GRAD_SF_LIST
from backend.q_2_sf import Q2_SF_LIST


def main() -> None:
    num_slabs = 10
    dom = Domain(1, upper_bdn=lambda x: x**2 / 10 + 0.2, lower_bdn=lambda x: -0.2)
    dom_coords, dom_ltg = dom.slice_domain(num_slabs)
    velo_ltg = generate_ltg(num_slabs=num_slabs, fe_order=(1, 4), velocity_ltg=True)
    pres_ltg = generate_ltg(num_slabs=num_slabs, fe_order=(2, 2))
    n_matrix = assemble_n(
        ref_ltg=velo_ltg,
        grad_sfs=Q14_GRAD_SF_LIST,
        domain_coords=dom_coords,
        domain_ltg=dom_ltg,
    )
    g_matrix = assemble_g(
        velo_ltg=velo_ltg,
        press_ltg=pres_ltg,
        press_sfs=Q2_SF_LIST,
        grad_vel_sfs=Q14_GRAD_SF_LIST,
        domain_coords=dom_coords,
        domain_ltg=dom_ltg,
    )
    print(n_matrix)
    print(g_matrix)
    map_k = Mapping(slab_coord=dom_coords[dom_ltg[3]])
    ref_x, ref_y = 0.55, 1
    p_x, p_y = map_k.slab_map(ref_x, ref_y)
    _, ax = dom.visualise_domain(coords=dom_coords, ltg=dom_ltg)  # type: ignore
    ax.scatter(p_x, p_y, c="red")  # type: ignore
    plt.show()  # type: ignore


if __name__ == "__main__":
    main()
