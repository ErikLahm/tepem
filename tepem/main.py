import matplotlib.pyplot as plt
from backend.domain import Domain
from backend.mapping import Mapping


def main() -> None:
    dom = Domain(1, upper_bdn=lambda x: x**2 / 10 + 0.2, lower_bdn=lambda x: -0.2)
    coords, ltg = dom.slice_domain(10)
    map_k = Mapping(slab_coord=coords[ltg[3]])
    print(coords[ltg[3]])
    ref_x, ref_y = 0.55, 1
    p_x, p_y = map_k.slab_map(ref_x, ref_y)
    _, ax = dom.visualise_domain(coords=coords, ltg=ltg)  # type: ignore
    ax.scatter(p_x, p_y)  # type: ignore
    plt.show()  # type: ignore


if __name__ == "__main__":
    main()
