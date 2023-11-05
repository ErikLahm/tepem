from dataclasses import dataclass
from typing import Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from backend.mapping import Mapping
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def get_repeating_pattern(num_slices: int) -> npt.NDArray[np.int32]:
    pattern = np.ones(2 * num_slices + 1, dtype=int)
    for i, _ in enumerate(pattern, 1):
        if i % 2 == 0:
            pattern[i] += 1
        pattern[-1] = 1
    return pattern


def get_cheby_points(start: float, end: float, num: int) -> npt.NDArray[np.float64]:
    ref_points = [-np.cos(np.pi * i / num) for i in range(num + 1)]
    interval_points = [map_11_to_ab(x, start, end) for x in ref_points]
    return np.array(interval_points)


def map_11_to_ab(x: float, a: float, b: float) -> float:
    return (b - a) / 2 * x + (a + b) / 2


@dataclass
class Domain:
    length: float
    upper_bdn: Callable[[float], Tuple[float, float]]
    lower_bdn: Callable[[float], Tuple[float, float]]

    def slice_domain(
        self, num_slices: int
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
        """
        Method slices the domain in slabs. Each Slab is identified with the points making up the
        slab.

        Parameters
        ----------
        num_slices: int
            number indicating in how many slabs the domain shall be sliced

        Returns
        -------
        Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
            first array: shape(4n+2, 2) is an array containing tuple of all coordinates (x,y) of
            the points
            second array: shape(n,6) is an array containing the indices of all points in the
            first array that belong to each slab. Each row in the array contains all indices
            that belong to the slab.
        """

        # x_coords = np.linspace(0, self.length, 2 * num_slices + 1)
        # upper_y = np.array([self.upper_bdn(x_coord) for x_coord in x_coords])
        # lower_y = np.array([self.lower_bdn(x_coord) for x_coord in x_coords])
        # upper_coords = np.vstack((x_coords, upper_y)).T
        # lower_coords = np.vstack((x_coords, lower_y)).T
        # all_coords = np.vstack((lower_coords, upper_coords))

        dx = np.linspace(0, self.length, 2 * num_slices + 1)
        upper_coords = np.array(
            [self.upper_bdn(slice_coord) for slice_coord in dx], dtype=np.float64
        )
        lower_coords = np.array(
            [self.lower_bdn(slice_coord) for slice_coord in dx], dtype=np.float64
        )
        all_coords = np.vstack((lower_coords, upper_coords))

        ltg_lower = np.arange(2 * num_slices + 1)
        ltg_upper = np.arange(2 * num_slices + 1, 4 * num_slices + 2)
        repeating_pattern = get_repeating_pattern(num_slices)
        ltg_upper = np.repeat(ltg_upper, repeating_pattern).reshape(num_slices, 3)
        ltg_lower = np.repeat(ltg_lower, repeating_pattern).reshape(num_slices, 3)
        ltg = np.hstack((ltg_lower, ltg_upper))
        return all_coords, ltg

    def get_phy_dof_coords(
        self,
        num_slabs: int,
        sf_shape: Tuple[int, int],
        velo_sf: bool = False,
        cheby_points: bool = False,
    ) -> npt.NDArray[np.float64]:
        ref_x_coords = np.linspace(0, 1, sf_shape[0] + 1)
        ref_y_coords = np.linspace(0, 1, sf_shape[1] + 1)
        if cheby_points:
            ref_x_coords = get_cheby_points(0, 1, sf_shape[0])
            ref_y_coords = get_cheby_points(0, 1, sf_shape[1])
        if velo_sf:
            ref_y_coords = np.delete(ref_y_coords, [-1, 0])
        phy_dof_coords = np.zeros(
            shape=((sf_shape[0] * (num_slabs) + 1) * len(ref_y_coords), 2)
        )
        phy_dof_idx = 0
        all_map_coords, map_ltg = self.slice_domain(num_slices=num_slabs)
        for j, row in enumerate(map_ltg):
            aff_map = Mapping(all_map_coords[row])
            for i, ref_x in enumerate(ref_x_coords):
                if j != 0 and i == 0:
                    continue
                for ref_y in ref_y_coords:
                    current_phy_coords = aff_map.slab_map(ref_x=ref_x, ref_y=ref_y)
                    phy_dof_coords[phy_dof_idx][0] = current_phy_coords[0]
                    phy_dof_coords[phy_dof_idx][1] = current_phy_coords[1]
                    phy_dof_idx += 1
        return phy_dof_coords

    def visualise_domain(
        self, coords: npt.NDArray[np.float64], ltg: npt.NDArray[np.int32]
    ) -> Tuple[Figure, Axes]:
        fig, ax = plt.subplots(figsize=(15, 10))  # type: ignore
        dx = np.linspace(0, self.length, 500)
        upper_coords = np.array([self.upper_bdn(point) for point in dx])
        lower_coords = np.array([self.lower_bdn(point) for point in dx])
        ax.plot(upper_coords[:, 0], upper_coords[:, 1], "grey")  # type: ignore
        ax.plot(lower_coords[:, 0], lower_coords[:, 1], "grey")  # type: ignore
        for slab in ltg:
            ax.plot(  # type: ignore
                [coords[slab[0]][0], coords[slab[3]][0]],
                [coords[slab[0]][1], coords[slab[3]][1]],
                color="grey",
                linestyle="dashdot",
            )
            ax.plot(  # type: ignore
                [coords[slab[2]][0], coords[slab[5]][0]],
                [coords[slab[2]][1], coords[slab[5]][1]],
                color="grey",
                linestyle="dashdot",
            )
        ax.scatter(coords[:, 0], coords[:, 1], c="grey", marker=".")  # type: ignore
        # plt.show()  # type: ignore
        return fig, ax
