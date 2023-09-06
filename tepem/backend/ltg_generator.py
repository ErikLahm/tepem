from typing import Tuple

import numpy as np
import numpy.typing as npt


def generate_ltg(
    num_slabs: int, fe_order: Tuple[int, int], velocity_ltg: bool = False
) -> npt.NDArray[np.int64]:
    num_dof_rhs = fe_order[1] + 1
    if velocity_ltg:
        num_dof_rhs -= 2
    num_loc_dof = (fe_order[0] + 1) * (num_dof_rhs)
    ltg = np.arange(num_loc_dof * num_slabs).reshape((num_slabs, num_loc_dof))
    for i, _ in enumerate(ltg):
        ltg[i] = ltg[i] - (num_dof_rhs) * i
    return ltg
