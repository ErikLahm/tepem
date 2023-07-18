from dataclasses import dataclass
from typing import Tuple

import numpy as np
import numpy.typing as npt


@dataclass
class Mapping:
    slab_coord: npt.NDArray[np.float64]

    def slab_map(self, ref_x: float, ref_y: float) -> Tuple[float, float]:
        phys_x = (
            self.slab_coord[0][0] * (1 - ref_y) * (ref_x - 1) * (2 * ref_x - 1)
            - 4 * self.slab_coord[1][0] * ref_x * (1 - ref_y) * (ref_x - 1)
            + self.slab_coord[2][0] * ref_x * (1 - ref_y) * (2 * ref_x - 1)
            + self.slab_coord[3][0] * ref_y * (ref_x - 1) * (2 * ref_x - 1)
            - 4 * self.slab_coord[4][0] * ref_x * ref_y * (ref_x - 1)
            + self.slab_coord[5][0] * ref_x * ref_y * (2 * ref_x - 1)
        )
        phys_y = (
            self.slab_coord[0][1] * (1 - ref_y) * (ref_x - 1) * (2 * ref_x - 1)
            - 4 * self.slab_coord[1][1] * ref_x * (1 - ref_y) * (ref_x - 1)
            + self.slab_coord[2][1] * ref_x * (1 - ref_y) * (2 * ref_x - 1)
            + self.slab_coord[3][1] * ref_y * (ref_x - 1) * (2 * ref_x - 1)
            - 4 * self.slab_coord[4][1] * ref_x * ref_y * (ref_x - 1)
            + self.slab_coord[5][1] * ref_x * ref_y * (2 * ref_x - 1)
        )
        return phys_x, phys_y

    def jacobian(self, ref_x: float, ref_y: float) -> npt.NDArray[np.float64]:
        df1_dx, df2_dx = self.eval_df_dx(ref_x=ref_x, ref_y=ref_y)
        df1_dy, df2_dy = self.eval_df_dy(ref_x=ref_x)
        return np.array([[df1_dx, df1_dy], [df2_dx, df2_dy]])

    def inv_jacob(self, jacobian: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.linalg.inv(jacobian)

    def get_det_jacob(self, jacobian: npt.NDArray[np.float64]) -> float:
        det: float = jacobian[0][0] * jacobian[1][1] - jacobian[0][1] * jacobian[1][0]
        return det

    def eval_df_dx(self, ref_x: float, ref_y: float) -> Tuple[float, float]:
        df1_dx = (
            2 * self.slab_coord[0][0] * (1 - ref_y) * (ref_x - 1)
            + self.slab_coord[0][0] * (1 - ref_y) * (2 * ref_x - 1)
            - 4 * self.slab_coord[1][0] * ref_x * (1 - ref_y)
            - 4 * self.slab_coord[1][0] * (1 - ref_y) * (ref_x - 1)
            + 2 * self.slab_coord[2][0] * ref_x * (1 - ref_y)
            + self.slab_coord[2][0] * (1 - ref_y) * (2 * ref_x - 1)
            + 2 * self.slab_coord[3][0] * ref_y * (ref_x - 1)
            + self.slab_coord[3][0] * ref_y * (2 * ref_x - 1)
            - 4 * self.slab_coord[4][0] * ref_x * ref_y
            - 4 * self.slab_coord[4][0] * ref_y * (ref_x - 1)
            + 2 * self.slab_coord[5][0] * ref_x * ref_y
            + self.slab_coord[5][0] * ref_y * (2 * ref_x - 1)
        )
        df2_dx = (
            2 * self.slab_coord[0][1] * (1 - ref_y) * (ref_x - 1)
            + self.slab_coord[0][1] * (1 - ref_y) * (2 * ref_x - 1)
            - 4 * self.slab_coord[1][1] * ref_x * (1 - ref_y)
            - 4 * self.slab_coord[1][1] * (1 - ref_y) * (ref_x - 1)
            + 2 * self.slab_coord[2][1] * ref_x * (1 - ref_y)
            + self.slab_coord[2][1] * (1 - ref_y) * (2 * ref_x - 1)
            + 2 * self.slab_coord[3][1] * ref_y * (ref_x - 1)
            + self.slab_coord[3][1] * ref_y * (2 * ref_x - 1)
            - 4 * self.slab_coord[4][1] * ref_x * ref_y
            - 4 * self.slab_coord[4][1] * ref_y * (ref_x - 1)
            + 2 * self.slab_coord[5][1] * ref_x * ref_y
            + self.slab_coord[5][1] * ref_y * (2 * ref_x - 1)
        )
        return df1_dx, df2_dx

    def eval_df_dy(self, ref_x: float) -> Tuple[float, float]:
        df1_dy = (
            -self.slab_coord[0][0] * (ref_x - 1) * (2 * ref_x - 1)
            + 4 * self.slab_coord[1][0] * ref_x * (ref_x - 1)
            - self.slab_coord[2][0] * ref_x * (2 * ref_x - 1)
            + self.slab_coord[3][0] * (ref_x - 1) * (2 * ref_x - 1)
            - 4 * self.slab_coord[4][0] * ref_x * (ref_x - 1)
            + self.slab_coord[5][0] * ref_x * (2 * ref_x - 1)
        )
        df2_dy = (
            -self.slab_coord[0][1] * (ref_x - 1) * (2 * ref_x - 1)
            + 4 * self.slab_coord[1][1] * ref_x * (ref_x - 1)
            - self.slab_coord[2][1] * ref_x * (2 * ref_x - 1)
            + self.slab_coord[3][1] * (ref_x - 1) * (2 * ref_x - 1)
            - 4 * self.slab_coord[4][1] * ref_x * (ref_x - 1)
            + self.slab_coord[5][1] * ref_x * (2 * ref_x - 1)
        )
        return df1_dy, df2_dy
