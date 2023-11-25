from dataclasses import dataclass
from typing import Tuple

from backend.curve_shapes.boundary_shapes import bulge
from backend.curve_shapes.curves import straight_curve, straight_curve_normal


@dataclass
class SuddenChangeDomain:
    dom_length: float
    max_height: float
    init_radius: float

    def sudden_bdn_upper(self, s: float) -> Tuple[float, float]:
        x = (
            straight_curve(s)[0]
            + bulge(
                s=s,
                dom_length=self.dom_length,
                init_radius=self.init_radius,
                max_height=self.max_height,
            )
            * straight_curve_normal(s)[0]
        )
        y = (
            straight_curve(s)[1]
            + bulge(
                s=s,
                dom_length=self.dom_length,
                init_radius=self.init_radius,
                max_height=self.max_height,
            )
            * straight_curve_normal(s)[1]
        )
        return x, y

    def sudden_bdn_lower(self, s: float) -> Tuple[float, float]:
        x = (
            straight_curve(s)[0]
            - bulge(
                s=s,
                dom_length=self.dom_length,
                init_radius=self.init_radius,
                max_height=self.max_height,
            )
            * straight_curve_normal(s)[0]
        )
        y = (
            straight_curve(s)[1]
            - bulge(
                s=s,
                dom_length=self.dom_length,
                init_radius=self.init_radius,
                max_height=self.max_height,
            )
            * straight_curve_normal(s)[1]
        )
        return x, y
