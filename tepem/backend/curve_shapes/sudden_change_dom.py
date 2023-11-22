from dataclasses import dataclass
from typing import Tuple

import numpy as np
from backend.curve_shapes.boundary_shapes import sudden_linear
from backend.curve_shapes.curves import straight_curve, straight_curve_normal


def angle_to_gradient(angle: float) -> float:
    rad = angle_to_radian(angle=angle)
    return np.tan(rad)


def angle_to_radian(angle: float) -> float:
    return angle / 180 * np.pi


@dataclass
class SuddenChangeDomain:
    length_str_inlet: float
    start_str_outlet: float
    start_change: float
    end_change: float
    dx_change: float
    init_radius: float
    gradient: float

    def sudden_bdn_upper(self, s: float) -> Tuple[float, float]:
        x = (
            straight_curve(s)[0]
            + sudden_linear(
                s,
                radius=self.init_radius,
                start_change=self.start_change,
                end_change=self.end_change,
                dx_change=self.dx_change,
                gradient=self.gradient,
            )
            * straight_curve_normal(s)[0]
        )
        y = (
            straight_curve(s)[1]
            + sudden_linear(
                s,
                radius=self.init_radius,
                start_change=self.start_change,
                end_change=self.end_change,
                dx_change=self.dx_change,
                gradient=self.gradient,
            )
            * straight_curve_normal(s)[1]
        )
        return x, y

    def sudden_bdn_lower(self, s: float) -> Tuple[float, float]:
        x = (
            straight_curve(s)[0]
            - sudden_linear(
                s,
                radius=self.init_radius,
                start_change=self.start_change,
                end_change=self.end_change,
                dx_change=self.dx_change,
                gradient=self.gradient,
            )
            * straight_curve_normal(s)[0]
        )
        y = (
            straight_curve(s)[1]
            - sudden_linear(
                s,
                radius=self.init_radius,
                start_change=self.start_change,
                end_change=self.end_change,
                dx_change=self.dx_change,
                gradient=self.gradient,
            )
            * straight_curve_normal(s)[1]
        )
        return x, y
