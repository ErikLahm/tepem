from dataclasses import dataclass
from typing import Tuple

import numpy as np
from backend.curve_shapes.boundary_shapes import linear_boundary
from backend.curve_shapes.curves import straight_curve, straight_curve_normal


def angle_to_gradient(angle: float) -> float:
    rad = angle_to_radian(angle=angle)
    return np.tan(rad)


def angle_to_radian(angle: float) -> float:
    return angle / 180 * np.pi


@dataclass
class NozzleDomain:
    length_str_inlet: float
    start_str_outlet: float
    init_radius: float
    angle: float

    def straight_conv_upper(self, s: float) -> Tuple[float, float]:
        if s <= self.length_str_inlet:
            x = (
                straight_curve(s)[0]
                + linear_boundary(s, n=self.init_radius) * straight_curve_normal(s)[0]
            )
            y = (
                straight_curve(s)[1]
                + linear_boundary(s, n=self.init_radius) * straight_curve_normal(s)[1]
            )
        elif s > self.length_str_inlet and s <= self.start_str_outlet:
            x = (
                straight_curve(s)[0]
                + (
                    linear_boundary(
                        s, m=angle_to_gradient(self.angle), n=self.init_radius
                    )
                    - self.length_str_inlet * angle_to_gradient(self.angle)
                )
                * straight_curve_normal(s)[0]
            )
            y = (
                straight_curve(s)[1]
                + (
                    linear_boundary(
                        s, m=angle_to_gradient(self.angle), n=self.init_radius
                    )
                    - self.length_str_inlet * angle_to_gradient(self.angle)
                )
                * straight_curve_normal(s)[1]
            )
        else:
            n_decl = (
                self.init_radius - angle_to_gradient(self.angle) * self.length_str_inlet
            )
            n_out = angle_to_gradient(self.angle) * self.start_str_outlet + n_decl
            x = (
                straight_curve(s)[0]
                + linear_boundary(s, n=n_out) * straight_curve_normal(s)[0]
            )
            y = (
                straight_curve(s)[1]
                + linear_boundary(s, n=n_out) * straight_curve_normal(s)[1]
            )
        return x, y

    def straight_conv_lower(self, s: float) -> Tuple[float, float]:
        if s <= self.length_str_inlet:
            x = (
                straight_curve(s)[0]
                - linear_boundary(s, n=self.init_radius) * straight_curve_normal(s)[0]
            )
            y = (
                straight_curve(s)[1]
                - linear_boundary(s, n=self.init_radius) * straight_curve_normal(s)[1]
            )
        elif s > self.length_str_inlet and s <= self.start_str_outlet:
            x = (
                straight_curve(s)[0]
                - (
                    linear_boundary(
                        s, m=angle_to_gradient(self.angle), n=self.init_radius
                    )
                    - self.length_str_inlet * angle_to_gradient(self.angle)
                )
                * straight_curve_normal(s)[0]
            )
            y = (
                straight_curve(s)[1]
                - (
                    linear_boundary(
                        s, m=angle_to_gradient(self.angle), n=self.init_radius
                    )
                    - self.length_str_inlet * angle_to_gradient(self.angle)
                )
                * straight_curve_normal(s)[1]
            )
        else:
            n_decl = (
                self.init_radius - angle_to_gradient(self.angle) * self.length_str_inlet
            )
            n_out = angle_to_gradient(self.angle) * self.start_str_outlet + n_decl
            x = (
                straight_curve(s)[0]
                - linear_boundary(s, n=n_out) * straight_curve_normal(s)[0]
            )
            y = (
                straight_curve(s)[1]
                - linear_boundary(s, n=n_out) * straight_curve_normal(s)[1]
            )
        return x, y
