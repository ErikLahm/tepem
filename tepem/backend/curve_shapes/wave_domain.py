from dataclasses import dataclass
from typing import Tuple

from backend.curve_shapes.boundary_shapes import linear_boundary
from backend.curve_shapes.curves import (
    cos_curve,
    cos_curve_normal,
    straight_curve,
    straight_curve_normal,
)


@dataclass
class WaveDomain:
    length_str_inlet: float
    start_str_outlet: float
    init_radius: float
    gradient: float
    period: float
    nozzle: bool = False

    def cos_upper(self, s: float) -> Tuple[float, float]:
        if s <= self.length_str_inlet:
            x = (
                straight_curve(s)[0]
                + linear_boundary(s, n=self.init_radius) * straight_curve_normal(s)[0]
            )
            y = (
                straight_curve(s)[1]
                + linear_boundary(s, n=self.init_radius) * straight_curve_normal(s)[1]
            )
        elif s <= self.start_str_outlet:
            x = (
                cos_curve(s, self.length_str_inlet, period=self.period)[0]
                + linear_boundary(
                    s,
                    n=self.init_radius - self.gradient * self.length_str_inlet,
                    m=self.gradient,
                )
                * cos_curve_normal(s, self.length_str_inlet, period=self.period)[0]
            )
            y = (
                cos_curve(s, self.length_str_inlet, period=self.period)[1]
                + linear_boundary(
                    s,
                    n=self.init_radius - self.gradient * self.length_str_inlet,
                    m=self.gradient,
                )
                * cos_curve_normal(s, self.length_str_inlet, period=self.period)[1]
            )
        else:
            x = straight_curve(s)[0] + linear_boundary(s) * straight_curve_normal(s)[0]
            y = straight_curve(s)[1] + linear_boundary(s) * straight_curve_normal(s)[1]
        return x, y

    def cos_lower(self, s: float) -> Tuple[float, float]:
        if s <= self.length_str_inlet:
            x = (
                straight_curve(s)[0]
                - linear_boundary(s, n=self.init_radius) * straight_curve_normal(s)[0]
            )
            y = (
                straight_curve(s)[1]
                - linear_boundary(s, n=self.init_radius) * straight_curve_normal(s)[1]
            )
        elif s <= self.start_str_outlet:
            x = (
                cos_curve(s, self.length_str_inlet, period=self.period)[0]
                - linear_boundary(
                    s,
                    n=self.init_radius - self.gradient * self.length_str_inlet,
                    m=self.gradient,
                )
                * cos_curve_normal(s, self.length_str_inlet, period=self.period)[0]
            )
            y = (
                cos_curve(s, self.length_str_inlet, period=self.period)[1]
                - linear_boundary(
                    s,
                    n=self.init_radius - self.gradient * self.length_str_inlet,
                    m=self.gradient,
                )
                * cos_curve_normal(s, self.length_str_inlet, period=self.period)[1]
            )
        else:
            x = straight_curve(s)[0] - linear_boundary(s) * straight_curve_normal(s)[0]
            y = straight_curve(s)[1] - linear_boundary(s) * straight_curve_normal(s)[1]
        return x, y
