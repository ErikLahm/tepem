from dataclasses import dataclass
from typing import Tuple

from backend.curve_shapes.boundary_shapes import linear_boundary
from backend.curve_shapes.curves import straight_curve, straight_curve_normal


@dataclass
class StraightPipeDomain:
    def straight_upper(self, s: float) -> Tuple[float, float]:
        x = straight_curve(s)[0] + linear_boundary(s) * straight_curve_normal(s)[0]
        y = straight_curve(s)[1] + linear_boundary(s) * straight_curve_normal(s)[1]
        return x, y

    def straight_lower(self, s: float) -> Tuple[float, float]:
        x = straight_curve(s)[0] - linear_boundary(s) * straight_curve_normal(s)[0]
        y = straight_curve(s)[1] - linear_boundary(s) * straight_curve_normal(s)[1]
        return x, y
