from typing import Tuple

import numpy as np

DEFAULT_GRAD = 0


def straight_curve(s: float, m: float = DEFAULT_GRAD) -> Tuple[float, float]:
    return (s, m * s)


def straight_curve_normal(s: float, m: float = DEFAULT_GRAD) -> Tuple[float, float]:
    return (-m, 1)


# ------------------------------------------------------------------------------------------------


def curved_upward(s: float, m: float = 1.0) -> Tuple[float, float]:
    if s < 0.5:
        return (s, s**2)
    else:
        return (s, m * s - 1 / 4)


def curved_upward_normal(s: float, m: float = 1.0) -> Tuple[float, float]:
    if s < 0.5:
        x = 1
        y = 2 * s
        norm = np.sqrt(x**2 + y**2)
        return (-y / norm, x / norm)
    else:
        x = 1
        y = m
        norm = np.sqrt(x**2 + y**2)
        return (-y / norm, x / norm)
