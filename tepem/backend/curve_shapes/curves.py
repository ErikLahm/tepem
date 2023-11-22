from typing import Tuple

import numpy as np

DEFAULT_GRAD = 0


def straight_curve(s: float, m: float = DEFAULT_GRAD) -> Tuple[float, float]:
    return (s, m * s)


def straight_curve_normal(s: float, m: float = DEFAULT_GRAD) -> Tuple[float, float]:
    return (-m, 1)


# ------------------------------------------------------------------------------------------------


def cos_curve(
    s: float, hori_offset: float = 0, amplitude: float = 0.01
) -> Tuple[float, float]:
    return (s, amplitude * np.cos(2 * np.pi / 0.1 * (s - 0.01)) - amplitude)


def cos_curve_normal(
    s: float, hori_offset: float = 0, amplitude: float = 0.01
) -> Tuple[float, float]:
    x = 1
    y = -2 * np.pi / 0.1 * amplitude * np.sin(2 * np.pi / 0.1 * (s - 0.01))
    norm = np.sqrt(x**2 + y**2)
    return (-y / norm, x / norm)


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
