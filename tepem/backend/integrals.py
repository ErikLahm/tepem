import numpy as np
import numpy.typing as npt
from q_1_4_sf import (
    grad_phi_0,
    grad_phi_1,
    grad_phi_2,
    grad_phi_3,
    grad_phi_4,
    grad_phi_5,
    grad_phi_6,
    grad_phi_7,
    grad_phi_8,
    grad_phi_9,
)
from q_1_sf import psi_0, psi_1, psi_2, psi_3

WEIGHTS = [
    0.317460317460320,
    0.317460317460319,
    0.555555555555555,
    0.555555555555555,
    0.555555555555555,
    0.555555555555555,
    1.142857142857139,
]

POINTS = [
    (0.5, 0.9830458915396469),
    (0.5, 0.016954108460353),
    (0.8872983346207419, 0.21132486540518702),
    (0.8872983346207419, 0.788675134594813),
    (0.11270166537925802, 0.788675134594813),
    (0.11270166537925802, 0.21132486540518702),
    (0.5, 0.5),
]

GRAD_LIST_Q14 = [
    grad_phi_0,
    grad_phi_1,
    grad_phi_2,
    grad_phi_3,
    grad_phi_4,
    grad_phi_5,
    grad_phi_6,
    grad_phi_7,
    grad_phi_8,
    grad_phi_9,
]

SF_LIST_Q1 = [psi_0, psi_1, psi_2, psi_3]


def integral_a(
    inv_jacob: npt.NDArray[np.float64], jac_det: float, k: int, j: int
) -> float:
    integral_sum = 0
    for i, weight in enumerate(WEIGHTS):
        partial_sum = weight * np.dot(
            GRAD_LIST_Q14[k](POINTS[i][0], POINTS[i][1]),
            np.dot(
                np.dot(inv_jacob, inv_jacob.T),
                GRAD_LIST_Q14[j](POINTS[i][0], POINTS[i][1]).T,
            ),
        )
        integral_sum += partial_sum
    integral = integral_sum * abs(jac_det)
    return integral


def integral_b(
    inv_jacob: npt.NDArray[np.float64], jac_det: float, k: int, j: int, i: int
) -> float:
    integral_sum = 0
    for s, weight in enumerate(WEIGHTS):
        partial_sum = (
            weight
            * SF_LIST_Q1[k](POINTS[s][0], POINTS[s][1])
            * (
                np.dot(
                    inv_jacob[:, i],
                    GRAD_LIST_Q14[j](POINTS[s][0], POINTS[s][1]),
                )
            )
        )
        integral_sum += partial_sum
    integral = integral_sum * abs(jac_det)
    return integral
