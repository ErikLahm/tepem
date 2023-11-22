import numpy as np
import numpy.typing as npt


def phi_0(x: float, y: float) -> float:
    return (1.0 - 1.0 * x) * (
        -5.33333333333333 * y**3
        + 10.6666666666667 * y**2
        - 6.33333333333333 * y
        + 1.0
    )


def phi_1(x: float, y: float) -> float:
    return 5.33333333333333 * y * (1.0 - 1.0 * x) * (y - 1) * (2.0 * y - 1.5)


def phi_2(x: float, y: float) -> float:
    return (
        y
        * (1.0 - 1.0 * x)
        * (-10.6666666666667 * y**2 + 13.3333333333333 * y - 2.66666666666667)
    )


def phi_3(x: float, y: float) -> float:
    return (
        y
        * (1.0 - 1.0 * x)
        * (5.33333333333333 * y**2 - 5.33333333333333 * y + 0.999999999999999)
    )


def phi_4(x: float, y: float) -> float:
    return (
        1.0
        * x
        * (
            -5.33333333333333 * y**3
            + 10.6666666666667 * y**2
            - 6.33333333333333 * y
            + 1.0
        )
    )


def phi_5(x: float, y: float) -> float:
    return 5.33333333333333 * x * y * (y - 1) * (2.0 * y - 1.5)


def phi_6(x: float, y: float) -> float:
    return (
        1.0
        * x
        * y
        * (-10.6666666666667 * y**2 + 13.3333333333333 * y - 2.66666666666667)
    )


def phi_7(x: float, y: float) -> float:
    return (
        1.0
        * x
        * y
        * (5.33333333333333 * y**2 - 5.33333333333333 * y + 0.999999999999999)
    )


def grad_phi_0(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = (
        5.33333333333333 * y**3
        - 10.6666666666667 * y**2
        + 6.33333333333333 * y
        - 1.0
    )
    phi_dy = (1.0 - 1.0 * x) * (
        -16.0 * y**2 + 21.3333333333333 * y - 6.33333333333333
    )
    return np.array([phi_dx, phi_dy])


def grad_phi_1(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = -5.33333333333333 * y * (y - 1) * (2.0 * y - 1.5)
    phi_dy = (
        10.6666666666667 * y * (1.0 - 1.0 * x) * (y - 1)
        + 5.33333333333333 * y * (1.0 - 1.0 * x) * (2.0 * y - 1.5)
        + 5.33333333333333 * (1.0 - 1.0 * x) * (y - 1) * (2.0 * y - 1.5)
    )
    return np.array([phi_dx, phi_dy])


def grad_phi_2(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = (
        -1.0
        * y
        * (-10.6666666666667 * y**2 + 13.3333333333333 * y - 2.66666666666667)
    )
    phi_dy = y * (1.0 - 1.0 * x) * (13.3333333333333 - 21.3333333333333 * y) + (
        1.0 - 1.0 * x
    ) * (-10.6666666666667 * y**2 + 13.3333333333333 * y - 2.66666666666667)
    return np.array([phi_dx, phi_dy])


def grad_phi_3(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = (
        -1.0
        * y
        * (5.33333333333333 * y**2 - 5.33333333333333 * y + 0.999999999999999)
    )
    phi_dy = y * (1.0 - 1.0 * x) * (10.6666666666667 * y - 5.33333333333333) + (
        1.0 - 1.0 * x
    ) * (5.33333333333333 * y**2 - 5.33333333333333 * y + 0.999999999999999)
    return np.array([phi_dx, phi_dy])


def grad_phi_4(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = (
        -5.33333333333333 * y**3
        + 10.6666666666667 * y**2
        - 6.33333333333333 * y
        + 1.0
    )
    phi_dy = 1.0 * x * (-16.0 * y**2 + 21.3333333333333 * y - 6.33333333333333)
    return np.array([phi_dx, phi_dy])


def grad_phi_5(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = 5.33333333333333 * y * (y - 1) * (2.0 * y - 1.5)
    phi_dy = (
        10.6666666666667 * x * y * (y - 1)
        + 5.33333333333333 * x * y * (2.0 * y - 1.5)
        + 5.33333333333333 * x * (y - 1) * (2.0 * y - 1.5)
    )
    return np.array([phi_dx, phi_dy])


def grad_phi_6(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = (
        1.0 * y * (-10.6666666666667 * y**2 + 13.3333333333333 * y - 2.66666666666667)
    )
    phi_dy = 1.0 * x * y * (13.3333333333333 - 21.3333333333333 * y) + 1.0 * x * (
        -10.6666666666667 * y**2 + 13.3333333333333 * y - 2.66666666666667
    )
    return np.array([phi_dx, phi_dy])


def grad_phi_7(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = (
        1.0 * y * (5.33333333333333 * y**2 - 5.33333333333333 * y + 0.999999999999999)
    )
    phi_dy = 1.0 * x * y * (10.6666666666667 * y - 5.33333333333333) + 1.0 * x * (
        5.33333333333333 * y**2 - 5.33333333333333 * y + 0.999999999999999
    )
    return np.array([phi_dx, phi_dy])


Q13_SF_CHEB_LIST = [phi_0, phi_1, phi_2, phi_3, phi_4, phi_5, phi_6, phi_7]

Q13_GRAD_SF_CHEB_LIST = [
    grad_phi_0,
    grad_phi_1,
    grad_phi_2,
    grad_phi_3,
    grad_phi_4,
    grad_phi_5,
    grad_phi_6,
    grad_phi_7,
]
