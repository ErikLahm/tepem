import numpy as np
import numpy.typing as npt


def phi_0(x: float, y: float) -> float:
    return (
        y
        * (1.0 - 1.0 * x)
        * (-42.6666666666667 * y**3 + 96.0 * y**2 - 69.3333333333333 * y + 16.0)
    )


def phi_1(x: float, y: float) -> float:
    return 4.0 * y * (1.0 - 1.0 * x) * (y - 1) * (4.0 * y - 3.0) * (4.0 * y - 1.0)


def phi_2(x: float, y: float) -> float:
    return (
        y
        * (1.0 - 1.0 * x)
        * (
            -42.6666666666667 * y**3
            + 74.6666666666667 * y**2
            - 37.3333333333333 * y
            + 5.33333333333333
        )
    )


def phi_3(x: float, y: float) -> float:
    return (
        1.0
        * x
        * y
        * (-42.6666666666667 * y**3 + 96.0 * y**2 - 69.3333333333333 * y + 16.0)
    )


def phi_4(x: float, y: float) -> float:
    return 4.0 * x * y * (y - 1) * (4.0 * y - 3.0) * (4.0 * y - 1.0)


def phi_5(x: float, y: float) -> float:
    return (
        1.0
        * x
        * y
        * (
            -42.6666666666667 * y**3
            + 74.6666666666667 * y**2
            - 37.3333333333333 * y
            + 5.33333333333333
        )
    )


def grad_phi_0(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = (
        -1.0
        * y
        * (-42.6666666666667 * y**3 + 96.0 * y**2 - 69.3333333333333 * y + 16.0)
    )
    phi_dy = y * (1.0 - 1.0 * x) * (-128.0 * y**2 + 192.0 * y - 69.3333333333333) + (
        1.0 - 1.0 * x
    ) * (-42.6666666666667 * y**3 + 96.0 * y**2 - 69.3333333333333 * y + 16.0)
    return np.array([phi_dx, phi_dy])


def grad_phi_1(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = -4.0 * y * (y - 1) * (4.0 * y - 3.0) * (4.0 * y - 1.0)
    phi_dy = (
        16.0 * y * (1.0 - 1.0 * x) * (y - 1) * (4.0 * y - 3.0)
        + 16.0 * y * (1.0 - 1.0 * x) * (y - 1) * (4.0 * y - 1.0)
        + 4.0 * y * (1.0 - 1.0 * x) * (4.0 * y - 3.0) * (4.0 * y - 1.0)
        + 4.0 * (1.0 - 1.0 * x) * (y - 1) * (4.0 * y - 3.0) * (4.0 * y - 1.0)
    )
    return np.array([phi_dx, phi_dy])


def grad_phi_2(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = (
        -1.0
        * y
        * (
            -42.6666666666667 * y**3
            + 74.6666666666667 * y**2
            - 37.3333333333333 * y
            + 5.33333333333333
        )
    )
    phi_dy = y * (1.0 - 1.0 * x) * (
        -128.0 * y**2 + 149.333333333333 * y - 37.3333333333333
    ) + (1.0 - 1.0 * x) * (
        -42.6666666666667 * y**3
        + 74.6666666666667 * y**2
        - 37.3333333333333 * y
        + 5.33333333333333
    )
    return np.array([phi_dx, phi_dy])


def grad_phi_3(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = (
        1.0
        * y
        * (-42.6666666666667 * y**3 + 96.0 * y**2 - 69.3333333333333 * y + 16.0)
    )
    phi_dy = 1.0 * x * y * (
        -128.0 * y**2 + 192.0 * y - 69.3333333333333
    ) + 1.0 * x * (
        -42.6666666666667 * y**3 + 96.0 * y**2 - 69.3333333333333 * y + 16.0
    )
    return np.array([phi_dx, phi_dy])


def grad_phi_4(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = 4.0 * y * (y - 1) * (4.0 * y - 3.0) * (4.0 * y - 1.0)
    phi_dy = (
        16.0 * x * y * (y - 1) * (4.0 * y - 3.0)
        + 16.0 * x * y * (y - 1) * (4.0 * y - 1.0)
        + 4.0 * x * y * (4.0 * y - 3.0) * (4.0 * y - 1.0)
        + 4.0 * x * (y - 1) * (4.0 * y - 3.0) * (4.0 * y - 1.0)
    )
    return np.array([phi_dx, phi_dy])


def grad_phi_5(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = (
        1.0
        * y
        * (
            -42.6666666666667 * y**3
            + 74.6666666666667 * y**2
            - 37.3333333333333 * y
            + 5.33333333333333
        )
    )
    phi_dy = 1.0 * x * y * (
        -128.0 * y**2 + 149.333333333333 * y - 37.3333333333333
    ) + 1.0 * x * (
        -42.6666666666667 * y**3
        + 74.6666666666667 * y**2
        - 37.3333333333333 * y
        + 5.33333333333333
    )
    return np.array([phi_dx, phi_dy])


Q14_SF_LIST = [phi_0, phi_1, phi_2, phi_3, phi_4, phi_5]

Q14_GRAD_SF_LIST = [
    grad_phi_0,
    grad_phi_1,
    grad_phi_2,
    grad_phi_3,
    grad_phi_4,
    grad_phi_5,
]
