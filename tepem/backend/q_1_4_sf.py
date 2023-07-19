import numpy as np
import numpy.typing as npt

# ________________q_1_4_shape_functions_velocity_____________________________________


def phi_0(x: float, y: float) -> float:
    return (
        1.0
        * (1 - x)
        * (y - 1)
        * (1.33333333333333 * y - 1.0)
        * (2.0 * y - 1.0)
        * (4.0 * y - 1.0)
    )


def phi_1(x: float, y: float) -> float:
    return (
        y
        * (1 - x)
        * (-42.6666666666667 * y**3 + 96.0 * y**2 - 69.3333333333333 * y + 16.0)
    )


def phi_2(x: float, y: float) -> float:
    return 4.0 * y * (1 - x) * (y - 1) * (4.0 * y - 3.0) * (4.0 * y - 1.0)


def phi_3(x: float, y: float) -> float:
    return (
        y
        * (1 - x)
        * (
            -42.6666666666667 * y**3
            + 74.6666666666667 * y**2
            - 37.3333333333333 * y
            + 5.33333333333333
        )
    )


def phi_4(x: float, y: float) -> float:
    return (
        y
        * (1 - x)
        * (10.6666666666667 * y**3 - 16.0 * y**2 + 7.33333333333333 * y - 1.0)
    )


def phi_5(x: float, y: float) -> float:
    return (
        1.0
        * x
        * (y - 1)
        * (1.33333333333333 * y - 1.0)
        * (2.0 * y - 1.0)
        * (4.0 * y - 1.0)
    )


def phi_6(x: float, y: float) -> float:
    return (
        x
        * y
        * (-42.6666666666667 * y**3 + 96.0 * y**2 - 69.3333333333333 * y + 16.0)
    )


def phi_7(x: float, y: float) -> float:
    return 4.0 * x * y * (y - 1) * (4.0 * y - 3.0) * (4.0 * y - 1.0)


def phi_8(x: float, y: float) -> float:
    return (
        x
        * y
        * (
            -42.6666666666667 * y**3
            + 74.6666666666667 * y**2
            - 37.3333333333333 * y
            + 5.33333333333333
        )
    )


def phi_9(x: float, y: float) -> float:
    return (
        x * y * (10.6666666666667 * y**3 - 16.0 * y**2 + 7.33333333333333 * y - 1.0)
    )


def grad_phi_0(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = (
        -1.0
        * (y - 1)
        * (1.33333333333333 * y - 1.0)
        * (2.0 * y - 1.0)
        * (4.0 * y - 1.0)
    )
    phi_dy = (
        4.0 * (1 - x) * (y - 1) * (1.33333333333333 * y - 1.0) * (2.0 * y - 1.0)
        + 2.0 * (1 - x) * (y - 1) * (1.33333333333333 * y - 1.0) * (4.0 * y - 1.0)
        + 1.33333333333333 * (1 - x) * (y - 1) * (2.0 * y - 1.0) * (4.0 * y - 1.0)
        + 1.0
        * (1 - x)
        * (1.33333333333333 * y - 1.0)
        * (2.0 * y - 1.0)
        * (4.0 * y - 1.0)
    )
    return np.array([phi_dx, phi_dy])


def grad_phi_1(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = -y * (
        -42.6666666666667 * y**3 + 96.0 * y**2 - 69.3333333333333 * y + 16.0
    )
    phi_dy = y * (1 - x) * (-128.0 * y**2 + 192.0 * y - 69.3333333333333) + (
        1 - x
    ) * (-42.6666666666667 * y**3 + 96.0 * y**2 - 69.3333333333333 * y + 16.0)
    return np.array([phi_dx, phi_dy])


def grad_phi_2(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = -4.0 * y * (y - 1) * (4.0 * y - 3.0) * (4.0 * y - 1.0)
    phi_dy = (
        16.0 * y * (1 - x) * (y - 1) * (4.0 * y - 3.0)
        + 16.0 * y * (1 - x) * (y - 1) * (4.0 * y - 1.0)
        + 4.0 * y * (1 - x) * (4.0 * y - 3.0) * (4.0 * y - 1.0)
        + 4.0 * (1 - x) * (y - 1) * (4.0 * y - 3.0) * (4.0 * y - 1.0)
    )
    return np.array([phi_dx, phi_dy])


def grad_phi_3(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = -y * (
        -42.6666666666667 * y**3
        + 74.6666666666667 * y**2
        - 37.3333333333333 * y
        + 5.33333333333333
    )
    phi_dy = y * (1 - x) * (
        -128.0 * y**2 + 149.333333333333 * y - 37.3333333333333
    ) + (1 - x) * (
        -42.6666666666667 * y**3
        + 74.6666666666667 * y**2
        - 37.3333333333333 * y
        + 5.33333333333333
    )
    return np.array([phi_dx, phi_dy])


def grad_phi_4(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = -y * (
        10.6666666666667 * y**3 - 16.0 * y**2 + 7.33333333333333 * y - 1.0
    )
    phi_dy = y * (1 - x) * (32.0 * y**2 - 32.0 * y + 7.33333333333333) + (1 - x) * (
        10.6666666666667 * y**3 - 16.0 * y**2 + 7.33333333333333 * y - 1.0
    )
    return np.array([phi_dx, phi_dy])


def grad_phi_5(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = (
        1.0 * (y - 1) * (1.33333333333333 * y - 1.0) * (2.0 * y - 1.0) * (4.0 * y - 1.0)
    )
    phi_dy = (
        4.0 * x * (y - 1) * (1.33333333333333 * y - 1.0) * (2.0 * y - 1.0)
        + 2.0 * x * (y - 1) * (1.33333333333333 * y - 1.0) * (4.0 * y - 1.0)
        + 1.33333333333333 * x * (y - 1) * (2.0 * y - 1.0) * (4.0 * y - 1.0)
        + 1.0 * x * (1.33333333333333 * y - 1.0) * (2.0 * y - 1.0) * (4.0 * y - 1.0)
    )
    return np.array([phi_dx, phi_dy])


def grad_phi_6(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = y * (
        -42.6666666666667 * y**3 + 96.0 * y**2 - 69.3333333333333 * y + 16.0
    )
    phi_dy = x * y * (-128.0 * y**2 + 192.0 * y - 69.3333333333333) + x * (
        -42.6666666666667 * y**3 + 96.0 * y**2 - 69.3333333333333 * y + 16.0
    )
    return np.array([phi_dx, phi_dy])


def grad_phi_7(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = 4.0 * y * (y - 1) * (4.0 * y - 3.0) * (4.0 * y - 1.0)
    phi_dy = (
        16.0 * x * y * (y - 1) * (4.0 * y - 3.0)
        + 16.0 * x * y * (y - 1) * (4.0 * y - 1.0)
        + 4.0 * x * y * (4.0 * y - 3.0) * (4.0 * y - 1.0)
        + 4.0 * x * (y - 1) * (4.0 * y - 3.0) * (4.0 * y - 1.0)
    )
    return np.array([phi_dx, phi_dy])


def grad_phi_8(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = y * (
        -42.6666666666667 * y**3
        + 74.6666666666667 * y**2
        - 37.3333333333333 * y
        + 5.33333333333333
    )
    phi_dy = x * y * (-128.0 * y**2 + 149.333333333333 * y - 37.3333333333333) + x * (
        -42.6666666666667 * y**3
        + 74.6666666666667 * y**2
        - 37.3333333333333 * y
        + 5.33333333333333
    )
    return np.array([phi_dx, phi_dy])


def grad_phi_9(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = y * (
        10.6666666666667 * y**3 - 16.0 * y**2 + 7.33333333333333 * y - 1.0
    )
    phi_dy = x * y * (32.0 * y**2 - 32.0 * y + 7.33333333333333) + x * (
        10.6666666666667 * y**3 - 16.0 * y**2 + 7.33333333333333 * y - 1.0
    )
    return np.array([phi_dx, phi_dy])


# ______________________________end________________________________________
