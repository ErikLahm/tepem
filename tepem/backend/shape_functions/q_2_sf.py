import numpy as np
import numpy.typing as npt


def phi_0(x: float, y: float) -> float:
    return 1.0 * (x - 1) * (2.0 * x - 1.0) * (y - 1) * (2.0 * y - 1.0)


def phi_1(x: float, y: float) -> float:
    return 4.0 * x * (1 - x) * (y - 1) * (2.0 * y - 1.0)


def phi_2(x: float, y: float) -> float:
    return 1.0 * x * (2.0 * x - 1.0) * (y - 1) * (2.0 * y - 1.0)


def phi_3(x: float, y: float) -> float:
    return 4.0 * y * (1 - y) * (x - 1) * (2.0 * x - 1.0)


def phi_4(x: float, y: float) -> float:
    return 16.0 * x * y * (1 - x) * (1 - y)


def phi_5(x: float, y: float) -> float:
    return 4.0 * x * y * (1 - y) * (2.0 * x - 1.0)


def phi_6(x: float, y: float) -> float:
    return 1.0 * y * (x - 1) * (2.0 * x - 1.0) * (2.0 * y - 1.0)


def phi_7(x: float, y: float) -> float:
    return 4.0 * x * y * (1 - x) * (2.0 * y - 1.0)


def phi_8(x: float, y: float) -> float:
    return x * y * (2.0 * x - 1.0) * (2.0 * y - 1.0)


def grad_phi_0(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = 2.0 * (x - 1) * (y - 1) * (2.0 * y - 1.0) + 1.0 * (2.0 * x - 1.0) * (
        y - 1
    ) * (2.0 * y - 1.0)
    phi_dy = 2.0 * (x - 1) * (2.0 * x - 1.0) * (y - 1) + 1.0 * (x - 1) * (
        2.0 * x - 1.0
    ) * (2.0 * y - 1.0)
    return np.array([phi_dx, phi_dy])


def grad_phi_1(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = -4.0 * x * (y - 1) * (2.0 * y - 1.0) + 4.0 * (1 - x) * (y - 1) * (
        2.0 * y - 1.0
    )
    phi_dy = 8.0 * x * (1 - x) * (y - 1) + 4.0 * x * (1 - x) * (2.0 * y - 1.0)
    return np.array([phi_dx, phi_dy])


def grad_phi_2(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = 2.0 * x * (y - 1) * (2.0 * y - 1.0) + 1.0 * (2.0 * x - 1.0) * (y - 1) * (
        2.0 * y - 1.0
    )
    phi_dy = 2.0 * x * (2.0 * x - 1.0) * (y - 1) + 1.0 * x * (2.0 * x - 1.0) * (
        2.0 * y - 1.0
    )
    return np.array([phi_dx, phi_dy])


def grad_phi_3(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = 8.0 * y * (1 - y) * (x - 1) + 4.0 * y * (1 - y) * (2.0 * x - 1.0)
    phi_dy = -4.0 * y * (x - 1) * (2.0 * x - 1.0) + 4.0 * (1 - y) * (x - 1) * (
        2.0 * x - 1.0
    )
    return np.array([phi_dx, phi_dy])


def grad_phi_4(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = -16.0 * x * y * (1 - y) + 16.0 * y * (1 - x) * (1 - y)
    phi_dy = -16.0 * x * y * (1 - x) + 16.0 * x * (1 - x) * (1 - y)
    return np.array([phi_dx, phi_dy])


def grad_phi_5(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = 8.0 * x * y * (1 - y) + 4.0 * y * (1 - y) * (2.0 * x - 1.0)
    phi_dy = -4.0 * x * y * (2.0 * x - 1.0) + 4.0 * x * (1 - y) * (2.0 * x - 1.0)
    return np.array([phi_dx, phi_dy])


def grad_phi_6(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = 2.0 * y * (x - 1) * (2.0 * y - 1.0) + 1.0 * y * (2.0 * x - 1.0) * (
        2.0 * y - 1.0
    )
    phi_dy = 2.0 * y * (x - 1) * (2.0 * x - 1.0) + 1.0 * (x - 1) * (2.0 * x - 1.0) * (
        2.0 * y - 1.0
    )
    return np.array([phi_dx, phi_dy])


def grad_phi_7(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = -4.0 * x * y * (2.0 * y - 1.0) + 4.0 * y * (1 - x) * (2.0 * y - 1.0)
    phi_dy = 8.0 * x * y * (1 - x) + 4.0 * x * (1 - x) * (2.0 * y - 1.0)
    return np.array([phi_dx, phi_dy])


def grad_phi_8(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = 2.0 * x * y * (2.0 * y - 1.0) + y * (2.0 * x - 1.0) * (2.0 * y - 1.0)
    phi_dy = 2.0 * x * y * (2.0 * x - 1.0) + x * (2.0 * x - 1.0) * (2.0 * y - 1.0)
    return np.array([phi_dx, phi_dy])


Q2_SF_LIST = [phi_0, phi_1, phi_2, phi_3, phi_4, phi_5, phi_6, phi_7, phi_8]
