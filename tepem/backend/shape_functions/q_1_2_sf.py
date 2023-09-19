import numpy as np
import numpy.typing as npt


def phi_0(x: float, y: float) -> float:
    return 1.0 * (1.0 - 1.0 * x) * (y - 1) * (2.0 * y - 1.0)


def phi_1(x: float, y: float) -> float:
    return 4.0 * y * (1.0 - 1.0 * x) * (1 - y)


def phi_2(x: float, y: float) -> float:
    return y * (1.0 - 1.0 * x) * (2.0 * y - 1.0)


def phi_3(x: float, y: float) -> float:
    return 1.0 * x * (y - 1) * (2.0 * y - 1.0)


def phi_4(x: float, y: float) -> float:
    return 4.0 * x * y * (1 - y)


def phi_5(x: float, y: float) -> float:
    return 1.0 * x * y * (2.0 * y - 1.0)


def grad_phi_0(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = -1.0 * (y - 1) * (2.0 * y - 1.0)
    phi_dy = 2.0 * (1.0 - 1.0 * x) * (y - 1) + 1.0 * (1.0 - 1.0 * x) * (2.0 * y - 1.0)
    return np.array([phi_dx, phi_dy])


def grad_phi_1(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = -4.0 * y * (1 - y)
    phi_dy = -4.0 * y * (1.0 - 1.0 * x) + 4.0 * (1.0 - 1.0 * x) * (1 - y)
    return np.array([phi_dx, phi_dy])


def grad_phi_2(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = -1.0 * y * (2.0 * y - 1.0)
    phi_dy = 2.0 * y * (1.0 - 1.0 * x) + (1.0 - 1.0 * x) * (2.0 * y - 1.0)
    return np.array([phi_dx, phi_dy])


def grad_phi_3(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = 1.0 * (y - 1) * (2.0 * y - 1.0)
    phi_dy = 2.0 * x * (y - 1) + 1.0 * x * (2.0 * y - 1.0)
    return np.array([phi_dx, phi_dy])


def grad_phi_4(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = 4.0 * y * (1 - y)
    phi_dy = -4.0 * x * y + 4.0 * x * (1 - y)
    return np.array([phi_dx, phi_dy])


def grad_phi_5(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = 1.0 * y * (2.0 * y - 1.0)
    phi_dy = 2.0 * x * y + 1.0 * x * (2.0 * y - 1.0)
    return np.array([phi_dx, phi_dy])


Q12_SF_LIST = [phi_0, phi_1, phi_2, phi_3, phi_4, phi_5]

Q12_GRAD_SF_LIST = [
    grad_phi_0,
    grad_phi_1,
    grad_phi_2,
    grad_phi_3,
    grad_phi_4,
    grad_phi_5,
]
