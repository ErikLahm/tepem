import numpy as np
import numpy.typing as npt


def psi_0(x: float, y: float) -> float:
    return (1 - x) * (1 - y)


def psi_1(x: float, y: float) -> float:
    return x * (1 - y)


def psi_2(x: float, y: float) -> float:
    return y * (1 - x)


def psi_3(x: float, y: float) -> float:
    return x * y


def grad_psi_0(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = y - 1
    phi_dy = x - 1
    return np.array([phi_dx, phi_dy])


def grad_psi_1(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = 1 - y
    phi_dy = -x
    return np.array([phi_dx, phi_dy])


def grad_psi_2(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = -y
    phi_dy = 1 - x
    return np.array([phi_dx, phi_dy])


def grad_psi_3(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = y
    phi_dy = x
    return np.array([phi_dx, phi_dy])
