import numpy as np
import numpy.typing as npt


def phi_0(x: float, y: float) -> float:
    return (1.0 - 1.0 * x) * (-4.5 * y**3 + 9.0 * y**2 - 5.5 * y + 1.0)


def phi_1(x: float, y: float) -> float:
    return 4.5 * y * (1.0 - 1.0 * x) * (y - 1) * (3.0 * y - 2.0)


def phi_2(x: float, y: float) -> float:
    return y * (1.0 - 1.0 * x) * (-13.5 * y**2 + 18.0 * y - 4.5)


def phi_3(x: float, y: float) -> float:
    return y * (1.0 - 1.0 * x) * (4.5 * y**2 - 4.5 * y + 1.0)


def phi_4(x: float, y: float) -> float:
    return 1.0 * x * (-4.5 * y**3 + 9.0 * y**2 - 5.5 * y + 1.0)


def phi_5(x: float, y: float) -> float:
    return 4.5 * x * y * (y - 1) * (3.0 * y - 2.0)


def phi_6(x: float, y: float) -> float:
    return 1.0 * x * y * (-13.5 * y**2 + 18.0 * y - 4.5)


def phi_7(x: float, y: float) -> float:
    return 1.0 * x * y * (4.5 * y**2 - 4.5 * y + 1.0)


def grad_phi_0(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = 4.5 * y**3 - 9.0 * y**2 + 5.5 * y - 1.0
    phi_dy = (1.0 - 1.0 * x) * (-13.5 * y**2 + 18.0 * y - 5.5)
    return np.array([phi_dx, phi_dy])


def grad_phi_1(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = -4.5 * y * (y - 1) * (3.0 * y - 2.0)
    phi_dy = (
        13.5 * y * (1.0 - 1.0 * x) * (y - 1)
        + 4.5 * y * (1.0 - 1.0 * x) * (3.0 * y - 2.0)
        + 4.5 * (1.0 - 1.0 * x) * (y - 1) * (3.0 * y - 2.0)
    )
    return np.array([phi_dx, phi_dy])


def grad_phi_2(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = -1.0 * y * (-13.5 * y**2 + 18.0 * y - 4.5)
    phi_dy = y * (1.0 - 1.0 * x) * (18.0 - 27.0 * y) + (1.0 - 1.0 * x) * (
        -13.5 * y**2 + 18.0 * y - 4.5
    )
    return np.array([phi_dx, phi_dy])


def grad_phi_3(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = -1.0 * y * (4.5 * y**2 - 4.5 * y + 1.0)
    phi_dy = y * (1.0 - 1.0 * x) * (9.0 * y - 4.5) + (1.0 - 1.0 * x) * (
        4.5 * y**2 - 4.5 * y + 1.0
    )
    return np.array([phi_dx, phi_dy])


def grad_phi_4(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = -4.5 * y**3 + 9.0 * y**2 - 5.5 * y + 1.0
    phi_dy = 1.0 * x * (-13.5 * y**2 + 18.0 * y - 5.5)
    return np.array([phi_dx, phi_dy])


def grad_phi_5(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = 4.5 * y * (y - 1) * (3.0 * y - 2.0)
    phi_dy = (
        13.5 * x * y * (y - 1)
        + 4.5 * x * y * (3.0 * y - 2.0)
        + 4.5 * x * (y - 1) * (3.0 * y - 2.0)
    )
    return np.array([phi_dx, phi_dy])


def grad_phi_6(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = 1.0 * y * (-13.5 * y**2 + 18.0 * y - 4.5)
    phi_dy = 1.0 * x * y * (18.0 - 27.0 * y) + 1.0 * x * (
        -13.5 * y**2 + 18.0 * y - 4.5
    )
    return np.array([phi_dx, phi_dy])


def grad_phi_7(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = 1.0 * y * (4.5 * y**2 - 4.5 * y + 1.0)
    phi_dy = 1.0 * x * y * (9.0 * y - 4.5) + 1.0 * x * (4.5 * y**2 - 4.5 * y + 1.0)
    return np.array([phi_dx, phi_dy])


Q13_SF_LIST = [phi_0, phi_1, phi_2, phi_3, phi_4, phi_5, phi_6, phi_7]

Q13_GRAD_SF_LIST = [
    grad_phi_0,
    grad_phi_1,
    grad_phi_2,
    grad_phi_3,
    grad_phi_4,
    grad_phi_5,
    grad_phi_6,
    grad_phi_7,
]
