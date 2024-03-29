import numpy as np
import numpy.typing as npt


def phi_0(x: float, y: float) -> float:
    return (
        y
        * (-4.5 * x**3 + 9.0 * x**2 - 5.5 * x + 1.0)
        * (-42.6666666666667 * y**3 + 96.0 * y**2 - 69.3333333333333 * y + 16.0)
    )


def phi_1(x: float, y: float) -> float:
    return (
        4.0
        * y
        * (y - 1)
        * (4.0 * y - 3.0)
        * (4.0 * y - 1.0)
        * (-4.5 * x**3 + 9.0 * x**2 - 5.5 * x + 1.0)
    )


def phi_2(x: float, y: float) -> float:
    return (
        y
        * (-4.5 * x**3 + 9.0 * x**2 - 5.5 * x + 1.0)
        * (
            -42.6666666666667 * y**3
            + 74.6666666666667 * y**2
            - 37.3333333333333 * y
            + 5.33333333333333
        )
    )


def phi_3(x: float, y: float) -> float:
    return (
        4.5
        * x
        * y
        * (x - 1)
        * (3.0 * x - 2.0)
        * (-42.6666666666667 * y**3 + 96.0 * y**2 - 69.3333333333333 * y + 16.0)
    )


def phi_4(x: float, y: float) -> float:
    return (
        18.0
        * x
        * y
        * (x - 1)
        * (3.0 * x - 2.0)
        * (y - 1)
        * (4.0 * y - 3.0)
        * (4.0 * y - 1.0)
    )


def phi_5(x: float, y: float) -> float:
    return (
        4.5
        * x
        * y
        * (x - 1)
        * (3.0 * x - 2.0)
        * (
            -42.6666666666667 * y**3
            + 74.6666666666667 * y**2
            - 37.3333333333333 * y
            + 5.33333333333333
        )
    )


def phi_6(x: float, y: float) -> float:
    return (
        x
        * y
        * (-13.5 * x**2 + 18.0 * x - 4.5)
        * (-42.6666666666667 * y**3 + 96.0 * y**2 - 69.3333333333333 * y + 16.0)
    )


def phi_7(x: float, y: float) -> float:
    return (
        4.0
        * x
        * y
        * (y - 1)
        * (4.0 * y - 3.0)
        * (4.0 * y - 1.0)
        * (-13.5 * x**2 + 18.0 * x - 4.5)
    )


def phi_8(x: float, y: float) -> float:
    return (
        x
        * y
        * (-13.5 * x**2 + 18.0 * x - 4.5)
        * (
            -42.6666666666667 * y**3
            + 74.6666666666667 * y**2
            - 37.3333333333333 * y
            + 5.33333333333333
        )
    )


def phi_9(x: float, y: float) -> float:
    return (
        x
        * y
        * (4.5 * x**2 - 4.5 * x + 1.0)
        * (-42.6666666666667 * y**3 + 96.0 * y**2 - 69.3333333333333 * y + 16.0)
    )


def phi_10(x: float, y: float) -> float:
    return (
        4.0
        * x
        * y
        * (y - 1)
        * (4.0 * y - 3.0)
        * (4.0 * y - 1.0)
        * (4.5 * x**2 - 4.5 * x + 1.0)
    )


def phi_11(x: float, y: float) -> float:
    return (
        x
        * y
        * (4.5 * x**2 - 4.5 * x + 1.0)
        * (
            -42.6666666666667 * y**3
            + 74.6666666666667 * y**2
            - 37.3333333333333 * y
            + 5.33333333333333
        )
    )


def grad_phi_0(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = (
        y
        * (-13.5 * x**2 + 18.0 * x - 5.5)
        * (-42.6666666666667 * y**3 + 96.0 * y**2 - 69.3333333333333 * y + 16.0)
    )
    phi_dy = y * (-128.0 * y**2 + 192.0 * y - 69.3333333333333) * (
        -4.5 * x**3 + 9.0 * x**2 - 5.5 * x + 1.0
    ) + (-4.5 * x**3 + 9.0 * x**2 - 5.5 * x + 1.0) * (
        -42.6666666666667 * y**3 + 96.0 * y**2 - 69.3333333333333 * y + 16.0
    )
    return np.array([phi_dx, phi_dy])


def grad_phi_1(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = (
        4.0
        * y
        * (y - 1)
        * (4.0 * y - 3.0)
        * (4.0 * y - 1.0)
        * (-13.5 * x**2 + 18.0 * x - 5.5)
    )
    phi_dy = (
        16.0
        * y
        * (y - 1)
        * (4.0 * y - 3.0)
        * (-4.5 * x**3 + 9.0 * x**2 - 5.5 * x + 1.0)
        + 16.0
        * y
        * (y - 1)
        * (4.0 * y - 1.0)
        * (-4.5 * x**3 + 9.0 * x**2 - 5.5 * x + 1.0)
        + 4.0
        * y
        * (4.0 * y - 3.0)
        * (4.0 * y - 1.0)
        * (-4.5 * x**3 + 9.0 * x**2 - 5.5 * x + 1.0)
        + 4.0
        * (y - 1)
        * (4.0 * y - 3.0)
        * (4.0 * y - 1.0)
        * (-4.5 * x**3 + 9.0 * x**2 - 5.5 * x + 1.0)
    )
    return np.array([phi_dx, phi_dy])


def grad_phi_2(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = (
        y
        * (-13.5 * x**2 + 18.0 * x - 5.5)
        * (
            -42.6666666666667 * y**3
            + 74.6666666666667 * y**2
            - 37.3333333333333 * y
            + 5.33333333333333
        )
    )
    phi_dy = y * (-128.0 * y**2 + 149.333333333333 * y - 37.3333333333333) * (
        -4.5 * x**3 + 9.0 * x**2 - 5.5 * x + 1.0
    ) + (-4.5 * x**3 + 9.0 * x**2 - 5.5 * x + 1.0) * (
        -42.6666666666667 * y**3
        + 74.6666666666667 * y**2
        - 37.3333333333333 * y
        + 5.33333333333333
    )
    return np.array([phi_dx, phi_dy])


def grad_phi_3(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = (
        13.5
        * x
        * y
        * (x - 1)
        * (-42.6666666666667 * y**3 + 96.0 * y**2 - 69.3333333333333 * y + 16.0)
        + 4.5
        * x
        * y
        * (3.0 * x - 2.0)
        * (-42.6666666666667 * y**3 + 96.0 * y**2 - 69.3333333333333 * y + 16.0)
        + 4.5
        * y
        * (x - 1)
        * (3.0 * x - 2.0)
        * (-42.6666666666667 * y**3 + 96.0 * y**2 - 69.3333333333333 * y + 16.0)
    )
    phi_dy = 4.5 * x * y * (x - 1) * (3.0 * x - 2.0) * (
        -128.0 * y**2 + 192.0 * y - 69.3333333333333
    ) + 4.5 * x * (x - 1) * (3.0 * x - 2.0) * (
        -42.6666666666667 * y**3 + 96.0 * y**2 - 69.3333333333333 * y + 16.0
    )
    return np.array([phi_dx, phi_dy])


def grad_phi_4(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = (
        54.0 * x * y * (x - 1) * (y - 1) * (4.0 * y - 3.0) * (4.0 * y - 1.0)
        + 18.0 * x * y * (3.0 * x - 2.0) * (y - 1) * (4.0 * y - 3.0) * (4.0 * y - 1.0)
        + 18.0
        * y
        * (x - 1)
        * (3.0 * x - 2.0)
        * (y - 1)
        * (4.0 * y - 3.0)
        * (4.0 * y - 1.0)
    )
    phi_dy = (
        72.0 * x * y * (x - 1) * (3.0 * x - 2.0) * (y - 1) * (4.0 * y - 3.0)
        + 72.0 * x * y * (x - 1) * (3.0 * x - 2.0) * (y - 1) * (4.0 * y - 1.0)
        + 18.0 * x * y * (x - 1) * (3.0 * x - 2.0) * (4.0 * y - 3.0) * (4.0 * y - 1.0)
        + 18.0
        * x
        * (x - 1)
        * (3.0 * x - 2.0)
        * (y - 1)
        * (4.0 * y - 3.0)
        * (4.0 * y - 1.0)
    )
    return np.array([phi_dx, phi_dy])


def grad_phi_5(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = (
        13.5
        * x
        * y
        * (x - 1)
        * (
            -42.6666666666667 * y**3
            + 74.6666666666667 * y**2
            - 37.3333333333333 * y
            + 5.33333333333333
        )
        + 4.5
        * x
        * y
        * (3.0 * x - 2.0)
        * (
            -42.6666666666667 * y**3
            + 74.6666666666667 * y**2
            - 37.3333333333333 * y
            + 5.33333333333333
        )
        + 4.5
        * y
        * (x - 1)
        * (3.0 * x - 2.0)
        * (
            -42.6666666666667 * y**3
            + 74.6666666666667 * y**2
            - 37.3333333333333 * y
            + 5.33333333333333
        )
    )
    phi_dy = 4.5 * x * y * (x - 1) * (3.0 * x - 2.0) * (
        -128.0 * y**2 + 149.333333333333 * y - 37.3333333333333
    ) + 4.5 * x * (x - 1) * (3.0 * x - 2.0) * (
        -42.6666666666667 * y**3
        + 74.6666666666667 * y**2
        - 37.3333333333333 * y
        + 5.33333333333333
    )
    return np.array([phi_dx, phi_dy])


def grad_phi_6(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = x * y * (18.0 - 27.0 * x) * (
        -42.6666666666667 * y**3 + 96.0 * y**2 - 69.3333333333333 * y + 16.0
    ) + y * (-13.5 * x**2 + 18.0 * x - 4.5) * (
        -42.6666666666667 * y**3 + 96.0 * y**2 - 69.3333333333333 * y + 16.0
    )
    phi_dy = x * y * (-13.5 * x**2 + 18.0 * x - 4.5) * (
        -128.0 * y**2 + 192.0 * y - 69.3333333333333
    ) + x * (-13.5 * x**2 + 18.0 * x - 4.5) * (
        -42.6666666666667 * y**3 + 96.0 * y**2 - 69.3333333333333 * y + 16.0
    )
    return np.array([phi_dx, phi_dy])


def grad_phi_7(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = 4.0 * x * y * (18.0 - 27.0 * x) * (y - 1) * (4.0 * y - 3.0) * (
        4.0 * y - 1.0
    ) + 4.0 * y * (y - 1) * (4.0 * y - 3.0) * (4.0 * y - 1.0) * (
        -13.5 * x**2 + 18.0 * x - 4.5
    )
    phi_dy = (
        16.0 * x * y * (y - 1) * (4.0 * y - 3.0) * (-13.5 * x**2 + 18.0 * x - 4.5)
        + 16.0 * x * y * (y - 1) * (4.0 * y - 1.0) * (-13.5 * x**2 + 18.0 * x - 4.5)
        + 4.0
        * x
        * y
        * (4.0 * y - 3.0)
        * (4.0 * y - 1.0)
        * (-13.5 * x**2 + 18.0 * x - 4.5)
        + 4.0
        * x
        * (y - 1)
        * (4.0 * y - 3.0)
        * (4.0 * y - 1.0)
        * (-13.5 * x**2 + 18.0 * x - 4.5)
    )
    return np.array([phi_dx, phi_dy])


def grad_phi_8(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = x * y * (18.0 - 27.0 * x) * (
        -42.6666666666667 * y**3
        + 74.6666666666667 * y**2
        - 37.3333333333333 * y
        + 5.33333333333333
    ) + y * (-13.5 * x**2 + 18.0 * x - 4.5) * (
        -42.6666666666667 * y**3
        + 74.6666666666667 * y**2
        - 37.3333333333333 * y
        + 5.33333333333333
    )
    phi_dy = x * y * (-13.5 * x**2 + 18.0 * x - 4.5) * (
        -128.0 * y**2 + 149.333333333333 * y - 37.3333333333333
    ) + x * (-13.5 * x**2 + 18.0 * x - 4.5) * (
        -42.6666666666667 * y**3
        + 74.6666666666667 * y**2
        - 37.3333333333333 * y
        + 5.33333333333333
    )
    return np.array([phi_dx, phi_dy])


def grad_phi_9(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = x * y * (9.0 * x - 4.5) * (
        -42.6666666666667 * y**3 + 96.0 * y**2 - 69.3333333333333 * y + 16.0
    ) + y * (4.5 * x**2 - 4.5 * x + 1.0) * (
        -42.6666666666667 * y**3 + 96.0 * y**2 - 69.3333333333333 * y + 16.0
    )
    phi_dy = x * y * (4.5 * x**2 - 4.5 * x + 1.0) * (
        -128.0 * y**2 + 192.0 * y - 69.3333333333333
    ) + x * (4.5 * x**2 - 4.5 * x + 1.0) * (
        -42.6666666666667 * y**3 + 96.0 * y**2 - 69.3333333333333 * y + 16.0
    )
    return np.array([phi_dx, phi_dy])


def grad_phi_10(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = 4.0 * x * y * (9.0 * x - 4.5) * (y - 1) * (4.0 * y - 3.0) * (
        4.0 * y - 1.0
    ) + 4.0 * y * (y - 1) * (4.0 * y - 3.0) * (4.0 * y - 1.0) * (
        4.5 * x**2 - 4.5 * x + 1.0
    )
    phi_dy = (
        16.0 * x * y * (y - 1) * (4.0 * y - 3.0) * (4.5 * x**2 - 4.5 * x + 1.0)
        + 16.0 * x * y * (y - 1) * (4.0 * y - 1.0) * (4.5 * x**2 - 4.5 * x + 1.0)
        + 4.0
        * x
        * y
        * (4.0 * y - 3.0)
        * (4.0 * y - 1.0)
        * (4.5 * x**2 - 4.5 * x + 1.0)
        + 4.0
        * x
        * (y - 1)
        * (4.0 * y - 3.0)
        * (4.0 * y - 1.0)
        * (4.5 * x**2 - 4.5 * x + 1.0)
    )
    return np.array([phi_dx, phi_dy])


def grad_phi_11(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = x * y * (9.0 * x - 4.5) * (
        -42.6666666666667 * y**3
        + 74.6666666666667 * y**2
        - 37.3333333333333 * y
        + 5.33333333333333
    ) + y * (4.5 * x**2 - 4.5 * x + 1.0) * (
        -42.6666666666667 * y**3
        + 74.6666666666667 * y**2
        - 37.3333333333333 * y
        + 5.33333333333333
    )
    phi_dy = x * y * (4.5 * x**2 - 4.5 * x + 1.0) * (
        -128.0 * y**2 + 149.333333333333 * y - 37.3333333333333
    ) + x * (4.5 * x**2 - 4.5 * x + 1.0) * (
        -42.6666666666667 * y**3
        + 74.6666666666667 * y**2
        - 37.3333333333333 * y
        + 5.33333333333333
    )
    return np.array([phi_dx, phi_dy])


Q34_SF_LIST = [
    phi_0,
    phi_1,
    phi_2,
    phi_3,
    phi_4,
    phi_5,
    phi_6,
    phi_7,
    phi_8,
    phi_9,
    phi_10,
    phi_11,
]

Q34_GRAD_SF_LIST = [
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
    grad_phi_10,
    grad_phi_11,
]
