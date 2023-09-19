import numpy as np
import numpy.typing as npt


def phi_0(x: float, y: float) -> float:
    return (
        6.25
        * y
        * (1.0 - 1.0 * x)
        * (y - 1)
        * (1.66666666666667 * y - 1.33333333333333)
        * (2.5 * y - 1.5)
        * (5.0 * y - 2.0)
    )


def phi_1(x: float, y: float) -> float:
    return (
        y
        * (1.0 - 1.0 * x)
        * (
            -260.416666666667 * y**4
            + 677.083333333333 * y**3
            - 614.583333333333 * y**2
            + 222.916666666667 * y
            - 25.0
        )
    )


def phi_2(x: float, y: float) -> float:
    return (
        4.16666666666667
        * y
        * (1.0 - 1.0 * x)
        * (y - 1)
        * (2.5 * y - 0.5)
        * (5.0 * y - 2.0)
        * (5.0 * y - 4.0)
    )


def phi_3(x: float, y: float) -> float:
    return (
        y
        * (1.0 - 1.0 * x)
        * (
            -130.208333333333 * y**4
            + 286.458333333333 * y**3
            - 213.541666666667 * y**2
            + 63.5416666666667 * y
            - 6.25
        )
    )


def phi_4(x: float, y: float) -> float:
    return (
        6.25
        * x
        * y
        * (y - 1)
        * (1.66666666666667 * y - 1.33333333333333)
        * (2.5 * y - 1.5)
        * (5.0 * y - 2.0)
    )


def phi_5(x: float, y: float) -> float:
    return (
        1.0
        * x
        * y
        * (
            -260.416666666667 * y**4
            + 677.083333333333 * y**3
            - 614.583333333333 * y**2
            + 222.916666666667 * y
            - 25.0
        )
    )


def phi_6(x: float, y: float) -> float:
    return (
        4.16666666666667
        * x
        * y
        * (y - 1)
        * (2.5 * y - 0.5)
        * (5.0 * y - 2.0)
        * (5.0 * y - 4.0)
    )


def phi_7(x: float, y: float) -> float:
    return (
        1.0
        * x
        * y
        * (
            -130.208333333333 * y**4
            + 286.458333333333 * y**3
            - 213.541666666667 * y**2
            + 63.5416666666667 * y
            - 6.25
        )
    )


def grad_phi_0(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = (
        -6.25
        * y
        * (y - 1)
        * (1.66666666666667 * y - 1.33333333333333)
        * (2.5 * y - 1.5)
        * (5.0 * y - 2.0)
    )
    phi_dy = (
        31.25
        * y
        * (1.0 - 1.0 * x)
        * (y - 1)
        * (1.66666666666667 * y - 1.33333333333333)
        * (2.5 * y - 1.5)
        + 15.625
        * y
        * (1.0 - 1.0 * x)
        * (y - 1)
        * (1.66666666666667 * y - 1.33333333333333)
        * (5.0 * y - 2.0)
        + 10.4166666666667
        * y
        * (1.0 - 1.0 * x)
        * (y - 1)
        * (2.5 * y - 1.5)
        * (5.0 * y - 2.0)
        + 6.25
        * y
        * (1.0 - 1.0 * x)
        * (1.66666666666667 * y - 1.33333333333333)
        * (2.5 * y - 1.5)
        * (5.0 * y - 2.0)
        + 6.25
        * (1.0 - 1.0 * x)
        * (y - 1)
        * (1.66666666666667 * y - 1.33333333333333)
        * (2.5 * y - 1.5)
        * (5.0 * y - 2.0)
    )
    return np.array([phi_dx, phi_dy])


def grad_phi_1(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = (
        -1.0
        * y
        * (
            -260.416666666667 * y**4
            + 677.083333333333 * y**3
            - 614.583333333333 * y**2
            + 222.916666666667 * y
            - 25.0
        )
    )
    phi_dy = y * (1.0 - 1.0 * x) * (
        -1041.66666666667 * y**3
        + 2031.25 * y**2
        - 1229.16666666667 * y
        + 222.916666666667
    ) + (1.0 - 1.0 * x) * (
        -260.416666666667 * y**4
        + 677.083333333333 * y**3
        - 614.583333333333 * y**2
        + 222.916666666667 * y
        - 25.0
    )
    return np.array([phi_dx, phi_dy])


def grad_phi_2(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = (
        -4.16666666666667
        * y
        * (y - 1)
        * (2.5 * y - 0.5)
        * (5.0 * y - 2.0)
        * (5.0 * y - 4.0)
    )
    phi_dy = (
        20.8333333333333
        * y
        * (1.0 - 1.0 * x)
        * (y - 1)
        * (2.5 * y - 0.5)
        * (5.0 * y - 2.0)
        + 20.8333333333333
        * y
        * (1.0 - 1.0 * x)
        * (y - 1)
        * (2.5 * y - 0.5)
        * (5.0 * y - 4.0)
        + 10.4166666666667
        * y
        * (1.0 - 1.0 * x)
        * (y - 1)
        * (5.0 * y - 2.0)
        * (5.0 * y - 4.0)
        + 4.16666666666667
        * y
        * (1.0 - 1.0 * x)
        * (2.5 * y - 0.5)
        * (5.0 * y - 2.0)
        * (5.0 * y - 4.0)
        + 4.16666666666667
        * (1.0 - 1.0 * x)
        * (y - 1)
        * (2.5 * y - 0.5)
        * (5.0 * y - 2.0)
        * (5.0 * y - 4.0)
    )
    return np.array([phi_dx, phi_dy])


def grad_phi_3(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = (
        -1.0
        * y
        * (
            -130.208333333333 * y**4
            + 286.458333333333 * y**3
            - 213.541666666667 * y**2
            + 63.5416666666667 * y
            - 6.25
        )
    )
    phi_dy = y * (1.0 - 1.0 * x) * (
        -520.833333333333 * y**3
        + 859.375 * y**2
        - 427.083333333333 * y
        + 63.5416666666667
    ) + (1.0 - 1.0 * x) * (
        -130.208333333333 * y**4
        + 286.458333333333 * y**3
        - 213.541666666667 * y**2
        + 63.5416666666667 * y
        - 6.25
    )
    return np.array([phi_dx, phi_dy])


def grad_phi_4(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = (
        6.25
        * y
        * (y - 1)
        * (1.66666666666667 * y - 1.33333333333333)
        * (2.5 * y - 1.5)
        * (5.0 * y - 2.0)
    )
    phi_dy = (
        31.25
        * x
        * y
        * (y - 1)
        * (1.66666666666667 * y - 1.33333333333333)
        * (2.5 * y - 1.5)
        + 15.625
        * x
        * y
        * (y - 1)
        * (1.66666666666667 * y - 1.33333333333333)
        * (5.0 * y - 2.0)
        + 10.4166666666667 * x * y * (y - 1) * (2.5 * y - 1.5) * (5.0 * y - 2.0)
        + 6.25
        * x
        * y
        * (1.66666666666667 * y - 1.33333333333333)
        * (2.5 * y - 1.5)
        * (5.0 * y - 2.0)
        + 6.25
        * x
        * (y - 1)
        * (1.66666666666667 * y - 1.33333333333333)
        * (2.5 * y - 1.5)
        * (5.0 * y - 2.0)
    )
    return np.array([phi_dx, phi_dy])


def grad_phi_5(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = (
        1.0
        * y
        * (
            -260.416666666667 * y**4
            + 677.083333333333 * y**3
            - 614.583333333333 * y**2
            + 222.916666666667 * y
            - 25.0
        )
    )
    phi_dy = 1.0 * x * y * (
        -1041.66666666667 * y**3
        + 2031.25 * y**2
        - 1229.16666666667 * y
        + 222.916666666667
    ) + 1.0 * x * (
        -260.416666666667 * y**4
        + 677.083333333333 * y**3
        - 614.583333333333 * y**2
        + 222.916666666667 * y
        - 25.0
    )
    return np.array([phi_dx, phi_dy])


def grad_phi_6(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = (
        4.16666666666667
        * y
        * (y - 1)
        * (2.5 * y - 0.5)
        * (5.0 * y - 2.0)
        * (5.0 * y - 4.0)
    )
    phi_dy = (
        20.8333333333333 * x * y * (y - 1) * (2.5 * y - 0.5) * (5.0 * y - 2.0)
        + 20.8333333333333 * x * y * (y - 1) * (2.5 * y - 0.5) * (5.0 * y - 4.0)
        + 10.4166666666667 * x * y * (y - 1) * (5.0 * y - 2.0) * (5.0 * y - 4.0)
        + 4.16666666666667 * x * y * (2.5 * y - 0.5) * (5.0 * y - 2.0) * (5.0 * y - 4.0)
        + 4.16666666666667
        * x
        * (y - 1)
        * (2.5 * y - 0.5)
        * (5.0 * y - 2.0)
        * (5.0 * y - 4.0)
    )
    return np.array([phi_dx, phi_dy])


def grad_phi_7(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = (
        1.0
        * y
        * (
            -130.208333333333 * y**4
            + 286.458333333333 * y**3
            - 213.541666666667 * y**2
            + 63.5416666666667 * y
            - 6.25
        )
    )
    phi_dy = 1.0 * x * y * (
        -520.833333333333 * y**3
        + 859.375 * y**2
        - 427.083333333333 * y
        + 63.5416666666667
    ) + 1.0 * x * (
        -130.208333333333 * y**4
        + 286.458333333333 * y**3
        - 213.541666666667 * y**2
        + 63.5416666666667 * y
        - 6.25
    )
    return np.array([phi_dx, phi_dy])


Q15_SF_LIST = [phi_0, phi_1, phi_2, phi_3, phi_4, phi_5, phi_6, phi_7]

Q15_GRAD_SF_LIST = [
    grad_phi_0,
    grad_phi_1,
    grad_phi_2,
    grad_phi_3,
    grad_phi_4,
    grad_phi_5,
    grad_phi_6,
    grad_phi_7,
]