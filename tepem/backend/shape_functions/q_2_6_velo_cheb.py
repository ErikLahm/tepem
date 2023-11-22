import numpy as np
import numpy.typing as npt


def phi_0(x: float, y: float) -> float:
    return (
        1.0
        * y
        * (x - 1)
        * (2.0 * x - 1.0)
        * (
            -341.333333333334 * y**5
            + 1171.80166891254 * y**4
            - 1542.83750561469 * y**3
            + 963.316150746191 * y**2
            - 280.803387171259 * y
            + 29.856406460551
        )
    )


def phi_1(x: float, y: float) -> float:
    return (
        5.33333333333333
        * y
        * (x - 1)
        * (2.0 * x - 1.0)
        * (y - 1)
        * (1.46410161513775 * y - 1.36602540378444)
        * (2.0 * y - 1.5)
        * (4.0 * y - 2.0)
        * (5.46410161513776 * y - 0.366025403784439)
    )


def phi_2(x: float, y: float) -> float:
    return (
        1.0
        * y
        * (x - 1)
        * (2.0 * x - 1.0)
        * (
            -341.333333333333 * y**5
            + 1024.0 * y**4
            - 1109.33333333333 * y**3
            + 512.0 * y**2
            - 89.3333333333333 * y
            + 4.0
        )
    )


def phi_3(x: float, y: float) -> float:
    return (
        5.33333333333333
        * y
        * (x - 1)
        * (2.0 * x - 1.0)
        * (y - 1)
        * (1.46410161513775 * y - 0.0980762113533159)
        * (2.0 * y - 0.5)
        * (4.0 * y - 2.0)
        * (5.46410161513775 * y - 5.09807621135331)
    )


def phi_4(x: float, y: float) -> float:
    return (
        1.0
        * y
        * (x - 1)
        * (2.0 * x - 1.0)
        * (
            -341.333333333333 * y**5
            + 876.198331087455 * y**4
            - 803.829161051972 * y**3
            + 316.683849253809 * y**2
            - 49.8632794954081 * y
            + 2.14359353944898
        )
    )


def phi_5(x: float, y: float) -> float:
    return (
        4.0
        * x
        * y
        * (1 - x)
        * (
            -341.333333333334 * y**5
            + 1171.80166891254 * y**4
            - 1542.83750561469 * y**3
            + 963.316150746191 * y**2
            - 280.803387171259 * y
            + 29.856406460551
        )
    )


def phi_6(x: float, y: float) -> float:
    return (
        21.3333333333333
        * x
        * y
        * (1 - x)
        * (y - 1)
        * (1.46410161513775 * y - 1.36602540378444)
        * (2.0 * y - 1.5)
        * (4.0 * y - 2.0)
        * (5.46410161513776 * y - 0.366025403784439)
    )


def phi_7(x: float, y: float) -> float:
    return (
        4.0
        * x
        * y
        * (1 - x)
        * (
            -341.333333333333 * y**5
            + 1024.0 * y**4
            - 1109.33333333333 * y**3
            + 512.0 * y**2
            - 89.3333333333333 * y
            + 4.0
        )
    )


def phi_8(x: float, y: float) -> float:
    return (
        21.3333333333333
        * x
        * y
        * (1 - x)
        * (y - 1)
        * (1.46410161513775 * y - 0.0980762113533159)
        * (2.0 * y - 0.5)
        * (4.0 * y - 2.0)
        * (5.46410161513775 * y - 5.09807621135331)
    )


def phi_9(x: float, y: float) -> float:
    return (
        4.0
        * x
        * y
        * (1 - x)
        * (
            -341.333333333333 * y**5
            + 876.198331087455 * y**4
            - 803.829161051972 * y**3
            + 316.683849253809 * y**2
            - 49.8632794954081 * y
            + 2.14359353944898
        )
    )


def phi_10(x: float, y: float) -> float:
    return (
        x
        * y
        * (2.0 * x - 1.0)
        * (
            -341.333333333334 * y**5
            + 1171.80166891254 * y**4
            - 1542.83750561469 * y**3
            + 963.316150746191 * y**2
            - 280.803387171259 * y
            + 29.856406460551
        )
    )


def phi_11(x: float, y: float) -> float:
    return (
        5.33333333333333
        * x
        * y
        * (2.0 * x - 1.0)
        * (y - 1)
        * (1.46410161513775 * y - 1.36602540378444)
        * (2.0 * y - 1.5)
        * (4.0 * y - 2.0)
        * (5.46410161513776 * y - 0.366025403784439)
    )


def phi_12(x: float, y: float) -> float:
    return (
        x
        * y
        * (2.0 * x - 1.0)
        * (
            -341.333333333333 * y**5
            + 1024.0 * y**4
            - 1109.33333333333 * y**3
            + 512.0 * y**2
            - 89.3333333333333 * y
            + 4.0
        )
    )


def phi_13(x: float, y: float) -> float:
    return (
        5.33333333333333
        * x
        * y
        * (2.0 * x - 1.0)
        * (y - 1)
        * (1.46410161513775 * y - 0.0980762113533159)
        * (2.0 * y - 0.5)
        * (4.0 * y - 2.0)
        * (5.46410161513775 * y - 5.09807621135331)
    )


def phi_14(x: float, y: float) -> float:
    return (
        x
        * y
        * (2.0 * x - 1.0)
        * (
            -341.333333333333 * y**5
            + 876.198331087455 * y**4
            - 803.829161051972 * y**3
            + 316.683849253809 * y**2
            - 49.8632794954081 * y
            + 2.14359353944898
        )
    )


def grad_phi_0(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = 2.0 * y * (x - 1) * (
        -341.333333333334 * y**5
        + 1171.80166891254 * y**4
        - 1542.83750561469 * y**3
        + 963.316150746191 * y**2
        - 280.803387171259 * y
        + 29.856406460551
    ) + 1.0 * y * (2.0 * x - 1.0) * (
        -341.333333333334 * y**5
        + 1171.80166891254 * y**4
        - 1542.83750561469 * y**3
        + 963.316150746191 * y**2
        - 280.803387171259 * y
        + 29.856406460551
    )
    phi_dy = 1.0 * y * (x - 1) * (2.0 * x - 1.0) * (
        -1706.66666666667 * y**4
        + 4687.20667565018 * y**3
        - 4628.51251684408 * y**2
        + 1926.63230149238 * y
        - 280.803387171259
    ) + 1.0 * (x - 1) * (2.0 * x - 1.0) * (
        -341.333333333334 * y**5
        + 1171.80166891254 * y**4
        - 1542.83750561469 * y**3
        + 963.316150746191 * y**2
        - 280.803387171259 * y
        + 29.856406460551
    )
    return np.array([phi_dx, phi_dy])


def grad_phi_1(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = 10.6666666666667 * y * (x - 1) * (y - 1) * (
        1.46410161513775 * y - 1.36602540378444
    ) * (2.0 * y - 1.5) * (4.0 * y - 2.0) * (
        5.46410161513776 * y - 0.366025403784439
    ) + 5.33333333333333 * y * (
        2.0 * x - 1.0
    ) * (
        y - 1
    ) * (
        1.46410161513775 * y - 1.36602540378444
    ) * (
        2.0 * y - 1.5
    ) * (
        4.0 * y - 2.0
    ) * (
        5.46410161513776 * y - 0.366025403784439
    )
    phi_dy = (
        29.1418752807347
        * y
        * (x - 1)
        * (2.0 * x - 1.0)
        * (y - 1)
        * (1.46410161513775 * y - 1.36602540378444)
        * (2.0 * y - 1.5)
        * (4.0 * y - 2.0)
        + 21.3333333333333
        * y
        * (x - 1)
        * (2.0 * x - 1.0)
        * (y - 1)
        * (1.46410161513775 * y - 1.36602540378444)
        * (2.0 * y - 1.5)
        * (5.46410161513776 * y - 0.366025403784439)
        + 10.6666666666667
        * y
        * (x - 1)
        * (2.0 * x - 1.0)
        * (y - 1)
        * (1.46410161513775 * y - 1.36602540378444)
        * (4.0 * y - 2.0)
        * (5.46410161513776 * y - 0.366025403784439)
        + 7.80854194740136
        * y
        * (x - 1)
        * (2.0 * x - 1.0)
        * (y - 1)
        * (2.0 * y - 1.5)
        * (4.0 * y - 2.0)
        * (5.46410161513776 * y - 0.366025403784439)
        + 5.33333333333333
        * y
        * (x - 1)
        * (2.0 * x - 1.0)
        * (1.46410161513775 * y - 1.36602540378444)
        * (2.0 * y - 1.5)
        * (4.0 * y - 2.0)
        * (5.46410161513776 * y - 0.366025403784439)
        + 5.33333333333333
        * (x - 1)
        * (2.0 * x - 1.0)
        * (y - 1)
        * (1.46410161513775 * y - 1.36602540378444)
        * (2.0 * y - 1.5)
        * (4.0 * y - 2.0)
        * (5.46410161513776 * y - 0.366025403784439)
    )
    return np.array([phi_dx, phi_dy])


def grad_phi_2(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = 2.0 * y * (x - 1) * (
        -341.333333333333 * y**5
        + 1024.0 * y**4
        - 1109.33333333333 * y**3
        + 512.0 * y**2
        - 89.3333333333333 * y
        + 4.0
    ) + 1.0 * y * (2.0 * x - 1.0) * (
        -341.333333333333 * y**5
        + 1024.0 * y**4
        - 1109.33333333333 * y**3
        + 512.0 * y**2
        - 89.3333333333333 * y
        + 4.0
    )
    phi_dy = 1.0 * y * (x - 1) * (2.0 * x - 1.0) * (
        -1706.66666666667 * y**4
        + 4096.0 * y**3
        - 3328.0 * y**2
        + 1024.0 * y
        - 89.3333333333333
    ) + 1.0 * (x - 1) * (2.0 * x - 1.0) * (
        -341.333333333333 * y**5
        + 1024.0 * y**4
        - 1109.33333333333 * y**3
        + 512.0 * y**2
        - 89.3333333333333 * y
        + 4.0
    )
    return np.array([phi_dx, phi_dy])


def grad_phi_3(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = 10.6666666666667 * y * (x - 1) * (y - 1) * (
        1.46410161513775 * y - 0.0980762113533159
    ) * (2.0 * y - 0.5) * (4.0 * y - 2.0) * (
        5.46410161513775 * y - 5.09807621135331
    ) + 5.33333333333333 * y * (
        2.0 * x - 1.0
    ) * (
        y - 1
    ) * (
        1.46410161513775 * y - 0.0980762113533159
    ) * (
        2.0 * y - 0.5
    ) * (
        4.0 * y - 2.0
    ) * (
        5.46410161513775 * y - 5.09807621135331
    )
    phi_dy = (
        29.1418752807346
        * y
        * (x - 1)
        * (2.0 * x - 1.0)
        * (y - 1)
        * (1.46410161513775 * y - 0.0980762113533159)
        * (2.0 * y - 0.5)
        * (4.0 * y - 2.0)
        + 21.3333333333333
        * y
        * (x - 1)
        * (2.0 * x - 1.0)
        * (y - 1)
        * (1.46410161513775 * y - 0.0980762113533159)
        * (2.0 * y - 0.5)
        * (5.46410161513775 * y - 5.09807621135331)
        + 10.6666666666667
        * y
        * (x - 1)
        * (2.0 * x - 1.0)
        * (y - 1)
        * (1.46410161513775 * y - 0.0980762113533159)
        * (4.0 * y - 2.0)
        * (5.46410161513775 * y - 5.09807621135331)
        + 7.80854194740136
        * y
        * (x - 1)
        * (2.0 * x - 1.0)
        * (y - 1)
        * (2.0 * y - 0.5)
        * (4.0 * y - 2.0)
        * (5.46410161513775 * y - 5.09807621135331)
        + 5.33333333333333
        * y
        * (x - 1)
        * (2.0 * x - 1.0)
        * (1.46410161513775 * y - 0.0980762113533159)
        * (2.0 * y - 0.5)
        * (4.0 * y - 2.0)
        * (5.46410161513775 * y - 5.09807621135331)
        + 5.33333333333333
        * (x - 1)
        * (2.0 * x - 1.0)
        * (y - 1)
        * (1.46410161513775 * y - 0.0980762113533159)
        * (2.0 * y - 0.5)
        * (4.0 * y - 2.0)
        * (5.46410161513775 * y - 5.09807621135331)
    )
    return np.array([phi_dx, phi_dy])


def grad_phi_4(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = 2.0 * y * (x - 1) * (
        -341.333333333333 * y**5
        + 876.198331087455 * y**4
        - 803.829161051972 * y**3
        + 316.683849253809 * y**2
        - 49.8632794954081 * y
        + 2.14359353944898
    ) + 1.0 * y * (2.0 * x - 1.0) * (
        -341.333333333333 * y**5
        + 876.198331087455 * y**4
        - 803.829161051972 * y**3
        + 316.683849253809 * y**2
        - 49.8632794954081 * y
        + 2.14359353944898
    )
    phi_dy = 1.0 * y * (x - 1) * (2.0 * x - 1.0) * (
        -1706.66666666667 * y**4
        + 3504.79332434982 * y**3
        - 2411.48748315592 * y**2
        + 633.367698507619 * y
        - 49.8632794954081
    ) + 1.0 * (x - 1) * (2.0 * x - 1.0) * (
        -341.333333333333 * y**5
        + 876.198331087455 * y**4
        - 803.829161051972 * y**3
        + 316.683849253809 * y**2
        - 49.8632794954081 * y
        + 2.14359353944898
    )
    return np.array([phi_dx, phi_dy])


def grad_phi_5(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = -4.0 * x * y * (
        -341.333333333334 * y**5
        + 1171.80166891254 * y**4
        - 1542.83750561469 * y**3
        + 963.316150746191 * y**2
        - 280.803387171259 * y
        + 29.856406460551
    ) + 4.0 * y * (1 - x) * (
        -341.333333333334 * y**5
        + 1171.80166891254 * y**4
        - 1542.83750561469 * y**3
        + 963.316150746191 * y**2
        - 280.803387171259 * y
        + 29.856406460551
    )
    phi_dy = 4.0 * x * y * (1 - x) * (
        -1706.66666666667 * y**4
        + 4687.20667565018 * y**3
        - 4628.51251684408 * y**2
        + 1926.63230149238 * y
        - 280.803387171259
    ) + 4.0 * x * (1 - x) * (
        -341.333333333334 * y**5
        + 1171.80166891254 * y**4
        - 1542.83750561469 * y**3
        + 963.316150746191 * y**2
        - 280.803387171259 * y
        + 29.856406460551
    )
    return np.array([phi_dx, phi_dy])


def grad_phi_6(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = -21.3333333333333 * x * y * (y - 1) * (
        1.46410161513775 * y - 1.36602540378444
    ) * (2.0 * y - 1.5) * (4.0 * y - 2.0) * (
        5.46410161513776 * y - 0.366025403784439
    ) + 21.3333333333333 * y * (
        1 - x
    ) * (
        y - 1
    ) * (
        1.46410161513775 * y - 1.36602540378444
    ) * (
        2.0 * y - 1.5
    ) * (
        4.0 * y - 2.0
    ) * (
        5.46410161513776 * y - 0.366025403784439
    )
    phi_dy = (
        116.567501122939
        * x
        * y
        * (1 - x)
        * (y - 1)
        * (1.46410161513775 * y - 1.36602540378444)
        * (2.0 * y - 1.5)
        * (4.0 * y - 2.0)
        + 85.3333333333334
        * x
        * y
        * (1 - x)
        * (y - 1)
        * (1.46410161513775 * y - 1.36602540378444)
        * (2.0 * y - 1.5)
        * (5.46410161513776 * y - 0.366025403784439)
        + 42.6666666666667
        * x
        * y
        * (1 - x)
        * (y - 1)
        * (1.46410161513775 * y - 1.36602540378444)
        * (4.0 * y - 2.0)
        * (5.46410161513776 * y - 0.366025403784439)
        + 31.2341677896054
        * x
        * y
        * (1 - x)
        * (y - 1)
        * (2.0 * y - 1.5)
        * (4.0 * y - 2.0)
        * (5.46410161513776 * y - 0.366025403784439)
        + 21.3333333333333
        * x
        * y
        * (1 - x)
        * (1.46410161513775 * y - 1.36602540378444)
        * (2.0 * y - 1.5)
        * (4.0 * y - 2.0)
        * (5.46410161513776 * y - 0.366025403784439)
        + 21.3333333333333
        * x
        * (1 - x)
        * (y - 1)
        * (1.46410161513775 * y - 1.36602540378444)
        * (2.0 * y - 1.5)
        * (4.0 * y - 2.0)
        * (5.46410161513776 * y - 0.366025403784439)
    )
    return np.array([phi_dx, phi_dy])


def grad_phi_7(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = -4.0 * x * y * (
        -341.333333333333 * y**5
        + 1024.0 * y**4
        - 1109.33333333333 * y**3
        + 512.0 * y**2
        - 89.3333333333333 * y
        + 4.0
    ) + 4.0 * y * (1 - x) * (
        -341.333333333333 * y**5
        + 1024.0 * y**4
        - 1109.33333333333 * y**3
        + 512.0 * y**2
        - 89.3333333333333 * y
        + 4.0
    )
    phi_dy = 4.0 * x * y * (1 - x) * (
        -1706.66666666667 * y**4
        + 4096.0 * y**3
        - 3328.0 * y**2
        + 1024.0 * y
        - 89.3333333333333
    ) + 4.0 * x * (1 - x) * (
        -341.333333333333 * y**5
        + 1024.0 * y**4
        - 1109.33333333333 * y**3
        + 512.0 * y**2
        - 89.3333333333333 * y
        + 4.0
    )
    return np.array([phi_dx, phi_dy])


def grad_phi_8(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = -21.3333333333333 * x * y * (y - 1) * (
        1.46410161513775 * y - 0.0980762113533159
    ) * (2.0 * y - 0.5) * (4.0 * y - 2.0) * (
        5.46410161513775 * y - 5.09807621135331
    ) + 21.3333333333333 * y * (
        1 - x
    ) * (
        y - 1
    ) * (
        1.46410161513775 * y - 0.0980762113533159
    ) * (
        2.0 * y - 0.5
    ) * (
        4.0 * y - 2.0
    ) * (
        5.46410161513775 * y - 5.09807621135331
    )
    phi_dy = (
        116.567501122939
        * x
        * y
        * (1 - x)
        * (y - 1)
        * (1.46410161513775 * y - 0.0980762113533159)
        * (2.0 * y - 0.5)
        * (4.0 * y - 2.0)
        + 85.3333333333333
        * x
        * y
        * (1 - x)
        * (y - 1)
        * (1.46410161513775 * y - 0.0980762113533159)
        * (2.0 * y - 0.5)
        * (5.46410161513775 * y - 5.09807621135331)
        + 42.6666666666667
        * x
        * y
        * (1 - x)
        * (y - 1)
        * (1.46410161513775 * y - 0.0980762113533159)
        * (4.0 * y - 2.0)
        * (5.46410161513775 * y - 5.09807621135331)
        + 31.2341677896054
        * x
        * y
        * (1 - x)
        * (y - 1)
        * (2.0 * y - 0.5)
        * (4.0 * y - 2.0)
        * (5.46410161513775 * y - 5.09807621135331)
        + 21.3333333333333
        * x
        * y
        * (1 - x)
        * (1.46410161513775 * y - 0.0980762113533159)
        * (2.0 * y - 0.5)
        * (4.0 * y - 2.0)
        * (5.46410161513775 * y - 5.09807621135331)
        + 21.3333333333333
        * x
        * (1 - x)
        * (y - 1)
        * (1.46410161513775 * y - 0.0980762113533159)
        * (2.0 * y - 0.5)
        * (4.0 * y - 2.0)
        * (5.46410161513775 * y - 5.09807621135331)
    )
    return np.array([phi_dx, phi_dy])


def grad_phi_9(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = -4.0 * x * y * (
        -341.333333333333 * y**5
        + 876.198331087455 * y**4
        - 803.829161051972 * y**3
        + 316.683849253809 * y**2
        - 49.8632794954081 * y
        + 2.14359353944898
    ) + 4.0 * y * (1 - x) * (
        -341.333333333333 * y**5
        + 876.198331087455 * y**4
        - 803.829161051972 * y**3
        + 316.683849253809 * y**2
        - 49.8632794954081 * y
        + 2.14359353944898
    )
    phi_dy = 4.0 * x * y * (1 - x) * (
        -1706.66666666667 * y**4
        + 3504.79332434982 * y**3
        - 2411.48748315592 * y**2
        + 633.367698507619 * y
        - 49.8632794954081
    ) + 4.0 * x * (1 - x) * (
        -341.333333333333 * y**5
        + 876.198331087455 * y**4
        - 803.829161051972 * y**3
        + 316.683849253809 * y**2
        - 49.8632794954081 * y
        + 2.14359353944898
    )
    return np.array([phi_dx, phi_dy])


def grad_phi_10(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = 2.0 * x * y * (
        -341.333333333334 * y**5
        + 1171.80166891254 * y**4
        - 1542.83750561469 * y**3
        + 963.316150746191 * y**2
        - 280.803387171259 * y
        + 29.856406460551
    ) + y * (2.0 * x - 1.0) * (
        -341.333333333334 * y**5
        + 1171.80166891254 * y**4
        - 1542.83750561469 * y**3
        + 963.316150746191 * y**2
        - 280.803387171259 * y
        + 29.856406460551
    )
    phi_dy = x * y * (2.0 * x - 1.0) * (
        -1706.66666666667 * y**4
        + 4687.20667565018 * y**3
        - 4628.51251684408 * y**2
        + 1926.63230149238 * y
        - 280.803387171259
    ) + x * (2.0 * x - 1.0) * (
        -341.333333333334 * y**5
        + 1171.80166891254 * y**4
        - 1542.83750561469 * y**3
        + 963.316150746191 * y**2
        - 280.803387171259 * y
        + 29.856406460551
    )
    return np.array([phi_dx, phi_dy])


def grad_phi_11(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = 10.6666666666667 * x * y * (y - 1) * (
        1.46410161513775 * y - 1.36602540378444
    ) * (2.0 * y - 1.5) * (4.0 * y - 2.0) * (
        5.46410161513776 * y - 0.366025403784439
    ) + 5.33333333333333 * y * (
        2.0 * x - 1.0
    ) * (
        y - 1
    ) * (
        1.46410161513775 * y - 1.36602540378444
    ) * (
        2.0 * y - 1.5
    ) * (
        4.0 * y - 2.0
    ) * (
        5.46410161513776 * y - 0.366025403784439
    )
    phi_dy = (
        29.1418752807347
        * x
        * y
        * (2.0 * x - 1.0)
        * (y - 1)
        * (1.46410161513775 * y - 1.36602540378444)
        * (2.0 * y - 1.5)
        * (4.0 * y - 2.0)
        + 21.3333333333333
        * x
        * y
        * (2.0 * x - 1.0)
        * (y - 1)
        * (1.46410161513775 * y - 1.36602540378444)
        * (2.0 * y - 1.5)
        * (5.46410161513776 * y - 0.366025403784439)
        + 10.6666666666667
        * x
        * y
        * (2.0 * x - 1.0)
        * (y - 1)
        * (1.46410161513775 * y - 1.36602540378444)
        * (4.0 * y - 2.0)
        * (5.46410161513776 * y - 0.366025403784439)
        + 7.80854194740136
        * x
        * y
        * (2.0 * x - 1.0)
        * (y - 1)
        * (2.0 * y - 1.5)
        * (4.0 * y - 2.0)
        * (5.46410161513776 * y - 0.366025403784439)
        + 5.33333333333333
        * x
        * y
        * (2.0 * x - 1.0)
        * (1.46410161513775 * y - 1.36602540378444)
        * (2.0 * y - 1.5)
        * (4.0 * y - 2.0)
        * (5.46410161513776 * y - 0.366025403784439)
        + 5.33333333333333
        * x
        * (2.0 * x - 1.0)
        * (y - 1)
        * (1.46410161513775 * y - 1.36602540378444)
        * (2.0 * y - 1.5)
        * (4.0 * y - 2.0)
        * (5.46410161513776 * y - 0.366025403784439)
    )
    return np.array([phi_dx, phi_dy])


def grad_phi_12(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = 2.0 * x * y * (
        -341.333333333333 * y**5
        + 1024.0 * y**4
        - 1109.33333333333 * y**3
        + 512.0 * y**2
        - 89.3333333333333 * y
        + 4.0
    ) + y * (2.0 * x - 1.0) * (
        -341.333333333333 * y**5
        + 1024.0 * y**4
        - 1109.33333333333 * y**3
        + 512.0 * y**2
        - 89.3333333333333 * y
        + 4.0
    )
    phi_dy = x * y * (2.0 * x - 1.0) * (
        -1706.66666666667 * y**4
        + 4096.0 * y**3
        - 3328.0 * y**2
        + 1024.0 * y
        - 89.3333333333333
    ) + x * (2.0 * x - 1.0) * (
        -341.333333333333 * y**5
        + 1024.0 * y**4
        - 1109.33333333333 * y**3
        + 512.0 * y**2
        - 89.3333333333333 * y
        + 4.0
    )
    return np.array([phi_dx, phi_dy])


def grad_phi_13(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = 10.6666666666667 * x * y * (y - 1) * (
        1.46410161513775 * y - 0.0980762113533159
    ) * (2.0 * y - 0.5) * (4.0 * y - 2.0) * (
        5.46410161513775 * y - 5.09807621135331
    ) + 5.33333333333333 * y * (
        2.0 * x - 1.0
    ) * (
        y - 1
    ) * (
        1.46410161513775 * y - 0.0980762113533159
    ) * (
        2.0 * y - 0.5
    ) * (
        4.0 * y - 2.0
    ) * (
        5.46410161513775 * y - 5.09807621135331
    )
    phi_dy = (
        29.1418752807346
        * x
        * y
        * (2.0 * x - 1.0)
        * (y - 1)
        * (1.46410161513775 * y - 0.0980762113533159)
        * (2.0 * y - 0.5)
        * (4.0 * y - 2.0)
        + 21.3333333333333
        * x
        * y
        * (2.0 * x - 1.0)
        * (y - 1)
        * (1.46410161513775 * y - 0.0980762113533159)
        * (2.0 * y - 0.5)
        * (5.46410161513775 * y - 5.09807621135331)
        + 10.6666666666667
        * x
        * y
        * (2.0 * x - 1.0)
        * (y - 1)
        * (1.46410161513775 * y - 0.0980762113533159)
        * (4.0 * y - 2.0)
        * (5.46410161513775 * y - 5.09807621135331)
        + 7.80854194740136
        * x
        * y
        * (2.0 * x - 1.0)
        * (y - 1)
        * (2.0 * y - 0.5)
        * (4.0 * y - 2.0)
        * (5.46410161513775 * y - 5.09807621135331)
        + 5.33333333333333
        * x
        * y
        * (2.0 * x - 1.0)
        * (1.46410161513775 * y - 0.0980762113533159)
        * (2.0 * y - 0.5)
        * (4.0 * y - 2.0)
        * (5.46410161513775 * y - 5.09807621135331)
        + 5.33333333333333
        * x
        * (2.0 * x - 1.0)
        * (y - 1)
        * (1.46410161513775 * y - 0.0980762113533159)
        * (2.0 * y - 0.5)
        * (4.0 * y - 2.0)
        * (5.46410161513775 * y - 5.09807621135331)
    )
    return np.array([phi_dx, phi_dy])


def grad_phi_14(x: float, y: float) -> npt.NDArray[np.float64]:
    phi_dx = 2.0 * x * y * (
        -341.333333333333 * y**5
        + 876.198331087455 * y**4
        - 803.829161051972 * y**3
        + 316.683849253809 * y**2
        - 49.8632794954081 * y
        + 2.14359353944898
    ) + y * (2.0 * x - 1.0) * (
        -341.333333333333 * y**5
        + 876.198331087455 * y**4
        - 803.829161051972 * y**3
        + 316.683849253809 * y**2
        - 49.8632794954081 * y
        + 2.14359353944898
    )
    phi_dy = x * y * (2.0 * x - 1.0) * (
        -1706.66666666667 * y**4
        + 3504.79332434982 * y**3
        - 2411.48748315592 * y**2
        + 633.367698507619 * y
        - 49.8632794954081
    ) + x * (2.0 * x - 1.0) * (
        -341.333333333333 * y**5
        + 876.198331087455 * y**4
        - 803.829161051972 * y**3
        + 316.683849253809 * y**2
        - 49.8632794954081 * y
        + 2.14359353944898
    )
    return np.array([phi_dx, phi_dy])


Q26_SF_CHEB_LIST = [
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
    phi_12,
    phi_13,
    phi_14,
]

Q26_GRAD_SF_CHEB_LIST = [
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
    grad_phi_12,
    grad_phi_13,
    grad_phi_14,
]
