def linear_boundary(s: float, m: float = 0, n: float = 0.0125) -> float:
    return m * s + n


def sudden_linear(
    s: float,
    m: float = 0,
    radius: float = 0.0125,
    start_change: float = 0.02,
    end_change: float = 0.09,
    dx_change: float = 0.001,
    gradient: float = 8,
) -> float:
    if s <= start_change or s > end_change:
        return m * s + radius
    n_incl = 0.0125 - gradient * start_change
    if s > start_change and s <= start_change + dx_change:
        return gradient * s + n_incl
    upper_point = gradient * (start_change + dx_change) + n_incl
    n_decl = 0.0125 + gradient * end_change
    if s > start_change + dx_change and s <= end_change - dx_change:
        return m * s + upper_point
    else:
        return -gradient * s + n_decl
