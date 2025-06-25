import numpy as np


def ashby_grid(param_values, shrinkage_curves, x, y) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    log_param_values = np.log(param_values)
    log_x = np.log(x)

    times = np.array([np.interp(y, s, t) for t, s in shrinkage_curves]).T
    times = np.exp([np.interp(log_x, log_param_values, np.log(t)) for t in times])
    grid_x, grid_y = np.meshgrid(x, y)

    return grid_x, grid_y, times
