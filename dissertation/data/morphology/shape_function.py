import numpy as np


def particle_shape_function(angles: np.ndarray, o: float, n: int, h: float, p: float):
    return np.sqrt(o / (o**2 * np.sin(angles) ** 2 + np.cos(angles) ** 2)) * (
        h * np.cos(n * angles + 2 * np.pi * p) + 1
    )
