import numpy as np
import numba


@numba.njit(inline='always')
def find_index(arr, val, max_index):
    lo = 0
    hi = max_index + 1 

    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] <= val:
            lo = mid + 1
        else:
            hi = mid

    idx = lo - 1

    if idx < 0:
        return 0
    elif idx > max_index:
        return max_index
    else:
        return idx


@numba.njit(
    numba.float64[:, :](
        numba.float64[:],        # x
        numba.float64[:],        # y
        numba.float64[:, :, :],  # values
        numba.float64[:, :],     # points
        numba.int64,             # x_max_index
        numba.int64              # y_max_index
    ),
    cache=True
)
def evaluate_numba(x, y, values, points, x_max_index, y_max_index):
    n = points.shape[0]
    nv = values.shape[2]

    results = np.empty((n, nv))

    for k in range(n):
        px = points[k, 0]
        py = points[k, 1]

        i = find_index(x, px, x_max_index)
        j = find_index(y, py, y_max_index)

        x0 = x[i]
        x1 = x[i + 1]
        y0 = y[j]
        y1 = y[j + 1]

        tx = (px - x0) / (x1 - x0) if x1 != x0 else 0.0
        ty = (py - y0) / (y1 - y0) if y1 != y0 else 0.0

        v00 = values[i,     j    ]
        v10 = values[i + 1, j    ]
        v01 = values[i,     j + 1]
        v11 = values[i + 1, j + 1]

        results[k] = (
            (1 - tx) * (1 - ty) * v00 +
            tx * (1 - ty) * v10 +
            (1 - tx) * ty * v01 +
            tx * ty * v11
        )

    return results


class LinearInterpolator2D:
    def __init__(self, x_vals, y_vals, values_grid):
        self.x = np.asarray(x_vals)
        self.y = np.asarray(y_vals)
        self.values = np.asarray(values_grid)

        self.x_max_index = len(self.x) - 2
        self.y_max_index = len(self.y) - 2

    def evaluate(self, points: np.ndarray):
        points = np.asarray(points)
        return evaluate_numba(
            self.x,
            self.y,
            self.values,
            points,
            self.x_max_index,
            self.y_max_index
        )
    