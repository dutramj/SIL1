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
        numba.float64[:],           # x: from the interpolation table
        numba.float64[:],           # y: from the interpolation table
        numba.float64[:],           # z: from the interpolation table
        numba.float64[:, :, :, :],  # values: from the interpolation table
        numba.float64[:, :],        # points: to be interpolated
        numba.int64,                # x_max_index
        numba.int64,                # y_max_index
        numba.int64                 # z_max_index
    ),
    cache=True
)
def evaluate_numba(x, y, z, values, points, x_max_index, y_max_index, z_max_index):
    n = points.shape[0]
    nv = values.shape[3]

    results = np.empty((n, nv))

    for m in range(n):
        px = points[m, 0]
        py = points[m, 1]
        pz = points[m, 2]

        i = find_index(x, px, x_max_index)
        j = find_index(y, py, y_max_index)
        k = find_index(z, pz, z_max_index)

        x0 = x[i]
        x1 = x[i + 1]
        y0 = y[j]
        y1 = y[j + 1]
        z0 = z[k]
        z1 = z[k+1]

        tx = (px - x0) / (x1 - x0) if x1 != x0 else 0.0
        ty = (py - y0) / (y1 - y0) if y1 != y0 else 0.0
        
        tz0 = (pz - z0) / (z1 - z0) if z1 != z0 else 0.0
        tz1 = (z1 - pz) / (z1 - z0) if z1 != z0 else 1.0

        txy1 = (1 - tx) * (1 - ty)
        txy2 = tx * (1 - ty)
        txy3 = (1 - tx) * ty
        txy4 = tx * ty 

        for v in range(nv):

            v000 = values[i,     j,     k,     v]
            v100 = values[i + 1, j,     k,     v]
            v010 = values[i,     j + 1, k,     v]
            v110 = values[i + 1, j + 1, k,     v]

            v001 = values[i,     j,     k + 1, v]
            v101 = values[i + 1, j,     k + 1, v]
            v011 = values[i,     j + 1, k + 1, v]
            v111 = values[i + 1, j + 1, k + 1, v]

            v0 = txy1 * v000 + txy2 * v100 + txy3 * v010 + txy4 * v110
            v1 = txy1 * v001 + txy2 * v101 + txy3 * v011 + txy4 * v111

            results[m, v] = tz1*v0 + tz0*v1

    return results

class LinearInterpolator3D:
    def __init__(self, x_vals, y_vals, z_vals, values_grid):
        self.x = np.ascontiguousarray(x_vals, dtype=np.float64)
        self.y = np.ascontiguousarray(y_vals, dtype=np.float64)
        self.z = np.ascontiguousarray(z_vals, dtype=np.float64)
        self.values = np.ascontiguousarray(values_grid, dtype=np.float64)

        self.x_max_index = len(self.x) - 2
        self.y_max_index = len(self.y) - 2
        self.z_max_index = len(self.z) - 2

    def evaluate(self, points: np.ndarray):
        points = np.asarray(points)
        return evaluate_numba(
            self.x,
            self.y,
            self.z,
            self.values,
            points,
            self.x_max_index,
            self.y_max_index,
            self.z_max_index
        )
    