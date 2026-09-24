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
        numba.float64[:],              # x
        numba.float64[:],              # y
        numba.float64[:],              # z
        numba.float64[:],              # w
        numba.float64[:, :, :, :, :],  # values
        numba.float64[:, :],           # points
        numba.int64,
        numba.int64,
        numba.int64,
        numba.int64
    ),
    cache=True
)
def evaluate_numba_4d(x, y, z, w, values, points,
                     x_max_index, y_max_index, z_max_index, w_max_index):

    n = points.shape[0]
    nv = values.shape[4]

    results = np.empty((n, nv))

    for m in range(n):
        px = points[m, 0]
        py = points[m, 1]
        pz = points[m, 2]
        pw = points[m, 3]

        i = find_index(x, px, x_max_index)
        j = find_index(y, py, y_max_index)
        k = find_index(z, pz, z_max_index)
        l = find_index(w, pw, w_max_index)

        x0 = x[i]
        x1 = x[i + 1]
        y0 = y[j]
        y1 = y[j + 1]
        z0 = z[k]
        z1 = z[k + 1]
        w0 = w[l]
        w1 = w[l + 1]

        tx = (px - x0) / (x1 - x0) if x1 != x0 else 0.0
        ty = (py - y0) / (y1 - y0) if y1 != y0 else 0.0
        tz = (pz - z0) / (z1 - z0) if z1 != z0 else 0.0
        tw = (pw - w0) / (w1 - w0) if w1 != w0 else 0.0

        # XY weights (same as before)
        txy1 = (1 - tx) * (1 - ty)
        txy2 = tx * (1 - ty)
        txy3 = (1 - tx) * ty
        txy4 = tx * ty

        for v in range(nv):

            # ---- 16 CORNERS ----

            v0000 = values[i,     j,     k,     l,     v]
            v1000 = values[i + 1, j,     k,     l,     v]
            v0100 = values[i,     j + 1, k,     l,     v]
            v1100 = values[i + 1, j + 1, k,     l,     v]

            v0010 = values[i,     j,     k + 1, l,     v]
            v1010 = values[i + 1, j,     k + 1, l,     v]
            v0110 = values[i,     j + 1, k + 1, l,     v]
            v1110 = values[i + 1, j + 1, k + 1, l,     v]

            v0001 = values[i,     j,     k,     l + 1, v]
            v1001 = values[i + 1, j,     k,     l + 1, v]
            v0101 = values[i,     j + 1, k,     l + 1, v]
            v1101 = values[i + 1, j + 1, k,     l + 1, v]

            v0011 = values[i,     j,     k + 1, l + 1, v]
            v1011 = values[i + 1, j,     k + 1, l + 1, v]
            v0111 = values[i,     j + 1, k + 1, l + 1, v]
            v1111 = values[i + 1, j + 1, k + 1, l + 1, v]

            # ---- interpolate XY (4 planes) ----
            v00 = txy1*v0000 + txy2*v1000 + txy3*v0100 + txy4*v1100
            v01 = txy1*v0010 + txy2*v1010 + txy3*v0110 + txy4*v1110
            v10 = txy1*v0001 + txy2*v1001 + txy3*v0101 + txy4*v1101
            v11 = txy1*v0011 + txy2*v1011 + txy3*v0111 + txy4*v1111

            # ---- interpolate Z ----
            vz0 = (1 - tz) * v00 + tz * v01
            vz1 = (1 - tz) * v10 + tz * v11

            # ---- interpolate W ----
            results[m, v] = (1 - tw) * vz0 + tw * vz1

    return results


class LinearInterpolator4D:
    def __init__(self, x_vals, y_vals, z_vals, w_vals, values_grid):
        self.x = np.ascontiguousarray(x_vals, dtype=np.float64)
        self.y = np.ascontiguousarray(y_vals, dtype=np.float64)
        self.z = np.ascontiguousarray(z_vals, dtype=np.float64)
        self.w = np.ascontiguousarray(w_vals, dtype=np.float64)

        self.values = np.ascontiguousarray(values_grid, dtype=np.float64)

        self.x_max_index = len(self.x) - 2
        self.y_max_index = len(self.y) - 2
        self.z_max_index = len(self.z) - 2
        self.w_max_index = len(self.w) - 2

    def evaluate(self, points: np.ndarray):
        points = np.asarray(points)
        return evaluate_numba_4d(
            self.x,
            self.y,
            self.z,
            self.w,
            self.values,
            points,
            self.x_max_index,
            self.y_max_index,
            self.z_max_index,
            self.w_max_index
        )
