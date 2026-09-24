import numpy as np
import numba
import pandas as pd
from typing_extensions import Self
from pathlib import Path

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
        numba.float64[:],     # x
        numba.float64[:, :],  # values
        numba.float64[:],     # points
        numba.int64,          # x_max_index
    ),
    cache=True
)
def evaluate_numba(x, values, points, x_max_index):
    n = points.shape[0]
    nv = values.shape[1]

    results = np.empty((n, nv))

    for k in range(n):
        px = points[k]

        i = find_index(x, px, x_max_index)

        x0 = x[i]
        x1 = x[i + 1]

        v0 = values[i ]
        v1 = values[i + 1]

        tx1 = (x1 - px)/(x1 - x0) if x1 != x0 else 1.0
        tx2 = (px - x0)/(x1 - x0) if x1 != x0 else 0.0

        results[k] = v0*tx1 + v1*tx2

    return results

class LinearInterpolator1D:
    def __init__(self, x_vals, values_grid):
        self.x = np.asarray(x_vals)
        self.values = np.asarray(values_grid)

        self.x_max_index = len(self.x) - 2


    def evaluate(self, points: np.ndarray):
        points = np.asarray(points)
        return evaluate_numba(
            self.x,
            self.values,
            points,
            self.x_max_index,
        )
    
    
    @classmethod
    def from_csv(cls, csv_file: Path, key: str) -> Self:
        df = pd.read_csv(csv_file).sort_values(key).reset_index(drop=True)
        return cls.from_df(df,key)
    
    @classmethod
    def from_df(cls, df: pd.DataFrame, key:str) -> Self:
        
        COEFF_KEYS = df.columns.to_list()[1:]
        key_vals = np.sort(df[key].unique())

        n_key = len(key_vals)
        n_coeff = len(COEFF_KEYS)

        index = {v:i for i,v in enumerate(key_vals)}

        values_grid = np.full((n_key, n_coeff), np.nan)

        # Generate a regular grid based on altitude values
        for _, row in df.iterrows():

            i = index[row[key]]

            for j, k in enumerate(COEFF_KEYS):

                val = row[k]

                if not np.isnan(val):
                    values_grid[i, j] = val

        return cls(key_vals, values_grid)