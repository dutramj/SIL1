# Python standard libraries
from typing_extensions import ClassVar

# 3rd party libraries
import numpy as np
import pandas as pd

# VAHSimulator library
from ..interpolate import LinearInterpolator4D


class ControlSurfaceInterpolator():

    interpolator: LinearInterpolator4D
    COEFF_KEYS: ClassVar[list[str]] = ['DCA','DCY','DCN','DCLL','DCM','DCLN','DPANL']

    def __init__(self, aerodeck_df: pd.DataFrame, fin_key: str):

        aerodeck_df = aerodeck_df[aerodeck_df["FIN"] == fin_key].reset_index(drop=True)

        alpha_vals = np.sort(aerodeck_df['ALPHA'].unique())
        mach_vals  = np.sort(aerodeck_df['MACH'].unique())
        beta_vals  = np.sort(aerodeck_df['BETA'].unique())
        dcmd_vals  = np.sort(aerodeck_df['D_CMD'].unique())

        n_alpha = len(alpha_vals)
        n_mach  = len(mach_vals)
        n_beta  = len(beta_vals)
        n_dcmd  = len(dcmd_vals)
        n_coeff = len(self.COEFF_KEYS)

        alpha_index = {v:i for i,v in enumerate(alpha_vals)}
        mach_index  = {v:i for i,v in enumerate(mach_vals)}
        beta_index  = {v:i for i,v in enumerate(beta_vals)}
        d_cmd_index = {v:i for i,v in enumerate(dcmd_vals)}

        values_grid = np.full((n_alpha, n_beta, n_mach, n_dcmd, n_coeff), np.nan)

        for _, row in aerodeck_df.iterrows():

            i = alpha_index[row['ALPHA']]
            j = beta_index[row['BETA']]
            k = mach_index[row['MACH']]
            l = d_cmd_index[row['D_CMD']]

            for m, key in enumerate(self.COEFF_KEYS):

                val = row[key]

                if not np.isnan(val):
                    values_grid[i, j, k, l, m] = val

        self.interpolator = LinearInterpolator4D(alpha_vals, beta_vals, mach_vals, dcmd_vals, values_grid)

    def evaluate(self, points: np.ndarray) -> pd.DataFrame:
        values = self.interpolator.evaluate(points)
        return dict(zip(self.COEFF_KEYS, values.T))
