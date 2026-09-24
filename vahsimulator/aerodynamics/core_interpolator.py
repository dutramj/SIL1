# 3rd party libraries
import numpy as np
import pandas as pd

# VAHSimulator library
from .. import performance_decorator
from ..interpolate import LinearInterpolator4D


class AerodynamicCoreInterpolator:

    interpolator: LinearInterpolator4D

    COEFF_KEYS = [
        'CN','CM','CA','CY','CLN','CLL','X-C.P.','XCG',
        'CNA','CMA','CYB','CLNB','CLLB','CNQ','CMQ','CNAD','CMAD','CYR','CLNR','CLLR','CYP','CLNP','CLLP'
        # To be implemented
        # 'CL','CD','CL/CD', 'PANL 1','PANL 2','PANL 3','PANL 4',
    ]

    def __init__(self, aerodeck_df: pd.DataFrame):

        # Creating Coefficient Interpolator
        alpha_vals = np.sort(aerodeck_df['ALPHA'].unique())
        beta_vals  = np.sort(aerodeck_df['BETA'].unique())
        mach_vals  = np.sort(aerodeck_df['MACH'].unique())
        alt_vals   = np.sort(aerodeck_df['ALTITUDE'].unique())

        n_alpha = len(alpha_vals)
        n_beta  = len(beta_vals)
        n_mach  = len(mach_vals)
        n_alt   = len(alt_vals)

        n_coeff = len(self.COEFF_KEYS)

        alpha_index = {v:i for i,v in enumerate(alpha_vals)}
        beta_index = {v:i for i,v in enumerate(beta_vals)}
        mach_index  = {v:i for i,v in enumerate(mach_vals)}
        alt_index   = {v:i for i,v in enumerate(alt_vals)}

        values_grid = np.full((n_alpha, n_beta, n_mach, n_alt, n_coeff), np.nan)

        for _, row in aerodeck_df.iterrows():

            i = alpha_index[row['ALPHA']]
            j = beta_index[row['BETA']]
            k = mach_index[row['MACH']]
            l = alt_index[row['ALTITUDE']]

            for m, key in enumerate(self.COEFF_KEYS):

                val = row[key]

                if not np.isnan(val):
                    values_grid[i, j, k, l, m] = val

        self.interpolator = LinearInterpolator4D(alpha_vals, beta_vals, mach_vals, alt_vals, values_grid)

    @performance_decorator.time_execution_stats
    def evaluate(self, points: np.ndarray):
        values = self.interpolator.evaluate(points)
        return dict(zip(self.COEFF_KEYS, values.T))
