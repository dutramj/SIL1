# 3rd party libraries
import numpy as np
import pandas as pd

# VAHSimulator library
from . import ControlSurfaceInterpolator
from .. import performance_decorator

    
class ControlSurfacesSetInterpolator:

    interpolator_set: dict[str, ControlSurfaceInterpolator]

    def __init__(self, fin_keys: list[str], aerodeck_df: pd.DataFrame):

        self.interpolator_set = {}
        for fin_key in fin_keys:
            self.interpolator_set[fin_key] = ControlSurfaceInterpolator(aerodeck_df, fin_key)

    @performance_decorator.time_execution_stats
    def evaluate(self, points: np.ndarray, fin_commands: dict[str, np.ndarray]):
        
        coeffs = {}
        for key, command in fin_commands.items():
            points_new = np.column_stack((points, command))
            fin_coeffs = self.interpolator_set[key].evaluate(points_new)
            coeffs[key] = fin_coeffs
            
        coeffs["total"] = {
            coeff_key: sum(
                fin_coeffs[coeff_key]
                for fin_coeffs in coeffs.values()
            )
            for coeff_key in ControlSurfaceInterpolator.COEFF_KEYS
        }

        return coeffs
