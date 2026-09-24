import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class LaunchReference:
    lat: np.float64
    lon: np.float64
    alt: np.float64
    azimuth: np.float64
