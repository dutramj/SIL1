# Wikipedia contributors. (2025, March 15). Stagnation temperature. In Wikipedia, The Free Encyclopedia. Retrieved 11:28, July 22, 2025, from https://en.wikipedia.org/w/index.php?title=Stagnation_temperature&oldid=1280690173
# 3rd party libraries
import numpy as np

# VAHSimulator library
from .parameters import gamma
from . import performance_decorator
from .atmosphere import AtmosphereData


class Heating:
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed + 7)

        self.Ts = 0.0
    
    def _stagnation_temperature(self, T, mach):   
        """
        Calculate the stagnation temperature based on the given static temperature and Mach number.

        Args:
            T (float): Static temperature in Kelvin.
            mach (float): Mach number, representing the ratio of the object's speed to the speed of sound.
        """
        self.Ts = T*(1 + mach**2*(gamma - 1.0) / 2.0)
    
    @performance_decorator.time_execution_stats
    def step(self, atmosphere: AtmosphereData, mach):
        self._stagnation_temperature(atmosphere.temperature, mach)
        return self.Ts
