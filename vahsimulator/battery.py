# Python standard libraries
import logging

# 3rd party libraries
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

# VAHSimulator library
from .parameters import c_1, series, parallel, soc
from . import performance_decorator

logger = logging.getLogger(__name__)


class Battery:
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed + 3)
            
        # read battery data
        self.v_data = []
        for i in range(0, 4):
            battery_data = pd.read_excel('./datasets/Battery_Parameters_26650.xlsx', sheet_name=i).to_numpy()
            self.v_data.append(battery_data[:, 2])
        self.v_data = np.array(self.v_data)
        self.capacity_data = battery_data[:, 0]
        self.soc_data = battery_data[:, 1]
        c_data = np.array([0.0, 2.0, 4.0, 7.0])

        # update the total battery
        self.v_data = series*self.v_data
        self.capacity_data = parallel*self.capacity_data
        self.c_total = 3600*parallel*c_1

        # initial conditions: state of charge, capacity and tensions
        self.c_rating = c_data[0]  # C-rating
        self.soc = soc  # state of charge
        self.capacity = np.interp(self.soc, self.soc_data, self.capacity_data, period=np.inf)
        self.interp_v = RegularGridInterpolator((c_data, self.soc_data), self.v_data, method='linear')
        self.v = self.interp_v((self.c_rating, self.soc))  # tension
        self.i = 0.0  # current
    
    def _update_state(self, power, dt):
        self.capacity = self.capacity - (1.0 / self.c_total)*self.i*dt
        self.soc = np.interp(self.capacity, self.capacity_data, self.soc_data, period=np.inf)
        if self.soc < 0.0:
            self.soc = 0.0
        self.v = self.interp_v((self.c_rating, self.soc))
        self.i = power / self.v
        self.c_rating = self.i / c_1
    
    def get_soc(self):
        return self.soc

    @performance_decorator.time_execution_stats
    def step(self, power, dt):
        self._update_state(power, dt)
        if self.soc < 0.2:
            logger.warning("Battery drawn!")
        return self.soc, self.v, self.i, self.c_rating
