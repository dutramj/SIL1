# STEVENS, Brian L.; LEWIS, Frank L.; JOHNSON, Eric N. Aircraft control and simulation: dynamics, controls design, and autonomous systems. John Wiley & Sons, 2015.

# 3rd party libraries
import numpy as np
from numpy import sin
from numpy.linalg import norm

# VAHSimulator library
from .utils import skew_matrix, eci_to_ecef, geodetic_to_geocentric_latitude, eci_to_body_quaternion
from .parameters import earth_rate__rad_s, GM, a, J2
from . import performance_decorator
from .vehicle_state import VehicleState
from .mass_properties import MassPropertiesData
from .loads import Loads

class Gravity:
    def _gravity_wgs84(self, lat, alt, pe):
        px = pe[0, 0]
        py = pe[1, 0]
        pz = pe[2, 0]
        r = norm(pe)
        lat = geodetic_to_geocentric_latitude(lat, alt)

        px_bar = px * (1 + 1.5 * J2 * ((a / r) ** 2.0) * (1 - 5 * ((sin(lat))** 2.0)))
        py_bar = py * (1 + 1.5 * J2 * ((a / r) ** 2.0) * (1 - 5 * ((sin(lat))** 2.0)))
        pz_bar = pz * (1 + 1.5 * J2 * ((a / r) ** 2.0) * (3 - 5 * ((sin(lat))** 2.0)))
        
        return (-GM / (r**3.0)) * np.array([[px_bar],
                                            [py_bar],
                                            [pz_bar]])
    
    def _gravitational_forces(self, state: VehicleState, mpd: MassPropertiesData) -> tuple[Loads, np.ndarray]:
        """
        Returns the gravitational force in the body frame.
        """
        q = state.quaternion
        lat, _, alt = state.position_geodetic
        LEI = eci_to_ecef(earth_rate__rad_s * state.time)
        pe = np.matmul(LEI, state.position_eci)

        g_e = self._gravity_wgs84(lat, alt, pe)
        
        wee = np.array([[0], [0], [earth_rate__rad_s]])
        wee_sm = skew_matrix(wee)
        g_e = np.add(g_e, - np.matmul(wee_sm, np.matmul(wee_sm, pe)))

        LBI = eci_to_body_quaternion(q)
        LIE = np.transpose(LEI)
        g_b = LBI @ LIE @ g_e

        fg_b = g_b * mpd.mass
        loads = Loads(
            fx=np.float64(fg_b[0,0]),
            fy=np.float64(fg_b[1,0]), 
            fz=np.float64(fg_b[2,0]),
            l=np.float64(0.0),
            m=np.float64(0.0),
            n=np.float64(0.0)
        )

        return loads, g_b
    
    @performance_decorator.time_execution_stats
    def evaluate(self, state: VehicleState, mass_properties: MassPropertiesData) -> tuple[Loads, np.ndarray]:
        grav_loads, grav_acc = self._gravitational_forces(state, mass_properties)
        return grav_loads, grav_acc
