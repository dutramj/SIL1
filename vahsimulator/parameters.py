# 3rd party libraries
import numpy as np
from numpy import sqrt

# Atmospheric constants
p0 = 101314.628  # pressure in Pa -> 1013.2 mbar -> 29.92 inHg
R = 286.99236  # universal gas constant (dry air), J/K.kg
gamma = 1.4  # heat capacity ratio cp/cv, cp: specific heat capacity (heat capacity per unit mass) at constant pressure, cv: specific heat capacity at constant volume
a0 = 340.3  # sound speed, m/s, a0 = sqrt(gamma*R*T0)
T0 = 288.1667  # standard temperature at sea level, K
T_ref = 273.15  # reference temperature, K
S = 110.4 # Sutherland temperature, K
mu_ref = 1.716e-5  # reference dynamic viscosity, kg/(m*s)
M0 = 28.9644  # molar mass of dry air, kg/kmol
Rstar = 8.31432e3  # (N m/kmol K)

# Earth's properties
earth_rate__rad_s = 7.2921158e-5  # Earth's rotation rate in rad/s
GM = 3986004.418e8  # Earth's gravitational constant, m³/s²
J2 = 1.082626684e-3  # Earth's second zonal harmonic coefficient
G = 6.67430e-11  # gravitational constant, 1/kg.s²
Me = 5.972e24  # Earth's mass, kg
Re = 6356766.0  # Earth's radius, m
Re_equatorial = 6378137  # Earth's equatorial radius, m
g0 = 9.80665  # standard value of gravity acceleration - m/s²
WGS84_A = 6378137.0  # Semi-major axis (m)
WGS84_B = 6356752.314245  # Semi-minor axis (m)
a = 6378137.0  # WGS84 semi-major axis, m
f = 1.0 / 298.257223563  # WGS84 flattening factor
e = sqrt(2*f - f**2)  # WGS84 first eccentricity
b = a*sqrt(1.0 - e**2)  # WGS84 semi-minor axis, m
A = a*e**2
B = b*e**2 / (1.0 - e**2)

# Battery parameters
c_1 = 1.0  # battery capacity, Ah
series = 10  # number of branches in series
parallel = 2  # number of branches in parallel
soc = 0.9  # state of charge
