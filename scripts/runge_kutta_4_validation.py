import numpy as np
import numba
import control as ct
import matplotlib.pyplot as plt

from vahsimulator import rk4_factory

# Test Model: mass-spring-damper system
## Ref: katsuhiko, Ogata - Modern Control Engineering (5ed) - Example 2-2, pag. 32
mass = 20
damping_constant = 4
stiffness = 2

# Simulation setup
sim_time = 60
dt = 0.05
state_vector = np.array([0, 0])
input = 1 #Unity step
y_rk4 = np.array([])
time_array = np.arange(0, sim_time, dt)

#-------------- RUNGE KUTTA 4th Method ------------------------#
@numba.njit(cache=True)
def mass_spring_damper_diff_eq(state_vector, input, m, k, b):
    x1 = state_vector[0]
    x2 = state_vector[1]
    u = input

    x1_dot = x2
    x2_dot = (-k/m)*x1 + (-b/m)*x2 + (1/m)*u

    state_derivative = np.array([x1_dot, x2_dot])

    return state_derivative

rk4_step = rk4_factory(mass_spring_damper_diff_eq)

time = 0
while(time < sim_time):
    state_vector = rk4_step(state_vector, input, dt, mass, stiffness, damping_constant)
    y_rk4 = np.append(y_rk4, state_vector[0]) #x1
    time+=dt

y_rk4 = np.delete(y_rk4, -1)

#-------------- Control Lib Method ------------------------#
A = np.array([[0, 1],[-stiffness/mass, -damping_constant/mass]])
B = np.array([[0], [1/mass]])
C = np.array([[1, 0]])
D = 0

sys = ct.ss(A, B, C, D, outputs=['y'], name="spring mass damper")

_,y_clib = ct.step_response(sys, timepts=time_array)

# Plotting
fig, ax = plt.subplots()
ax.plot(time_array, y_clib, label='Control lib')
ax.plot(time_array, y_rk4, '--', label='RungeKutta4')
ax.grid()
ax.set_title('Mass Spring Damper system response')
ax.set_xlabel('time [s]')
ax.set_ylabel('position')
ax.legend()

plt.show()
