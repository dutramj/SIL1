import numpy as np
import numba
from typing import Callable


def rk4_factory(f: Callable) -> Callable:
    """
    Creates a runge kutta 4th order integration function, based on the

    provided differential equations function

    Parameters
    ----------
    f : Callable
        Function that computes the state derivative. It must have the form

            f(x, u, *args) -> np.ndarray

        where `x` is the state vector and `u` is the system input.

    Returns
    -------
    Callable
        Runge Kutta integration function
    """

    def rk4_step(
            x: np.ndarray,
            u: np.float64,
            dt: np.float64,
            *args,
        ) -> np.ndarray:
        """
        Perform one integration step using the classical 4th-order Runge–Kutta method.

        The function advances the state of a dynamical system described by

            dx/dt = f(x, u, *args)

        by one timestep `dt`.

        Parameters
        ----------
        x : np.ndarray
            Current state vector.
        u : np.float64
            System input applied during the integration step.
        dt : np.float64
            Integration timestep.
        *args
            Additional parameters passed directly to `f`.

        Returns
        -------
        np.ndarray
            State vector at the next timestep.

        Notes
        -----
        This implementation uses the classical RK4 scheme:

            k1 = f(x)
            k2 = f(x + dt/2 * k1)
            k3 = f(x + dt/2 * k2)
            k4 = f(x + dt   * k3)

            x_{k+1} = x + dt/6 * (k1 + 2k2 + 2k3 + k4)
        """

        k1 = f(x, u, *args)
        k2 = f(x + 0.5 * dt * k1, u, *args)
        k3 = f(x + 0.5 * dt * k2, u, *args)
        k4 = f(x + dt * k3, u, *args)

        dxdt = (1.0 / 6.0) * (k1 + 2.0 * (k2 + k3) + k4)

        new_x = x + dt * dxdt

        return new_x
    
    return rk4_step
