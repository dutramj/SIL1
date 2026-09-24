import numpy as np


class PIDGains:
    """
    Container for PID controller gain parameters.

    This class groups all controller tuning parameters together so they
    can be easily replaced, scheduled, or modified without affecting the
    controller logic.

    Attributes
    ----------
    kp : np.float64
        Proportional gain.
    ki : np.float64
        Integral gain.
    kd : np.float64
        Derivative gain.
    tau_d : np.float64
        Derivative filter time constant. If zero, no derivative filtering
        is applied.
    """

    def __init__(
        self,
        kp: np.float64,
        ki: np.float64,
        kd: np.float64,
        tau_d: np.float64 = np.float64(0.0),
    ) -> None:
        """
        Initialize PID gains.

        Parameters
        ----------
        kp : np.float64
            Proportional gain.
        ki : np.float64
            Integral gain.
        kd : np.float64
            Derivative gain.
        tau_d : np.float64, optional
            Derivative filter time constant. Default is 0.0.
        """

        self.kp: np.float64 = kp
        self.ki: np.float64 = ki
        self.kd: np.float64 = kd
        self.tau_d: np.float64 = tau_d


class PIDState:
    """
    Dynamic state of a PID controller.

    This class stores the internal variables required for the controller
    to operate between timesteps.

    Attributes
    ----------
    integral : np.float64
        Accumulated integral of the control error.
    prev_measurement : np.float64
        Measurement value from the previous timestep.
    d_term : np.float64
        Internal state of the derivative filter.
    """

    def __init__(self) -> None:
        """
        Initialize the controller state variables.
        """

        self.integral: np.float64 = np.float64(0.0)
        self.prev_measurement: np.float64 = np.float64(0.0)
        self.d_term: np.float64 = np.float64(0.0)

    def reset(self) -> None:
        """
        Reset the controller state.

        This clears the integral accumulator and derivative filter state.
        """

        self.integral = np.float64(0.0)
        self.prev_measurement = np.float64(0.0)
        self.d_term = np.float64(0.0)


class PIDController:
    """
    Discrete-time PID controller with output saturation and anti-windup.

    The controller computes control commands based on the difference
    between a desired setpoint and the measured value of a controlled
    variable.

    The derivative term is computed using the measurement rather than
    the error in order to avoid derivative spikes when the setpoint
    changes.

    Attributes
    ----------
    gains : PIDGains
        Controller gain parameters.
    state : PIDState
        Internal controller state variables.
    dt : np.float64
        Controller timestep.
    u_min : np.float64
        Minimum allowable controller output.
    u_max : np.float64
        Maximum allowable controller output.
    """

    def __init__(
        self,
        gains: PIDGains,
        u_min: np.float64 = np.float64(-np.inf),
        u_max: np.float64 = np.float64(np.inf),
    ) -> None:
        """
        Initialize the PID controller.

        Parameters
        ----------
        gains : PIDGains
            Controller gain parameters.
        u_min : np.float64, optional
            Minimum allowable controller output. Default is negative infinity.
        u_max : np.float64, optional
            Maximum allowable controller output. Default is positive infinity.
        """

        self.gains: PIDGains = gains
        self.state: PIDState = PIDState()

        self.u_min: np.float64 = u_min
        self.u_max: np.float64 = u_max

    def reset(self) -> None:
        """
        Reset the internal controller state.

        This method clears the integral and derivative memory so the
        controller behaves as if it had just been initialized.
        """

        self.state.reset()

    def step(
        self,
        setpoint: np.float64,
        measurement: np.float64,
        dt: np.float64,
    ) -> np.float64:
        """
        Compute the PID controller output.

        Parameters
        ----------
        setpoint : np.float64
            Desired reference value for the controlled variable.
        measurement : np.float64
            Current measured value of the controlled variable.
        dt : np.float64
            Controller timestep.

        Returns
        -------
        np.float64
            Saturated controller output.
        """

        error: np.float64 = setpoint - measurement

        # Proportional term
        p_term: np.float64 = self.gains.kp * error

        # Integral term
        self.state.integral += error * dt
        i_term: np.float64 = self.gains.ki * self.state.integral

        # Derivative (computed on measurement)
        derivative: np.float64 = (
            measurement - self.state.prev_measurement
        ) / dt

        if self.gains.tau_d > np.float64(0.0):
            alpha: np.float64 = self.gains.tau_d / (
                self.gains.tau_d + dt
            )

            self.state.d_term = (
                alpha * self.state.d_term
                + (np.float64(1.0) - alpha) * derivative
            )

            d_term: np.float64 = -self.gains.kd * self.state.d_term

        else:
            d_term = -self.gains.kd * derivative

        # Unsaturated controller output
        u: np.float64 = p_term + i_term + d_term

        # Apply output saturation
        u_sat: np.float64 = np.clip(u, self.u_min, self.u_max)

        # Anti-windup: rollback integrator if saturation occurs
        if u != u_sat:
            self.state.integral -= error * dt

        # Store measurement for next derivative computation
        self.state.prev_measurement = measurement

        return np.float64(u_sat)
