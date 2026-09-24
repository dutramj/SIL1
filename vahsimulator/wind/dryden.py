from __future__ import annotations

from numpy.linalg import norm
from numpy.random import randn, uniform

import numpy as np
from numpy import pi, pow, sqrt

from .. import performance_decorator
from ..utils import TransferFunction
from ..vehicle_state import VehicleState


class DrydenTurbulence:
    """
    Modelo de turbulência atmosférica de Dryden.

    References
    ----------
    Beard, R. W.; McLain, T. W.
        Small Unmanned Aircraft: Theory and Practice.
        Princeton University Press, 2012.

    Zipfel, P. H.
        Modeling and Simulation of Aerospace Vehicle Dynamics.
        AIAA, 2007.

    MathWorks.
        Dryden Wind Turbulence Model (Continuous).

    Ichwanul Hakim, T. M.; Arifianto, O.
        Implementation of Dryden continuous turbulence model into
        Simulink for LSA-02 flight test simulation.
        Journal of Physics: Conference Series, 2018.

    Marciotto, E. R.; Medeiros, G. F.; Fisch, L. E.
        Characterization of surface level wind in the Centro de
        Lançamento de Alcântara for use in rocket structure loading
        and dispersion studies.
        Journal of Aerospace Technology and Management, 2012.

    Notes
    -----
    Este componente modela somente a turbulência.

    O vento médio e a transformação de referência são responsabilidades
    de outros componentes do pacote ``wind``.

    """

    def __init__(self, turbulence: str = "moderate") -> None:
        """
        Parameters
        ----------
        turbulence : {"light", "moderate", "severe"}, optional
            Intensidade da turbulência.
        """
        if turbulence not in {"light", "moderate", "severe"}:
            raise ValueError("turbulence deve ser 'light', 'moderate' ou 'severe'.")

        self.turbulence = turbulence

        self.dt = 0.0
        self.b = 0.0
        self.Va = 1.0

        self.Lu = 0.0
        self.Lv = 0.0
        self.Lw = 0.0

        self.sigma_u = 0.0
        self.sigma_v = 0.0
        self.sigma_w = 0.0

        self.w6 = 0.0

        self.u_w = None
        self.v_w = None
        self.w_w = None
        self.p_w = None
        self.q_w = None
        self.r_w = None

        self._initialized = False

    def initialize(self, config_data: dict, dt: float, seed: int | None = None) -> None:
        """
        Inicializa o modelo Dryden.

        Parameters
        ----------
        config_data : dict
            Dados gerais de configuração do simulador.
        dt : float
            Período de amostragem.
        seed : int or None, optional
            Semente do gerador pseudoaleatório.
        """
        if seed is not None:
            np.random.seed(seed + 12)

        self.dt = float(dt)

        geom_data = config_data["geometric_data"]

        phase_data = next(
            (data for data in geom_data if data.get("phase_id") == 1), None
        )

        if phase_data is None:
            raise ValueError("Não foi encontrada phase_id=1 em geometric_data.")

        self.b = float(phase_data["main_parameters"]["l_ref_2"])

        if self.turbulence == "light":
            self.w6 = uniform(2.2, 4.3)
        elif self.turbulence == "moderate":
            self.w6 = 2.0 * uniform(2.2, 4.3)
        else:
            self.w6 = 3.0 * uniform(2.2, 4.3)

    def _update_dryden_parameters(self, altitude_m: float) -> None:
        """
        Atualiza escalas de comprimento e desvios padrão do Dryden.

        Parameters
        ----------
        altitude_m : float
            Altitude geométrica, em metros.
        """
        h = max(float(altitude_m), 1e-3)

        if h <= 304.8:
            self.Lw = h

            self.Lu = self.Lv = h / pow(0.177 + 0.000823 * h, 1.2)

            self.sigma_w = 0.1 * self.w6

            self.sigma_u = self.sigma_v = 0.1 * self.w6 / pow(0.177 + 0.000823 * h, 0.4)

        elif h < 609.6:
            self.Lw = 0.75 * h + 76.26096

            self.Lu = self.Lv = -1.0198080247515782 * h + 1155.074971888562

            self.sigma_w = (3.0 - 0.1 * self.w6) / 304.8 * (h - 609.6) + 3.0

            self.sigma_u = self.sigma_v = (
                3.0 - 0.1 * self.w6 / 0.7120603065939916
            ) / 304.8 * (h - 609.6) + 3.0

        else:
            self.Lu = self.Lv = self.Lw = 533.4
            self.sigma_u = self.sigma_v = self.sigma_w = 3.0

    def _create_filters(self) -> None:
        """
        Cria os filtros de transferência do modelo Dryden.
        """
        Va = max(float(self.Va), 1.0)

        num_u_w = np.array([[self.sigma_u * sqrt(2.0 * self.Lu / (pi * Va))]])

        den_u_w = np.array([[1.0, self.Lu / Va]])

        self.u_w = TransferFunction(num_u_w, den_u_w, self.dt)

        num_v_w = np.array(
            [
                [
                    self.sigma_v * sqrt(3.0 * self.Lv / (pi * Va)),
                    self.sigma_v
                    * sqrt(3.0 * self.Lv / (pi * Va))
                    * sqrt(3.0)
                    * self.Lv
                    / Va,
                ]
            ]
        )

        den_v_w = np.array([[1.0, 2.0 * self.Lv / Va, (self.Lv / Va) ** 2]])

        self.v_w = TransferFunction(num_v_w, den_v_w, self.dt)

        num_w_w = np.array(
            [
                [
                    self.sigma_w * sqrt(3.0 * self.Lw / (pi * Va)),
                    self.sigma_w
                    * sqrt(3.0 * self.Lw / (pi * Va))
                    * sqrt(3.0)
                    * self.Lw
                    / Va,
                ]
            ]
        )

        den_w_w = np.array([[1.0, 2.0 * self.Lw / Va, (self.Lw / Va) ** 2]])

        self.w_w = TransferFunction(num_w_w, den_w_w, self.dt)

        num_p_w = np.array(
            [[self.sigma_w * sqrt(0.8 / Va) * pow(pi / (4.0 * self.b), 1.0 / 6.0)]]
        )

        den_p_w = np.array(
            [
                [
                    pow(2.0 * self.Lw, 1.0 / 3.0),
                    pow(2.0 * self.Lw, 1.0 / 3.0) * 4.0 * self.b / (pi * Va),
                ]
            ]
        )

        self.p_w = TransferFunction(num_p_w, den_p_w, self.dt)

        num_q_w = np.array(
            [
                [
                    0.0,
                    self.sigma_w * sqrt(2.0 * self.Lw / (pi * Va)) / Va,
                    self.sigma_w
                    * sqrt(2.0 * self.Lw / (pi * Va))
                    * 2.0
                    * sqrt(3.0)
                    * self.Lw
                    / Va**2,
                ]
            ]
        )

        den_q_w = np.array(
            [
                [
                    1.0,
                    4.0 * self.Lw / Va + 4.0 * self.b / (pi * Va),
                    4.0 * self.Lw**2 / Va**2
                    + 16.0 * self.b * self.Lw / (pi * Va**2),
                    16.0 * self.b * self.Lw**2 / (pi * Va**3),
                ]
            ]
        )

        self.q_w = TransferFunction(num_q_w, den_q_w, self.dt)

        num_r_w = np.array(
            [
                [
                    0.0,
                    self.sigma_v * sqrt(2.0 * self.Lv / (pi * Va)) / Va,
                    self.sigma_v
                    * sqrt(2.0 * self.Lv / (pi * Va))
                    * 2.0
                    * sqrt(3.0)
                    * self.Lv
                    / Va**2,
                ]
            ]
        )

        den_r_w = np.array(
            [
                [
                    1.0,
                    3.0 * self.Lv / Va + 3.0 * self.b / (pi * Va),
                    3.0 * self.Lv**2 / Va**2
                    + 12.0 * self.b * self.Lv / (pi * Va**2),
                    12.0 * self.b * self.Lv**2 / (pi * Va**3),
                ]
            ]
        )

        self.r_w = TransferFunction(num_r_w, den_r_w, self.dt)

    def evaluate_ic(self, state: VehicleState, constant_wind_body: np.ndarray) -> None:
        """
        Inicializa os filtros no instante inicial.

        Parameters
        ----------
        state : VehicleState
            Estado inicial do veículo.
        constant_wind_body : numpy.ndarray
            Vento constante na referência body, ``3 x 1``.
        """
        self._update_dryden_parameters(state.alt)

        velocity_body = np.asarray(state.velocity_body, dtype=float)

        air_velocity_body = velocity_body - constant_wind_body

        self.Va = max(float(norm(air_velocity_body)), 1.0)

        self._create_filters()

        self._initialized = True

    def _update_filters(self, altitude_m: float, Va: float) -> np.ndarray:
        """
        Atualiza os filtros Dryden.

        Parameters
        ----------
        altitude_m : float
            Altitude atual, em metros.
        Va : float
            Velocidade aerodinâmica, em m/s.

        Returns
        -------
        numpy.ndarray
            Vetor ``6 x 1`` com a turbulência.
        """
        self._update_dryden_parameters(altitude_m)

        self.Va = max(float(Va), 1.0)

        if self.u_w is None:
            self._create_filters()

        Va = self.Va

        num_u_w = np.array([[self.sigma_u * sqrt(2.0 * self.Lu / (pi * Va))]])

        den_u_w = np.array([[1.0, self.Lu / Va]])

        u_w = self.u_w.step(randn(), num_u_w, den_u_w)

        num_v_w = np.array(
            [
                [
                    self.sigma_v * sqrt(3.0 * self.Lv / (pi * Va)),
                    self.sigma_v
                    * sqrt(3.0 * self.Lv / (pi * Va))
                    * sqrt(3.0)
                    * self.Lv
                    / Va,
                ]
            ]
        )

        den_v_w = np.array([[1.0, 2.0 * self.Lv / Va, (self.Lv / Va) ** 2]])

        v_w = self.v_w.step(randn(), num_v_w, den_v_w)

        num_w_w = np.array(
            [
                [
                    self.sigma_w * sqrt(3.0 * self.Lw / (pi * Va)),
                    self.sigma_w
                    * sqrt(3.0 * self.Lw / (pi * Va))
                    * sqrt(3.0)
                    * self.Lw
                    / Va,
                ]
            ]
        )

        den_w_w = np.array([[1.0, 2.0 * self.Lw / Va, (self.Lw / Va) ** 2]])

        w_w = self.w_w.step(randn(), num_w_w, den_w_w)

        num_p_w = np.array(
            [[self.sigma_w * sqrt(0.8 / Va) * pow(pi / (4.0 * self.b), 1.0 / 6.0)]]
        )

        den_p_w = np.array(
            [
                [
                    pow(2.0 * self.Lw, 1.0 / 3.0),
                    pow(2.0 * self.Lw, 1.0 / 3.0) * 4.0 * self.b / (pi * Va),
                ]
            ]
        )

        p_w = self.p_w.step(randn(), num_p_w, den_p_w)

        num_q_w = np.array(
            [
                [
                    0.0,
                    self.sigma_w * sqrt(2.0 * self.Lw / (pi * Va)) / Va,
                    self.sigma_w
                    * sqrt(2.0 * self.Lw / (pi * Va))
                    * 2.0
                    * sqrt(3.0)
                    * self.Lw
                    / Va**2,
                ]
            ]
        )

        den_q_w = np.array(
            [
                [
                    1.0,
                    4.0 * self.Lw / Va + 4.0 * self.b / (pi * Va),
                    4.0 * self.Lw**2 / Va**2
                    + 16.0 * self.b * self.Lw / (pi * Va**2),
                    16.0 * self.b * self.Lw**2 / (pi * Va**3),
                ]
            ]
        )

        q_w = self.q_w.step(randn(), num_q_w, den_q_w)

        num_r_w = np.array(
            [
                [
                    0.0,
                    self.sigma_v * sqrt(2.0 * self.Lv / (pi * Va)) / Va,
                    self.sigma_v
                    * sqrt(2.0 * self.Lv / (pi * Va))
                    * 2.0
                    * sqrt(3.0)
                    * self.Lv
                    / Va**2,
                ]
            ]
        )

        den_r_w = np.array(
            [
                [
                    1.0,
                    3.0 * self.Lv / Va + 3.0 * self.b / (pi * Va),
                    3.0 * self.Lv**2 / Va**2
                    + 12.0 * self.b * self.Lv / (pi * Va**2),
                    12.0 * self.b * self.Lv**2 / (pi * Va**3),
                ]
            ]
        )

        r_w = self.r_w.step(randn(), num_r_w, den_r_w)

        return np.array([[u_w], [v_w], [w_w], [p_w], [q_w], [r_w]])

    @performance_decorator.time_execution_stats
    def evaluate(
        self, state: VehicleState, Va: float, phase: int, config_data: dict
    ) -> np.ndarray:
        """
        Avalia a turbulência Dryden.

        Parameters
        ----------
        state : VehicleState
            Estado atual.
        Va : float
            Velocidade aerodinâmica.
        phase : int
            Fase atual do voo.
        config_data : dict
            Configuração geral do simulador.

        Returns
        -------
        numpy.ndarray
            Vetor ``6 x 1`` com a turbulência.
        """
        geom_data = config_data["geometric_data"]

        phase_data = next(
            (data for data in geom_data if data.get("phase_id") == phase), None
        )

        if phase_data is None:
            raise ValueError(f"Não foi encontrada phase_id={phase}.")

        self.b = float(phase_data["main_parameters"]["l_ref_2"])

        if state.alt <= 0.0 or state.alt > 80000.0:
            return np.zeros((6, 1))

        return self._update_filters(state.alt, Va)
