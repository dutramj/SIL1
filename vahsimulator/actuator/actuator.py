from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ..runge_kutta import rk4_factory
from .linear_model import linear_model_2nd_order
from .nonlinearities import nonlinearities_pipeline_factory_from_dict


# =====================================================================
# Parâmetros
# =====================================================================


@dataclass(slots=True, frozen=True)
class ActuatorParameters:
    """
    Armazena os parâmetros dinâmicos do atuador.

    Parameters
    ----------
    natural_frequency : float
        Frequência natural não amortecida do atuador [rad/s].
    damping_ratio : float
        Razão de amortecimento do atuador [-].

    Examples
    --------
    >>> parameters = ActuatorParameters(
    ...     natural_frequency=62.8,
    ...     damping_ratio=0.7,
    ... )
    """

    natural_frequency: float
    damping_ratio: float

    def to_array(self) -> np.ndarray:
        """
        Converte os parâmetros para um ``numpy.ndarray``.

        Returns
        -------
        numpy.ndarray
            Vetor contendo:

            ``[natural_frequency, damping_ratio]``.
        """

        return np.array([self.natural_frequency, self.damping_ratio], dtype=np.float64)


# =====================================================================
# Entrada
# =====================================================================


@dataclass(slots=True)
class ActuatorInput:
    """
    Representa a referência de posição do atuador.

    Parameters
    ----------
    position_reference : float, optional
        Posição de referência desejada [rad].
        O valor padrão é ``0.0``.

    Examples
    --------
    >>> actuator_input = ActuatorInput(position_reference=0.1)
    """

    position_reference: float = 0.0

    def to_array(self) -> np.ndarray:
        """
        Converte a entrada para um ``numpy.ndarray``.

        Returns
        -------
        numpy.ndarray
            Vetor contendo a posição de referência:

            ``[position_reference]``.
        """

        return np.array([self.position_reference], dtype=np.float64)


# =====================================================================
# Estado
# =====================================================================


@dataclass(slots=True)
class ActuatorState:
    """
    Representa o estado dinâmico do atuador.

    Para um atuador de segunda ordem, o estado é composto por posição
    e velocidade.

    Parameters
    ----------
    position : float, optional
        Posição atual do atuador [rad].
        O valor padrão é ``0.0``.
    velocity : float, optional
        Velocidade atual do atuador [rad/s].
        O valor padrão é ``0.0``.
    """

    position: float = 0.0
    velocity: float = 0.0

    def to_array(self) -> np.ndarray:
        """
        Converte o estado para um ``numpy.ndarray``.

        Returns
        -------
        numpy.ndarray
            Vetor de estado:

            ``[position, velocity]``.
        """

        return np.array([self.position, self.velocity], dtype=np.float64)

    @classmethod
    def from_array(cls, array: np.ndarray) -> ActuatorState:
        """
        Cria um estado a partir de um ``numpy.ndarray``.

        Parameters
        ----------
        array : numpy.ndarray
            Vetor contendo:

            ``[position, velocity]``.

        Returns
        -------
        ActuatorState
            Estado correspondente ao vetor fornecido.
        """

        return cls(position=array[0], velocity=array[1])


# =====================================================================
# Dinâmica não ideal
# =====================================================================


def nonlinear_model_factory(linear_model: Callable) -> Callable:
    """
    Cria um modelo dinâmico não ideal a partir de um modelo linear.

    As não-linearidades de estado são aplicadas à derivada calculada
    pelo modelo linear.

    Parameters
    ----------
    linear_model : Callable
        Modelo dinâmico linear.

        Deve possuir a assinatura::

            derivative = linear_model(
                state,
                reference,
                parameters,
            )

    Returns
    -------
    Callable
        Modelo dinâmico não linear.

    Notes
    -----
    As não-linearidades de referência não são aplicadas nesta função.
    Elas são processadas pelo ``ExplicitActuatorSolver`` antes da
    integração.
    """

    def nonlinear_model(
        state: np.ndarray,
        reference: np.ndarray,
        parameters: np.ndarray,
        nonlinearities: Callable,
    ) -> np.ndarray:
        """
        Avalia a dinâmica linear e aplica as não-linearidades de estado.

        Parameters
        ----------
        state : numpy.ndarray
            Estado atual do atuador.

        reference : numpy.ndarray
            Referência efetiva aplicada ao modelo.

        parameters : numpy.ndarray
            Parâmetros dinâmicos do atuador.

        nonlinearities : Callable
            Pipeline de não-linearidades de estado.

        Returns
        -------
        numpy.ndarray
            Derivada do estado após a aplicação das não-linearidades.
        """

        derivative = linear_model(state, reference, parameters)

        _, derivative = nonlinearities(state, derivative)

        return derivative

    return nonlinear_model


# =====================================================================
# Solver
# =====================================================================


class ExplicitActuatorSolver:
    """
    Integra explicitamente a dinâmica de um atuador.

    O solver separa as não-linearidades em dois grupos:

    * não-linearidades de referência, processadas uma vez por passo;
    * não-linearidades de estado, avaliadas durante a integração.

    Essa separação permite implementar corretamente não-linearidades
    com memória interna, como backlash.

    Parameters
    ----------
    linear_model : Callable
        Modelo dinâmico linear do atuador.

    parameters : ActuatorParameters
        Parâmetros dinâmicos do atuador.

    dt_s : float
        Passo de integração [s].

    state_nonlinearities : Callable, optional
        Pipeline de não-linearidades de estado.

        Assinatura::

            state, derivative = nonlinearities(
                state,
                derivative,
            )

    reference_nonlinearities : Callable, optional
        Pipeline de não-linearidades de referência.

        Assinatura::

            reference, memory = nonlinearities(
                reference,
                memory,
            )

    reference_memory : numpy.ndarray, optional
        Estado interno das não-linearidades de referência.

    Notes
    -----
    As não-linearidades de referência são executadas somente uma vez
    por passo de integração.

    A referência efetiva resultante é então mantida constante durante
    as quatro avaliações do método RK4.
    """

    def __init__(
        self,
        linear_model: Callable,
        parameters: ActuatorParameters,
        dt_s: float,
        *,
        state_nonlinearities: Callable | None = None,
        reference_nonlinearities: Callable | None = None,
        reference_memory: np.ndarray | None = None,
    ) -> None:
        self._linear_model = linear_model
        self._parameters = parameters.to_array()
        self._dt_s = dt_s

        self._state_nonlinearities = state_nonlinearities
        self._reference_nonlinearities = reference_nonlinearities

        if reference_memory is None:
            self._reference_memory = np.empty(0, dtype=np.float64)
        else:
            self._reference_memory = reference_memory

        # --------------------------------------------------------------
        # Modelo dinâmico
        # --------------------------------------------------------------

        if self._state_nonlinearities is None:
            self._model = self._linear_model
        else:
            self._model = nonlinear_model_factory(self._linear_model)

        # --------------------------------------------------------------
        # Integrador
        # --------------------------------------------------------------

        self._integrator = rk4_factory(self._model)

    def step(self, state: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """
        Executa um passo de integração.

        Parameters
        ----------
        state : numpy.ndarray
            Estado atual do atuador.

        reference : numpy.ndarray
            Referência de posição desejada [rad].

        Returns
        -------
        numpy.ndarray
            Estado do atuador no próximo instante de tempo.
        """

        # ==============================================================
        # 1. Processamento da referência
        #
        # Não-linearidades aplicáveis ao sinal de referência são executadas
        # aqui (ex: backlash).
        # ==============================================================

        effective_reference = reference

        if self._reference_nonlinearities is not None:
            (
                effective_reference,
                self._reference_memory,
            ) = self._reference_nonlinearities(reference, self._reference_memory)

        self._effective_reference = effective_reference

        # ==============================================================
        # 2. Integração RK4
        # ==============================================================

        if self._state_nonlinearities is None:
            next_state = self._integrator(
                state, effective_reference, self._dt_s, self._parameters
            )
        else:
            next_state = self._integrator(
                state,
                effective_reference,
                self._dt_s,
                self._parameters,
                self._state_nonlinearities,
            )

        # ==============================================================
        # 3. Pós-processamento do estado
        #
        # Garante que limitadores de posição/velocidade/aceleração possam
        # atuar diretamente sobre o estado final.
        # ==============================================================

        if self._state_nonlinearities is not None:
            derivative = self._model(
                next_state,
                effective_reference,
                self._parameters,
                self._state_nonlinearities,
            )

            next_state, _ = self._state_nonlinearities(next_state, derivative)

        return next_state

    def set_dt(self, dt_s: float) -> None:
        """
        Atualiza o passo de integração.

        Parameters
        ----------
        dt_s : float
            Novo passo de integração [s].
        """

        self._dt_s = dt_s

    @property
    def reference_memory(self) -> np.ndarray:
        """
        Retorna a memória das não-linearidades de referência.

        Returns
        -------
        numpy.ndarray
            Estado interno persistente das não-linearidades de referência.
        """

        return self._reference_memory

    @property
    def effective_reference(self) -> np.ndarray:
        """Retorna a última referência efetiva aplicada ao modelo."""
        return self._effective_reference


# =====================================================================
# Atuador
# =====================================================================


class Actuator:
    """
    Modelo de atuador de segunda ordem.

    O atuador encapsula:

    * parâmetros dinâmicos;
    * estado atual;
    * modelo dinâmico;
    * solver numérico;
    * não-linearidades de referência;
    * não-linearidades de estado.

    Parameters
    ----------
    parameters : ActuatorParameters
        Parâmetros dinâmicos do atuador.

    dt : float
        Passo de integração [s].

    initial_state : ActuatorState, optional
        Estado inicial do atuador.

    nonlinearities : Callable, optional
        Pipeline de não-linearidades de estado.

        Este argumento é mantido por compatibilidade com a API anterior.
        Para novos modelos, prefira utilizar explicitamente
        ``state_nonlinearities``.

    state_nonlinearities : Callable, optional
        Pipeline de não-linearidades aplicadas ao estado e à derivada.

    reference_nonlinearities : Callable, optional
        Pipeline de não-linearidades aplicadas à referência.

        É neste pipeline que devem ser colocadas não-linearidades com
        memória, como backlash.

    linear_model : Callable, optional
        Modelo linear do atuador.

        Por padrão, utiliza :func:`linear_model_2nd_order`.
    """

    def __init__(
        self,
        parameters: ActuatorParameters,
        dt: float,
        *,
        initial_state: ActuatorState | None = None,
        nonlinearities: Callable | None = None,
        state_nonlinearities: Callable | None = None,
        reference_nonlinearities: Callable | None = None,
        reference_memory: np.ndarray | None = None,
        linear_model: Callable = linear_model_2nd_order,
    ) -> None:
        """
        @brief Inicializa o atuador.

        Parameters
        ----------
        parameters : ActuatorParameters
            Parâmetros dinâmicos do atuador.
        dt : float
            Passo de integração em segundos.
        initial_state : ActuatorState, optional
            Estado inicial do atuador.
        nonlinearities : Callable, optional
            Pipeline de não linearidades de estado.
            Mantido por compatibilidade com a API anterior.
        state_nonlinearities : Callable, optional
            Pipeline de não linearidades aplicadas ao estado.
        reference_nonlinearities : Callable, optional
            Pipeline de não linearidades aplicadas à referência.
        reference_memory : numpy.ndarray, optional
            Memória persistente utilizada pelas não linearidades da referência.
            Por exemplo, o backlash utiliza um vetor de um elemento.
        linear_model : Callable, optional
            Modelo linear utilizado pelo atuador.
        """
        self._linear_model = linear_model

        if initial_state is None:
            initial_state = ActuatorState()

        self._state_array = initial_state.to_array()

        if state_nonlinearities is None:
            state_nonlinearities = nonlinearities

        self.configure(
            parameters,
            dt,
            state_nonlinearities=state_nonlinearities,
            reference_nonlinearities=reference_nonlinearities,
            reference_memory=reference_memory,
        )

    def configure(
        self,
        parameters: ActuatorParameters,
        dt: float,
        *,
        nonlinearities: Callable | None = None,
        state_nonlinearities: Callable | None = None,
        reference_nonlinearities: Callable | None = None,
        reference_memory: np.ndarray | None = None,
    ) -> None:
        """
        @brief Configura os parâmetros e as não linearidades do atuador.

        Parameters
        ----------
        parameters : ActuatorParameters
            Parâmetros dinâmicos do atuador.
        dt : float
            Passo de integração em segundos.
        nonlinearities : Callable, optional
            Pipeline de não linearidades de estado.
            Mantido por compatibilidade com a API anterior.
        state_nonlinearities : Callable, optional
            Pipeline de não linearidades aplicadas ao estado.
        reference_nonlinearities : Callable, optional
            Pipeline de não linearidades aplicadas à referência.
        reference_memory : numpy.ndarray, optional
            Memória persistente utilizada pelas não linearidades da referência.
        """
        if state_nonlinearities is None:
            state_nonlinearities = nonlinearities

        self._parameters = parameters
        self._state_nonlinearities = state_nonlinearities
        self._reference_nonlinearities = reference_nonlinearities

        self._solver = ExplicitActuatorSolver(
            linear_model=self._linear_model,
            parameters=parameters,
            dt_s=dt,
            state_nonlinearities=state_nonlinearities,
            reference_nonlinearities=reference_nonlinearities,
            reference_memory=reference_memory,
        )

    @classmethod
    def from_dict(
        cls,
        config: dict[str, Any],
        dt: float,
        *,
        initial_state: ActuatorState | None = None,
        linear_model: Callable = linear_model_2nd_order,
    ) -> Actuator:
        """
        Cria um atuador a partir de uma configuração.

        Parameters
        ----------
        config : dict
            Configuração do atuador.

            Exemplo::

                {
                    "natural_frequency": 100.0,
                    "damping_ratio": 0.7,
                    "nonlinearities": [
                        {
                            "type": "backlash",
                            "width": 0.001,
                        },
                        {
                            "type": "velocity_limiter",
                            "maximum_velocity": 0.26179,
                        },
                        {
                            "type": "position_limiter",
                            "maximum_position": 0.05235,
                        },
                    ],
                }

        dt : float
            Passo de integração [s].

        initial_state : ActuatorState, optional
            Estado inicial do atuador.

        linear_model : Callable, optional
            Modelo linear utilizado pelo atuador.

        Returns
        -------
        Actuator
            Atuador configurado.
        """

        parameters = ActuatorParameters(
            natural_frequency=config["natural_frequency"],
            damping_ratio=config["damping_ratio"],
        )

        nonlinearities_config = config.get("nonlinearities")

        state_nonlinearities = None
        reference_nonlinearities = None

        if nonlinearities_config is not None:
            (
                reference_nonlinearities,
                state_nonlinearities,
            ) = nonlinearities_pipeline_factory_from_dict(nonlinearities_config)

        return cls(
            parameters=parameters,
            dt=dt,
            initial_state=initial_state,
            state_nonlinearities=state_nonlinearities,
            reference_nonlinearities=reference_nonlinearities,
            linear_model=linear_model,
        )

    @property
    def state(self) -> ActuatorState:
        """
        Retorna o estado atual do atuador.

        Returns
        -------
        ActuatorState
            Estado atual do atuador.
        """

        return ActuatorState.from_array(self._state_array)

    def step(self, reference: ActuatorInput) -> ActuatorState:
        """
        Executa um passo de simulação do atuador.

        Parameters
        ----------
        reference : ActuatorInput
            Referência de posição aplicada ao atuador.

        Returns
        -------
        ActuatorState
            Estado do atuador após o passo de integração.
        """

        self._state_array = self._solver.step(
            state=self._state_array, reference=reference.to_array()
        )

        return self.state

    def set_dt(self, dt: float) -> None:
        """
        Atualiza o passo de integração.

        Parameters
        ----------
        dt : float
            Novo passo de integração [s].
        """

        self._solver.set_dt(dt)
