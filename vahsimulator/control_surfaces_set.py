from __future__ import annotations

from typing import Any

import numpy as np

from . import performance_decorator
from .actuator.actuator import Actuator, ActuatorInput


class ControlSurfacesSet:
    """
    Gerencia as quatro superfícies de controle e seus atuadores.

    A classe realiza a alocação dos comandos de controle longitudinal,
    lateral e direcional para as quatro superfícies, além de modelar a
    dinâmica de cada superfície por meio de um atuador.

    A configuração dos atuadores é fornecida por meio de um dicionário,
    normalmente carregado a partir de um arquivo YAML. A criação dos
    atuadores é delegada ao método :meth:`Actuator.from_dict`, evitando
    duplicação da lógica de configuração.

    A interface externa desta classe utiliza graus para posição e comando
    das superfícies. Internamente, os objetos :class:`Actuator` trabalham
    em radianos.

    Esse modelo de :class:`ControlSurfacesSet` não considera resistências
    nos componentes, como atrito, inércia dos atuadores ou torques
    resistivos.

    Parameters
    ----------
    dt : float
        Passo de integração dos atuadores [s].
    config_data : dict[str, Any]
        Configuração dos atuadores. Deve conter a chave
        ``"control_surfaces_set"`` com uma subseção ``"actuator"``
        compatível com :meth:`Actuator.from_dict`.
    seed : int or None, optional
        Semente utilizada para geração de números aleatórios. Se ``None``,
        nenhuma semente é configurada.

    Notes
    -----
    Cada superfície possui seu próprio atuador. Portanto, os estados
    dinâmicos dos quatro atuadores são independentes.
    """

    def __init__(
        self, dt: float, config_data: dict[str, Any], seed: int | None = None
    ) -> None:
        """
        Inicializa o conjunto de superfícies de controle.

        Parameters
        ----------
        dt : float
            Passo de integração dos atuadores [s].
        config_data : dict[str, Any]
            Configuração dos atuadores. Deve conter a chave
            ``"control_surfaces_set"`` com uma subseção ``"actuator"``
            compatível com :meth:`Actuator.from_dict`.
        seed : int or None, optional
            Semente utilizada para geração de números aleatórios. Se
            ``None``, nenhuma semente é configurada.

        Raises
        ------
        ValueError
            Se a configuração dos atuadores for inválida.
        """

        if seed is not None:
            np.random.seed(seed + 4)

        self.dt = dt
        self.actuator_config = config_data["control_surfaces_set"]["actuator"]

        # Cada superfície possui seu próprio atuador para garantir
        # estados dinâmicos independentes.
        self.actuators = [
            Actuator.from_dict(self.actuator_config, dt) for _ in range(4)
        ]

        # Posições das superfícies de controle [deg].
        self.d1 = 0.0
        self.d2 = 0.0
        self.d3 = 0.0
        self.d4 = 0.0

    def _fin_state(self, actuator: Actuator, command_degrees: float) -> float:
        """
        Atualiza o estado de uma superfície de controle.

        O atuador interno trabalha em radianos, enquanto a interface
        desta classe utiliza graus para os comandos e posições das
        superfícies.

        Parameters
        ----------
        actuator : Actuator
            Atuador associado à superfície de controle.
        command_degrees : float
            Comando de posição da superfície [deg].

        Returns
        -------
        float
            Posição atual da superfície de controle [deg].
        """

        command = ActuatorInput(position_reference=np.deg2rad(command_degrees))

        state = actuator.step(command)

        return np.rad2deg(state.position)

    def _fins_control_allocation(
        self,
        delta_p: float,
        delta_q: float,
        delta_r: float,
        error_d1: float,
        error_d2: float,
        error_d3: float,
        error_d4: float,
        cs_config_type: int,
    ) -> None:
        """
        Calcula os comandos das quatro superfícies e atualiza seus atuadores.

        A alocação dos comandos depende da configuração selecionada por
        ``cs_config_type``. Após a determinação dos comandos, cada
        superfície é atualizada pelo seu respectivo atuador.

        Parameters
        ----------
        delta_p : float
            Comando associado ao movimento de rolamento.
        delta_q : float
            Comando associado ao movimento de arfagem.
        delta_r : float
            Comando associado ao movimento de guinada.
        error_d1 : float
            Erro ou perturbação aplicado à superfície 1 [deg].
        error_d2 : float
            Erro ou perturbação aplicado à superfície 2 [deg].
        error_d3 : float
            Erro ou perturbação aplicado à superfície 3 [deg].
        error_d4 : float
            Erro ou perturbação aplicado à superfície 4 [deg].
        cs_config_type : int
            Tipo de configuração das superfícies:

            * ``0``: configuração X;
            * ``1``: configuração com as superfícies 2 e 4 inativas;
            * outros valores: todas as superfícies inativas.

        Returns
        -------
        None
            Os estados das superfícies são atualizados diretamente nos
            atributos ``d1``, ``d2``, ``d3`` e ``d4``.
        """

        if cs_config_type == 0:
            # Configuração X.
            d1_command = -delta_p + delta_q - delta_r
            d2_command = -delta_p + delta_q + delta_r
            d3_command = delta_p + delta_q - delta_r
            d4_command = delta_p + delta_q + delta_r

        elif cs_config_type == 1:
            # Configuração com as superfícies 2 e 4 inativas.
            d1_command = -delta_p + delta_q
            d2_command = 0.0
            d3_command = delta_p + delta_q
            d4_command = 0.0

            error_d2 = 0.0
            error_d4 = 0.0

        else:
            # Todas as superfícies inativas.
            d1_command = 0.0
            d2_command = 0.0
            d3_command = 0.0
            d4_command = 0.0

            error_d1 = 0.0
            error_d2 = 0.0
            error_d3 = 0.0
            error_d4 = 0.0

        # Atualiza os estados dos quatro atuadores.
        self.d1 = self._fin_state(self.actuators[0], d1_command)
        self.d2 = self._fin_state(self.actuators[1], d2_command)
        self.d3 = self._fin_state(self.actuators[2], d3_command)
        self.d4 = self._fin_state(self.actuators[3], d4_command)

        # Aplica os erros ou perturbações das superfícies.
        self.d1 += error_d1
        self.d2 += error_d2
        self.d3 += error_d3
        self.d4 += error_d4

    def set_dt(self, dt: float) -> None:
        """
        Atualiza o passo de integração dos atuadores.

        Parameters
        ----------
        dt : float
            Novo passo de integração [s].
        """

        self.dt = dt

        for actuator in self.actuators:
            actuator.set_dt(dt)

    @performance_decorator.time_execution_stats
    def step(
        self,
        delta_p: float,
        delta_q: float,
        delta_r: float,
        error_d1: float,
        error_d2: float,
        error_d3: float,
        error_d4: float,
        cs_config_type: int,
    ) -> tuple[float, float, float, float]:
        """
        Executa um passo de atualização das superfícies de controle.

        Calcula a alocação dos comandos para as quatro superfícies,
        atualiza os respectivos atuadores e aplica os erros ou
        perturbações especificados.

        Parameters
        ----------
        delta_p : float
            Comando associado ao movimento de rolamento.
        delta_q : float
            Comando associado ao movimento de arfagem.
        delta_r : float
            Comando associado ao movimento de guinada.
        error_d1 : float
            Erro ou perturbação da superfície 1 [deg].
        error_d2 : float
            Erro ou perturbação da superfície 2 [deg].
        error_d3 : float
            Erro ou perturbação da superfície 3 [deg].
        error_d4 : float
            Erro ou perturbação da superfície 4 [deg].
        cs_config_type : int
            Tipo de configuração das superfícies.

        Returns
        -------
        tuple[float, float, float, float]
            Posições das superfícies 1, 2, 3 e 4 [deg], respectivamente.
        """

        self._fins_control_allocation(
            delta_p,
            delta_q,
            delta_r,
            error_d1,
            error_d2,
            error_d3,
            error_d4,
            cs_config_type,
        )

        return self.d1, self.d2, self.d3, self.d4
