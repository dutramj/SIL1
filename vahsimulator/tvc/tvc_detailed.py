import numpy as np

from .. import performance_decorator
from .tvc import TVCBase
from .tvc_geometry import TVCGeometry
from vahsimulator.actuator.actuator import Actuator, ActuatorInput


# =====================================================================
# TVC
# =====================================================================


class TVCDetailed(TVCBase):
    """
    Segundo modelo do sistema Thrust Vector Control (TVC).

    Esse TVC é composto por dois conjuntos independentes, um para pitch e
    outro para yaw. Cada conjunto possui:

    * atuador angular;
    * redução por correia;
    * conversão rotação-deslocamento;
    * atuador linear;
    * geometria cinemática;
    * deflexão final da tubeira.

    O comando de entrada representa diretamente o ângulo desejado de
    deflexão da tubeira. Esse comando é convertido pela geometria
    inversa no deslocamento linear necessário do atuador.

    A cadeia cinemática de cada eixo é:

        TVC angle command
            -> TVC geometry inverse
            -> actuator displacement
            -> lead screw
            -> angular actuator reference
            -> angular actuator
            -> belt reduction
            -> linear actuator reference
            -> linear actuator
            -> TVC geometry direct
            -> actual TVC angle

    A convenção de sinais utilizada pela geometria é:

        tvc_angle > 0
            -> actuator displacement < 0
            -> actuator retracts

        tvc_angle < 0
            -> actuator displacement > 0
            -> actuator extends

        tvc_angle = 0
            -> actuator displacement = 0

    Portanto, esse modelo de TVC leva em consideração a
    dinâmica dos atuadores e as geometrias entre os
    componentes.
    
    Porém, esse modelo de TVC não considera a dinâmica
    do divergente (ex: inércia da tubeira, torque de mola
    torsional) e nem as resistências nos componentes (ex:
    atritos, inércias dos atuadores linear e angular,
    torques resistivos, etc).

    Parameters
    ----------
    dt : float
        Passo de integração utilizado pelos atuadores [s].

    Attributes
    ----------
    delta_q : float
        Deflexão atual da tubeira em pitch [rad].

    delta_r : float
        Deflexão atual da tubeira em yaw [rad].
    """

    def __init__(self, dt: float):
        """
        Inicializa o modelo do TVC.

        Parameters
        ----------
        dt : float
            Passo de integração dos atuadores [s].
        """

        self._tvc_config = {}
        self._dt = dt
        self._phase_id = None

        # Geometria cinemática compartilhada pelos eixos de pitch e yaw.
        self._geometry = None

        # Relação de redução entre o atuador angular e o atuador linear.
        self._belt_ratio = 1.0

        # Avanço do mecanismo rotação-deslocamento [m/rev].
        self._linear_lead = 0.0

        # Atuadores angulares.
        self._pitch_angular_actuator = None
        self._yaw_angular_actuator = None

        # Atuadores lineares.
        self._pitch_linear_actuator = None
        self._yaw_linear_actuator = None

        # Deflexões atuais da tubeira.
        self._delta_q = 0.0
        self._delta_r = 0.0

    @property
    def delta_q(self) -> float:
        """
        Retorna a deflexão atual da tubeira em pitch.

        Returns
        -------
        float
            Deflexão em pitch [rad].
        """

        return self._delta_q

    @property
    def delta_r(self) -> float:
        """
        Retorna a deflexão atual da tubeira em yaw.

        Returns
        -------
        float
            Deflexão em yaw [rad].
        """

        return self._delta_r

    def _configure(self, tvc_config: dict) -> None:
        """
        Configura o TVC para uma determinada fase de voo.

        A configuração é realizada somente quando a fase recebida é
        diferente da fase atualmente configurada.

        A geometria inicial da TVC é definida pela posição das duas
        juntas e pelo ângulo inicial do atuador.

        A junta estrutural é calculada internamente por ``TVCGeometry``.
        Portanto, ``structure_joint_y`` não precisa ser fornecido na
        configuração.

        Parameters
        ----------
        tvc_config : dict
            Configuração do TVC correspondente à fase de voo.
        """

        phase_id = tvc_config["phase_id"]

        # Evita reconstruir os atuadores e a geometria quando a fase
        # de voo não mudou.
        if phase_id == self._phase_id:
            return

        self._phase_id = phase_id

        self._tvc_config.update(tvc_config)

        # --------------------------------------------------------------
        # Geometria do TVC
        # --------------------------------------------------------------
        #
        # As coordenadas da junta da tubeira são as coordenadas
        # correspondentes à posição neutra:
        #
        #     tvc_angle = 0
        #
        # Por isso utilizamos explicitamente o prefixo "initial".
        # --------------------------------------------------------------

        geometry_config = self._tvc_config["geometry"]

        self._geometry = TVCGeometry(
            structure_joint_x=geometry_config["structure_joint_x"],
            initial_nozzle_joint_x=geometry_config["initial_nozzle_joint_x"],
            initial_nozzle_joint_y=geometry_config["initial_nozzle_joint_y"],
            initial_actuator_angle=np.deg2rad(
                geometry_config["initial_actuator_angle"]
            ),
        )

        # --------------------------------------------------------------
        # Redução por correia
        # --------------------------------------------------------------

        self._belt_ratio = self._tvc_config["belt"]["reduction_ratio"]

        # --------------------------------------------------------------
        # Atuadores
        # --------------------------------------------------------------

        angular_config = self._tvc_config["angular_actuator"]
        
        linear_config = self._tvc_config["linear_actuator"]

        self._pitch_angular_actuator = self._create_actuator(angular_config)

        self._yaw_angular_actuator = self._create_actuator(angular_config)

        self._pitch_linear_actuator = self._create_actuator(linear_config)

        self._yaw_linear_actuator = self._create_actuator(linear_config)

        self._linear_lead = linear_config["lead"]

    def _create_actuator(self, actuator_config: dict) -> Actuator:
        """
        Cria um atuador a partir da configuração fornecida.

        Parameters
        ----------
        actuator_config : dict
            Configuração do atuador.

        Returns
        -------
        Actuator
            Atuador configurado.
        """

        return Actuator.from_dict(config=actuator_config, dt=self._dt)

    def _rotation_to_linear_displacement(self, angle: float) -> float:
        """
        Converte rotação em deslocamento linear utilizando o avanço
        do mecanismo.

        Para um mecanismo com avanço ``lead`` [m/rev], uma rotação
        de uma volta completa produz um deslocamento:

        .. math::

            x = lead

        Como o ângulo recebido está em radianos:

        .. math::

            N = \\frac{\\theta}{2\\pi}

        onde ``N`` é o número de revoluções.

        Portanto:

        .. math::

            x =
            lead \\frac{\\theta}{2\\pi}

        Parameters
        ----------
        angle : float
            Rotação do mecanismo [rad].

        Returns
        -------
        float
            Deslocamento linear [m].
        """

        return self._linear_lead * angle / (2.0 * np.pi)

    def _tvc_angle_to_angular_actuator_reference(self, tvc_angle: float) -> float:
        """
        Converte um comando de ângulo de TVC na referência angular
        do atuador angular.

        A conversão é realizada em duas etapas.

        Primeiro, a geometria inversa converte o ângulo da tubeira
        no deslocamento linear requerido pelo atuador:

        .. math::

            d =
            f(\\delta)

        através de:

        ``TVCGeometry.tvc_angle_to_actuator_displacement()``.

        Em seguida, o deslocamento linear é convertido em rotação
        através do avanço do mecanismo:

        .. math::

            \\theta_l =
            \\frac{2\\pi d}{lead}

        Por fim, a redução por correia é aplicada:

        .. math::

            \\theta_a =
            ratio\\,\\theta_l

        onde:

        * ``d`` é o deslocamento linear do atuador;
        * ``theta_l`` é a rotação do mecanismo linear;
        * ``theta_a`` é a referência do atuador angular.

        A convenção de sinal da geometria é preservada durante toda
        essa conversão. Portanto, para a geometria atual:

        .. math::

            \\delta > 0
            \\Rightarrow d < 0

        Parameters
        ----------
        tvc_angle : float
            Comando de deflexão da tubeira [rad].

        Returns
        -------
        float
            Posição angular de referência do atuador angular [rad].
        """

        # --------------------------------------------------------------
        # TVC angle -> actuator linear displacement
        # --------------------------------------------------------------

        actuator_displacement = self._geometry.tvc_angle_to_actuator_displacement(
            tvc_angle
        )

        # --------------------------------------------------------------
        # Actuator linear displacement -> linear actuator rotation
        #
        #     displacement = lead * angle / (2*pi)
        #
        # portanto:
        #
        #     angle = displacement * 2*pi / lead
        # --------------------------------------------------------------

        linear_actuator_angle = actuator_displacement * 2.0 * np.pi / self._linear_lead

        # --------------------------------------------------------------
        # Linear actuator rotation -> angular actuator rotation
        #
        # A redução por correia é aplicada conforme a convenção
        # definida em belt_ratio.
        # --------------------------------------------------------------

        return self._belt_ratio * linear_actuator_angle

    @performance_decorator.time_execution_stats
    def step(
        self, config_data: dict, delta_q_cmd: float, delta_r_cmd: float, phase: int
    ) -> tuple[float, float]:
        """
        Executa um passo de simulação do sistema TVC.

        Os comandos ``delta_q_cmd`` e ``delta_r_cmd`` representam
        diretamente os ângulos desejados de deflexão da tubeira.

        Para cada eixo, a cadeia de conversão é:

        .. code-block:: text

            TVC angle command
                |
                v
            TVC geometry inverse
                |
                v
            actuator displacement
                |
                v
            linear lead
                |
                v
            angular actuator reference
                |
                v
            angular actuator
                |
                v
            belt reduction
                |
                v
            linear actuator reference
                |
                v
            linear actuator
                |
                v
            TVC geometry direct
                |
                v
            actual TVC angle

        A geometria inversa converte o comando de TVC em deslocamento
        linear:

        .. math::

            d_{cmd} =
            f^{-1}(\\delta_{cmd})

        O avanço do mecanismo converte esse deslocamento em rotação:

        .. math::

            \\theta_l =
            \\frac{2\\pi d_{cmd}}{lead}

        e a redução por correia converte a rotação do mecanismo na
        referência do atuador angular:

        .. math::

            \\theta_a =
            ratio\\,\\theta_l

        Depois que os atuadores evoluem um passo de integração, a
        posição do atuador linear é convertida novamente em ângulo
        de TVC através da geometria direta.

        Parameters
        ----------
        config_data : dict
            Configuração do simulador contendo os parâmetros do TVC.

        delta_q_cmd : float
            Comando de deflexão desejada da tubeira em pitch [deg].

        delta_r_cmd : float
            Comando de deflexão desejada da tubeira em yaw [deg].

        phase : int
            Identificador da fase de voo.

        Returns
        -------
        tuple[float, float]
            Deflexões atuais da tubeira:

            ``(delta_q, delta_r)`` em [rad].

        Raises
        ------
        ValueError
            Caso não exista configuração de TVC para a fase informada.
        """

        # --------------------------------------------------------------
        # Localiza a configuração correspondente à fase atual.
        # --------------------------------------------------------------

        tvc_config = next(
            (
                item
                for item in config_data["tvc_parameters"]
                if item["phase_id"] == phase
            ),
            None,
        )

        if tvc_config is None:
            raise ValueError(
                f"Nenhuma configuração de TVC encontrada " f"para phase_id={phase}."
            )

        # --------------------------------------------------------------
        # Configura a geometria e os atuadores somente se a fase mudou.
        # --------------------------------------------------------------

        self._configure(tvc_config)

        # ==============================================================
        # 1. TVC COMMAND -> ANGULAR ACTUATOR REFERENCE
        # ==============================================================

        pitch_angular_reference = self._tvc_angle_to_angular_actuator_reference(
            np.deg2rad(delta_q_cmd)
        )

        yaw_angular_reference = self._tvc_angle_to_angular_actuator_reference(
            np.deg2rad(delta_r_cmd)
        )

        # ==============================================================
        # 2. ANGULAR ACTUATOR
        # ==============================================================

        pitch_angular_state = self._pitch_angular_actuator.step(
            ActuatorInput(position_reference=pitch_angular_reference)
        )

        yaw_angular_state = self._yaw_angular_actuator.step(
            ActuatorInput(position_reference=yaw_angular_reference)
        )

        # ==============================================================
        # 3. ANGULAR ACTUATOR -> LINEAR ACTUATOR ROTATION
        # ==============================================================

        # O atuador angular está conectado ao mecanismo linear através
        # da redução por correia.
        #
        #     theta_linear = theta_angular / belt_ratio
        #
        # Essa relação deve ser consistente com a definição de
        # reduction_ratio utilizada na configuração do TVC.

        pitch_linear_actuator_angle = pitch_angular_state.position / self._belt_ratio

        yaw_linear_actuator_angle = yaw_angular_state.position / self._belt_ratio

        # ==============================================================
        # 4. LINEAR ACTUATOR ROTATION -> LINEAR DISPLACEMENT
        # ==============================================================

        pitch_linear_reference = self._rotation_to_linear_displacement(
            pitch_linear_actuator_angle
        )

        yaw_linear_reference = self._rotation_to_linear_displacement(
            yaw_linear_actuator_angle
        )

        # ==============================================================
        # 5. LINEAR ACTUATOR
        # ==============================================================

        pitch_linear_state = self._pitch_linear_actuator.step(
            ActuatorInput(position_reference=pitch_linear_reference)
        )

        yaw_linear_state = self._yaw_linear_actuator.step(
            ActuatorInput(position_reference=yaw_linear_reference)
        )

        # ==============================================================
        # 6. LINEAR ACTUATOR DISPLACEMENT -> TVC ANGLE
        # ==============================================================

        # A posição do atuador linear é medida em relação à posição
        # inicial. A geometria direta converte esse deslocamento no
        # ângulo efetivamente produzido pela tubeira.

        self._delta_q = self._geometry.actuator_displacement_to_tvc_angle(
            pitch_linear_state.position
        )

        self._delta_r = self._geometry.actuator_displacement_to_tvc_angle(
            yaw_linear_state.position
        )

        return self._delta_q, self._delta_r

    def set_dt(self, dt: float) -> None:
        """
        Atualiza o passo de integração de todos os atuadores do TVC.

        Parameters
        ----------
        dt : float
            Novo passo de integração [s].
        """

        self._dt = dt

        if self._pitch_angular_actuator is not None:
            self._pitch_angular_actuator.set_dt(dt)

        if self._yaw_angular_actuator is not None:
            self._yaw_angular_actuator.set_dt(dt)

        if self._pitch_linear_actuator is not None:
            self._pitch_linear_actuator.set_dt(dt)

        if self._yaw_linear_actuator is not None:
            self._yaw_linear_actuator.set_dt(dt)
