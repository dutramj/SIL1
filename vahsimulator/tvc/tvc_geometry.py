# =====================================================================
# Geometria
# =====================================================================


from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True, frozen=True)
class TVCGeometry:
    """
    Representa a geometria cinemática de um sistema de Thrust Vector
    Control (TVC) acionado por um atuador linear.

    O sistema de coordenadas 2-D possui sua origem no ponto de rotação
    da tubeira (divergente). A coordenada ``x`` representa a direção
    longitudinal do foguete e a coordenada ``y`` representa a direção
    transversal.

    A geometria possui duas juntas:

    * uma junta fixa, conectada à estrutura do foguete;
    * uma junta móvel, conectada à tubeira.

    A junta da tubeira é definida em sua posição inicial, correspondente
    a ``tvc_angle = 0``. Para uma deflexão ``tvc_angle`` diferente de
    zero, sua posição é obtida pela rotação da posição inicial em torno
    da origem.

    O deslocamento linear do atuador é definido em relação ao
    comprimento inicial do atuador:

    .. math::

        d = L(\\delta) - L_0

    onde:

    * ``d`` é o deslocamento linear do atuador;
    * ``L(delta)`` é o comprimento do atuador para a deflexão ``delta``;
    * ``L_0`` é o comprimento do atuador para ``delta = 0``.

    Portanto, pela convenção adotada nesta implementação:

    * ``d > 0``: o atuador aumenta de comprimento (extensão);
    * ``d < 0``: o atuador diminui de comprimento (retração);
    * ``d = 0``: o atuador está na posição inicial.

    Para a geometria definida pelo usuário, um ângulo de TVC positivo
    provoca a retração do atuador. Consequentemente:

    .. math::

        \\delta > 0 \\quad \\Rightarrow \\quad d < 0

    Parameters
    ----------
    structure_joint_x : float
        Coordenada longitudinal da junta do atuador conectada à
        estrutura do foguete [m].

    initial_nozzle_joint_x : float
        Coordenada longitudinal da junta do atuador conectada à
        tubeira para ``tvc_angle = 0`` [m].

    initial_nozzle_joint_y : float
        Coordenada transversal da junta do atuador conectada à tubeira
        para ``tvc_angle = 0`` [m].

    initial_actuator_angle : float
        Ângulo entre o eixo longitudinal do foguete e o eixo do
        atuador linear para ``tvc_angle = 0`` [rad].

    Notes
    -----
    A coordenada ``structure_joint_y`` não é um parâmetro independente.
    Ela é calculada a partir da geometria inicial e do ângulo inicial
    do atuador.

    Isso garante que o atuador apresente exatamente
    ``initial_actuator_angle`` na configuração neutra.

    Todas as grandezas geométricas que não dependem de ``tvc_angle``
    são calculadas uma única vez durante a construção do objeto.
    """

    structure_joint_x: float
    initial_nozzle_joint_x: float
    initial_nozzle_joint_y: float
    initial_actuator_angle: float

    # ------------------------------------------------------------------
    # Grandezas geométricas calculadas uma única vez.
    #
    # A classe é frozen=True, portanto esses valores são preenchidos
    # em __post_init__ usando object.__setattr__().
    # ------------------------------------------------------------------

    _structure_joint_y: float = field(init=False)
    _initial_actuator_length: float = field(init=False)

    # Distância da origem até cada uma das juntas.
    _nozzle_radius: float = field(init=False)
    _structure_radius: float = field(init=False)

    # Ângulo polar de cada junta em relação à origem.
    _initial_nozzle_angle: float = field(init=False)
    _structure_angle: float = field(init=False)

    def __post_init__(self):
        """
        Calcula as grandezas geométricas invariantes do mecanismo.

        A junta estrutural é definida como:

        .. math::

            A = (x_s, y_s)

        enquanto a junta da tubeira em sua posição inicial é:

        .. math::

            P_0 = (x_{n0}, y_{n0})

        Como o atuador deve apresentar ``initial_actuator_angle``
        em relação ao eixo longitudinal na posição inicial:

        .. math::

            y_s =
            y_{n0} +
            \\frac{x_s - x_{n0}}
                 {\\tan(\\alpha_0)}

        O comprimento inicial do atuador é então:

        .. math::

            L_0 =
            \\sqrt{
                (x_s-x_{n0})^2 +
                (y_s-y_{n0})^2
            }

        Esses valores são constantes durante a simulação e, portanto,
        são calculados somente uma vez.
        """

        # --------------------------------------------------------------
        # Calcula a coordenada transversal da junta estrutural.
        #
        # O sinal da expressão é determinado pela definição do eixo
        # longitudinal e pelo ângulo positivo adotado para o atuador.
        # --------------------------------------------------------------

        structure_joint_y = self.initial_nozzle_joint_y + (
            self.structure_joint_x - self.initial_nozzle_joint_x
        ) / np.tan(self.initial_actuator_angle)

        # --------------------------------------------------------------
        # Comprimento do atuador na configuração inicial.
        #
        # IMPORTANTE:
        # Esta é a distância entre as DUAS JUNTAS do atuador:
        #
        #       L0 = |P0 - A|
        #
        # e não a distância entre a junta estrutural e a origem.
        # --------------------------------------------------------------

        initial_actuator_length = np.hypot(
            self.structure_joint_x - self.initial_nozzle_joint_x,
            structure_joint_y - self.initial_nozzle_joint_y,
        )

        # --------------------------------------------------------------
        # Distância da origem até a junta da tubeira.
        #
        # Essa distância permanece constante porque a junta apenas
        # gira em torno da origem.
        # --------------------------------------------------------------

        nozzle_radius = np.hypot(
            self.initial_nozzle_joint_x, self.initial_nozzle_joint_y
        )

        # --------------------------------------------------------------
        # Distância da origem até a junta estrutural.
        #
        # Como a junta estrutural é fixa, essa distância também é
        # constante.
        # --------------------------------------------------------------

        structure_radius = np.hypot(self.structure_joint_x, structure_joint_y)

        # --------------------------------------------------------------
        # Ângulos polares das duas juntas.
        #
        # Esses valores são utilizados posteriormente na conversão
        # inversa:
        #
        # actuator displacement -> TVC angle
        # --------------------------------------------------------------

        initial_nozzle_angle = np.arctan2(
            self.initial_nozzle_joint_y, self.initial_nozzle_joint_x
        )

        structure_angle = np.arctan2(structure_joint_y, self.structure_joint_x)

        # --------------------------------------------------------------
        # Como a classe é frozen=True, os valores calculados precisam
        # ser atribuídos usando object.__setattr__().
        # --------------------------------------------------------------

        object.__setattr__(self, "_structure_joint_y", structure_joint_y)

        object.__setattr__(self, "_initial_actuator_length", initial_actuator_length)

        object.__setattr__(self, "_nozzle_radius", nozzle_radius)

        object.__setattr__(self, "_structure_radius", structure_radius)

        object.__setattr__(self, "_initial_nozzle_angle", initial_nozzle_angle)

        object.__setattr__(self, "_structure_angle", structure_angle)

    @property
    def structure_joint_y(self) -> float:
        """
        Retorna a coordenada transversal da junta estrutural.

        Returns
        -------
        float
            Coordenada ``y`` da junta estrutural [m].
        """

        return self._structure_joint_y

    @property
    def initial_actuator_length(self) -> float:
        """
        Retorna o comprimento do atuador na posição inicial.

        A posição inicial corresponde a ``tvc_angle = 0``.

        Returns
        -------
        float
            Comprimento inicial do atuador [m].
        """

        return self._initial_actuator_length

    @property
    def initial_nozzle_joint_position(self) -> np.ndarray:
        """
        Retorna a posição inicial da junta do atuador na tubeira.

        Essa posição corresponde a ``tvc_angle = 0``.

        Returns
        -------
        numpy.ndarray
            Vetor ``[x, y]`` da junta da tubeira [m].
        """

        return np.array(
            [self.initial_nozzle_joint_x, self.initial_nozzle_joint_y], dtype=np.float64
        )

    @property
    def structure_joint_position(self) -> np.ndarray:
        """
        Retorna a posição da junta do atuador conectada à estrutura.

        A junta estrutural é fixa durante a operação do TVC.

        Returns
        -------
        numpy.ndarray
            Vetor ``[x, y]`` da junta estrutural [m].
        """

        return np.array(
            [self.structure_joint_x, self.structure_joint_y], dtype=np.float64
        )

    def nozzle_joint_position_at_angle(self, tvc_angle: float) -> np.ndarray:
        """
        Calcula a posição da junta da tubeira para uma determinada
        deflexão do TVC.

        A junta da tubeira gira rigidamente em torno da origem.
        Portanto, sua posição é obtida pela matriz de rotação
        bidimensional:

        .. math::

            P(\\delta) =
            R(\\delta) P_0

        com:

        .. math::

            R(\\delta) =
            \\begin{bmatrix}
                \\cos\\delta & -\\sin\\delta \\\\
                \\sin\\delta &  \\cos\\delta
            \\end{bmatrix}

        Assim:

        .. math::

            x_n =
            x_{n0}\\cos\\delta
            -
            y_{n0}\\sin\\delta

        .. math::

            y_n =
            x_{n0}\\sin\\delta
            +
            y_{n0}\\cos\\delta

        Parameters
        ----------
        tvc_angle : float
            Ângulo de deflexão da tubeira em relação à posição neutra
            [rad].

        Returns
        -------
        numpy.ndarray
            Posição ``[x, y]`` da junta da tubeira [m].
        """

        cos_angle = np.cos(tvc_angle)
        sin_angle = np.sin(tvc_angle)

        x = (
            cos_angle * self.initial_nozzle_joint_x
            - sin_angle * self.initial_nozzle_joint_y
        )

        y = (
            sin_angle * self.initial_nozzle_joint_x
            + cos_angle * self.initial_nozzle_joint_y
        )

        return np.array([x, y], dtype=np.float64)

    def actuator_length(self, tvc_angle: float) -> float:
        """
        Calcula o comprimento do atuador para uma determinada
        deflexão da tubeira.

        O comprimento é simplesmente a distância euclidiana entre a
        junta fixa da estrutura e a junta móvel da tubeira:

        .. math::

            L(\\delta) =
            \\left\\|
                P(\\delta) - A
            \\right\\|

        ou, explicitamente:

        .. math::

            L(\\delta) =
            \\sqrt{
                [x_n(\\delta)-x_s]^2 +
                [y_n(\\delta)-y_s]^2
            }

        Parameters
        ----------
        tvc_angle : float
            Ângulo de deflexão da tubeira [rad].

        Returns
        -------
        float
            Comprimento instantâneo do atuador [m].
        """

        nozzle_position = self.nozzle_joint_position_at_angle(tvc_angle)

        return np.hypot(
            nozzle_position[0] - self.structure_joint_x,
            nozzle_position[1] - self.structure_joint_y,
        )

    def tvc_angle_to_actuator_displacement(self, tvc_angle: float) -> float:
        """
        Converte o ângulo de TVC em deslocamento linear do atuador.

        O deslocamento é definido em relação ao comprimento inicial:

        .. math::

            d(\\delta) =
            L(\\delta) - L_0

        onde:

        * ``L(delta)`` é o comprimento do atuador para o ângulo
          ``delta``;
        * ``L_0`` é o comprimento do atuador para ``delta = 0``.

        Essa definição estabelece a seguinte convenção de sinal:

        .. math::

            d > 0
            \\quad\\Rightarrow\\quad
            \\text{extensão do atuador}

        .. math::

            d < 0
            \\quad\\Rightarrow\\quad
            \\text{retração do atuador}

        Para a geometria e convenção de sinais adotadas pelo sistema:

        .. math::

            \\delta > 0
            \\quad\\Rightarrow\\quad
            d < 0

        Portanto, um comando de TVC positivo corresponde a uma redução
        do comprimento do atuador.

        Parameters
        ----------
        tvc_angle : float
            Ângulo de deflexão da tubeira [rad].

        Returns
        -------
        float
            Deslocamento do atuador em relação à posição inicial [m].
            Valores negativos representam retração e valores positivos
            representam extensão.
        """

        return self.actuator_length(tvc_angle) - self.initial_actuator_length

    def actuator_displacement_to_tvc_angle(self, actuator_displacement: float) -> float:
        """
        Converte o deslocamento linear do atuador em ângulo de TVC.

        O comprimento desejado do atuador é inicialmente obtido a partir
        do deslocamento em relação à posição inicial:

        .. math::

            L =
            L_0 + d

        onde ``d`` é o deslocamento do atuador.

        A conversão inversa é então obtida pela lei dos cossenos.

        Seja:

        * ``r_n`` a distância da origem até a junta da tubeira;
        * ``r_s`` a distância da origem até a junta estrutural;
        * ``L`` o comprimento desejado do atuador.

        Então:

        .. math::

            L^2 =
            r_n^2 +
            r_s^2 -
            2r_nr_s\\cos(\\theta)

        Portanto:

        .. math::

            \\theta =
            \\arccos
            \\left(
                \\frac{
                    r_n^2+r_s^2-L^2
                }{
                    2r_nr_s
                }
            \\right)

        Os ângulos absolutos das juntas permitem relacionar ``theta``
        com o ângulo de deflexão da tubeira:

        .. math::

            \\theta =
            \\delta +
            \\phi_n -
            \\phi_s

        resultando em duas possíveis soluções geométricas:

        .. math::

            \\delta_1 =
            \\phi_s - \\phi_n + \\theta

        e

        .. math::

            \\delta_2 =
            \\phi_s - \\phi_n - \\theta

        A solução mais próxima de ``delta = 0`` é selecionada. Isso
        corresponde ao ramo geométrico associado à configuração
        operacional próxima da posição neutra.

        Parameters
        ----------
        actuator_displacement : float
            Deslocamento linear do atuador em relação à posição inicial
            [m].

            Valores negativos representam retração do atuador e,
            conforme a convenção adotada, correspondem a ângulos de TVC
            positivos.

        Returns
        -------
        float
            Ângulo de deflexão da tubeira [rad].

        Raises
        ------
        ValueError
            Se o comprimento resultante do atuador for negativo.

        ValueError
            Se o comprimento solicitado não for geometricamente
            realizável pelo mecanismo TVC.
        """

        # --------------------------------------------------------------
        # Recupera o comprimento absoluto correspondente ao
        # deslocamento solicitado.
        #
        # L = L0 + displacement
        # --------------------------------------------------------------

        target_length = self.initial_actuator_length + actuator_displacement

        if target_length < 0.0:
            raise ValueError(
                "O comprimento resultante do atuador linear " "não pode ser negativo."
            )

        # --------------------------------------------------------------
        # Lei dos cossenos.
        #
        # Para um triângulo formado pela origem, pela junta estrutural
        # e pela junta da tubeira:
        #
        #       L² = rn² + rs² - 2 rn rs cos(theta)
        #
        # Isolando cos(theta):
        #
        #       cos(theta) =
        #           (rn² + rs² - L²) / (2 rn rs)
        # --------------------------------------------------------------

        cosine_argument = (
            self._nozzle_radius**2 + self._structure_radius**2 - target_length**2
        ) / (2.0 * self._nozzle_radius * self._structure_radius)

        # --------------------------------------------------------------
        # Um valor fora de [-1, 1] indica que o comprimento solicitado
        # não pode ser obtido pela geometria atual.
        #
        # A pequena tolerância numérica pode ser tratada pelo clip
        # abaixo quando o valor estiver ligeiramente fora do intervalo
        # devido a erro de ponto flutuante.
        # --------------------------------------------------------------

        if cosine_argument < -1.0 or cosine_argument > 1.0:
            raise ValueError(
                "O deslocamento linear informado não é "
                "geometricamente realizável pelo mecanismo TVC."
            )

        cosine_argument = np.clip(cosine_argument, -1.0, 1.0)

        relative_angle = np.arccos(cosine_argument)

        # --------------------------------------------------------------
        # Existem duas configurações geométricas possíveis para o mesmo
        # comprimento do atuador.
        #
        # A configuração desejada é obtida relacionando os ângulos
        # absolutos das duas juntas:
        #
        #       delta = structure_angle
        #               - nozzle_angle
        #               +/- relative_angle
        # --------------------------------------------------------------

        delta_1 = self._structure_angle - self._initial_nozzle_angle + relative_angle

        delta_2 = self._structure_angle - self._initial_nozzle_angle - relative_angle

        # --------------------------------------------------------------
        # Seleciona o ramo mais próximo da configuração neutra.
        #
        # Isso evita selecionar a solução geometricamente espelhada
        # quando o mecanismo está operando próximo de delta = 0.
        # --------------------------------------------------------------

        if abs(delta_1) < abs(delta_2):
            return delta_1

        return delta_2
    