import pytest
import matplotlib.pyplot as plt
import numpy as np

from dataclasses import dataclass
from scipy.integrate import solve_ivp
from scipy.ndimage import binary_dilation

from vahsimulator.actuator.actuator import Actuator, ActuatorInput, ActuatorParameters
from vahsimulator.actuator.nonlinearities import (
    acceleration_limiter_factory,
    backlash_factory,
    nonlinearities_pipeline_factory,
    position_limiter_factory,
    velocity_dependent_acceleration_limiter_factory,
    velocity_limiter_factory,
)


# =====================================================================
# Configuração
# =====================================================================


@dataclass
class ActuatorTestConfig:
    """Configuração compartilhada dos testes do atuador.

    Armazena os parâmetros numéricos da simulação, os parâmetros
    dinâmicos do atuador e os limites físicos utilizados nos testes
    de validação.

    Attributes
    ----------
    dt : float, optional
        Passo de integração utilizado na simulação [s].
    duration : float, optional
        Duração total da simulação [s].
    wn : float, optional
        Frequência natural do atuador [rad/s].
    zeta : float, optional
        Razão de amortecimento do atuador [-].
    max_position_deg : float, optional
        Limite absoluto de posição do atuador [deg].
    max_velocity_deg : float, optional
        Limite absoluto de velocidade do atuador [deg/s].
    max_acceleration_deg : float, optional
        Limite absoluto de aceleração do atuador [deg/s²].
    """

    dt: float = 0.0001
    duration: float = 0.2

    wn: float = 62.8
    zeta: float = 0.7

    max_position_deg: float = 10.0
    max_velocity_deg: float = 100.0
    max_acceleration_deg: float = 5000.0


# =====================================================================
# Simulação
# =====================================================================


def run_step_response(
    *,
    command_deg: float = 1.0,
    config: ActuatorTestConfig = ActuatorTestConfig(),
    nonlinearities=None,
):
    """Executa a resposta ao degrau de um atuador.

    Cria um atuador com os parâmetros especificados, aplica uma
    referência constante de posição e executa a simulação durante o
    intervalo de tempo definido pela configuração.

    A posição calculada pelo atuador é convertida de radianos para
    graus antes de ser armazenada no resultado.

    Uma segunda resposta é calculada por meio de uma integração
    contínua de alta precisão da mesma dinâmica de segunda ordem.
    Essa resposta é utilizada como referência nos testes do atuador
    ideal.

    Parameters
    ----------
    command_deg : float, optional
        Comando de posição aplicado ao atuador [deg].
    config : ActuatorTestConfig, optional
        Configuração utilizada para a simulação.
    nonlinearities : Callable, optional
        Pipeline de não linearidades aplicada ao estado do atuador.
        Quando ``None``, o atuador é considerado ideal e sua dinâmica
        corresponde diretamente ao modelo linear de segunda ordem.

    Returns
    -------
    time : numpy.ndarray
        Vetor de tempo da simulação [s].
    reference : numpy.ndarray
        Resposta contínua de referência da dinâmica de segunda ordem
        [deg].
    actual_actuator : numpy.ndarray
        Resposta de posição obtida pelo modelo do atuador [deg].

    Notes
    -----
    A dinâmica interna do atuador utiliza radianos, enquanto os
    resultados retornados por esta função são expressos em graus.
    """

    command_rad = np.deg2rad(command_deg)

    actuator_input = ActuatorInput(position_reference=command_rad)

    # Tempo de simulação.
    n_steps = int(round(config.duration / config.dt))
    time = np.arange(n_steps) * config.dt

    # Parâmetros do atuador.
    parameters = ActuatorParameters(
        natural_frequency=config.wn, damping_ratio=config.zeta
    )

    # Criação do atuador.
    actuator = Actuator(
        parameters=parameters, dt=config.dt, nonlinearities=nonlinearities
    )

    # Simulação.
    actual_actuator = np.zeros(n_steps)

    for k in range(n_steps):
        state = actuator.step(actuator_input)
        actual_actuator[k] = np.rad2deg(state.position)

    # Referência contínua.
    reference = second_order_reference(
        time=time, wn=config.wn, zeta=config.zeta, command=command_deg
    )

    return time, reference, actual_actuator


# =====================================================================
# Referência contínua
# =====================================================================


def second_order_reference(
    time: np.ndarray, wn: float, zeta: float, command: float
) -> np.ndarray:
    """Calcula a resposta de referência de um sistema contínuo de segunda ordem.

    A resposta é obtida por integração numérica da seguinte equação
    diferencial:

    .. math::

        \\ddot{x} + 2 \\zeta \\omega_n \\dot{x}
        + \\omega_n^2 x = \\omega_n^2 u

    Na forma de espaço de estados:

    .. math::

        \\dot{x} = v

        \\dot{v} =
        \\omega_n^2 (u - x) - 2 \\zeta \\omega_n v

    O sistema parte das condições iniciais:

    .. math::

        x(0) = 0

        v(0) = 0

    Essa solução é utilizada como referência para verificar se o
    atuador ideal reproduz corretamente a dinâmica contínua esperada.

    Parameters
    ----------
    time : numpy.ndarray
        Vetor de instantes de tempo nos quais a solução deve ser
        avaliada [s].
    wn : float
        Frequência natural do sistema [rad/s].
    zeta : float
        Razão de amortecimento do sistema [-].
    command : float
        Comando constante de posição aplicado ao sistema [deg].

    Returns
    -------
    numpy.ndarray
        Posição do sistema nos instantes especificados em ``time`` [deg].

    Notes
    -----
    A integração é realizada com :func:`scipy.integrate.solve_ivp`
    utilizando tolerâncias significativamente menores que as
    utilizadas na simulação principal. Dessa forma, a solução
    obtida serve como referência numérica de alta precisão.
    """

    def dynamics(t, x):
        position = x[0]
        velocity = x[1]

        acceleration = wn**2 * (command - position) - 2.0 * zeta * wn * velocity

        return [velocity, acceleration]

    solution = solve_ivp(
        dynamics,
        (time[0], time[-1]),
        y0=[0.0, 0.0],
        t_eval=time,
        rtol=1e-11,
        atol=1e-13,
    )

    return solution.y[0]


# =====================================================================
# Plot
# =====================================================================


def plot_results(
    time: np.ndarray,
    reference: np.ndarray,
    actual_actuator: np.ndarray,
    *,
    derivative: int = 0,
    prefix_title: str = "",
    prefix_ylabel: str = "",
) -> None:
    """Plota a resposta do atuador e o erro em relação à referência.

    A figura possui dois gráficos:

    - resposta do atuador e referência;
    - erro entre a resposta do atuador e a referência.

    O parâmetro ``derivative`` identifica a grandeza apresentada no
    eixo vertical:

    - ``derivative=0``: posição;
    - ``derivative=1``: velocidade;
    - ``derivative=2``: aceleração.

    Parameters
    ----------
    time : numpy.ndarray
        Vetor de tempo da simulação [s].
    reference : numpy.ndarray
        Resposta de referência da grandeza analisada.
    actual_actuator : numpy.ndarray
        Resposta correspondente obtida pelo modelo do atuador.
    derivative : int, optional
        Ordem da derivada representada no gráfico. O valor padrão
        ``0`` representa a posição.
    prefix_title : str, optional
        Texto adicionado ao início do título da figura.

    Returns
    -------
    None

    Notes
    -----
    A unidade apresentada no eixo vertical é construída
    automaticamente a partir da ordem da derivada.
    """

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

    fig.suptitle(prefix_title + "Resposta ao degrau do atuador de segunda ordem")

    unit = "graus"

    for _ in range(derivative):
        unit += "/s"

    ax1.set_ylabel(f"{prefix_ylabel}atuador [{unit}]")

    ax1.plot(time, actual_actuator, label="Atuador")

    ax1.plot(time, reference, ":", label="Referência - segunda ordem")

    ax2.set_ylabel(f"Erro do atuador [{unit}]")

    ax2.plot(time, actual_actuator - reference, "--", label="Erro")

    for ax in (ax1, ax2):
        ax.set_xlabel("Tempo [s]")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()


# =====================================================================
# Teste: resposta ideal
# =====================================================================


def test_step_response_matches_second_order_reference(
    request: pytest.FixtureRequest,
) -> None:
    """Verifica a resposta ao degrau de um atuador ideal de segunda ordem.

    O teste compara a resposta obtida pelo atuador com a resposta
    contínua de referência calculada numericamente por
    :func:`second_order_reference`.

    Como nenhuma não linearidade é fornecida, o atuador deve reproduzir
    a dinâmica linear ideal de segunda ordem.

    A comparação é realizada utilizando
    :func:`numpy.testing.assert_allclose`, com tolerâncias relativa e
    absoluta definidas para acomodar o erro numérico introduzido pela
    integração discreta.

    Quando a opção ``--plot`` do pytest está habilitada, também são
    apresentados os gráficos de posição, velocidade e aceleração.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Fixture do pytest utilizada para verificar se a opção
        ``--plot`` foi habilitada.

    Raises
    ------
    AssertionError
        Se a resposta do atuador não for suficientemente próxima da
        resposta contínua de referência.
    """

    command_deg = 1.0
    config = ActuatorTestConfig()

    time, ref_pos, act_pos = run_step_response(
        command_deg=command_deg, config=config, nonlinearities=None
    )

    # Derivadas numéricas.
    act_vel = np.gradient(act_pos) / config.dt
    ref_vel = np.gradient(ref_pos) / config.dt

    act_acc = np.gradient(act_vel) / config.dt
    ref_acc = np.gradient(ref_vel) / config.dt

    # Plot.
    if request.config.getoption("--plot"):
        prefix_title = "Ausência de interferência dos limites - "

        plot_results(
            time,
            ref_pos,
            act_pos,
            prefix_title=prefix_title,
            prefix_ylabel="Posição do ",
        )

        plot_results(
            time,
            ref_vel,
            act_vel,
            prefix_title=prefix_title,
            prefix_ylabel="Velocidade do ",
            derivative=1,
        )

        plot_results(
            time,
            ref_acc,
            act_acc,
            prefix_title=prefix_title,
            prefix_ylabel="Aceleração do ",
            derivative=2,
        )

    # Verificação.
    np.testing.assert_allclose(
        act_pos,
        ref_pos,
        rtol=1e-3,
        atol=1e-2,
        err_msg=(
            "O atuador não corresponde à referência contínua " "de segunda ordem."
        ),
    )


# =====================================================================
# Teste: todos os limites
# =====================================================================


def test_actuator_all_limits(request: pytest.FixtureRequest) -> None:
    """Verifica simultaneamente os limites físicos do atuador.

    O teste aplica um comando de posição superior ao limite máximo
    permitido e verifica se o atuador respeita as restrições físicas
    configuradas.

    São verificadas:

    - limite absoluto de posição;
    - limite absoluto de velocidade;
    - limite de aceleração dependente da velocidade.

    A pipeline de não linearidades utilizada no teste é composta por:

    - limitador de aceleração dependente da velocidade;
    - limitador de velocidade;
    - limitador de posição.

    O teste também verifica se o atuador efetivamente alcança o
    batente positivo de posição.

    Quando a opção ``--plot`` do pytest está habilitada, são
    apresentados os gráficos de posição, velocidade e aceleração.

    A aceleração é avaliada somente fora da região de saturação
    de posição, pois a derivada numérica de uma posição saturada
    não representa diretamente a aceleração física do atuador.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Fixture utilizada para verificar se a opção ``--plot`` foi
        habilitada.

    Raises
    ------
    AssertionError
        Se qualquer limite físico for violado ou se o atuador não
        alcançar o batente positivo configurado.
    """

    command_deg = 11.0
    config = ActuatorTestConfig()

    # Conversão dos limites para radianos.
    max_position_rad = np.deg2rad(config.max_position_deg)
    max_velocity_rad = np.deg2rad(config.max_velocity_deg)

    # Curva de aceleração máxima dependente da velocidade.
    #
    # A curva abaixo representa um modelo simplificado de uma
    # característica de atuador. Em um teste baseado em dados
    # reais de fabricante, estes valores devem ser substituídos
    # pelos dados correspondentes.
    #
    # velocidade [rad/s] -> aceleração máxima [rad/s²]
    #
    # A aceleração máxima diminui à medida que a velocidade
    # aumenta, representando uma característica torque x velocidade.

    velocity_lut = np.deg2rad(np.array([0.0, 25.0, 50.0, 75.0, 100.0]))

    acceleration_lut = np.deg2rad(np.array([5000.0, 4500.0, 3500.0, 2500.0, 1500.0]))

    # Pipeline de não linearidades.
    nonlinearities = nonlinearities_pipeline_factory(
        velocity_dependent_acceleration_limiter_factory(
            velocity_lut=velocity_lut, acceleration_lut=acceleration_lut
        ),
        velocity_limiter_factory(max_velocity_rad),
        position_limiter_factory(max_position_rad),
    )

    # Simulação.
    time, ref_pos, act_pos = run_step_response(
        command_deg=command_deg, config=config, nonlinearities=nonlinearities
    )

    # Derivadas numéricas.
    act_vel = np.gradient(act_pos, config.dt)
    ref_vel = np.gradient(ref_pos, config.dt)

    act_acc = np.gradient(act_vel, config.dt)
    ref_acc = np.gradient(ref_vel, config.dt)

    # Plot.
    if request.config.getoption("--plot"):
        prefix_title = "Presença de interferência dos limites - "

        plot_results(
            time,
            ref_pos,
            act_pos,
            prefix_title=prefix_title,
            prefix_ylabel="Posição do ",
        )

        plot_results(
            time,
            ref_vel,
            act_vel,
            prefix_title=prefix_title,
            prefix_ylabel="Velocidade do ",
            derivative=1,
        )

        plot_results(
            time,
            ref_acc,
            act_acc,
            prefix_title=prefix_title,
            prefix_ylabel="Aceleração do ",
            derivative=2,
        )

    # Limite de posição.
    tol_abs = 1e-8
    tol_rel = 1e-5

    position_tolerance = tol_abs + tol_rel * abs(config.max_position_deg)

    assert np.all(
        np.abs(act_pos) <= config.max_position_deg + position_tolerance
    ), "A posição do atuador excedeu o limite configurado."

    # O atuador deve atingir o batente positivo.
    np.testing.assert_allclose(
        act_pos[-1],
        config.max_position_deg,
        rtol=tol_rel,
        atol=tol_abs,
        err_msg=("O atuador não atingiu o limite positivo " "de posição configurado."),
    )

    # Limite de velocidade.
    velocity_tolerance = tol_abs + tol_rel * abs(config.max_velocity_deg)

    assert np.all(
        np.abs(act_vel) <= config.max_velocity_deg + velocity_tolerance
    ), "A velocidade do atuador excedeu o limite configurado."

    # Limite de aceleração dependente da velocidade.
    #
    # Não avaliar a aceleração na região de saturação de posição.
    valid_acceleration = np.abs(act_pos) < config.max_position_deg - position_tolerance

    saturated = ~valid_acceleration

    # np.gradient utiliza pontos vizinhos. Portanto, dilatamos
    # a região de saturação para incluir todas as amostras
    # utilizadas pela diferenciação finita.
    saturated = binary_dilation(saturated, structure=np.ones(5, dtype=bool))

    valid_acceleration = ~saturated

    # A aceleração máxima depende da magnitude da velocidade.
    expected_max_acceleration = np.interp(
        np.abs(np.deg2rad(act_vel[valid_acceleration])), velocity_lut, acceleration_lut
    )

    measured_acceleration = np.abs(np.deg2rad(act_acc[valid_acceleration]))

    acceleration_tolerance = np.deg2rad(tol_abs + tol_rel * config.max_acceleration_deg)

    assert np.all(
        measured_acceleration <= expected_max_acceleration + acceleration_tolerance
    ), (
        "A aceleração do atuador excedeu o limite "
        "dependente da velocidade configurado."
    )


# =====================================================================
# Referência trapezoidal
# =====================================================================


def trapezoidal_reference(
    *,
    time: np.ndarray,
    amplitude_deg: float = 5.0,
    rise_time_s: float = 0.005,
    hold_time_s: float = 0.05,
) -> np.ndarray:
    """Gera uma referência trapezoidal com inversão de comando.

    A referência inicia em zero, sobe rapidamente até uma amplitude
    positiva, permanece nesse valor, inverte rapidamente para a
    amplitude negativa, permanece nesse valor e retorna a zero.

    A rampa de inversão é particularmente útil para visualizar o
    efeito da banda morta (backlash), pois o comando de referência
    pode atravessar o zero enquanto a saída efetiva permanece
    temporariamente próxima do valor anterior.

    Parameters
    ----------
    time : numpy.ndarray
        Vetor de tempo em segundos [s].
    amplitude_deg : float, optional
        Amplitude máxima da referência [deg].
    rise_time_s : float, optional
        Duração das rampas de subida e retorno [s].
    hold_time_s : float, optional
        Duração de cada patamar da referência [s].

    Returns
    -------
    numpy.ndarray
        Referência angular em graus [deg].

    Raises
    ------
    ValueError
        Se ``rise_time_s`` ou ``hold_time_s`` for negativo.
    """

    if rise_time_s < 0.0:
        raise ValueError("rise_time_s must be non-negative.")

    if hold_time_s < 0.0:
        raise ValueError("hold_time_s must be non-negative.")

    time = np.asarray(time, dtype=np.float64)
    reference = np.zeros_like(time)

    # Define os instantes de início e fim de cada trecho.
    t_rise_end = rise_time_s
    t_positive_hold_end = t_rise_end + hold_time_s
    t_fall_end = t_positive_hold_end + 2.0 * rise_time_s
    t_negative_hold_end = t_fall_end + hold_time_s
    t_return_end = t_negative_hold_end + rise_time_s

    # Rampa de 0 para +amplitude.
    mask = (time >= 0.0) & (time < t_rise_end)

    if rise_time_s > 0.0:
        reference[mask] = amplitude_deg * time[mask] / rise_time_s

    # Patamar positivo.
    mask = (time >= t_rise_end) & (time < t_positive_hold_end)

    reference[mask] = amplitude_deg

    # Rampa de +amplitude para -amplitude.
    mask = (time >= t_positive_hold_end) & (time < t_fall_end)

    if rise_time_s > 0.0:
        normalized_time = (time[mask] - t_positive_hold_end) / (2.0 * rise_time_s)

        reference[mask] = amplitude_deg * (1.0 - 2.0 * normalized_time)

    # Patamar negativo.
    mask = (time >= t_fall_end) & (time < t_negative_hold_end)

    reference[mask] = -amplitude_deg

    # Rampa de -amplitude para zero.
    mask = (time >= t_negative_hold_end) & (time < t_return_end)

    if rise_time_s > 0.0:
        normalized_time = (time[mask] - t_negative_hold_end) / rise_time_s

        reference[mask] = -amplitude_deg * (1.0 - normalized_time)

    return reference


# =====================================================================
# Teste: backlash
# =====================================================================


def test_actuator_backlash_interferes_with_dynamic_response(
    request: pytest.FixtureRequest,
) -> None:
    """Verifica o efeito do backlash sobre uma referência com inversão.

    O teste aplica uma referência trapezoidal ao atuador. A referência
    apresenta rampas rápidas entre valores positivos e negativos,
    permitindo visualizar o efeito da banda morta durante a inversão
    do comando.

    Durante a inversão, a referência efetiva deve permanecer
    temporariamente próxima do valor anterior enquanto a variação
    do comando permanecer dentro da banda de backlash.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Fixture utilizada para verificar se a opção ``--plot`` foi
        habilitada.

    Raises
    ------
    AssertionError
        Se o backlash não produzir a diferença esperada entre a
        referência original e a referência efetiva, ou se a posição
        final do atuador não corresponder ao deslocamento esperado
        devido ao backlash.
    """

    config = ActuatorTestConfig()
    config.duration = 1.0

    backlash_width_deg = 1.0
    backlash_width_rad = np.deg2rad(backlash_width_deg)

    reference_nonlinearity = backlash_factory(backlash_width_rad)

    parameters = ActuatorParameters(
        natural_frequency=config.wn, damping_ratio=config.zeta
    )

    actuator = Actuator(
        parameters=parameters,
        dt=config.dt,
        reference_nonlinearities=reference_nonlinearity,
        reference_memory=np.zeros(1, dtype=np.float64),
    )

    n_steps = int(round(config.duration / config.dt))

    time = np.arange(n_steps) * config.dt

    reference_deg = trapezoidal_reference(
        time=time, amplitude_deg=5.0, rise_time_s=0.1, hold_time_s=0.2
    )

    actuator_position_deg = np.zeros(n_steps)
    effective_reference_deg = np.zeros(n_steps)

    reference_memory = np.zeros(1, dtype=np.float64)

    for k, reference_value_deg in enumerate(reference_deg):
        reference_rad = np.deg2rad(reference_value_deg)

        effective_reference, reference_memory = reference_nonlinearity(
            np.array([reference_rad]), reference_memory
        )

        state = actuator.step(ActuatorInput(position_reference=reference_rad))

        actuator_position_deg[k] = np.rad2deg(state.position)

        effective_reference_deg[k] = np.rad2deg(effective_reference[0])

    if request.config.getoption("--plot"):
        plt.figure()

        plt.plot(time, reference_deg, label="Reference")

        plt.plot(time, effective_reference_deg, label="Effective reference")

        plt.plot(time, actuator_position_deg, label="Posição do atuador")

        plt.xlabel("Tempo [s]")
        plt.ylabel("Posição [graus]")
        plt.title("Resposta do atuador com folga (backlash)")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()

    # O backlash deve produzir uma diferença entre a referência
    # original e a referência efetivamente aplicada ao modelo.
    assert (
        np.max(np.abs(reference_deg - effective_reference_deg))
        > 0.5 * backlash_width_deg
    )

    # Em regime, a referência efetiva deve permanecer deslocada
    # pela metade da largura da banda de backlash em relação à
    # referência original.
    #
    # Como a referência final é zero e a última inversão foi de
    # uma posição negativa para zero, a saída deve permanecer em
    # aproximadamente -backlash_width / 2.
    expected_final_position_deg = -0.5 * backlash_width_deg

    assert np.isclose(actuator_position_deg[-1], expected_final_position_deg, atol=0.1)
