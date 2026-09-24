import numpy as np
import pytest
import matplotlib.pyplot as plt

from vahsimulator.tvc.tvc_detailed import TVCDetailed


class IdealActuatorState:
    """
    Estado mínimo necessário pelo TVC para um atuador ideal.
    """

    def __init__(self, position: float):
        self.position = position


class IdealActuator:
    """
    Atuador ideal utilizado exclusivamente em alguns testes.

    O atuador atinge instantaneamente a referência solicitada.
    """

    def __init__(self, dt: float):
        self.dt = dt
        self.position = 0.0

    def step(self, actuator_input):
        self.position = actuator_input.position_reference

        return IdealActuatorState(self.position)

    def set_dt(self, dt: float):
        self.dt = dt


class IdealActuatorTVC(TVCDetailed):
    """
    Versão do TVC que utiliza atuadores ideais em alguns testes.

    Isso permite testar a cadeia cinemática sem depender da dinâmica
    dos atuadores reais.
    """

    def _create_actuator(self, actuator_config):
        return IdealActuator(dt=self._dt)


@pytest.fixture
def tvc():
    return IdealActuatorTVC(dt=0.001)


@pytest.fixture
def tvc_config():
    """
    Configuração TVC mínima para os testes.

    Os nomes dos parâmetros geométricos correspondem à nova
    TVCGeometry.
    """

    return {
        "tvc_parameters": [
            {
                "phase_id": 1,
                "type": "tvc_detailed",
                "geometry": {
                    "structure_joint_x": 0.650,
                    "initial_nozzle_joint_x": 0.300,
                    "initial_nozzle_joint_y": -0.260,
                    "initial_actuator_angle": 0.060,
                },
                "belt": {"reduction_ratio": 2.0},
                "angular_actuator": {"natural_frequency": 100.0, "damping_ratio": 0.7},
                "linear_actuator": {
                    "natural_frequency": 90.0,
                    "damping_ratio": 0.7,
                    "lead": 0.01,
                    "nonlinearities": [
                        {
                            "type": "position_limiter",
                            "maximum_position": 0.01939,  # (roughly 3 degrees TVC deflexion according to TVC geometry)
                        }
                    ],
                },
            }
        ]
    }


def test_tvc_configures_correctly(tvc, tvc_config):
    """
    Verifica se o TVC cria corretamente sua geometria e atuadores
    durante a configuração.
    """

    tvc._configure(tvc_config["tvc_parameters"][0])

    assert tvc._geometry is not None

    assert tvc._pitch_angular_actuator is not None
    assert tvc._yaw_angular_actuator is not None

    assert tvc._pitch_linear_actuator is not None
    assert tvc._yaw_linear_actuator is not None

    assert tvc._belt_ratio == pytest.approx(2.0)
    assert tvc._linear_lead == pytest.approx(0.01)


def test_tvc_does_not_reconfigure_same_phase(tvc, tvc_config):
    """
    Verifica que uma mesma fase não provoca reconstrução dos
    componentes do TVC.
    """

    config = tvc_config["tvc_parameters"][0]

    tvc._configure(config)

    geometry = tvc._geometry
    pitch_actuator = tvc._pitch_angular_actuator

    tvc._configure(config)

    assert tvc._geometry is geometry
    assert tvc._pitch_angular_actuator is pitch_actuator


def test_rotation_to_linear_displacement(tvc, tvc_config):
    """
    Verifica a conversão:

        x = lead * theta / (2*pi)
    """

    tvc._configure(tvc_config["tvc_parameters"][0])

    angle = 2.0 * np.pi

    displacement = tvc._rotation_to_linear_displacement(angle)

    assert displacement == pytest.approx(tvc._linear_lead)


def test_rotation_to_linear_displacement_half_turn(tvc, tvc_config):
    """
    Uma rotação de pi rad corresponde a meia volta.

        x = lead / 2
    """

    tvc._configure(tvc_config["tvc_parameters"][0])

    angle = np.pi

    displacement = tvc._rotation_to_linear_displacement(angle)

    assert displacement == pytest.approx(tvc._linear_lead / 2.0)


@pytest.mark.parametrize(
    "tvc_angle",
    [np.deg2rad(-10.0), np.deg2rad(-5.0), 0.0, np.deg2rad(5.0), np.deg2rad(10.0)],
)
def test_tvc_angle_to_angular_actuator_reference(tvc, tvc_config, tvc_angle):
    """
    Verifica a cadeia:

        TVC angle
            -> actuator displacement
            -> linear actuator angle
            -> angular actuator angle

    Com:

        theta_linear =
            displacement * 2*pi / lead

        theta_angular =
            belt_ratio * theta_linear
    """

    tvc._configure(tvc_config["tvc_parameters"][0])

    geometry = tvc._geometry

    displacement = geometry.tvc_angle_to_actuator_displacement(tvc_angle)

    expected_linear_angle = displacement * 2.0 * np.pi / tvc._linear_lead

    expected_angular_angle = tvc._belt_ratio * expected_linear_angle

    actual = tvc._tvc_angle_to_angular_actuator_reference(tvc_angle)

    assert actual == pytest.approx(expected_angular_angle)


def test_zero_command_produces_zero_deflection(tvc, tvc_config):
    """
    Com atuadores ideais, um comando nulo deve produzir:

        delta_q = delta_r = 0
    """

    delta_q, delta_r = tvc.step(
        config_data=tvc_config, delta_q_cmd=0.0, delta_r_cmd=0.0, phase=1
    )

    assert delta_q == pytest.approx(0.0)
    assert delta_r == pytest.approx(0.0)


@pytest.mark.parametrize(
    "delta_cmd",
    # [np.deg2rad(-10.0), np.deg2rad(-5.0), np.deg2rad(5.0), np.deg2rad(10.0)],
    [-10.0, -5.0, 5.0, 10.0],
)
def test_pitch_tracks_command_with_ideal_actuators(tvc, tvc_config, delta_cmd):
    """
    Com atuadores ideais, o pitch deve reproduzir exatamente o
    comando aplicado.
    """

    delta_q, delta_r = tvc.step(
        config_data=tvc_config, delta_q_cmd=delta_cmd, delta_r_cmd=0.0, phase=1
    )

    assert delta_q == pytest.approx(np.deg2rad(delta_cmd), abs=1e-10)

    assert delta_r == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize(
    "delta_cmd",
    # [np.deg2rad(-10.0), np.deg2rad(-5.0), np.deg2rad(5.0), np.deg2rad(10.0)],
    [-10.0, -5.0, 5.0, 10.0],
)
def test_yaw_tracks_command_with_ideal_actuators(tvc, tvc_config, delta_cmd):
    """
    Com atuadores ideais, o yaw deve reproduzir exatamente o
    comando aplicado.
    """

    delta_q, delta_r = tvc.step(
        config_data=tvc_config, delta_q_cmd=0.0, delta_r_cmd=delta_cmd, phase=1
    )

    assert delta_q == pytest.approx(0.0, abs=1e-10)

    assert delta_r == pytest.approx(np.deg2rad(delta_cmd), abs=1e-10)


def test_pitch_and_yaw_are_independent(tvc, tvc_config):
    """
    Verifica que os canais de pitch e yaw são independentes.

    Um comando em pitch não deve produzir deflexão em yaw e vice-versa.
    """

    pitch_cmd = 5.0
    yaw_cmd = -3.0

    delta_q, delta_r = tvc.step(
        config_data=tvc_config, delta_q_cmd=pitch_cmd, delta_r_cmd=yaw_cmd, phase=1
    )

    assert delta_q == pytest.approx(np.deg2rad(pitch_cmd), abs=1e-10)

    assert delta_r == pytest.approx(np.deg2rad(yaw_cmd), abs=1e-10)


def test_missing_phase_raises_error(tvc, tvc_config):
    """
    Verifica que uma fase sem configuração de TVC gera ValueError.
    """

    with pytest.raises(ValueError, match="phase_id=99"):
        tvc.step(config_data=tvc_config, delta_q_cmd=0.0, delta_r_cmd=0.0, phase=99)


def test_set_dt_updates_all_actuators(tvc, tvc_config):
    """
    Verifica que set_dt() atualiza todos os quatro atuadores.
    """

    tvc._configure(tvc_config["tvc_parameters"][0])

    new_dt = 0.0001

    tvc.set_dt(new_dt)

    assert tvc._dt == pytest.approx(new_dt)

    assert tvc._pitch_angular_actuator.dt == pytest.approx(new_dt)

    assert tvc._yaw_angular_actuator.dt == pytest.approx(new_dt)

    assert tvc._pitch_linear_actuator.dt == pytest.approx(new_dt)

    assert tvc._yaw_linear_actuator.dt == pytest.approx(new_dt)


@pytest.mark.parametrize("axis, command_deg", [("pitch", 2.0), ("yaw", 4.0)])
def test_tvc_step_response(tvc_config, request, axis, command_deg):
    """
    Verifica a resposta temporal do TVC a um comando degrau.

    O teste verifica se a resposta final do TVC converge para o
    comando solicitado quando este está dentro do batente mecânico.
    Quando o comando excede o batente, a resposta final deve convergir
    para o ângulo correspondente ao limite de posição do atuador.

    Casos testados:

    * Pitch: comando de 2 grau, abaixo do batente de 3 graus.
    * Yaw: comando de 4 graus, acima do batente de 3 graus.

    O gráfico pode ser habilitado utilizando a opção ``--plot``.

    :param tvc_config: Configuração do TVC, incluindo a dinâmica dos
        atuadores e os limites de posição.
    :param request: Fixture do pytest utilizada para acessar a opção
        ``--plot``.
    :param axis: Eixo do comando, ``"pitch"`` ou ``"yaw"``.
    :param command_deg: Magnitude do comando degrau em graus.
    """

    dt = 0.001
    simulation_time = 0.2
    step_time = 0.01

    tvc = TVCDetailed(dt=dt)

    number_of_steps = int(simulation_time / dt)
    time = np.arange(number_of_steps) * dt

    pitch_command_history = np.zeros(number_of_steps)
    yaw_command_history = np.zeros(number_of_steps)

    pitch_response = np.zeros(number_of_steps)
    yaw_response = np.zeros(number_of_steps)

    # command = np.deg2rad(command_deg)
    command = command_deg

    for index, current_time in enumerate(time):
        if current_time < step_time:
            pitch_command = 0.0
            yaw_command = 0.0

        elif axis == "pitch":
            pitch_command = command
            yaw_command = 0.0

        else:
            pitch_command = 0.0
            yaw_command = command

        pitch_command_history[index] = pitch_command
        yaw_command_history[index] = yaw_command

        delta_q, delta_r = tvc.step(
            config_data=tvc_config,
            delta_q_cmd=pitch_command,
            delta_r_cmd=yaw_command,
            phase=1,
        )

        pitch_response[index] = delta_q
        yaw_response[index] = delta_r

    # Seleciona o histórico e a resposta do eixo testado.
    if axis == "pitch":
        command_history = pitch_command_history
        response = pitch_response
    else:
        command_history = yaw_command_history
        response = yaw_response

    # Obtém o limite de posição do atuador linear.
    nonlinearities = tvc_config["tvc_parameters"][0]["linear_actuator"][
        "nonlinearities"
    ]

    maximum_actuator_position = next(
        item["maximum_position"]
        for item in nonlinearities
        if item["type"] == "position_limiter"
    )

    # Pela convenção da TVCGeometry, uma deflexão positiva do TVC
    # corresponde a uma redução no comprimento do atuador.
    stop_angle = tvc._geometry.actuator_displacement_to_tvc_angle(
        -maximum_actuator_position
    )
    stop_angle = np.rad2deg(stop_angle)

    # Determina o valor esperado para o estado final.
    #
    # Se o comando estiver dentro do batente, a resposta deve atingir
    # o comando solicitado.
    #
    # Se o comando exceder o batente, a resposta deve ser limitada
    # pelo ângulo correspondente ao batente.
    if abs(command) <= abs(stop_angle):
        expected_final_response = command
    else:
        expected_final_response = np.sign(command) * abs(stop_angle)

    # Verifica a resposta final.
    last_angle = np.rad2deg(response[-1])
    assert last_angle == pytest.approx(expected_final_response, abs=1e-3)

    if request.config.getoption("--plot"):
        plt.figure(figsize=(10, 5))

        plt.plot(
            time,
            command_history,
            "--",
            label=f"{axis.capitalize()} command",
        )

        plt.plot(time, np.rad2deg(response), label=f"{axis.capitalize()} response")

        plt.axhline(stop_angle, linestyle=":", label="TVC position stop")

        plt.xlabel("Time [s]")
        plt.ylabel("TVC deflection [deg]")
        plt.title(f"TVC {axis.capitalize()} Step Response " f"({command_deg:.1f} deg)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
