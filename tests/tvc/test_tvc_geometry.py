import numpy as np
import pytest

from vahsimulator.tvc import TVCGeometry


@pytest.fixture
def geometry() -> TVCGeometry:
    """
    Cria uma geometria TVC de referência para os testes.

    A configuração corresponde a:

        structure_joint_x = 650 mm
        initial_nozzle_joint = (300, -260) mm
        initial_actuator_angle = 60 deg

    A origem é o ponto de rotação da tubeira.
    """

    return TVCGeometry(
        structure_joint_x=650.0,
        initial_nozzle_joint_x=300.0,
        initial_nozzle_joint_y=-260.0,
        initial_actuator_angle=np.deg2rad(60.0),
    )


def test_structure_joint_y_is_calculated_correctly(geometry):
    """
    Verifica o cálculo da coordenada y da junta estrutural.

    A relação geométrica é:

        y_s = y_n0 + (x_s - x_n0) / tan(alpha_0)
    """

    expected = -260.0 + (650.0 - 300.0) / np.tan(np.deg2rad(60.0))

    assert geometry.structure_joint_y == pytest.approx(expected)


def test_initial_actuator_length_is_correct(geometry):
    """
    Verifica o comprimento inicial do atuador.

    O comprimento inicial é a distância entre as duas juntas:

        L0 = |P0 - A|
    """

    expected = np.hypot(650.0 - 300.0, geometry.structure_joint_y - (-260.0))

    assert geometry.initial_actuator_length == pytest.approx(expected)


def test_initial_nozzle_joint_position_is_correct(geometry):
    """
    Verifica a posição inicial da junta da tubeira.
    """

    expected = np.array([300.0, -260.0])

    np.testing.assert_allclose(geometry.initial_nozzle_joint_position, expected)


def test_structure_joint_position_is_correct(geometry):
    """
    Verifica a posição da junta estrutural.
    """

    expected = np.array([650.0, geometry.structure_joint_y])

    np.testing.assert_allclose(geometry.structure_joint_position, expected)


def test_nozzle_joint_position_at_zero_angle_is_initial_position(geometry):
    """
    Para tvc_angle = 0, a rotação deve ser a identidade.

        P(0) = P0
    """

    position = geometry.nozzle_joint_position_at_angle(0.0)

    np.testing.assert_allclose(position, geometry.initial_nozzle_joint_position)


@pytest.mark.parametrize(
    "angle",
    [
        np.deg2rad(-10.0),
        np.deg2rad(-5.0),
        np.deg2rad(5.0),
        np.deg2rad(10.0),
        np.deg2rad(20.0),
    ],
)
def test_nozzle_joint_radius_is_constant(geometry, angle):
    """
    Verifica que a junta da tubeira permanece à mesma distância
    da origem após a rotação.

        |P(delta)| = |P0|
    """

    initial_radius = np.linalg.norm(geometry.initial_nozzle_joint_position)

    rotated_radius = np.linalg.norm(geometry.nozzle_joint_position_at_angle(angle))

    assert rotated_radius == pytest.approx(initial_radius)


def test_actuator_length_at_zero_angle_equals_initial_length(geometry):
    """
    Para tvc_angle = 0:

        L(0) = L0
    """

    assert geometry.actuator_length(0.0) == pytest.approx(
        geometry.initial_actuator_length
    )


def test_zero_angle_produces_zero_actuator_displacement(geometry):
    """
    Para tvc_angle = 0:

        displacement = L(0) - L0 = 0
    """

    displacement = geometry.tvc_angle_to_actuator_displacement(0.0)

    assert displacement == pytest.approx(0.0)


def test_positive_tvc_angle_retracts_actuator(geometry):
    """
    Verifica a convenção de sinais adotada.

    Pela convenção do modelo:

        tvc_angle > 0
            -> actuator displacement < 0

    """

    displacement = geometry.tvc_angle_to_actuator_displacement(np.deg2rad(5.0))

    assert displacement < 0.0


def test_negative_tvc_angle_extends_actuator(geometry):
    """
    Verifica a convenção de sinais para ângulos negativos.

        tvc_angle < 0
            -> actuator displacement > 0
    """

    displacement = geometry.tvc_angle_to_actuator_displacement(np.deg2rad(-5.0))

    assert displacement > 0.0


@pytest.mark.parametrize(
    "angle",
    [np.deg2rad(-10.0), np.deg2rad(-5.0), 0.0, np.deg2rad(5.0), np.deg2rad(10.0)],
)
def test_angle_to_displacement_matches_direct_length_calculation(geometry, angle):
    """
    Verifica que a conversão de ângulo para deslocamento corresponde
    diretamente à definição geométrica:

        displacement = L(delta) - L0
    """

    expected = geometry.actuator_length(angle) - geometry.initial_actuator_length

    actual = geometry.tvc_angle_to_actuator_displacement(angle)

    assert actual == pytest.approx(expected)


@pytest.mark.parametrize(
    "angle",
    [
        np.deg2rad(-10.0),
        np.deg2rad(-5.0),
        np.deg2rad(-1.0),
        0.0,
        np.deg2rad(1.0),
        np.deg2rad(5.0),
        np.deg2rad(10.0),
    ],
)
def test_round_trip_angle_to_displacement_to_angle(geometry, angle):
    """
    Verifica a consistência da conversão direta e inversa:

        delta
          -> displacement
          -> delta

    O resultado deve retornar ao ângulo original.
    """

    displacement = geometry.tvc_angle_to_actuator_displacement(angle)

    recovered_angle = geometry.actuator_displacement_to_tvc_angle(displacement)

    assert recovered_angle == pytest.approx(angle, abs=1e-10)


@pytest.mark.parametrize("displacement", [-0.5, -0.1, -0.01, 0.0, 0.01, 0.1, 0.5])
def test_round_trip_displacement_to_angle_to_displacement(geometry, displacement):
    """
    Verifica a consistência da conversão inversa:

        displacement
            -> angle
            -> displacement
    """

    angle = geometry.actuator_displacement_to_tvc_angle(displacement)

    recovered_displacement = geometry.tvc_angle_to_actuator_displacement(angle)

    assert recovered_displacement == pytest.approx(displacement, abs=1e-10)


def test_negative_actuator_length_raises_error(geometry):
    """
    Verifica que um deslocamento que resultaria em comprimento
    negativo é rejeitado.
    """

    displacement = -geometry.initial_actuator_length - 1.0

    with pytest.raises(ValueError):
        geometry.actuator_displacement_to_tvc_angle(displacement)


def test_unreachable_actuator_length_raises_error(geometry):
    """
    Verifica que um comprimento geometricamente impossível é
    rejeitado pela lei dos cossenos.

    Para dois pontos a distâncias rn e rs da origem, o comprimento
    deve satisfazer:

        |rs - rn| <= L <= rs + rn
    """

    minimum_length = abs(geometry._structure_radius - geometry._nozzle_radius)

    maximum_length = geometry._structure_radius + geometry._nozzle_radius

    unreachable_length = maximum_length + 1.0

    assert unreachable_length > maximum_length

    displacement = unreachable_length - geometry.initial_actuator_length

    with pytest.raises(ValueError):
        geometry.actuator_displacement_to_tvc_angle(displacement)


def test_initial_geometry_has_expected_actuator_angle(geometry):
    """
    Verifica o ângulo geométrico do atuador em relação ao eixo
    longitudinal.

    Como initial_actuator_angle é definido em relação ao eixo
    transversal/vertical:

        actuator_angle_x =
            pi/2 - initial_actuator_angle
    """

    dx = geometry.structure_joint_x - geometry.initial_nozzle_joint_x

    dy = geometry.structure_joint_y - geometry.initial_nozzle_joint_y

    actuator_angle = np.arctan2(dy, dx)

    expected = np.pi / 2.0 - geometry.initial_actuator_angle

    assert actuator_angle == pytest.approx(expected)
