from __future__ import annotations

import numpy as np
import pytest

from vahsimulator.actuator.nonlinearities import (
    acceleration_limiter_factory,
    backlash_factory,
    identity_nonlinearity,
    nonlinearities_pipeline_factory,
    nonlinearities_pipeline_factory_from_dict,
    position_limiter_factory,
    velocity_dependent_acceleration_limiter_factory,
    velocity_limiter_factory,
)

# =====================================================================
# Identity
# =====================================================================


def test_identity_nonlinearity_does_not_modify_state_or_derivative():
    """Verifica que a não-linearidade identidade não altera as entradas."""
    state = np.array([1.0, 2.0])
    derivative = np.array([2.0, 3.0])

    returned_state, returned_derivative = identity_nonlinearity(state, derivative)

    np.testing.assert_array_equal(returned_state, state)
    np.testing.assert_array_equal(returned_derivative, derivative)


# =====================================================================
# Acceleration limiter
# =====================================================================


def test_acceleration_limiter_limits_positive_acceleration():
    """Verifica o limite da aceleração no sentido positivo."""
    limiter = acceleration_limiter_factory(maximum_acceleration=10.0)

    state = np.array([1.0, 2.0])
    derivative = np.array([999.0, 20.0])

    _, derivative = limiter(state, derivative)

    assert derivative[1] == pytest.approx(10.0)
    assert derivative[0] == pytest.approx(2.0)


def test_acceleration_limiter_limits_negative_acceleration():
    """Verifica o limite da aceleração no sentido negativo."""
    limiter = acceleration_limiter_factory(maximum_acceleration=10.0)

    state = np.array([1.0, 2.0])
    derivative = np.array([999.0, -20.0])

    _, derivative = limiter(state, derivative)

    assert derivative[1] == pytest.approx(-10.0)
    assert derivative[0] == pytest.approx(2.0)


def test_acceleration_limiter_does_not_modify_acceleration_inside_limit():
    """Verifica que a aceleração dentro do limite não é alterada."""
    limiter = acceleration_limiter_factory(maximum_acceleration=10.0)

    state = np.array([1.0, 2.0])
    derivative = np.array([999.0, 5.0])

    _, derivative = limiter(state, derivative)

    assert derivative[1] == pytest.approx(5.0)
    assert derivative[0] == pytest.approx(2.0)


def test_acceleration_limiter_rejects_negative_limit():
    """Verifica a validação do limite de aceleração."""
    with pytest.raises(ValueError):
        acceleration_limiter_factory(-1.0)


# =====================================================================
# Velocity limiter
# =====================================================================


def test_velocity_limiter_limits_positive_velocity():
    """Verifica o batente de velocidade positivo."""
    limiter = velocity_limiter_factory(maximum_velocity=10.0)

    state = np.array([1.0, 20.0])
    derivative = np.array([999.0, 5.0])

    state, derivative = limiter(state, derivative)

    assert state[1] == pytest.approx(10.0)
    assert derivative[0] == pytest.approx(10.0)
    assert derivative[1] == pytest.approx(0.0)


def test_velocity_limiter_limits_negative_velocity():
    """Verifica o batente de velocidade negativo."""
    limiter = velocity_limiter_factory(maximum_velocity=10.0)

    state = np.array([1.0, -20.0])
    derivative = np.array([999.0, -5.0])

    state, derivative = limiter(state, derivative)

    assert state[1] == pytest.approx(-10.0)
    assert derivative[0] == pytest.approx(-10.0)
    assert derivative[1] == pytest.approx(0.0)


def test_velocity_limiter_allows_acceleration_towards_positive_limit():
    """Verifica que a aceleração em direção ao interior do limite é mantida."""
    limiter = velocity_limiter_factory(maximum_velocity=10.0)

    state = np.array([1.0, 20.0])
    derivative = np.array([999.0, -5.0])

    _, derivative = limiter(state, derivative)

    assert derivative[1] == pytest.approx(-5.0)


def test_velocity_limiter_allows_acceleration_towards_negative_limit():
    """Verifica que a aceleração em direção ao interior do limite é mantida."""
    limiter = velocity_limiter_factory(maximum_velocity=10.0)

    state = np.array([1.0, -20.0])
    derivative = np.array([999.0, 5.0])

    _, derivative = limiter(state, derivative)

    assert derivative[1] == pytest.approx(5.0)


def test_velocity_limiter_does_not_modify_velocity_inside_limit():
    """Verifica que a velocidade dentro do limite não é alterada."""
    limiter = velocity_limiter_factory(maximum_velocity=10.0)

    state = np.array([1.0, 5.0])
    derivative = np.array([999.0, 2.0])

    state, derivative = limiter(state, derivative)

    assert state[1] == pytest.approx(5.0)
    assert derivative[0] == pytest.approx(5.0)
    assert derivative[1] == pytest.approx(2.0)


def test_velocity_limiter_rejects_negative_limit():
    """Verifica a validação do limite de velocidade."""
    with pytest.raises(ValueError):
        velocity_limiter_factory(-1.0)


# =====================================================================
# Position limiter
# =====================================================================


def test_position_limiter_limits_positive_position():
    """Verifica o batente de posição positivo."""
    limiter = position_limiter_factory(maximum_position=10.0)

    state = np.array([20.0, 5.0])
    derivative = np.array([999.0, 3.0])

    state, derivative = limiter(state, derivative)

    assert state[0] == pytest.approx(10.0)
    assert state[1] == pytest.approx(0.0)
    assert derivative[0] == pytest.approx(0.0)
    assert derivative[1] == pytest.approx(0.0)


def test_position_limiter_limits_negative_position():
    """Verifica o batente de posição negativo."""
    limiter = position_limiter_factory(maximum_position=10.0)

    state = np.array([-20.0, -5.0])
    derivative = np.array([999.0, -3.0])

    state, derivative = limiter(state, derivative)

    assert state[0] == pytest.approx(-10.0)
    assert state[1] == pytest.approx(0.0)
    assert derivative[0] == pytest.approx(0.0)
    assert derivative[1] == pytest.approx(0.0)


def test_position_limiter_allows_motion_towards_positive_interior():
    """Verifica que o movimento em direção ao interior do batente positivo é mantido."""
    limiter = position_limiter_factory(maximum_position=10.0)

    state = np.array([20.0, -5.0])
    derivative = np.array([999.0, -3.0])

    state, derivative = limiter(state, derivative)

    assert state[0] == pytest.approx(10.0)
    assert state[1] == pytest.approx(-5.0)
    assert derivative[0] == pytest.approx(-5.0)
    assert derivative[1] == pytest.approx(-3.0)


def test_position_limiter_allows_motion_towards_negative_interior():
    """Verifica que o movimento em direção ao interior do batente negativo é mantido."""
    limiter = position_limiter_factory(maximum_position=10.0)

    state = np.array([-20.0, 5.0])
    derivative = np.array([999.0, 3.0])

    state, derivative = limiter(state, derivative)

    assert state[0] == pytest.approx(-10.0)
    assert state[1] == pytest.approx(5.0)
    assert derivative[0] == pytest.approx(5.0)
    assert derivative[1] == pytest.approx(3.0)


def test_position_limiter_does_not_modify_position_inside_limit():
    """Verifica que a posição dentro dos limites não seja alterada."""
    limiter = position_limiter_factory(maximum_position=10.0)

    state = np.array([5.0, 2.0])
    derivative = np.array([999.0, 3.0])

    state, derivative = limiter(state, derivative)

    assert state[0] == pytest.approx(5.0)
    assert state[1] == pytest.approx(2.0)
    assert derivative[0] == pytest.approx(2.0)
    assert derivative[1] == pytest.approx(3.0)


def test_position_limiter_rejects_negative_limit():
    """Verifica a validação do limite de posição."""
    with pytest.raises(ValueError):
        position_limiter_factory(-1.0)


# =====================================================================
# Velocity-dependent acceleration limiter
# =====================================================================


def test_velocity_dependent_acceleration_limiter_at_lut_point():
    """Verifica a aceleração máxima exatamente em um ponto da LUT."""
    limiter = velocity_dependent_acceleration_limiter_factory(
        velocity_lut=[0.0, 100.0, 200.0], acceleration_lut=[120.0, 100.0, 60.0]
    )

    state = np.array([0.0, 100.0])
    derivative = np.array([999.0, 150.0])

    _, derivative = limiter(state, derivative)

    assert derivative[1] == pytest.approx(100.0)
    assert derivative[0] == pytest.approx(100.0)


def test_velocity_dependent_acceleration_limiter_interpolates():
    """Verifica a interpolação linear entre dois pontos da LUT."""
    limiter = velocity_dependent_acceleration_limiter_factory(
        velocity_lut=[0.0, 100.0, 200.0], acceleration_lut=[120.0, 100.0, 60.0]
    )

    state = np.array([0.0, 150.0])
    derivative = np.array([999.0, 100.0])

    _, derivative = limiter(state, derivative)

    # Entre 100 e 200 rad/s:
    #
    # a_max = 100 + (60 - 100) / (200 - 100) * (150 - 100)
    #       = 80 rad/s²

    assert derivative[1] == pytest.approx(80.0)
    assert derivative[0] == pytest.approx(150.0)


def test_velocity_dependent_acceleration_limiter_limits_negative_acceleration():
    """Verifica o limite de aceleração no sentido negativo."""
    limiter = velocity_dependent_acceleration_limiter_factory(
        velocity_lut=[0.0, 100.0, 200.0], acceleration_lut=[120.0, 100.0, 60.0]
    )

    state = np.array([0.0, 150.0])
    derivative = np.array([999.0, -100.0])

    _, derivative = limiter(state, derivative)

    assert derivative[1] == pytest.approx(-80.0)


def test_velocity_dependent_acceleration_limiter_uses_velocity_magnitude():
    """
    Verifica que a curva de aceleração seja aplicada simetricamente
    para velocidades positivas e negativas.
    """
    limiter = velocity_dependent_acceleration_limiter_factory(
        velocity_lut=[0.0, 100.0, 200.0], acceleration_lut=[120.0, 100.0, 60.0]
    )

    positive_state = np.array([0.0, 150.0])
    positive_derivative = np.array([999.0, 100.0])

    negative_state = np.array([0.0, -150.0])
    negative_derivative = np.array([999.0, -100.0])

    _, positive_derivative = limiter(positive_state, positive_derivative)

    _, negative_derivative = limiter(negative_state, negative_derivative)

    assert positive_derivative[1] == pytest.approx(80.0)
    assert negative_derivative[1] == pytest.approx(-80.0)


def test_velocity_dependent_acceleration_limiter_allows_acceleration_inside_limit():
    """Verifica que a aceleração dentro do limite não seja alterada."""
    limiter = velocity_dependent_acceleration_limiter_factory(
        velocity_lut=[0.0, 100.0, 200.0], acceleration_lut=[120.0, 100.0, 60.0]
    )

    state = np.array([0.0, 150.0])
    derivative = np.array([999.0, 50.0])

    _, derivative = limiter(state, derivative)

    assert derivative[1] == pytest.approx(50.0)
    assert derivative[0] == pytest.approx(150.0)


@pytest.mark.parametrize(
    "velocity, expected_acceleration",
    [(-1000.0, 60.0), (-200.0, 60.0), (0.0, 120.0), (200.0, 60.0), (1000.0, 60.0)],
)
def test_velocity_dependent_acceleration_limiter_is_constant_outside_lut(
    velocity, expected_acceleration
):
    """Verifica a extrapolação constante fora dos limites da LUT."""
    limiter = velocity_dependent_acceleration_limiter_factory(
        velocity_lut=[0.0, 100.0, 200.0], acceleration_lut=[120.0, 100.0, 60.0]
    )

    state = np.array([0.0, velocity])
    derivative = np.array([999.0, 1000.0])

    _, derivative = limiter(state, derivative)

    assert derivative[1] == pytest.approx(expected_acceleration)


# =====================================================================
# Validation of velocity-dependent acceleration limiter
# =====================================================================


def test_velocity_dependent_acceleration_limiter_rejects_different_lengths():
    """Verifica LUTs com tamanhos diferentes."""
    with pytest.raises(ValueError):
        velocity_dependent_acceleration_limiter_factory(
            velocity_lut=[0.0, 100.0], acceleration_lut=[100.0]
        )


def test_velocity_dependent_acceleration_limiter_requires_two_points():
    """Verifica que pelo menos dois pontos sejam necessários."""
    with pytest.raises(ValueError):
        velocity_dependent_acceleration_limiter_factory(
            velocity_lut=[0.0], acceleration_lut=[100.0]
        )


def test_velocity_dependent_acceleration_limiter_rejects_non_increasing_velocity():
    """Verifica que a LUT de velocidade seja estritamente crescente."""
    with pytest.raises(ValueError):
        velocity_dependent_acceleration_limiter_factory(
            velocity_lut=[0.0, 100.0, 100.0], acceleration_lut=[120.0, 100.0, 80.0]
        )


def test_velocity_dependent_acceleration_limiter_rejects_decreasing_velocity():
    """Verifica que a LUT de velocidade não seja decrescente."""
    with pytest.raises(ValueError):
        velocity_dependent_acceleration_limiter_factory(
            velocity_lut=[0.0, 200.0, 100.0], acceleration_lut=[120.0, 60.0, 100.0]
        )


def test_velocity_dependent_acceleration_limiter_rejects_negative_acceleration():
    """Verifica que a LUT não aceite acelerações negativas."""
    with pytest.raises(ValueError):
        velocity_dependent_acceleration_limiter_factory(
            velocity_lut=[0.0, 100.0], acceleration_lut=[100.0, -10.0]
        )


# =====================================================================
# Backlash
# =====================================================================


def test_backlash_blocks_command_inside_deadband():
    """Verifica que comandos dentro da folga não movimentem a saída."""
    backlash = backlash_factory(backlash_width=0.2)

    memory = np.zeros(1, dtype=np.float64)

    reference = np.array([0.05])

    effective_reference, memory = backlash(reference, memory)

    assert effective_reference[0] == pytest.approx(0.0)
    assert memory[0] == pytest.approx(0.0)


def test_backlash_transmits_command_after_deadband():
    """Verifica que o comando seja transmitido quando ultrapassa a folga."""
    backlash = backlash_factory(backlash_width=0.2)

    memory = np.zeros(1, dtype=np.float64)

    reference = np.array([0.3])

    effective_reference, memory = backlash(reference, memory)

    # backlash_width = 0.2
    # half_width = 0.1
    # output = 0.3 - 0.1 = 0.2

    assert effective_reference[0] == pytest.approx(0.2)
    assert memory[0] == pytest.approx(0.2)


def test_backlash_holds_output_when_command_returns_inside_deadband():
    """Verifica a memória da saída quando o comando retorna para dentro da banda de folga."""
    backlash = backlash_factory(backlash_width=0.2)

    memory = np.zeros(1, dtype=np.float64)

    effective_reference, memory = backlash(np.array([0.3]), memory)

    assert effective_reference[0] == pytest.approx(0.2)

    effective_reference, memory = backlash(np.array([0.15]), memory)

    # A saída anterior era 0.2.
    # A nova referência ainda está dentro da banda:
    #
    # 0.2 - 0.1 <= 0.15 <= 0.2 + 0.1
    #
    # Portanto a saída permanece em 0.2.

    assert effective_reference[0] == pytest.approx(0.2)
    assert memory[0] == pytest.approx(0.2)


def test_backlash_reverses_output_after_deadband_is_crossed():
    """Verifica a reversão da saída após a folga ser atravessada."""
    backlash = backlash_factory(backlash_width=0.2)

    memory = np.zeros(1, dtype=np.float64)

    backlash(np.array([0.3]), memory)

    effective_reference, memory = backlash(np.array([-0.3]), memory)

    # Após a reversão:
    #
    # output = -0.3 + 0.1 = -0.2

    assert effective_reference[0] == pytest.approx(-0.2)
    assert memory[0] == pytest.approx(-0.2)


def test_backlash_rejects_negative_width():
    """Verifica a rejeição de uma largura de backlash negativa."""
    with pytest.raises(ValueError):
        backlash_factory(-0.1)


# =====================================================================

# Pipeline

# =====================================================================


def test_pipeline_with_no_nonlinearities_returns_identity():
    """Verifica a pipeline vazia."""
    pipeline = nonlinearities_pipeline_factory()

    state = np.array([1.0, 2.0])
    derivative = np.array([2.0, 3.0])

    returned_state, returned_derivative = pipeline(state, derivative)

    np.testing.assert_array_equal(returned_state, state)
    np.testing.assert_array_equal(returned_derivative, derivative)


def test_pipeline_with_one_nonlinearity_returns_same_function():
    """Verifica a pipeline contendo apenas uma não-linearidade."""
    limiter = acceleration_limiter_factory(10.0)

    pipeline = nonlinearities_pipeline_factory(limiter)

    assert pipeline is limiter


def test_pipeline_preserves_nonlinearity_order():
    """Verifica que as não-linearidades sejam executadas na ordem especificada."""
    execution_order = []

    def first(
        state: np.ndarray, derivative: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        execution_order.append("first")
        return state, derivative

    def second(
        state: np.ndarray, derivative: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        execution_order.append("second")
        return state, derivative

    def third(
        state: np.ndarray, derivative: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        execution_order.append("third")
        return state, derivative

    pipeline = nonlinearities_pipeline_factory(first, second, third)

    state = np.array([0.0, 0.0])
    derivative = np.array([0.0, 0.0])

    pipeline(state, derivative)

    assert execution_order == ["first", "second", "third"]


def test_pipeline_passes_modifications_between_nonlinearities():
    """Verifica que uma não-linearidade receba as modificações produzidas pela anterior."""

    def first(
        state: np.ndarray, derivative: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        derivative[1] = 10.0
        return state, derivative

    def second(
        state: np.ndarray, derivative: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        derivative[1] *= 2.0
        return state, derivative

    pipeline = nonlinearities_pipeline_factory(first, second)

    state = np.array([0.0, 0.0])
    derivative = np.array([0.0, 0.0])

    _, derivative = pipeline(state, derivative)

    assert derivative[1] == pytest.approx(20.0)


# =====================================================================

# Pipeline from dictionary/YAML

# =====================================================================


def test_pipeline_factory_from_dict_with_none_returns_no_pipelines():
    """Verifica configuração `None`."""
    reference_pipeline, state_pipeline = nonlinearities_pipeline_factory_from_dict(None)

    assert reference_pipeline is None
    assert state_pipeline is None


def test_pipeline_factory_from_dict_with_empty_list_returns_no_pipelines():
    """Verifica configuração vazia."""
    reference_pipeline, state_pipeline = nonlinearities_pipeline_factory_from_dict([])

    assert reference_pipeline is None
    assert state_pipeline is None


def test_pipeline_factory_from_dict_creates_acceleration_limiter():
    """Verifica a criação do limitador de aceleração via configuração."""
    config = [{"type": "acceleration_limiter", "maximum_acceleration": 10.0}]

    reference_pipeline, state_pipeline = nonlinearities_pipeline_factory_from_dict(
        config
    )

    assert reference_pipeline is None
    assert state_pipeline is not None

    state = np.array([0.0, 2.0])
    derivative = np.array([999.0, 20.0])

    _, derivative = state_pipeline(state, derivative)

    assert derivative[1] == pytest.approx(10.0)


def test_pipeline_factory_from_dict_creates_velocity_limiter():
    """Verifica a criação do limitador de velocidade via configuração."""
    config = [{"type": "velocity_limiter", "maximum_velocity": 10.0}]

    reference_pipeline, state_pipeline = nonlinearities_pipeline_factory_from_dict(
        config
    )

    assert reference_pipeline is None
    assert state_pipeline is not None

    state = np.array([0.0, 20.0])
    derivative = np.array([999.0, 5.0])

    state, derivative = state_pipeline(state, derivative)

    assert state[1] == pytest.approx(10.0)
    assert derivative[1] == pytest.approx(0.0)


def test_pipeline_factory_from_dict_creates_position_limiter():
    """Verifica a criação do limitador de posição via configuração."""
    config = [{"type": "position_limiter", "maximum_position": 10.0}]

    reference_pipeline, state_pipeline = nonlinearities_pipeline_factory_from_dict(
        config
    )

    assert reference_pipeline is None
    assert state_pipeline is not None

    state = np.array([20.0, 5.0])
    derivative = np.array([999.0, 3.0])

    state, derivative = state_pipeline(state, derivative)

    assert state[0] == pytest.approx(10.0)
    assert state[1] == pytest.approx(0.0)
    assert derivative[1] == pytest.approx(0.0)


def test_pipeline_factory_from_dict_creates_velocity_dependent_limiter():
    """
    Verifica a criação do limitador dependente da velocidade via
    configuração.
    """
    config = [
        {
            "type": "velocity_dependent_acceleration_limiter",
            "velocity_lut": [0.0, 100.0, 200.0],
            "acceleration_lut": [120.0, 100.0, 60.0],
        }
    ]

    reference_pipeline, state_pipeline = nonlinearities_pipeline_factory_from_dict(
        config
    )

    assert reference_pipeline is None
    assert state_pipeline is not None

    state = np.array([0.0, 150.0])
    derivative = np.array([999.0, 100.0])

    _, derivative = state_pipeline(state, derivative)

    assert derivative[1] == pytest.approx(80.0)


def test_pipeline_factory_from_dict_creates_backlash_as_reference_pipeline():
    """Verifica que o backlash seja separado para a pipeline de referência."""
    config = [{"type": "backlash", "backlash_width": 0.2}]

    reference_pipeline, state_pipeline = nonlinearities_pipeline_factory_from_dict(
        config
    )

    assert reference_pipeline is not None
    assert state_pipeline is None

    memory = np.zeros(1, dtype=np.float64)

    effective_reference, memory = reference_pipeline(np.array([0.3]), memory)

    assert effective_reference[0] == pytest.approx(0.2)
    assert memory[0] == pytest.approx(0.2)


def test_pipeline_factory_from_dict_separates_reference_and_state_nonlinearities():
    """
    Verifica que a fábrica separe corretamente as não-linearidades de
    referência das não-linearidades de estado.
    """
    config = [
        {"type": "backlash", "backlash_width": 0.2},
        {"type": "velocity_limiter", "maximum_velocity": 10.0},
        {"type": "position_limiter", "maximum_position": 5.0},
    ]

    reference_pipeline, state_pipeline = nonlinearities_pipeline_factory_from_dict(
        config
    )

    assert reference_pipeline is not None
    assert state_pipeline is not None

    # Verifica backlash.
    memory = np.zeros(1, dtype=np.float64)

    effective_reference, memory = reference_pipeline(np.array([0.3]), memory)

    assert effective_reference[0] == pytest.approx(0.2)

    # Verifica state pipeline.
    state = np.array([10.0, 20.0])
    derivative = np.array([999.0, 5.0])

    state, derivative = state_pipeline(state, derivative)

    assert state[0] == pytest.approx(5.0)
    assert state[1] == pytest.approx(0.0)

    assert derivative[0] == pytest.approx(0.0)
    assert derivative[1] == pytest.approx(0.0)


def test_pipeline_factory_from_dict_preserves_state_pipeline_order():
    """
    Verifica que a ordem definida na configuração seja preservada
    dentro da pipeline de estado.

    ```
    Neste caso, o limitador de aceleração é aplicado antes do limitador
    dependente da velocidade.
    """
    config = [
        {"type": "acceleration_limiter", "maximum_acceleration": 100.0},
        {
            "type": "velocity_dependent_acceleration_limiter",
            "velocity_lut": [0.0, 100.0],
            "acceleration_lut": [50.0, 50.0],
        },
    ]

    reference_pipeline, state_pipeline = nonlinearities_pipeline_factory_from_dict(
        config
    )

    assert reference_pipeline is None
    assert state_pipeline is not None

    state = np.array([0.0, 50.0])
    derivative = np.array([999.0, 200.0])

    _, derivative = state_pipeline(state, derivative)

    # Primeiro: 200 -> 100 rad/s²
    # Segundo: 100 -> 50 rad/s²

    assert derivative[1] == pytest.approx(50.0)


def test_pipeline_factory_from_dict_requires_type():
    """Verifica configuração sem a chave `type`."""
    config = [{"maximum_velocity": 10.0}]

    with pytest.raises(ValueError):
        nonlinearities_pipeline_factory_from_dict(config)


def test_pipeline_factory_from_dict_rejects_unknown_type():
    """Verifica uma não-linearidade desconhecida."""
    config = [{"type": "unknown_limiter"}]

    with pytest.raises(ValueError):
        nonlinearities_pipeline_factory_from_dict(config)


# =============================================================================
# Backlash
#
# Para rodar somente os testes relacionados ao backlash:
# pytest tests\test_nonlinearities.py -k backlash -v
# =============================================================================


#  `backlash_width` da implementação atual é a largura total da banda,
#  portanto:
#     backlash_width = 2.0
#     half_width     = 1.0
#
# Com memória inicialmente zero:
# reference = +1.0  → output =  0.0
# reference = +1.1  → output = +0.1
# reference = -1.0  → output =  0.0
# reference = -1.1  → output = -0.1
#
# Isso está diretamente alinhado com a implementação atual:
# half_width = 0.5 * backlash_width
#
# e com as condições estritas:
#
# if reference_value > output_value + half_width:
# elif reference_value < output_value - half_width:
#
# Ou seja, na fronteira exata da banda o backlash permanece parado.


def test_backlash_does_not_move_inside_deadband():
    """Backlash must preserve its previous output inside the deadband."""
    backlash = backlash_factory(backlash_width=2.0)
    memory = np.zeros(1)

    reference = np.array([0.5])

    output, memory = backlash(reference, memory)

    assert output[0] == pytest.approx(0.0)
    assert memory[0] == pytest.approx(0.0)


def test_backlash_moves_reference_when_positive_threshold_is_exceeded():
    """Backlash must move when the positive deadband threshold is exceeded."""
    backlash = backlash_factory(backlash_width=2.0)
    memory = np.zeros(1)

    reference = np.array([1.1])

    output, memory = backlash(reference, memory)

    assert output[0] == pytest.approx(0.1)
    assert memory[0] == pytest.approx(0.1)


def test_backlash_moves_reference_when_negative_threshold_is_exceeded():
    """Backlash must move when the negative deadband threshold is exceeded."""
    backlash = backlash_factory(backlash_width=2.0)
    memory = np.zeros(1)

    reference = np.array([-1.1])

    output, memory = backlash(reference, memory)

    assert output[0] == pytest.approx(-0.1)
    assert memory[0] == pytest.approx(-0.1)


def test_backlash_does_not_move_at_positive_deadband_boundary():
    """Backlash must preserve its output at the positive boundary."""
    backlash = backlash_factory(backlash_width=2.0)
    memory = np.zeros(1)

    reference = np.array([1.0])

    output, memory = backlash(reference, memory)

    assert output[0] == pytest.approx(0.0)
    assert memory[0] == pytest.approx(0.0)


def test_backlash_does_not_move_at_negative_deadband_boundary():
    """Backlash must preserve its output at the negative boundary."""
    backlash = backlash_factory(backlash_width=2.0)
    memory = np.zeros(1)

    reference = np.array([-1.0])

    output, memory = backlash(reference, memory)

    assert output[0] == pytest.approx(0.0)
    assert memory[0] == pytest.approx(0.0)


def test_backlash_preserves_memory_inside_deadband():
    """Backlash must preserve its previous output inside the deadband."""
    backlash = backlash_factory(backlash_width=2.0)
    memory = np.array([0.5])

    reference = np.array([1.0])

    output, memory = backlash(reference, memory)

    assert output[0] == pytest.approx(0.5)
    assert memory[0] == pytest.approx(0.5)


def test_backlash_memory_is_updated_after_positive_motion():
    """Backlash memory must store the effective positive reference."""
    backlash = backlash_factory(backlash_width=2.0)
    memory = np.zeros(1)

    reference = np.array([3.0])

    output, memory = backlash(reference, memory)

    assert output[0] == pytest.approx(2.0)
    assert memory[0] == pytest.approx(2.0)


def test_backlash_memory_is_updated_after_negative_motion():
    """Backlash memory must store the effective negative reference."""
    backlash = backlash_factory(backlash_width=2.0)
    memory = np.zeros(1)

    reference = np.array([-3.0])

    output, memory = backlash(reference, memory)

    assert output[0] == pytest.approx(-2.0)
    assert memory[0] == pytest.approx(-2.0)


def test_backlash_follows_reference_after_deadband_is_crossed():
    """Backlash output must track the reference after crossing the deadband."""
    backlash = backlash_factory(backlash_width=2.0)
    memory = np.zeros(1)

    output, memory = backlash(np.array([3.0]), memory)

    assert output[0] == pytest.approx(2.0)

    output, memory = backlash(np.array([4.0]), memory)

    assert output[0] == pytest.approx(3.0)

    output, memory = backlash(np.array([5.0]), memory)

    assert output[0] == pytest.approx(4.0)


def test_backlash_preserves_output_when_reference_returns_inside_deadband():
    """Backlash must hold its output when the reference returns into the band."""
    backlash = backlash_factory(backlash_width=2.0)
    memory = np.zeros(1)

    output, memory = backlash(np.array([3.0]), memory)

    assert output[0] == pytest.approx(2.0)

    output, memory = backlash(np.array([2.5]), memory)

    assert output[0] == pytest.approx(2.0)

    output, memory = backlash(np.array([2.0]), memory)

    assert output[0] == pytest.approx(2.0)


# Teste de inversão de direção, justamente aqui que a memória do backlash
# faz diferença. O comportamento esperado é:
#               backlash = 2.0
#                     │
#                     ▼
# reference       0 ────────► +3
#                          output = +2

# Depois começa a voltar:


# reference       +3 ─────► +1.5 ─────► +0.9 ─────► -1
#                                   │           │
# output          +2 ───────────────┘           │
#                                               │
#                                   começa a voltar
#
# Esse comportamento é mais representativo de um backlash mecânico real
#  do que simplesmente testar entradas positivas e negativas isoladamente.
def test_backlash_reverses_direction_only_after_deadband_is_crossed():
    """Backlash must resist motion until the opposite deadband is crossed."""
    backlash = backlash_factory(backlash_width=2.0)
    memory = np.zeros(1)

    output, memory = backlash(np.array([3.0]), memory)

    assert output[0] == pytest.approx(2.0)

    # The reference moves in the negative direction, but remains inside
    # the backlash band around the current output.
    output, memory = backlash(np.array([1.5]), memory)

    assert output[0] == pytest.approx(2.0)

    output, memory = backlash(np.array([0.9]), memory)

    assert output[0] == pytest.approx(1.9)

    output, memory = backlash(np.array([-1.0]), memory)

    assert output[0] == pytest.approx(0.0)


def test_backlash_is_symmetric():
    """Backlash must have symmetric behavior for positive and negative inputs."""
    positive_backlash = backlash_factory(backlash_width=2.0)
    negative_backlash = backlash_factory(backlash_width=2.0)

    positive_memory = np.zeros(1)
    negative_memory = np.zeros(1)

    positive_output, _ = positive_backlash(np.array([3.0]), positive_memory)

    negative_output, _ = negative_backlash(np.array([-3.0]), negative_memory)

    assert positive_output[0] == pytest.approx(-negative_output[0])


def test_backlash_zero_width_behaves_as_identity():
    """Zero backlash width must reproduce the reference exactly."""
    backlash = backlash_factory(backlash_width=0.0)
    memory = np.zeros(1)

    reference = np.array([2.5])

    output, memory = backlash(reference, memory)

    assert output[0] == pytest.approx(2.5)
    assert memory[0] == pytest.approx(2.5)


def test_backlash_rejects_negative_width():
    """Backlash must reject a negative backlash width."""
    with pytest.raises(ValueError):
        backlash_factory(backlash_width=-1.0)


def test_backlash_updates_memory_in_place():
    """Backlash must update the supplied memory array in place."""
    backlash = backlash_factory(backlash_width=2.0)
    memory = np.zeros(1)

    memory_id = id(memory)

    output, updated_memory = backlash(np.array([3.0]), memory)

    assert id(updated_memory) == memory_id
    assert output[0] == pytest.approx(2.0)
    assert memory[0] == pytest.approx(2.0)


def test_backlash_returns_new_reference_array():
    """Backlash must return an effective reference array."""
    backlash = backlash_factory(backlash_width=2.0)
    memory = np.zeros(1)

    reference = np.array([3.0])

    output, _ = backlash(reference, memory)

    assert output is not reference
    assert output[0] == pytest.approx(2.0)
