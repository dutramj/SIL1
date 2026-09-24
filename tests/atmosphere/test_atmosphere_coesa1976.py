from __future__ import annotations

import numpy as np
import pytest

from vahsimulator.atmosphere.atmosphere_coesa1976 import AtmosphereCOESA1976


@pytest.fixture
def atmosphere_model() -> AtmosphereCOESA1976:
    """
    Cria uma instância do modelo atmosférico COESA-1976.

    Returns
    -------
    AtmosphereCOESA1976
        Modelo atmosférico utilizado nos testes.
    """
    return AtmosphereCOESA1976()


# ============================================================================
# Reference values
# ============================================================================
#
# Os valores abaixo foram calculados reproduzindo diretamente as equações,
# constantes e tabelas presentes no COESA.jl fornecido como referência.
#
# IMPORTANTE:
# altitude_m representa altitude GEOMÉTRICA.
#
# Portanto, por exemplo, 11000 m de entrada não corresponde exatamente aos
# valores clássicos da Standard Atmosphere tabelados para 11 km de altitude
# geopotencial.
#
# Isso explica, por exemplo:
#
#   T(11000 m) = 216.7735127 K
#
# em vez de:
#
#   T = 216.65 K
#
# que é o valor convencional em 11 km de altitude geopotencial.
# ============================================================================


def test_sea_level(atmosphere_model: AtmosphereCOESA1976):
    data = atmosphere_model.evaluate(0.0)

    assert data.temperature_K == pytest.approx(288.15)
    assert data.pressure_Pa == pytest.approx(101325.0)
    assert data.density_kg_m3 == pytest.approx(1.225)


def test_temperature_at_11_km_geometric_altitude(atmosphere_model: AtmosphereCOESA1976):
    data = atmosphere_model.evaluate(11_000.0)

    assert data.temperature_K == pytest.approx(216.7735)


def test_geopotential_altitude_at_11_km(atmosphere_model: AtmosphereCOESA1976):
    geopotential_altitude_m = atmosphere_model._geopotential_altitude(11_000.0)

    assert geopotential_altitude_m == pytest.approx(10_980.99, abs=0.01)


def test_altitude_limits(atmosphere_model: AtmosphereCOESA1976):
    atmosphere_model.evaluate(-5_000.0)
    atmosphere_model.evaluate(1_000_000.0)


def test_altitude_below_lower_limit(atmosphere_model: AtmosphereCOESA1976):
    with pytest.raises(ValueError):
        atmosphere_model.evaluate(-5_000.1)


def test_altitude_above_upper_limit(atmosphere_model: AtmosphereCOESA1976):
    with pytest.raises(ValueError):
        atmosphere_model.evaluate(1_000_000.1)


@pytest.mark.parametrize(
    (
        "altitude_m",
        "expected_temperature_K",
        "expected_pressure_Pa",
        "expected_density_kg_m3",
        "expected_speed_of_sound_m_s",
    ),
    [
        (0.0, 288.150000000, 101325.000000000, 1.22499915589, 340.294107787),
        (5000.0, 255.675543222, 54048.2861458, 0.736428420780, 320.545519670),
        (10000.0, 223.252092648, 26499.8981393, 0.413510428899, 299.531765677),
        (11000.0, 216.773512704, 22699.9607392, 0.364801564187, 295.153695326),
        (15000.0, 216.650000000, 12111.8256981, 0.194755046444, 295.069597354),
        (20000.0, 216.650000000, 5529.31189230, 0.0889099150889, 295.069597354),
        (25000.0, 221.552064726, 2549.22299238, 0.0400838867181, 298.389143766),
        (32000.0, 228.489718656, 889.064417202, 0.0135551512224, 303.024992270),
        (47000.0, 269.684130854, 115.851113765, 0.00149652033496, 329.209844235),
        (51000.0, 270.650000000, 70.4580090284, 0.000906901533867, 329.798847071),
        (60000.0, 247.020884773, 21.9586661397, 0.000309677807648, 315.073555487),
        (71000.0, 216.845910679, 4.47956324620, 7.1965150355e-05, 295.202978897),
        (80000.0, 198.638576251, 1.05247354505, 1.84580320369e-05, 282.538030990),
        (85000.0, 188.835372378, 0.445680763008, 8.21950050435e-06, 275.520075701),
        (86000.0, 186.867300000, 0.373380000000, 6.95728145961e-06, 274.106766740),
        (87000.0, 186.867300000, 0.312590000000, 5.82456642418e-06, 274.106766740),
        (90000.0, 186.867300000, 0.183590000000, 3.41615106537e-06, 274.296328842),
        (91000.0, 186.867300000, 0.153810000000, 2.86003974124e-06, 274.391257493),
        (100000.0, 195.081344335, 0.0320767885738, 5.61552389223e-07, 282.790176495),
        (110000.0, 239.999727168, 0.00710420000000, 9.70873917102e-08, 320.066477797),
        (120000.0, 360.000000000, 0.00253820000000, 2.22176384305e-08, 399.924481421),
        (150000.0, 634.392033111, 0.000454220000000, 2.07538579393e-09, 553.538362280),
        (200000.0, 854.559090798, 8.47360000000e-05, 2.54026326818e-10, 683.374286042),
        (500000.0, 999.235601763, 3.02360000000e-07, 5.21525913445e-13, 900.924173398),
        (1000000.0, 999.999685598, 7.51380000000e-09, 3.56064973536e-15, 1718.815282311,
        ),
    ],
)
def test_reference_values(
    atmosphere_model: AtmosphereCOESA1976,
    altitude_m: float,
    expected_temperature_K: float,
    expected_pressure_Pa: float,
    expected_density_kg_m3: float,
    expected_speed_of_sound_m_s: float,
) -> None:
    """
    Verifica os principais resultados contra valores de referência.

    Os valores de referência de temperatura, pressão e densidade foram
    obtidos a partir das equações e tabelas do COESA.jl fornecido como
    referência para a implementação Python.

    A velocidade do som é calculada a partir da temperatura e da massa
    molecular locais, incluindo a região acima de 86 km.

    Parameters
    ----------
    atmosphere_model : AtmosphereCOESA1976
        Modelo atmosférico.
    altitude_m : float
        Altitude geométrica [m].
    expected_temperature_K : float
        Temperatura esperada [K].
    expected_pressure_Pa : float
        Pressão esperada [Pa].
    expected_density_kg_m3 : float
        Massa específica esperada [kg/m³].
    expected_speed_of_sound_m_s : float
        Velocidade do som esperada [m/s].
    """
    result = atmosphere_model.evaluate(altitude_m)

    assert result.temperature_K == pytest.approx(
        expected_temperature_K, rel=1e-8, abs=1e-10
    )

    assert result.pressure_Pa == pytest.approx(
        expected_pressure_Pa, rel=1e-8, abs=1e-12
    )

    assert result.density_kg_m3 == pytest.approx(
        expected_density_kg_m3, rel=1e-8, abs=1e-15
    )

    assert result.speed_of_sound_m_s == pytest.approx(
        expected_speed_of_sound_m_s, rel=1e-8, abs=1e-10
    )


# ============================================================================
# Atmospheric physical properties
# ============================================================================


@pytest.mark.parametrize(
    "altitude_m",
    [
        -5000.0,
        0.0,
        5000.0,
        11000.0,
        20000.0,
        32000.0,
        47000.0,
        51000.0,
        71000.0,
        80000.0,
        86000.0,
        100000.0,
        200000.0,
        500000.0,
        1000000.0,
    ],
)
def test_temperature_is_positive(
    atmosphere_model: AtmosphereCOESA1976, altitude_m: float
) -> None:
    """
    Verifica que a temperatura seja positiva em toda a faixa válida.

    Parameters
    ----------
    atmosphere_model : AtmosphereCOESA1976
        Modelo atmosférico.
    altitude_m : float
        Altitude geométrica [m].
    """
    result = atmosphere_model.evaluate(altitude_m)

    assert result.temperature_K > 0.0


@pytest.mark.parametrize(
    "altitude_m",
    [
        -5000.0,
        0.0,
        5000.0,
        11000.0,
        20000.0,
        32000.0,
        47000.0,
        51000.0,
        71000.0,
        80000.0,
        86000.0,
        100000.0,
        200000.0,
        500000.0,
        1000000.0,
    ],
)
def test_pressure_is_positive(
    atmosphere_model: AtmosphereCOESA1976, altitude_m: float
) -> None:
    """
    Verifica que a pressão seja positiva em toda a faixa válida.

    Parameters
    ----------
    atmosphere_model : AtmosphereCOESA1976
        Modelo atmosférico.
    altitude_m : float
        Altitude geométrica [m].
    """
    result = atmosphere_model.evaluate(altitude_m)

    assert result.pressure_Pa > 0.0


@pytest.mark.parametrize(
    "altitude_m",
    [
        -5000.0,
        0.0,
        5000.0,
        11000.0,
        20000.0,
        32000.0,
        47000.0,
        51000.0,
        71000.0,
        80000.0,
        86000.0,
        100000.0,
        200000.0,
        500000.0,
        1000000.0,
    ],
)
def test_density_is_positive(
    atmosphere_model: AtmosphereCOESA1976, altitude_m: float
) -> None:
    """
    Verifica que a massa específica seja positiva.

    Parameters
    ----------
    atmosphere_model : AtmosphereCOESA1976
        Modelo atmosférico.
    altitude_m : float
        Altitude geométrica [m].
    """
    result = atmosphere_model.evaluate(altitude_m)

    assert result.density_kg_m3 > 0.0


@pytest.mark.parametrize(
    "altitude_m",
    [
        -5000.0,
        0.0,
        5000.0,
        11000.0,
        20000.0,
        32000.0,
        47000.0,
        51000.0,
        71000.0,
        80000.0,
        86000.0,
        100000.0,
        200000.0,
        500000.0,
        1000000.0,
    ],
)
def test_speed_of_sound_is_positive(
    atmosphere_model: AtmosphereCOESA1976, altitude_m: float
) -> None:
    """
    Verifica que a velocidade do som seja positiva.

    Parameters
    ----------
    atmosphere_model : AtmosphereCOESA1976
        Modelo atmosférico.
    altitude_m : float
        Altitude geométrica [m].
    """
    result = atmosphere_model.evaluate(altitude_m)

    assert result.speed_of_sound_m_s > 0.0


# ============================================================================
# Monotonicity
# ============================================================================


def test_pressure_decreases_with_altitude(
    atmosphere_model: AtmosphereCOESA1976,
) -> None:
    """
    Verifica que a pressão diminui monotonicamente com a altitude.

    Parameters
    ----------
    atmosphere_model : AtmosphereCOESA1976
        Modelo atmosférico.
    """
    altitudes_m = np.linspace(-5000.0, 1_000_000.0, 1001)

    pressures_Pa = np.array(
        [
            atmosphere_model.evaluate(altitude_m).pressure_Pa
            for altitude_m in altitudes_m
        ]
    )

    pressure_differences_Pa = np.diff(pressures_Pa)

    assert np.all(pressure_differences_Pa < 0.0)


def test_density_decreases_with_altitude(atmosphere_model: AtmosphereCOESA1976) -> None:
    """
    Verifica que a massa específica diminua monotonicamente com a altitude.

    Parameters
    ----------
    atmosphere_model : AtmosphereCOESA1976
        Modelo atmosférico.
    """
    altitudes_m = np.linspace(-5000.0, 1_000_000.0, 1001)

    densities_kg_m3 = np.array(
        [
            atmosphere_model.evaluate(altitude_m).density_kg_m3
            for altitude_m in altitudes_m
        ]
    )

    density_differences = np.diff(densities_kg_m3)

    assert np.all(density_differences < 0.0)


# ============================================================================
# Geometric / geopotential altitude
# ============================================================================


@pytest.mark.parametrize(
    "geometric_altitude_m, expected_geopotential_altitude_m",
    [
        (0.0, 0.0),
        (1000.0, 999.842769),
        (5000.0, 4996.067),
        (10000.0, 9984.287),
        (11000.0, 10980.998045),
        (20000.0, 19937.219),
        (50000.0, 49609.919),
        (86000.0, 84852.0),
    ],
)
def test_geopotential_altitude_conversion(
    atmosphere_model: AtmosphereCOESA1976,
    geometric_altitude_m: float,
    expected_geopotential_altitude_m: float,
) -> None:
    """
    Verifica a conversão de altitude geométrica para geopotencial.

    Parameters
    ----------
    atmosphere_model : AtmosphereCOESA1976
        Modelo atmosférico.
    geometric_altitude_m : float
        Altitude geométrica [m].
    expected_geopotential_altitude_m : float
        Altitude geopotencial esperada [m].
    """
    result = atmosphere_model._geopotential_altitude(geometric_altitude_m)

    assert result == pytest.approx(expected_geopotential_altitude_m, rel=1e-5, abs=1e-3)


# ============================================================================
# Layer boundaries
# ============================================================================


@pytest.mark.parametrize(
    "boundary_altitude_m", [11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0]
)
def test_layer_boundaries_are_continuous(
    atmosphere_model: AtmosphereCOESA1976, boundary_altitude_m: float
) -> None:
    """
    Verifica a continuidade das propriedades próximas às interfaces.

    A avaliação é feita em pontos muito próximos da fronteira, evitando
    comparar diretamente os dois ramos da implementação no mesmo ponto.

    Parameters
    ----------
    atmosphere_model : AtmosphereCOESA1976
        Modelo atmosférico.
    boundary_altitude_m : float
        Altitude da fronteira [m].
    """
    delta_altitude_m = 1e-3

    below = atmosphere_model.evaluate(boundary_altitude_m - delta_altitude_m)
    above = atmosphere_model.evaluate(boundary_altitude_m + delta_altitude_m)

    assert below.temperature_K == pytest.approx(above.temperature_K, rel=1e-5, abs=1e-3)

    assert below.pressure_Pa == pytest.approx(above.pressure_Pa, rel=1e-5, abs=1e-8)

    assert below.density_kg_m3 == pytest.approx(
        above.density_kg_m3, rel=1e-5, abs=1e-12
    )


# ============================================================================
# 86 km transition
# ============================================================================


def test_speed_of_sound_varies_above_86_km():
    """
    Verify that the speed of sound is calculated from the local
    temperature and molecular weight above 86 km.
    """
    atmosphere = AtmosphereCOESA1976()

    speed_86_km = atmosphere.evaluate(86_000.0).speed_of_sound_m_s
    speed_100_km = atmosphere.evaluate(100_000.0).speed_of_sound_m_s
    speed_120_km = atmosphere.evaluate(120_000.0).speed_of_sound_m_s

    assert speed_86_km != pytest.approx(speed_100_km, rel=1e-10)

    assert speed_100_km != pytest.approx(speed_120_km, rel=1e-10)


# ============================================================================
# Upper atmosphere temperature
# ============================================================================


def test_temperature_is_constant_between_86_and_91_km(
    atmosphere_model: AtmosphereCOESA1976,
) -> None:
    """
    Verifica a região isotérmica de 86 a 91 km.

    Parameters
    ----------
    atmosphere_model : AtmosphereCOESA1976
        Modelo atmosférico.
    """
    expected_temperature_K = 186.8673

    for altitude_m in [86_000.0, 87_000.0, 88_000.0, 89_000.0, 90_000.0, 91_000.0]:
        result = atmosphere_model.evaluate(altitude_m)

        assert result.temperature_K == pytest.approx(
            expected_temperature_K, rel=1e-12, abs=1e-12
        )


def test_temperature_reaches_360_k_at_120_km(
    atmosphere_model: AtmosphereCOESA1976,
) -> None:
    """
    Verifica a temperatura de 360 K em 120 km.

    Parameters
    ----------
    atmosphere_model : AtmosphereCOESA1976
        Modelo atmosférico.
    """
    result = atmosphere_model.evaluate(120_000.0)

    assert result.temperature_K == pytest.approx(360.0, rel=1e-12, abs=1e-12)


def test_temperature_asymptotically_approaches_1000_k(
    atmosphere_model: AtmosphereCOESA1976,
) -> None:
    """
    Verifica o comportamento assintótico da temperatura acima de 120 km.

    Parameters
    ----------
    atmosphere_model : AtmosphereCOESA1976
        Modelo atmosférico.
    """
    temperature_200km_K = atmosphere_model.evaluate(200_000.0).temperature_K

    temperature_500km_K = atmosphere_model.evaluate(500_000.0).temperature_K

    temperature_1000km_K = atmosphere_model.evaluate(1_000_000.0).temperature_K

    assert temperature_200km_K < temperature_500km_K
    assert temperature_500km_K < temperature_1000km_K
    assert temperature_1000km_K < 1000.0


# ============================================================================
# Speed of sound equation
# ============================================================================


@pytest.mark.parametrize(
    "altitude_m, expected_speed_of_sound_m_s",
    [
        (-5000.0, 358.9864564272176),
        (0.0, 340.294107787),
        (5000.0, 320.5455196704035),
        (20_000.0, 295.0695973539042),
        (50_000.0, 329.7988470709885),
        (80_000.0, 282.538030990),
        (85_000.0, 275.520075701),
    ],
)
def test_speed_of_sound_reference_values(
    atmosphere_model: AtmosphereCOESA1976,
    altitude_m: float,
    expected_speed_of_sound_m_s: float,
) -> None:
    """
    Verifica a velocidade do som contra valores de referência do
    modelo atmosférico COESA 1976.

    Parameters
    ----------
    atmosphere_model : AtmosphereCOESA1976
        Modelo atmosférico COESA 1976.
    altitude_m : float
        Altitude geométrica [m].
    expected_speed_of_sound_m_s : float
        Velocidade do som de referência [m/s].
    """
    result = atmosphere_model.evaluate(altitude_m)

    assert result.speed_of_sound_m_s == pytest.approx(
        expected_speed_of_sound_m_s, rel=1e-10, abs=1e-10
    )


@pytest.mark.parametrize(
    "altitude_m, expected_density_kg_m3",
    [
        (-5000.0, 1.9311215702612288),
        (0.0, 1.22499915589),
        (10_000.0, 0.413510428899),
        (20_000.0, 0.0889099150889),
        (50_000.0, 0.0010268780342616321),
        (86_000.0, 6.95728145961e-06),
        (100_000.0, 5.61552389223e-07),
        (500_000.0, 5.21525913445e-13),
        (1_000_000.0, 3.56064973536e-15),
    ],
)
def test_density_reference_values(
    atmosphere_model: AtmosphereCOESA1976,
    altitude_m: float,
    expected_density_kg_m3: float,
) -> None:
    """
    Verifica a massa específica contra valores de referência do
    modelo atmosférico COESA 1976.

    Parameters
    ----------
    atmosphere_model : AtmosphereCOESA1976
        Modelo atmosférico COESA 1976.
    altitude_m : float
        Altitude geométrica [m].
    expected_density_kg_m3 : float
        Massa específica de referência [kg/m^3].
    """
    result = atmosphere_model.evaluate(altitude_m)

    assert result.density_kg_m3 == pytest.approx(
        expected_density_kg_m3, rel=1e-10, abs=1e-15
    )


# ============================================================================
# Dynamic viscosity
# ============================================================================


@pytest.mark.parametrize(
    "altitude_m", [-5000.0, 0.0, 5000.0, 20_000.0, 50_000.0, 80_000.0, 86_000.0]
)
def test_dynamic_viscosity_is_positive(
    atmosphere_model: AtmosphereCOESA1976, altitude_m: float
) -> None:
    """
    Verifica que a viscosidade dinâmica seja positiva até 86 km.

    Parameters
    ----------
    atmosphere_model : AtmosphereCOESA1976
        Modelo atmosférico.
    altitude_m : float
        Altitude geométrica [m].
    """
    result = atmosphere_model.evaluate(altitude_m)

    viscosity_Pa_s = atmosphere_model.dynamic_viscosity(altitude_m)

    assert viscosity_Pa_s > 0.0


def test_dynamic_viscosity_follows_sutherland_relation(
    atmosphere_model: AtmosphereCOESA1976,
) -> None:
    """
    Verifica a equação de viscosidade utilizada pelo COESA.jl.

    A implementação utiliza:

        mu = beta * T^(3/2) / (T + S)

    com:

        beta = 1.458e-6
        S = 110.4 K

    Parameters
    ----------
    atmosphere_model : AtmosphereCOESA1976
        Modelo atmosférico.
    """
    beta = 1.458e-6
    sutherland_constant_K = 110.4

    for altitude_m in [0.0, 20_000.0, 50_000.0, 86_000.0]:
        result = atmosphere_model.evaluate(altitude_m)

        expected_viscosity_Pa_s = (
            beta
            * result.temperature_K**1.5
            / (result.temperature_K + sutherland_constant_K)
        )

        actual_viscosity_Pa_s = atmosphere_model.dynamic_viscosity(altitude_m)

        assert actual_viscosity_Pa_s == pytest.approx(
            expected_viscosity_Pa_s, rel=1e-12, abs=1e-15
        )


def test_dynamic_viscosity_above_86_km_raises_error(
    atmosphere_model: AtmosphereCOESA1976,
) -> None:
    """
    Verifica que a viscosidade não seja calculada acima de 86 km.

    Esse comportamento é explicitamente definido pelo COESA.jl.

    Parameters
    ----------
    atmosphere_model : AtmosphereCOESA1976
        Modelo atmosférico.
    """
    with pytest.raises(ValueError):
        atmosphere_model.dynamic_viscosity(86_000.1)


# ============================================================================
# Altitude limits
# ============================================================================


@pytest.mark.parametrize("altitude_m", [-5000.1, -10_000.0])
def test_altitude_below_lower_limit_raises_error(
    atmosphere_model: AtmosphereCOESA1976, altitude_m: float
) -> None:
    """
    Verifica rejeição de altitudes abaixo de -5000 m.

    Parameters
    ----------
    atmosphere_model : AtmosphereCOESA1976
        Modelo atmosférico.
    altitude_m : float
        Altitude geométrica [m].
    """
    with pytest.raises(ValueError):
        atmosphere_model.evaluate(altitude_m)


@pytest.mark.parametrize("altitude_m", [1_000_000.1, 1_100_000.0])
def test_altitude_above_upper_limit_raises_error(
    atmosphere_model: AtmosphereCOESA1976, altitude_m: float
) -> None:
    """
    Verifica rejeição de altitudes acima de 1000 km.

    Parameters
    ----------
    atmosphere_model : AtmosphereCOESA1976
        Modelo atmosférico.
    altitude_m : float
        Altitude geométrica [m].
    """
    with pytest.raises(ValueError):
        atmosphere_model.evaluate(altitude_m)


def test_lower_altitude_limit_is_valid(atmosphere_model: AtmosphereCOESA1976) -> None:
    """
    Verifica que exatamente -5000 m seja uma entrada válida.

    Parameters
    ----------
    atmosphere_model : AtmosphereCOESA1976
        Modelo atmosférico.
    """
    result = atmosphere_model.evaluate(-5000.0)

    assert result.temperature_K > 0.0
    assert result.pressure_Pa > 0.0
    assert result.density_kg_m3 > 0.0


def test_upper_altitude_limit_is_valid(atmosphere_model: AtmosphereCOESA1976) -> None:
    """
    Verifica que exatamente 1000 km seja uma entrada válida.

    Parameters
    ----------
    atmosphere_model : AtmosphereCOESA1976
        Modelo atmosférico.
    """
    result = atmosphere_model.evaluate(1_000_000.0)

    assert result.temperature_K > 0.0
    assert result.pressure_Pa > 0.0
    assert result.density_kg_m3 > 0.0


# ============================================================================
# Mathematical consistency
# ============================================================================


def test_sea_level_standard_conditions(atmosphere_model: AtmosphereCOESA1976) -> None:
    """
    Verifica as condições atmosféricas padrão ao nível do mar.

    Parameters
    ----------
    atmosphere_model : AtmosphereCOESA1976
        Modelo atmosférico.
    """
    result = atmosphere_model.evaluate(0.0)

    assert result.temperature_K == pytest.approx(288.15, rel=1e-12)

    assert result.pressure_Pa == pytest.approx(101325.0, rel=1e-12)


def test_zero_temperature_delta_preserves_standard_atmosphere():
    """
    Verify that zero temperature deviation reproduces the
    standard COESA 1976 atmosphere.
    """
    atmosphere_standard = AtmosphereCOESA1976()
    atmosphere_zero_delta = AtmosphereCOESA1976(delta_temperature_C=0.0)

    standard = atmosphere_standard.evaluate(100_000.0)
    zero_delta = atmosphere_zero_delta.evaluate(100_000.0)

    assert zero_delta == standard


def test_temperature_delta_affects_atmospheric_temperature():
    """
    Verify that a temperature deviation changes the atmospheric
    temperature by the configured amount.
    """
    atmosphere_standard = AtmosphereCOESA1976()
    atmosphere_perturbed = AtmosphereCOESA1976(delta_temperature_C=10.0)

    standard = atmosphere_standard.evaluate(100_000.0)
    perturbed = atmosphere_perturbed.evaluate(100_000.0)

    assert perturbed.temperature_K == pytest.approx(
        standard.temperature_K + 10.0, rel=1e-10, abs=1e-10
    )


def test_temperature_delta_affects_speed_of_sound():
    """
    Verify that a positive temperature deviation increases
    the speed of sound.
    """
    atmosphere_standard = AtmosphereCOESA1976()
    atmosphere_perturbed = AtmosphereCOESA1976(delta_temperature_C=10.0)

    standard = atmosphere_standard.evaluate(100_000.0)
    perturbed = atmosphere_perturbed.evaluate(100_000.0)

    assert perturbed.speed_of_sound_m_s > standard.speed_of_sound_m_s


def test_speed_of_sound_follows_temperature_delta():
    """
    Verify the speed-of-sound response to a temperature deviation.

    For a fixed altitude, pressure, and molecular weight, the speed
    of sound is proportional to the square root of the absolute
    temperature.
    """
    delta_temperature_C = 10.0

    atmosphere_standard = AtmosphereCOESA1976()
    atmosphere_perturbed = AtmosphereCOESA1976(delta_temperature_C=delta_temperature_C)

    standard = atmosphere_standard.evaluate(100_000.0)
    perturbed = atmosphere_perturbed.evaluate(100_000.0)

    expected_speed_of_sound_m_s = standard.speed_of_sound_m_s * np.sqrt(
        perturbed.temperature_K / standard.temperature_K
    )

    assert perturbed.speed_of_sound_m_s == pytest.approx(
        expected_speed_of_sound_m_s, rel=1e-10, abs=1e-10
    )

    expected_speed_of_sound_m_s = standard.speed_of_sound_m_s * np.sqrt(
        perturbed.temperature_K / standard.temperature_K
    )

    assert perturbed.speed_of_sound_m_s == pytest.approx(
        expected_speed_of_sound_m_s, rel=1e-10
    )
