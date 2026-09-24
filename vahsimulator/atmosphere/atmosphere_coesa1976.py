from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
from pydantic import PrivateAttr

from .atmosphere_base import AtmosphereBase
from .atmosphere_data import AtmosphereData


class AtmosphereCOESA1976(AtmosphereBase):
    """
    U.S. Standard Atmosphere 1976 atmospheric model.

    This class implements the atmospheric model used by the COESA.jl
    reference implementation provided for this simulator.

    The model provides atmospheric temperature, pressure, density, and
    speed of sound as a function of geometric altitude.

    The implementation is divided into two regions:

    - 0 to 86 km: geopotential altitude formulation using the standard
    atmospheric lapse-rate layers.
    - 86 km to 1000 km: upper-atmosphere formulation using the
    temperature equations and tabulated pressure and molecular-weight
    data from the reference implementation.

    Notes
    -----
    The input altitude is geometric altitude, while the lower-atmosphere
    equations use geopotential altitude.

    The model supports altitudes from -5 km to 1000 km, matching the
    limits imposed by the reference COESA.jl implementation.

    A temperature deviation may be supplied through ``delta_temperature_C``.
    The deviation is added to the COESA temperature. Pressure is kept at
    the standard COESA value, while density and speed of sound are
    recalculated using the modified temperature.

    References
    ----------
    COESA.jl reference implementation supplied for this simulator.
    U.S. Standard Atmosphere 1976.
    https://github.com/danielmatz/COESA.jl/blob/master/README.md
    """

    # Effective Earth radius at 45 deg latitude [m].
    _EARTH_RADIUS_M: ClassVar[float] = 6_356_766.0

    # Standard gravitational acceleration [m/s^2].
    _STANDARD_GRAVITY_M_S2: ClassVar[float] = 9.80665

    # Molecular weight at lower altitudes [kg/kmol].
    _REFERENCE_MOLECULAR_WEIGHT_KG_KMOL: ClassVar[float] = 28.9644

    # Universal gas constant [N m/(kmol K)].
    _UNIVERSAL_GAS_CONSTANT_J_KMOL_K: ClassVar[float] = 8.31432e3

    # Ratio of specific heats [-].
    _GAMMA: ClassVar[float] = 1.4

    # Atmospheric layer geopotential base altitudes [m].
    _BASE_GEOPOTENTIAL_ALTITUDES_M: ClassVar[np.ndarray] = np.array(
        [0.0, 11_000.0, 20_000.0, 32_000.0, 47_000.0, 51_000.0, 71_000.0], dtype=float
    )

    # Atmospheric layer temperature lapse rates [K/m].
    _TEMPERATURE_LAPSE_RATES_K_M: ClassVar[np.ndarray] = np.array(
        [-6.5e-3, 0.0, 1.0e-3, 2.8e-3, 0.0, -2.8e-3, -2.0e-3], dtype=float
    )

    # Upper-atmosphere altitude table [m].
    _UPPER_ALTITUDES_M: ClassVar[np.ndarray] = np.array(
        [
            80_000.0,
            80_500.0,
            81_000.0,
            81_500.0,
            82_000.0,
            82_500.0,
            83_000.0,
            83_500.0,
            84_000.0,
            84_500.0,
            85_000.0,
            85_500.0,
            86_000.0,
        ],
        dtype=float,
    )

    # Ratio of molecular weight to the reference molecular weight [-].
    _MOLECULAR_WEIGHT_RATIOS: ClassVar[np.ndarray] = np.array(
        [
            1.0,
            0.999996,
            0.999989,
            0.999971,
            0.999941,
            0.999909,
            0.999870,
            0.999829,
            0.999786,
            0.999741,
            0.999694,
            0.999641,
            0.999579,
        ],
        dtype=float,
    )

    # Upper-atmosphere altitude table [m].
    _UPPER_PRESSURE_ALTITUDES_M: ClassVar[np.ndarray] = np.array(
        [
            86_000.0,
            87_000.0,
            88_000.0,
            89_000.0,
            90_000.0,
            91_000.0,
            93_000.0,
            95_000.0,
            97_000.0,
            99_000.0,
            101_000.0,
            103_000.0,
            105_000.0,
            107_000.0,
            109_000.0,
            110_000.0,
            111_000.0,
            112_000.0,
            113_000.0,
            114_000.0,
            115_000.0,
            116_000.0,
            117_000.0,
            118_000.0,
            119_000.0,
            120_000.0,
            125_000.0,
            130_000.0,
            135_000.0,
            140_000.0,
            145_000.0,
            150_000.0,
            160_000.0,
            170_000.0,
            180_000.0,
            190_000.0,
            200_000.0,
            210_000.0,
            220_000.0,
            230_000.0,
            240_000.0,
            250_000.0,
            260_000.0,
            270_000.0,
            280_000.0,
            290_000.0,
            300_000.0,
            310_000.0,
            320_000.0,
            330_000.0,
            340_000.0,
            350_000.0,
            360_000.0,
            370_000.0,
            380_000.0,
            390_000.0,
            400_000.0,
            410_000.0,
            420_000.0,
            430_000.0,
            440_000.0,
            450_000.0,
            460_000.0,
            470_000.0,
            480_000.0,
            490_000.0,
            500_000.0,
            525_000.0,
            550_000.0,
            575_000.0,
            600_000.0,
            625_000.0,
            650_000.0,
            675_000.0,
            700_000.0,
            725_000.0,
            750_000.0,
            775_000.0,
            800_000.0,
            825_000.0,
            850_000.0,
            875_000.0,
            900_000.0,
            925_000.0,
            950_000.0,
            975_000.0,
            1_000_000.0,
        ],
        dtype=float,
    )

    # Upper-atmosphere pressure table [Pa].
    _UPPER_PRESSURES_PA: ClassVar[np.ndarray] = np.array(
        [
            3.7338e-1,
            3.1259e-1,
            2.6173e-1,
            2.1919e-1,
            1.8359e-1,
            1.5381e-1,
            1.0801e-1,
            7.5966e-2,
            5.3571e-2,
            3.7948e-2,
            2.7192e-2,
            1.9742e-2,
            1.4477e-2,
            1.0751e-2,
            8.1142e-3,
            7.1042e-3,
            6.2614e-3,
            5.5547e-3,
            4.9570e-3,
            4.4473e-3,
            4.0096e-3,
            3.6312e-3,
            3.3022e-3,
            3.0144e-3,
            2.7615e-3,
            2.5382e-3,
            1.7354e-3,
            1.2505e-3,
            9.3568e-4,
            7.2028e-4,
            5.6691e-4,
            4.5422e-4,
            3.0395e-4,
            2.1210e-4,
            1.5271e-4,
            1.1266e-4,
            8.4736e-5,
            6.4756e-5,
            5.0149e-5,
            3.9276e-5,
            3.1059e-5,
            2.4767e-5,
            1.9894e-5,
            1.6083e-5,
            1.3076e-5,
            1.0683e-5,
            8.7704e-6,
            7.2285e-6,
            5.9796e-6,
            4.9630e-6,
            4.1320e-6,
            3.4498e-6,
            2.8878e-6,
            2.4234e-6,
            2.0384e-6,
            1.7184e-6,
            1.4518e-6,
            1.2291e-6,
            1.0427e-6,
            8.8645e-7,
            7.5517e-7,
            6.4468e-7,
            5.5155e-7,
            4.7292e-7,
            4.0642e-7,
            3.5011e-7,
            3.0236e-7,
            2.1200e-7,
            1.5137e-7,
            1.1028e-7,
            8.2130e-8,
            6.2601e-8,
            4.8865e-8,
            3.9048e-8,
            3.1908e-8,
            2.6611e-8,
            2.2599e-8,
            1.9493e-8,
            1.7036e-8,
            1.5051e-8,
            1.3415e-8,
            1.2043e-8,
            1.0873e-8,
            9.8635e-9,
            8.9816e-9,
            8.2043e-9,
            7.5138e-9,
        ],
        dtype=float,
    )

    # Upper-atmosphere mean molecular weight [kg/kmol].
    _UPPER_MOLECULAR_WEIGHTS_KG_KMOL: ClassVar[np.ndarray] = np.array(
        [
            28.95,
            28.95,
            28.94,
            28.93,
            28.91,
            28.89,
            28.82,
            28.73,
            28.62,
            28.48,
            28.30,
            28.10,
            27.88,
            27.64,
            27.39,
            27.27,
            27.14,
            27.02,
            26.90,
            26.79,
            26.68,
            26.58,
            26.48,
            26.38,
            26.29,
            26.20,
            25.80,
            25.44,
            25.09,
            24.75,
            24.42,
            24.10,
            23.49,
            22.90,
            22.34,
            21.81,
            21.30,
            20.83,
            20.37,
            19.95,
            19.56,
            19.19,
            18.85,
            18.53,
            18.24,
            17.97,
            17.73,
            17.50,
            17.29,
            17.09,
            16.91,
            16.74,
            16.57,
            16.42,
            16.27,
            16.13,
            15.98,
            15.84,
            15.70,
            15.55,
            15.40,
            15.25,
            15.08,
            14.91,
            14.73,
            14.54,
            14.33,
            13.76,
            13.09,
            12.34,
            11.51,
            10.62,
            9.72,
            8.83,
            8.00,
            7.24,
            6.58,
            6.01,
            5.54,
            5.16,
            4.85,
            4.60,
            4.40,
            4.25,
            4.12,
            4.02,
            3.94,
        ],
        dtype=float,
    )

    # Sutherland-law constants for dynamic viscosity.
    _VISCOSITY_BETA_KG_S_M_K_HALF: ClassVar[float] = 1.458e-6
    _VISCOSITY_SUTHERLAND_K: ClassVar[float] = 110.4

    # Model altitude limits [m].
    _MIN_ALTITUDE_M: ClassVar[float] = -5_000.0
    _MAX_ALTITUDE_M: ClassVar[float] = 1_000_000.0

    _delta_temperature_C: float = PrivateAttr(default=0.0)

    def __init__(self, delta_temperature_C: float = 0.0) -> None:
        """Initialize the COESA 1976 atmosphere model.

        Parameters
        ----------
        delta_temperature_C : float, optional
            Temperature deviation from the standard COESA 1976 atmosphere
            [°C]. The deviation is added to the COESA temperature.
        """
        super().__init__()

        self._base_temperatures_K: np.ndarray = self._calculate_base_temperatures()
        self._base_pressures_Pa: np.ndarray = self._calculate_base_pressures()
        self._delta_temperature_C: float = float(delta_temperature_C)

    @classmethod
    def _speed_of_sound(
        cls, temperature_K: float, molecular_weight_kg_kmol: float
    ) -> float:
        """
        Calculate the local speed of sound.

        Parameters
        ----------
        temperature_K : float
            Static atmospheric temperature [K].
        molecular_weight_kg_kmol : float
            Mean molecular weight [kg/kmol].

        Returns
        -------
        float
            Speed of sound [m/s].
        """
        return math.sqrt(
            cls._GAMMA
            * cls._UNIVERSAL_GAS_CONSTANT_J_KMOL_K
            * temperature_K
            / molecular_weight_kg_kmol
        )

    def evaluate(self, altitude_m: float) -> AtmosphereData:
        """
        Evaluate atmospheric properties at a given geometric altitude.

        Parameters
        ----------
        altitude_m : float
            Geometric altitude above mean sea level [m].

        Returns
        -------
        AtmosphereData
            Atmospheric temperature [K], pressure [Pa], density [kg/m^3],
            and speed of sound [m/s].

        Raises
        ------
        ValueError
            If the altitude is below -5 km or above 1000 km.

        Notes
        -----
        The implementation follows the same branch logic as the
        reference COESA.jl implementation:

        - Below 86 km, geopotential altitude is used.
        - From 86 km to 1000 km, the upper-atmosphere formulation is used.
        - ``self.delta_temperature_C`` is added to the COESA temperature.
        - Pressure remains unchanged by ``self.delta_temperature_C``.
        - Density and speed of sound are recalculated using the modified
        temperature.
        """
        altitude_m = float(altitude_m)

        self._check_altitude(altitude_m)

        if altitude_m < 86_000.0:
            geopotential_altitude_m = self._geopotential_altitude(altitude_m)

            molecular_weight_kg_kmol = self._mean_molecular_weight_lower(altitude_m)

            temperature_K = self._temperature_lower(
                geopotential_altitude_m, molecular_weight_kg_kmol
            )

            pressure_Pa = self._pressure_lower(geopotential_altitude_m)

        else:
            temperature_K = self._temperature_upper(altitude_m)

            pressure_Pa = self._pressure_upper(altitude_m)

            molecular_weight_kg_kmol = self._mean_molecular_weight_upper(altitude_m)

        # Apply temperature deviation relative to the COESA 1976
        # standard atmosphere.
        temperature_K += self._delta_temperature_C

        speed_of_sound_m_s = self._speed_of_sound(
            temperature_K, molecular_weight_kg_kmol
        )

        density_kg_m3 = self._density(
            pressure_Pa, molecular_weight_kg_kmol, temperature_K
        )

        return AtmosphereData(
            temperature_K=temperature_K,
            pressure_Pa=pressure_Pa,
            density_kg_m3=density_kg_m3,
            speed_of_sound_m_s=speed_of_sound_m_s,
        )

    def mean_molecular_weight(self, altitude_m: float) -> float:
        """
        Calculate mean molecular weight at a given altitude.

        Parameters
        ----------
        altitude_m : float
            Geometric altitude above mean sea level [m].

        Returns
        -------
        float
            Mean molecular weight [kg/kmol].

        Raises
        ------
        ValueError
            If the altitude is outside the model limits.
        """
        altitude_m = float(altitude_m)
        self._check_altitude(altitude_m)

        if altitude_m < 86_000.0:
            return self._mean_molecular_weight_lower(altitude_m)

        return self._mean_molecular_weight_upper(altitude_m)

    def dynamic_viscosity(self, altitude_m: float) -> float:
        """
        Calculate dynamic viscosity at a given altitude.

        Parameters
        ----------
        altitude_m : float
            Geometric altitude above mean sea level [m].

        Returns
        -------
        float
            Dynamic viscosity [kg/(m s)].

        Raises
        ------
        ValueError
            If the altitude is above 86 km.

        Notes
        -----
        The dynamic viscosity is calculated using the Sutherland-type
        expression used by the reference implementation.
        """
        altitude_m = float(altitude_m)

        self._check_altitude(altitude_m)

        if altitude_m > 86_000.0:
            raise ValueError("Dynamic viscosity cannot be calculated above 86 km.")

        temperature_K = self.evaluate(altitude_m).temperature_K

        beta = self._VISCOSITY_BETA_KG_S_M_K_HALF
        sutherland_temperature_K = self._VISCOSITY_SUTHERLAND_K

        return beta * temperature_K**1.5 / (temperature_K + sutherland_temperature_K)

    @classmethod
    def _check_altitude(cls, altitude_m: float) -> None:
        """
        Validate the requested altitude.

        Parameters
        ----------
        altitude_m : float
            Geometric altitude [m].

        Raises
        ------
        ValueError
            If altitude is outside the model limits.
        """
        if altitude_m < cls._MIN_ALTITUDE_M:
            raise ValueError(
                "Altitude is below the COESA 1976 lower limit " "of -5000 m."
            )

        if altitude_m > cls._MAX_ALTITUDE_M:
            raise ValueError(
                "Altitude is above the COESA 1976 upper limit " "of 1000000 m."
            )

    @classmethod
    def _geopotential_altitude(cls, geometric_altitude_m: float) -> float:
        """
        Convert geometric altitude to geopotential altitude.

        Parameters
        ----------
        geometric_altitude_m : float
            Geometric altitude [m].

        Returns
        -------
        float
            Geopotential altitude [m].
        """
        earth_radius_m = cls._EARTH_RADIUS_M

        return (
            earth_radius_m
            * geometric_altitude_m
            / (earth_radius_m + geometric_altitude_m)
        )

    @classmethod
    def _calculate_base_temperatures(cls) -> np.ndarray:
        """
        Calculate temperatures at atmospheric layer boundaries.

        Returns
        -------
        numpy.ndarray
            Base temperatures [K].
        """
        base_altitudes_m = cls._BASE_GEOPOTENTIAL_ALTITUDES_M
        lapse_rates_K_m = cls._TEMPERATURE_LAPSE_RATES_K_M

        base_temperatures_K = np.empty_like(base_altitudes_m)

        base_temperatures_K[0] = 288.15

        for index in range(len(base_altitudes_m) - 1):
            base_temperatures_K[index + 1] = base_temperatures_K[
                index
            ] + lapse_rates_K_m[index] * (
                base_altitudes_m[index + 1] - base_altitudes_m[index]
            )

        return base_temperatures_K

    def _calculate_base_pressures(self) -> np.ndarray:
        """
        Calculate pressures at atmospheric layer boundaries.

        Returns
        -------
        numpy.ndarray
            Base pressures [Pa].
        """
        base_altitudes_m = self._BASE_GEOPOTENTIAL_ALTITUDES_M
        lapse_rates_K_m = self._TEMPERATURE_LAPSE_RATES_K_M
        base_temperatures_K = self._base_temperatures_K

        base_pressures_Pa = np.empty_like(base_altitudes_m)

        base_pressures_Pa[0] = 101_325.0

        for index in range(len(base_altitudes_m) - 1):
            altitude_difference_m = (
                base_altitudes_m[index + 1] - base_altitudes_m[index]
            )

            if lapse_rates_K_m[index] == 0.0:
                base_pressures_Pa[index + 1] = base_pressures_Pa[index] * math.exp(
                    -self._STANDARD_GRAVITY_M_S2
                    * self._REFERENCE_MOLECULAR_WEIGHT_KG_KMOL
                    * altitude_difference_m
                    / (
                        self._UNIVERSAL_GAS_CONSTANT_J_KMOL_K
                        * base_temperatures_K[index]
                    )
                )

            else:
                temperature_ratio = base_temperatures_K[index] / (
                    base_temperatures_K[index]
                    + lapse_rates_K_m[index] * altitude_difference_m
                )

                exponent = (
                    self._STANDARD_GRAVITY_M_S2
                    * self._REFERENCE_MOLECULAR_WEIGHT_KG_KMOL
                    / (self._UNIVERSAL_GAS_CONSTANT_J_KMOL_K * lapse_rates_K_m[index])
                )

                base_pressures_Pa[index + 1] = (
                    base_pressures_Pa[index] * temperature_ratio**exponent
                )

        return base_pressures_Pa

    def _find_lower_layer(self, geopotential_altitude_m: float) -> int:
        """
        Find the atmospheric layer containing the altitude.

        Parameters
        ----------
        geopotential_altitude_m : float
            Geopotential altitude [m].

        Returns
        -------
        int
            Zero-based atmospheric layer index.
        """
        base_altitudes_m = self._BASE_GEOPOTENTIAL_ALTITUDES_M

        layer_index = 0

        while (
            layer_index < len(base_altitudes_m) - 1
            and geopotential_altitude_m > base_altitudes_m[layer_index + 1]
        ):
            layer_index += 1

        return layer_index

    def _temperature_lower(
        self, geopotential_altitude_m: float, molecular_weight_kg_kmol: float
    ) -> float:
        """
        Calculate lower-atmosphere temperature.

        Parameters
        ----------
        geopotential_altitude_m : float
            Geopotential altitude [m].
        molecular_weight_kg_kmol : float
            Mean molecular weight [kg/kmol].

        Returns
        -------
        float
            Temperature [K].
        """
        layer_index = self._find_lower_layer(geopotential_altitude_m)

        return (
            (
                self._base_temperatures_K[layer_index]
                + self._TEMPERATURE_LAPSE_RATES_K_M[layer_index]
                * (
                    geopotential_altitude_m
                    - self._BASE_GEOPOTENTIAL_ALTITUDES_M[layer_index]
                )
            )
            / self._REFERENCE_MOLECULAR_WEIGHT_KG_KMOL
            * (molecular_weight_kg_kmol)
        )

    def _pressure_lower(self, geopotential_altitude_m: float) -> float:
        """
        Calculate lower-atmosphere pressure.

        Parameters
        ----------
        geopotential_altitude_m : float
            Geopotential altitude [m].

        Returns
        -------
        float
            Pressure [Pa].
        """
        layer_index = self._find_lower_layer(geopotential_altitude_m)

        base_altitude_m = self._BASE_GEOPOTENTIAL_ALTITUDES_M[layer_index]
        base_temperature_K = self._base_temperatures_K[layer_index]
        base_pressure_Pa = self._base_pressures_Pa[layer_index]
        lapse_rate_K_m = self._TEMPERATURE_LAPSE_RATES_K_M[layer_index]

        altitude_difference_m = geopotential_altitude_m - base_altitude_m

        if lapse_rate_K_m == 0.0:
            return base_pressure_Pa * math.exp(
                -self._STANDARD_GRAVITY_M_S2
                * self._REFERENCE_MOLECULAR_WEIGHT_KG_KMOL
                * altitude_difference_m
                / (self._UNIVERSAL_GAS_CONSTANT_J_KMOL_K * base_temperature_K)
            )

        temperature_ratio = base_temperature_K / (
            base_temperature_K + lapse_rate_K_m * altitude_difference_m
        )

        exponent = (
            self._STANDARD_GRAVITY_M_S2
            * self._REFERENCE_MOLECULAR_WEIGHT_KG_KMOL
            / (self._UNIVERSAL_GAS_CONSTANT_J_KMOL_K * lapse_rate_K_m)
        )

        return base_pressure_Pa * temperature_ratio**exponent

    def _mean_molecular_weight_lower(self, geometric_altitude_m: float) -> float:
        """
        Calculate lower-atmosphere mean molecular weight.

        Parameters
        ----------
        geometric_altitude_m : float
            Geometric altitude [m].

        Returns
        -------
        float
            Mean molecular weight [kg/kmol].
        """
        altitude_table_m = self._UPPER_ALTITUDES_M
        molecular_weight_ratios = self._MOLECULAR_WEIGHT_RATIOS

        if geometric_altitude_m < altitude_table_m[0]:
            molecular_weight_ratio = 1.0

        elif geometric_altitude_m > altitude_table_m[-1]:
            raise ValueError(
                "Altitude is above the lower-atmosphere " "molecular-weight table."
            )

        else:
            molecular_weight_ratio = float(
                np.interp(
                    geometric_altitude_m, altitude_table_m, molecular_weight_ratios
                )
            )

        return self._REFERENCE_MOLECULAR_WEIGHT_KG_KMOL * molecular_weight_ratio

    @classmethod
    def _temperature_upper(cls, geometric_altitude_m: float) -> float:
        """
        Calculate upper-atmosphere temperature.

        Parameters
        ----------
        geometric_altitude_m : float
            Geometric altitude [m].

        Returns
        -------
        float
            Temperature [K].
        """
        altitude_m = geometric_altitude_m

        if altitude_m <= 91_000.0:
            return 186.8673

        if altitude_m <= 110_000.0:
            reference_temperature_K = 263.1905
            temperature_amplitude_K = -76.3232
            reference_altitude_m = -19_942.9

            normalized_altitude = (altitude_m - 91_000.0) / reference_altitude_m

            return reference_temperature_K + (
                temperature_amplitude_K * math.sqrt(1.0 - normalized_altitude**2)
            )

        if altitude_m <= 120_000.0:
            temperature_at_110_km_K = 240.0
            lapse_rate_K_m = 12e-3
            reference_altitude_m = 110_000.0

            return temperature_at_110_km_K + lapse_rate_K_m * (
                altitude_m - reference_altitude_m
            )

        temperature_at_120_km_K = 360.0
        reference_altitude_m = 120_000.0
        asymptotic_temperature_K = 1_000.0
        inverse_scale_length_per_m = 0.01875e-3

        geopotential_coordinate_m = (
            (altitude_m - reference_altitude_m)
            * (cls._EARTH_RADIUS_M + reference_altitude_m)
            / (cls._EARTH_RADIUS_M + altitude_m)
        )

        return asymptotic_temperature_K - (
            asymptotic_temperature_K - temperature_at_120_km_K
        ) * math.exp(-inverse_scale_length_per_m * geopotential_coordinate_m)

    @classmethod
    def _interpolation_index(cls, altitude_m: float) -> int:
        """
        Find the center index for upper-atmosphere interpolation.

        Parameters
        ----------
        altitude_m : float
            Geometric altitude [m].

        Returns
        -------
        int
            Zero-based center index used for quadratic interpolation.
        """
        altitude_table_m = cls._UPPER_PRESSURE_ALTITUDES_M

        index = 0

        while (
            index < len(altitude_table_m) - 1
            and altitude_m > altitude_table_m[index + 1]
        ):
            index += 1

        # The quadratic interpolation requires index - 1,
        # index, and index + 1.
        index = max(1, min(index, len(altitude_table_m) - 2))

        return index

    @classmethod
    def _interpolation_scale_factors(
        cls, center_index: int, altitude_m: float
    ) -> tuple[float, float, float]:
        """
        Calculate quadratic interpolation scale factors.

        Parameters
        ----------
        center_index : int
            Zero-based center interpolation index.
        altitude_m : float
            Geometric altitude [m].

        Returns
        -------
        tuple of float
            Lagrange interpolation scale factors.
        """
        altitude_table_m = cls._UPPER_PRESSURE_ALTITUDES_M

        altitude_0_m = altitude_table_m[center_index - 1]
        altitude_1_m = altitude_table_m[center_index]
        altitude_2_m = altitude_table_m[center_index + 1]

        scale_0 = (
            (altitude_m - altitude_1_m)
            * (altitude_m - altitude_2_m)
            / ((altitude_0_m - altitude_1_m) * (altitude_0_m - altitude_2_m))
        )

        scale_1 = (
            (altitude_m - altitude_0_m)
            * (altitude_m - altitude_2_m)
            / ((altitude_1_m - altitude_0_m) * (altitude_1_m - altitude_2_m))
        )

        scale_2 = (
            (altitude_m - altitude_0_m)
            * (altitude_m - altitude_1_m)
            / ((altitude_2_m - altitude_0_m) * (altitude_2_m - altitude_1_m))
        )

        return scale_0, scale_1, scale_2

    @classmethod
    def _pressure_upper(cls, altitude_m: float) -> float:
        """
        Calculate upper-atmosphere pressure.

        Pressure is interpolated quadratically in logarithmic space.

        Parameters
        ----------
        altitude_m : float
            Geometric altitude [m].

        Returns
        -------
        float
            Pressure [Pa].
        """
        center_index = cls._interpolation_index(altitude_m)

        scale_0, scale_1, scale_2 = cls._interpolation_scale_factors(
            center_index, altitude_m
        )

        log_pressures = np.log(cls._UPPER_PRESSURES_PA)

        log_pressure = (
            log_pressures[center_index - 1] * scale_0
            + log_pressures[center_index] * scale_1
            + log_pressures[center_index + 1] * scale_2
        )

        return math.exp(log_pressure)

    @classmethod
    def _mean_molecular_weight_upper(cls, altitude_m: float) -> float:
        """
        Calculate upper-atmosphere mean molecular weight.

        Parameters
        ----------
        altitude_m : float
            Geometric altitude [m].

        Returns
        -------
        float
            Mean molecular weight [kg/kmol].
        """
        center_index = cls._interpolation_index(altitude_m)

        scale_0, scale_1, scale_2 = cls._interpolation_scale_factors(
            center_index, altitude_m
        )

        molecular_weights = cls._UPPER_MOLECULAR_WEIGHTS_KG_KMOL

        return (
            molecular_weights[center_index - 1] * scale_0
            + molecular_weights[center_index] * scale_1
            + molecular_weights[center_index + 1] * scale_2
        )

    @classmethod
    def _density(
        cls, pressure_Pa: float, molecular_weight_kg_kmol: float, temperature_K: float
    ) -> float:
        """
        Calculate atmospheric density from the ideal gas law.

        Parameters
        ----------
        pressure_Pa : float
            Static pressure [Pa].
        molecular_weight_kg_kmol : float
            Mean molecular weight [kg/kmol].
        temperature_K : float
            Static temperature [K].

        Returns
        -------
        float
            Atmospheric density [kg/m^3].
        """
        return (
            pressure_Pa
            * molecular_weight_kg_kmol
            / (cls._UNIVERSAL_GAS_CONSTANT_J_KMOL_K * temperature_K)
        )
