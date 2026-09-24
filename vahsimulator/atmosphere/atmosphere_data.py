from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class AtmosphereData:
    """
    Atmospheric properties evaluated at a given geometric altitude.

    Parameters
    ----------
    temperature_K : float
        Static atmospheric temperature [K].
    pressure_Pa : float
        Static atmospheric pressure [Pa].
    density_kg_m3 : float
        Atmospheric density [kg/m^3].
    speed_of_sound_m_s : float
        Local speed of sound [m/s].
    """

    temperature_K: float
    pressure_Pa: float
    density_kg_m3: float
    speed_of_sound_m_s: float
