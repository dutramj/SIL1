from __future__ import annotations

from typing import Any

from .atmosphere_base import AtmosphereBase
from .atmosphere_coesa1976 import AtmosphereCOESA1976
from .atmosphere_lut import AtmosphereLUT

ATMOSPHERE_MODELS = {"lut": AtmosphereLUT, "coesa1976": AtmosphereCOESA1976}


def atmosphere_factory(config: dict[str, Any]) -> AtmosphereBase:
    """Create an atmospheric model from configuration data.

    ```
    Parameters
    ----------
    config : dict[str, Any]
        Atmospheric model configuration. The ``type`` field specifies
        the concrete atmospheric model.

    Returns
    -------
    AtmosphereBase
        Instantiated atmospheric model.

    Raises
    ------
    ValueError
        If the atmospheric model type is not supported.
    """

    model_name = config["type"]

    try:
        atmosphere_class = ATMOSPHERE_MODELS[model_name]

    except KeyError:
        raise ValueError(
            f"Unknown atmosphere model: {model_name!r}. "
            f"Available models: {list(ATMOSPHERE_MODELS)}"
        ) from None

    atmosphere_config = {key: value for key, value in config.items() if key != "type"}

    if model_name == "coesa1976":
        # COESA1976 is a physics-based model and does not use
        # the lookup-table interpolator.
        atmosphere_config.pop("interpolator", None)

        # Use delta_temperature_C = 0.0 if coesa_delta_temperature is ommited
        atmosphere_config["delta_temperature_C"] = atmosphere_config.pop(
            "coesa_delta_temperature", 0.0
        )

    elif model_name == "lut":
        # LUT does not use the COESA1976 temperature deviation.
        atmosphere_config.pop("coesa_delta_temperature", None)

    return atmosphere_class(**atmosphere_config)
