from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel

from .atmosphere_data import AtmosphereData


class AtmosphereBase(BaseModel, ABC):
    """Interface base para modelos atmosféricos."""

    model_config = {"arbitrary_types_allowed": True}

    @abstractmethod
    def evaluate(self, altitude_m: float) -> AtmosphereData:
        """Avalia as propriedades atmosféricas.

        Parameters
        ----------
        altitude_m : float
            Altitude do veículo [m].

        Returns
        -------
        AtmosphereData
            Propriedades atmosféricas na altitude solicitada.
        """
        raise NotImplementedError
