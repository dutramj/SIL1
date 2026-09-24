from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import norm


class GustComponentParameters(BaseModel):
    """Parâmetros de uma componente da rajada discreta.

    Parameters
    ----------
    standard_deviation : float
        Desvio padrão da componente da velocidade da rajada, em m/s.
    length_scale : float
        Escala de comprimento da componente, em metros.
    """

    model_config = ConfigDict(extra="forbid")

    standard_deviation: float = Field(
        ge=0.0, description="Desvio padrão da velocidade da rajada em m/s."
    )

    length_scale: float = Field(
        gt=0.0, description="Escala de comprimento da rajada em metros."
    )


class DiscreteGustParameters(BaseModel):
    """Parâmetros completos de uma rajada discreta.

    Parameters
    ----------
    leading_edge_altitude_m : float
        Altitude da borda dianteira da rajada, em metros.
    half_width : float
        Meia largura espacial da rajada, em metros.
        O comprimento total da rajada é ``2 * half_width``.
    risk : float
        Probabilidade de ocorrência da rajada.
    longitudinal : GustComponentParameters
        Parâmetros da componente longitudinal.
    lateral : GustComponentParameters
        Parâmetros da componente lateral.
    vertical : GustComponentParameters
        Parâmetros da componente vertical.
    """

    model_config = ConfigDict(extra="forbid")

    leading_edge_altitude_m: float = Field(
        description="Altitude da borda dianteira da rajada em metros."
    )

    half_width: float = Field(gt=0.0, description="Meia largura da rajada em metros.")

    risk: float = Field(
        ge=0.0, le=1.0, description="Probabilidade de ocorrência da rajada."
    )

    longitudinal: GustComponentParameters
    lateral: GustComponentParameters
    vertical: GustComponentParameters

    @property
    def total_length_m(self) -> float:
        """Retorna o comprimento total da rajada.

        Returns
        -------
        float
            Comprimento total da rajada, em metros.
        """

        return 2.0 * self.half_width


@dataclass(frozen=True)
class GustVector:
    """Vetor de velocidade produzido por uma rajada.

    Attributes
    ----------
    longitudinal : float
        Componente longitudinal, em m/s.
    lateral : float
        Componente lateral, em m/s.
    vertical : float
        Componente vertical, em m/s.
    """

    longitudinal: float
    lateral: float
    vertical: float

    def as_array(self) -> np.ndarray:
        """Converte o vetor para um array coluna.

        Returns
        -------
        numpy.ndarray
            Vetor ``3 x 1`` contendo as componentes da rajada.
        """

        return np.array(
            [[self.longitudinal], [self.lateral], [self.vertical]], dtype=np.float64
        )


class DiscreteGustWind:
    """Modelo de uma única rajada discreta.

    O modelo utiliza um perfil espacial 1-cosseno. A posição do veículo
    dentro da rajada é determinada diretamente pela altitude atual do
    veículo, não pelo avanço de um estado interno.

    Parameters
    ----------
    parameters : DiscreteGustParameters
        Parâmetros da rajada.

    Notes
    -----
    A borda dianteira da rajada ocorre em
    ``leading_edge_altitude_m``.

    A borda traseira ocorre em

    ``leading_edge_altitude_m + 2 * half_width``.

    O modelo não realiza transformação para o referencial do corpo.
    Essa responsabilidade pertence ao módulo de transformação do vento.
    """

    def __init__(self, parameters: DiscreteGustParameters) -> None:
        self.parameters = parameters

    @property
    def leading_edge_altitude_m(self) -> float:
        """Altitude da borda dianteira da rajada.

        Returns
        -------
        float
            Altitude em metros.
        """

        return self.parameters.leading_edge_altitude_m

    @property
    def trailing_edge_altitude_m(self) -> float:
        """Altitude da borda traseira da rajada.

        Returns
        -------
        float
            Altitude em metros.
        """

        return self.parameters.leading_edge_altitude_m + self.parameters.total_length_m

    def is_active(self, altitude_m: float) -> bool:
        """Verifica se a rajada está ativa na altitude informada.

        Parameters
        ----------
        altitude_m : float
            Altitude atual do veículo, em metros.

        Returns
        -------
        bool
            ``True`` se o veículo estiver dentro da rajada.
        """

        return (
            self.leading_edge_altitude_m <= altitude_m <= self.trailing_edge_altitude_m
        )

    def profile(self, distance_m: float, component: GustComponentParameters) -> float:
        """Avalia o perfil 1-cosseno da rajada discreta.

        O perfil é definido por:

        .. math::

            V(x) =
            \\frac{V_m}{2}
            \\left[
                1 -
                \\cos\\left(
                    \\frac{\\pi x}{d_m}
                \\right)
            \\right]

        para:

        .. math::

            0 \\leq x \\leq 2d_m

        onde ``V_m`` é a magnitude máxima calculada pelo modelo de
        Leahy (2008).

        Parameters
        ----------
        distance_m : float
            Distância percorrida desde a borda dianteira da rajada, em m.
        component : GustComponentParameters
            Parâmetros estatísticos da componente de turbulência.

        Returns
        -------
        float
            Velocidade da rajada em m/s.
        """
        if distance_m < 0.0 or distance_m > self.parameters.total_length_m:
            return 0.0

        magnitude = self._gust_magnitude(component)

        return (
            0.5
            * magnitude
            * (1.0 - np.cos(np.pi * distance_m / self.parameters.half_width))
        )

    def evaluate_components(self, altitude_m: float) -> np.ndarray:
        """Avalia as componentes da rajada.

        Parameters
        ----------
        altitude_m : float
            Altitude atual do veículo, em metros.

        Returns
        -------
        numpy.ndarray
            Vetor ``3 x 1`` com as componentes
            longitudinal, lateral e vertical.
        """

        distance_m = altitude_m - self.parameters.leading_edge_altitude_m

        return np.array(
            [
                [self.profile(distance_m, self.parameters.longitudinal)],
                [self.profile(distance_m, self.parameters.lateral)],
                [self.profile(distance_m, self.parameters.vertical)],
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_dict(cls, config: dict) -> "DiscreteGustWind":
        """Cria uma rajada a partir de um dicionário.

        Parameters
        ----------
        config : dict
            Configuração da rajada.

        Returns
        -------
        DiscreteGustWind
            Instância configurada da rajada.
        """

        parameters = DiscreteGustParameters.model_validate(config)

        return cls(parameters)

    def _gust_magnitude(self, component: GustComponentParameters) -> float:
        """Calcula a magnitude máxima da rajada segundo Leahy (2008).

        A magnitude máxima da rajada discreta é obtida a partir da
        distribuição condicional da velocidade de turbulência. O modelo
        utiliza a autocorrelação espacial do modelo Dryden para determinar
        a correlação entre o início da rajada e seu ponto de máxima
        magnitude.

        Para a componente longitudinal, a autocorrelação normalizada é:

        .. math::

            \\rho = \\exp\\left(-\\frac{d_m}{L}\\right)

        onde ``d_m`` é o ``half_width`` da rajada e ``L`` é a escala de
        comprimento da turbulência.

        A partir da distribuição condicional:

        .. math::

            \\sigma_c =
            \\sigma\\sqrt{1-\\rho^2}

        A probabilidade utilizada para determinar o quantil da distribuição
        normal é:

        .. math::

            P = 1 - \\frac{risk}{2}

        e a magnitude máxima é:

        .. math::

            V_m = \\Phi^{-1}(P)\\sigma_c

        Parameters
        ----------
        component : GustComponentParameters
            Parâmetros estatísticos da componente de turbulência.

        Returns
        -------
        float
            Magnitude máxima da rajada em m/s.

        Raises
        ------
        ValueError
            Se o risco não estiver estritamente entre 0 e 1.

        References
        ----------
        Leahy, F. B. (2008).
        ``Discrete Gust Model for Launch Vehicle Assessments``.
        13th Conference on Aviation, Range and Aerospace Meteorology,
        American Meteorological Society.
        """
        if not 0.0 < self.parameters.risk < 1.0:
            raise ValueError("risk must be strictly between 0 and 1.")

        sigma = component.standard_deviation
        length_scale = component.length_scale
        half_width = self.parameters.half_width

        rho = np.exp(-half_width / length_scale)

        conditional_standard_deviation = sigma * np.sqrt(1.0 - rho**2)

        probability = 1.0 - self.parameters.risk / 2.0

        magnitude = norm.ppf(probability) * conditional_standard_deviation

        return float(magnitude)
