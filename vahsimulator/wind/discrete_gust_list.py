from __future__ import annotations

import numpy as np

from ..launching_reference import LaunchReference
from .discrete_gust import DiscreteGustParameters, DiscreteGustWind


class DiscreteGustList:
    """Gerencia uma coleção de rajadas discretas.

    A classe é responsável por avaliar e somar todas as rajadas
    configuradas. A transformação das componentes longitudinal,
    lateral e vertical para NED é realizada aqui, enquanto a
    transformação NED para body é responsabilidade do
    ``WindFrameTransformer``.

    Parameters
    ----------
    gusts : list of DiscreteGustWind, optional
        Lista de modelos de rajada discreta.

    """

    def __init__(self, gusts: list[DiscreteGustWind] | None = None) -> None:
        """Inicializa a lista de rajadas.

        Parameters
        ----------
        gusts : list of DiscreteGustWind, optional
            Lista de rajadas. Caso ``None``, uma lista vazia é criada.
        """

        self.gusts = [] if gusts is None else list(gusts)

    @classmethod
    def from_parameters(
        cls, parameters: list[DiscreteGustParameters]
    ) -> "DiscreteGustList":
        """Cria uma lista a partir dos parâmetros Pydantic.

        Parameters
        ----------
        parameters : list of DiscreteGustParameters
            Parâmetros das rajadas.

        Returns
        -------
        DiscreteGustList
            Lista de rajadas configurada.
        """

        gusts = [DiscreteGustWind(parameter) for parameter in parameters]

        return cls(gusts)

    @classmethod
    def from_dict(cls, config: list[dict]) -> "DiscreteGustList":
        """Cria uma lista a partir de uma lista de dicionários.

        Parameters
        ----------
        config : list of dict
            Configuração das rajadas.

        Returns
        -------
        DiscreteGustList
            Lista de rajadas configurada.
        """

        parameters = [DiscreteGustParameters.model_validate(item) for item in config]

        return cls.from_parameters(parameters)

    def __len__(self) -> int:
        """Retorna o número de rajadas configuradas.

        Returns
        -------
        int
            Número de rajadas.
        """

        return len(self.gusts)

    def evaluate_components(self, altitude_m: float) -> np.ndarray:
        """Avalia e soma todas as componentes das rajadas.

        Parameters
        ----------
        altitude_m : float
            Altitude atual do veículo, em metros.

        Returns
        -------
        numpy.ndarray
            Vetor ``3 x 1`` contendo:

            ``[longitudinal, lateral, vertical]``.
        """

        total = np.zeros((3, 1), dtype=np.float64)

        for gust in self.gusts:
            total += gust.evaluate_components(altitude_m)

        return total

    def evaluate_ned(self, altitude_m: float, wind_direction_deg: float) -> np.ndarray:
        """Avalia e soma as rajadas no referencial NED.

        As componentes longitudinal e lateral da rajada são definidas
        em relação à direção do vento médio, conforme a convenção do
        modelo de rajada discreta de Leahy.

        A direção ``wind_direction_deg`` é interpretada como a direção
        meteorológica ``FROM``, isto é, a direção de onde o vento vem.
        Portanto, a componente longitudinal positiva aponta na direção
        ``TO`` do vento.

        Parameters
        ----------
        altitude_m : float
            Altitude atual do veículo, em metros.
        wind_direction_deg : float
            Direção meteorológica do vento médio, em graus.
            A convenção é:

            * ``0 deg``: vento vindo do Norte;
            * ``90 deg``: vento vindo do Leste;
            * ``180 deg``: vento vindo do Sul;
            * ``270 deg``: vento vindo do Oeste.

            A componente longitudinal positiva da rajada aponta no
            sentido oposto, ou seja, na direção para onde o vento sopra.

        Returns
        -------
        numpy.ndarray
            Vetor ``3 x 1`` contendo as componentes NED da rajada.

        Notes
        -----
        O modelo de Leahy define a componente longitudinal da turbulência
        paralela ao vento médio, enquanto as componentes lateral e vertical
        são definidas em relação a esse referencial.

        No referencial NED, a componente vertical positiva é definida para
        cima. Como o eixo ``Down`` é positivo em NED, essa direção é
        representada por ``[0, 0, -1]``.

        A velocidade do vento médio não é utilizada para definir a
        orientação dos eixos. Isso permite que, quando ``wind_speed = 0``,
        a orientação continue sendo determinada por ``wind_direction``.
        """
        components = self.evaluate_components(altitude_m)

        longitudinal = components[0, 0]
        lateral = components[1, 0]
        vertical = components[2, 0]

        # ``wind_direction_deg`` representa a direção FROM.
        # A componente longitudinal positiva aponta na direção TO.
        longitudinal_azimuth_rad = np.deg2rad((wind_direction_deg + 180.0) % 360.0)

        longitudinal_direction = np.array(
            [
                [np.cos(longitudinal_azimuth_rad)],
                [np.sin(longitudinal_azimuth_rad)],
                [0.0],
            ],
            dtype=np.float64,
        )

        # Componente lateral positiva: 90 graus à esquerda da direção
        # longitudinal no plano horizontal NED.
        lateral_direction = np.array(
            [
                [-np.sin(longitudinal_azimuth_rad)],
                [np.cos(longitudinal_azimuth_rad)],
                [0.0],
            ],
            dtype=np.float64,
        )

        # NED: Down é positivo. Portanto, positivo "para cima" é -Down.
        up_direction = np.array([[0.0], [0.0], [-1.0]], dtype=np.float64)

        return (
            longitudinal * longitudinal_direction
            + lateral * lateral_direction
            + vertical * up_direction
        )

    def active_gusts(self, altitude_m: float) -> list[DiscreteGustWind]:
        """Retorna as rajadas ativas na altitude informada.

        Parameters
        ----------
        altitude_m : float
            Altitude atual do veículo, em metros.

        Returns
        -------
        list of DiscreteGustWind
            Rajadas atualmente ativas.
        """

        return [gust for gust in self.gusts if gust.is_active(altitude_m)]
