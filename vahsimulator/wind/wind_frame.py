from __future__ import annotations

import numpy as np

from ..utils import ned_to_body


class WindFrameTransformer:
    """Realiza transformações entre referências do modelo de vento.

    A classe centraliza as transformações relacionadas ao referencial
    do vento. As componentes lineares ambientais são inicialmente
    expressas em NED e podem ser transformadas para o referencial body.

    Notes
    -----
    As componentes angulares do vento, ``p``, ``q`` e ``r``, são
    consideradas diretamente no referencial body e não passam pela
    transformação NED -> body.
    """

    @staticmethod
    def ned_to_body(
        wind_ned: np.ndarray, roll_rad: float, pitch_rad: float, yaw_rad: float
    ) -> np.ndarray:
        """Transforma uma velocidade de NED para body.

        Parameters
        ----------
        wind_ned : numpy.ndarray
            Vetor de velocidade NED ``3 x 1``.
        roll_rad : float
            Ângulo de roll em radianos.
        pitch_rad : float
            Ângulo de pitch em radianos.
        yaw_rad : float
            Ângulo de yaw em radianos.

        Returns
        -------
        numpy.ndarray
            Vetor de velocidade body ``3 x 1``.
        """
        lbn = ned_to_body(roll_rad, pitch_rad, yaw_rad)

        return np.matmul(lbn, wind_ned)

    @staticmethod
    def assemble_wind(
        wind_body: np.ndarray, angular_wind_body: np.ndarray | None = None
    ) -> np.ndarray:
        """Monta o vetor completo de vento no referencial body.

        Parameters
        ----------
        wind_body : numpy.ndarray
            Velocidade linear body ``3 x 1``.
        angular_wind_body : numpy.ndarray, optional
            Velocidade angular body ``3 x 1``.
            Caso não seja fornecida, será utilizado zero.

        Returns
        -------
        numpy.ndarray
            Vetor de vento ``6 x 1``.
        """
        if angular_wind_body is None:
            angular_wind_body = np.zeros((3, 1), dtype=np.float64)

        return np.vstack((wind_body, angular_wind_body))
