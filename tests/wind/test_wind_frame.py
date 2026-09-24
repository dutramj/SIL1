from __future__ import annotations

import numpy as np
import pytest

from vahsimulator.wind.wind_frame import WindFrameTransformer


class TestWindFrameTransformer:
    """Testes do transformador de referencial do vento."""

    @pytest.fixture
    def wind_ned(self) -> np.ndarray:
        """Retorna um vetor de vento NED de teste.

        Returns
        -------
        numpy.ndarray
            Vetor de vento NED ``3 x 1``.
        """
        return np.array([[10.0], [5.0], [-2.0]], dtype=np.float64)

    def test_ned_to_body_returns_expected_shape(self, wind_ned: np.ndarray) -> None:
        """Verifica a dimensão da transformação NED → body."""
        result = WindFrameTransformer.ned_to_body(
            wind_ned=wind_ned, roll_rad=0.0, pitch_rad=0.0, yaw_rad=0.0
        )

        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 1)

    def test_ned_to_body_zero_attitude_is_identity(self, wind_ned: np.ndarray) -> None:
        """Verifica a transformação para atitude nula."""
        result = WindFrameTransformer.ned_to_body(
            wind_ned=wind_ned, roll_rad=0.0, pitch_rad=0.0, yaw_rad=0.0
        )

        np.testing.assert_allclose(result, wind_ned, rtol=0.0, atol=1e-12)

    def test_ned_to_body_yaw_90_degrees(self) -> None:
        """Verifica a transformação para yaw de 90 graus."""
        wind_ned = np.array([[10.0], [0.0], [0.0]], dtype=np.float64)

        result = WindFrameTransformer.ned_to_body(
            wind_ned=wind_ned, roll_rad=0.0, pitch_rad=0.0, yaw_rad=np.deg2rad(90.0)
        )

        expected = np.array([[0.0], [-10.0], [0.0]], dtype=np.float64)

        np.testing.assert_allclose(result, expected, rtol=0.0, atol=1e-12)

    def test_ned_to_body_roll_90_degrees(self) -> None:
        """Verifica a transformação para roll de 90 graus."""
        wind_ned = np.array([[0.0], [10.0], [0.0]], dtype=np.float64)

        result = WindFrameTransformer.ned_to_body(
            wind_ned=wind_ned, roll_rad=np.deg2rad(90.0), pitch_rad=0.0, yaw_rad=0.0
        )

        expected = np.array([[0.0], [0.0], [-10.0]], dtype=np.float64)

        np.testing.assert_allclose(result, expected, rtol=0.0, atol=1e-12)

    def test_ned_to_body_pitch_90_degrees(self) -> None:
        """Verifica a transformação para pitch de 90 graus."""
        wind_ned = np.array([[10.0], [0.0], [0.0]], dtype=np.float64)

        result = WindFrameTransformer.ned_to_body(
            wind_ned=wind_ned, roll_rad=0.0, pitch_rad=np.deg2rad(90.0), yaw_rad=0.0
        )

        expected = np.array([[0.0], [0.0], [10.0]], dtype=np.float64)

        np.testing.assert_allclose(result, expected, rtol=0.0, atol=1e-12)

    def test_ned_to_body_preserves_vector_norm(self, wind_ned: np.ndarray) -> None:
        """Verifica que a rotação preserva a norma do vetor."""
        result = WindFrameTransformer.ned_to_body(
            wind_ned=wind_ned,
            roll_rad=np.deg2rad(20.0),
            pitch_rad=np.deg2rad(-15.0),
            yaw_rad=np.deg2rad(35.0),
        )

        assert np.linalg.norm(result) == pytest.approx(
            np.linalg.norm(wind_ned), rel=1e-12, abs=1e-12
        )

    def test_assemble_wind_without_angular_component(self) -> None:
        """Verifica a montagem sem componente angular."""
        wind_body = np.array([[10.0], [5.0], [-2.0]], dtype=np.float64)

        result = WindFrameTransformer.assemble_wind(wind_body=wind_body)

        expected = np.array(
            [[10.0], [5.0], [-2.0], [0.0], [0.0], [0.0]], dtype=np.float64
        )

        assert result.shape == (6, 1)

        np.testing.assert_allclose(result, expected, rtol=0.0, atol=1e-12)

    def test_assemble_wind_with_angular_component(self) -> None:
        """Verifica a montagem com componente angular."""
        wind_body = np.array([[10.0], [5.0], [-2.0]], dtype=np.float64)

        angular_wind_body = np.array([[0.1], [0.2], [0.3]], dtype=np.float64)

        result = WindFrameTransformer.assemble_wind(
            wind_body=wind_body, angular_wind_body=angular_wind_body
        )

        expected = np.array(
            [[10.0], [5.0], [-2.0], [0.1], [0.2], [0.3]], dtype=np.float64
        )

        assert result.shape == (6, 1)

        np.testing.assert_allclose(result, expected, rtol=0.0, atol=1e-12)

    def test_assemble_wind_preserves_angular_component(self) -> None:
        """Verifica que a componente angular não é transformada."""
        wind_body = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)

        angular_wind_body = np.array([[4.0], [5.0], [6.0]], dtype=np.float64)

        result = WindFrameTransformer.assemble_wind(
            wind_body=wind_body, angular_wind_body=angular_wind_body
        )

        np.testing.assert_allclose(result[3:6], angular_wind_body, rtol=0.0, atol=1e-12)

    def test_assemble_wind_returns_new_array(self) -> None:
        """Verifica que o resultado é um novo array."""
        wind_body = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)

        result = WindFrameTransformer.assemble_wind(wind_body=wind_body)

        result[0, 0] = 999.0

        assert wind_body[0, 0] == pytest.approx(1.0)
