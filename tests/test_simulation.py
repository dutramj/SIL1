# tests/test_simulator.py

from pathlib import Path

import numpy as np
import pytest

import vahsimulator as vah


MISSION_FILE = "rato_mission_3dof_skip_glide.yaml"


@pytest.fixture
def simulation_data():
    mission_file = (
        Path(__file__).resolve().parents[1]
        / "mission"
        / MISSION_FILE
    )

    mission_config = vah.load_config(mission_file=mission_file)

    simulator = vah.Simulator(**mission_config["simulator"])
    mission_plan = vah.MissionPlan(**mission_config["mission_plan"])

    return simulator.run(mission_plan)


def test_standard_flight_final_state(simulation_data):
    """Verifica o estado final da simulação."""

    final = simulation_data.iloc[-1]

    expected = {
        "TIME__s": 300.0,
        "PHASE": 4,
        "STATE_BODY_VEL_U__m_s": 1000.0,
        "STATE_BODY_VEL_V__m_s": 0.0,
        "STATE_BODY_VEL_W__m_s": 0.0,
        "ALTITUDE__m": 10000.0,
        "LATITUDE__deg": 0.0,
        "LONGITUDE__deg": 0.0,
        "DOWNRANGE__m": 10000.0,
        "CROSSRANGE__m": 0.0,
    }

    for column, expected_value in expected.items():
        assert final[column] == pytest.approx(
            expected_value,
            rel=1e-7,
            abs=1e-7,
        ), f"{column} terminou com valor {final[column]}, mas "
        f"não coincidiu com valor esperado {expected_value}."
