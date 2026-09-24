"""
Pytest suite for the VAHSimulator coordinate, attitude, quaternion,
utility, transfer-function and smoothing functions.

IMPORTANT
---------
Replace ``your_package.your_module`` below with the actual import path of
the module containing the functions under test.
"""

import numpy as np
import pytest
from types import SimpleNamespace


# ---------------------------------------------------------------------------
# Replace this import with the actual module under test.
# Example:
# from VAHSimulator.navigation import ...
# ---------------------------------------------------------------------------
import vahsimulator.utils as utils


RTOL = 1e-9
ATOL = 1e-9


def assert_dcm_is_rotation_matrix(dcm):
    """Check shape, orthogonality and determinant of a DCM."""
    assert dcm.shape == (3, 3)
    np.testing.assert_allclose(
        dcm @ dcm.T,
        np.eye(3),
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        np.linalg.det(dcm),
        1.0,
        rtol=RTOL,
        atol=ATOL,
    )


def assert_quaternion_is_unit(quaternion):
    """Check scalar-first quaternion shape and unit norm."""
    assert quaternion.shape == (4, 1)
    np.testing.assert_allclose(
        np.linalg.norm(quaternion),
        1.0,
        rtol=RTOL,
        atol=ATOL,
    )


def assert_same_rotation(q1, q2):
    """Check quaternion equivalence, accounting for q/-q ambiguity."""
    np.testing.assert_allclose(
        np.abs(np.dot(q1.ravel(), q2.ravel())),
        1.0,
        rtol=RTOL,
        atol=ATOL,
    )


# ===========================================================================
# ECI / ECEF / local-frame DCMs
# ===========================================================================

@pytest.mark.parametrize("angle_rad", [0.0, 0.3, -1.2, np.pi])
def test_eci_to_ecef_is_rotation(angle_rad):
    dcm = utils.eci_to_ecef(angle_rad)
    assert_dcm_is_rotation_matrix(dcm)


@pytest.mark.parametrize("angle_rad", [0.0, 0.3, -1.2, np.pi])
def test_ecef_to_eci_is_inverse(angle_rad):
    eci_to_ecef = utils.eci_to_ecef(angle_rad)
    ecef_to_eci = utils.ecef_to_eci(angle_rad)

    np.testing.assert_allclose(
        ecef_to_eci,
        eci_to_ecef.T,
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        ecef_to_eci @ eci_to_ecef,
        np.eye(3),
        rtol=RTOL,
        atol=ATOL,
    )


@pytest.mark.parametrize(
    "latitude_rad, longitude_rad, altitude_m",
    [
        (0.0, 0.0, 0.0),
        (np.deg2rad(30.0), np.deg2rad(45.0), 1000.0),
        (np.deg2rad(-55.0), np.deg2rad(120.0), 10000.0),
    ],
)
def test_ecef_vehicle_transformations_are_inverses(
    latitude_rad,
    longitude_rad,
    altitude_m,
):
    ecef_to_vehicle = utils.ecef_to_vehicle(
        latitude_rad,
        longitude_rad,
        altitude_m,
    )
    vehicle_to_ecef = utils.vehicle_to_ecef(
        latitude_rad,
        longitude_rad,
        altitude_m,
    )

    assert_dcm_is_rotation_matrix(ecef_to_vehicle)
    assert_dcm_is_rotation_matrix(vehicle_to_ecef)

    np.testing.assert_allclose(
        vehicle_to_ecef,
        ecef_to_vehicle.T,
        rtol=RTOL,
        atol=ATOL,
    )


@pytest.mark.parametrize(
    "latitude_rad, longitude_rad",
    [
        (0.0, 0.0),
        (0.4, 0.7),
        (-0.8, 2.1),
    ],
)
def test_enu_and_ned_matrices_are_rotation_matrices(
    latitude_rad,
    longitude_rad,
):
    ecef_to_enu = utils.ecef_to_enu_matrix(
        latitude_rad,
        longitude_rad,
    )
    enu_to_ecef = utils.enu_to_ecef_matrix(
        latitude_rad,
        longitude_rad,
    )
    ecef_to_ned = utils.ecef_to_ned_matrix(
        latitude_rad,
        longitude_rad,
    )
    ned_to_ecef = utils.ned_to_ecef_matrix(
        latitude_rad,
        longitude_rad,
    )

    for dcm in (
        ecef_to_enu,
        enu_to_ecef,
        ecef_to_ned,
        ned_to_ecef,
    ):
        assert_dcm_is_rotation_matrix(dcm)

    np.testing.assert_allclose(
        enu_to_ecef,
        ecef_to_enu.T,
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        ned_to_ecef,
        ecef_to_ned.T,
        rtol=RTOL,
        atol=ATOL,
    )


# ===========================================================================
# ECEF / ENU / NED position conversions
# ===========================================================================

def test_ecef_to_enu_and_enu_to_ecef_are_inverse():
    latitude_rad = np.deg2rad(35.0)
    longitude_rad = np.deg2rad(-20.0)
    altitude_m = 850.0

    east_m = 120.0
    north_m = -75.0
    up_m = 35.0

    ecef = utils.enu_to_ecef(
        east_m,
        north_m,
        up_m,
        latitude_rad,
        longitude_rad,
        altitude_m,
    )

    recovered_enu = utils.ecef_to_enu(
        *ecef,
        latitude_rad,
        longitude_rad,
        altitude_m,
    )

    np.testing.assert_allclose(
        recovered_enu,
        (east_m, north_m, up_m),
        rtol=RTOL,
        atol=ATOL,
    )


def test_ecef_to_ned_zero_displacement():
    latitude_rad = np.deg2rad(20.0)
    longitude_rad = np.deg2rad(30.0)
    altitude_m = 500.0

    ecef = utils.geodetic_to_ecef(
        latitude_rad,
        longitude_rad,
        altitude_m,
    )

    ned = utils.ecef_to_ned(
        *ecef,
        latitude_rad,
        longitude_rad,
        altitude_m,
    )

    np.testing.assert_allclose(
        ned,
        (0.0, 0.0, 0.0),
        atol=1e-6,
    )


def test_enu_and_ned_have_expected_axis_orientation():
    latitude_rad = 0.0
    longitude_rad = 0.0
    altitude_m = 0.0

    # At latitude = longitude = 0:
    # ECEF +Y is local East, ECEF +Z is local North, ECEF +X is Up.
    reference_ecef = np.array(
        utils.geodetic_to_ecef(
            latitude_rad,
            longitude_rad,
            altitude_m,
        )
    )

    east_ecef = reference_ecef + np.array([0.0, 10.0, 0.0])
    north_ecef = reference_ecef + np.array([0.0, 0.0, 10.0])
    up_ecef = reference_ecef + np.array([10.0, 0.0, 0.0])

    np.testing.assert_allclose(
        utils.ecef_to_enu(
            *east_ecef,
            latitude_rad,
            longitude_rad,
            altitude_m,
        ),
        (10.0, 0.0, 0.0),
        atol=1e-8,
    )
    np.testing.assert_allclose(
        utils.ecef_to_enu(
            *north_ecef,
            latitude_rad,
            longitude_rad,
            altitude_m,
        ),
        (0.0, 10.0, 0.0),
        atol=1e-8,
    )
    np.testing.assert_allclose(
        utils.ecef_to_enu(
            *up_ecef,
            latitude_rad,
            longitude_rad,
            altitude_m,
        ),
        (0.0, 0.0, 10.0),
        atol=1e-8,
    )


# ===========================================================================
# ECI / local position conversions
# ===========================================================================

def test_geodetic_ecef_roundtrip():
    latitude_rad = np.deg2rad(-35.0)
    longitude_rad = np.deg2rad(80.0)
    altitude_m = 2500.0

    ecef = utils.geodetic_to_ecef(
        latitude_rad,
        longitude_rad,
        altitude_m,
    )
    recovered = utils.ecef_to_geodetic(*ecef)

    np.testing.assert_allclose(
        recovered,
        (latitude_rad, longitude_rad, altitude_m),
        rtol=1e-8,
        atol=1e-7,
    )


def test_geodetic_eci_roundtrip():
    latitude_rad = np.deg2rad(15.0)
    longitude_rad = np.deg2rad(-70.0)
    altitude_m = 1500.0
    time_s = 1234.0

    eci = utils.geodetic_to_eci(
        latitude_rad,
        longitude_rad,
        altitude_m,
        time_s,
    )
    recovered = utils.eci_to_geodetic(
        *eci,
        time_s,
    )

    np.testing.assert_allclose(
        recovered,
        (latitude_rad, longitude_rad, altitude_m),
        rtol=1e-8,
        atol=1e-6,
    )


def test_enu_eci_roundtrip():
    reference_latitude_rad = np.deg2rad(25.0)
    reference_longitude_rad = np.deg2rad(-40.0)
    reference_altitude_m = 750.0
    time_s = 500.0

    enu = np.array([100.0, -50.0, 20.0])

    eci = utils.enu_to_eci(
        enu[0],
        enu[1],
        enu[2],
        reference_latitude_rad,
        reference_longitude_rad,
        reference_altitude_m,
        time_s,
    )

    recovered = utils.eci_to_enu(
        eci[0, 0],
        eci[1, 0],
        eci[2, 0],
        reference_latitude_rad,
        reference_longitude_rad,
        reference_altitude_m,
        time_s,
    )

    np.testing.assert_allclose(
        recovered.ravel(),
        enu,
        rtol=1e-8,
        atol=1e-5,
    )


def test_eci_to_ned_zero_reference_displacement():
    latitude_rad = np.deg2rad(20.0)
    longitude_rad = np.deg2rad(40.0)
    altitude_m = 1000.0
    time_s = 200.0

    eci = utils.geodetic_to_eci(
        latitude_rad,
        longitude_rad,
        altitude_m,
        time_s,
    )

    ned = utils.eci_to_ned(
        *eci,
        latitude_rad,
        longitude_rad,
        altitude_m,
        time_s,
    )

    np.testing.assert_allclose(
        ned,
        np.zeros(3),
        atol=1e-5,
    )


def test_eci_to_ned_velocity_matches_matrix_chain():
    velocity_eci = np.array([120.0, -40.0, 75.0])
    latitude_rad = np.deg2rad(30.0)
    longitude_rad = np.deg2rad(-60.0)
    time_s = 100.0

    expected = (
        utils.ecef_to_ned_matrix(latitude_rad, longitude_rad)
        @ utils.eci_to_ecef(utils.earth_rate__rad_s * time_s)
        @ velocity_eci.reshape(3, 1)
    ).ravel()

    actual = utils.eci_to_ned_velocity(
        *velocity_eci,
        latitude_rad,
        longitude_rad,
        time_s,
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=RTOL,
        atol=ATOL,
    )


# ===========================================================================
# Navigation / downrange
# ===========================================================================

def test_eci_to_nav_and_nav_to_eci_are_inverses():
    latitude_rad = 0.4
    longitude_rad = -0.7
    azimuth_rad = 1.1

    eci_to_nav = utils.eci_to_nav(
        latitude_rad,
        longitude_rad,
        azimuth_rad,
    )
    nav_to_eci = utils.nav_to_eci(
        latitude_rad,
        longitude_rad,
        azimuth_rad,
    )

    assert_dcm_is_rotation_matrix(eci_to_nav)
    assert_dcm_is_rotation_matrix(nav_to_eci)

    np.testing.assert_allclose(
        nav_to_eci,
        eci_to_nav.T,
        rtol=RTOL,
        atol=ATOL,
    )


def test_compute_downrange_crossrange_at_launch_reference():
    reference = SimpleNamespace(
        azimuth=np.deg2rad(35.0),
        lat=np.deg2rad(-20.0),
        lon=np.deg2rad(50.0),
        alt=250.0,
    )

    result = utils.compute_downrange_crossrange(
        np.rad2deg(reference.lat),
        np.rad2deg(reference.lon),
        reference.alt,
        reference,
    )

    np.testing.assert_allclose(
        result,
        (0.0, 0.0, 0.0),
        atol=1e-5,
    )


# ===========================================================================
# Body / ECI Euler rotations
# ===========================================================================

@pytest.mark.parametrize(
    "roll_rad,pitch_rad,yaw_rad",
    [
        (0.0, 0.0, 0.0),
        (0.2, -0.3, 0.7),
        (-1.0, 0.4, -2.0),
    ],
)
def test_body_eci_euler_inverse_pair(
    roll_rad,
    pitch_rad,
    yaw_rad,
):
    body_to_eci = utils.body_to_eci_euler(
        roll_rad,
        pitch_rad,
        yaw_rad,
    )
    eci_to_body = utils.eci_to_body_euler(
        roll_rad,
        pitch_rad,
        yaw_rad,
    )

    assert_dcm_is_rotation_matrix(body_to_eci)
    assert_dcm_is_rotation_matrix(eci_to_body)

    np.testing.assert_allclose(
        eci_to_body,
        body_to_eci.T,
        rtol=RTOL,
        atol=ATOL,
    )


@pytest.mark.parametrize(
    "roll_rad,pitch_rad,yaw_rad",
    [
        (0.2, -0.3, 0.7),
        (-0.8, 0.5, -1.4),
        (1.0, 0.2, 2.0),
    ],
)
def test_euler_quaternion_roundtrip(
    roll_rad,
    pitch_rad,
    yaw_rad,
):
    quaternion = utils.euler_to_quaternion(
        roll_rad,
        pitch_rad,
        yaw_rad,
    )

    assert_quaternion_is_unit(quaternion)

    recovered = utils.quaternion_to_euler_321(quaternion)

    np.testing.assert_allclose(
        recovered,
        (roll_rad, pitch_rad, yaw_rad),
        rtol=RTOL,
        atol=RTOL,
    )


def test_body_to_eci_quaternion_matches_euler_dcm():
    roll_rad = 0.3
    pitch_rad = -0.4
    yaw_rad = 0.8

    quaternion = utils.euler_to_quaternion(
        roll_rad,
        pitch_rad,
        yaw_rad,
    )
    dcm_from_quaternion = utils.body_to_eci_quaternion(quaternion)
    dcm_from_euler = utils.body_to_eci_euler(
        roll_rad,
        pitch_rad,
        yaw_rad,
    )

    np.testing.assert_allclose(
        dcm_from_quaternion,
        dcm_from_euler,
        rtol=RTOL,
        atol=ATOL,
    )


def test_eci_to_body_quaternion_is_transpose():
    quaternion = utils.euler_to_quaternion(0.2, -0.5, 1.0)

    expected = utils.body_to_eci_quaternion(quaternion).T
    actual = utils.eci_to_body_quaternion(quaternion)

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=RTOL,
        atol=ATOL,
    )


# ===========================================================================
# Quaternion algebra
# ===========================================================================

@pytest.mark.parametrize(
    "angle_rad,axis",
    [
        (0.0, 0),
        (np.pi / 2.0, 0),
        (-np.pi / 3.0, 1),
        (np.pi, 2),
    ],
)
def test_axis_angle_quaternion(angle_rad, axis):
    quaternion = utils.axis_angle_quaternion(angle_rad, axis)

    assert_quaternion_is_unit(quaternion)

    expected = np.zeros((4, 1))
    expected[0, 0] = np.cos(angle_rad / 2.0)
    expected[axis + 1, 0] = np.sin(angle_rad / 2.0)

    np.testing.assert_allclose(
        quaternion,
        expected,
        rtol=RTOL,
        atol=ATOL,
    )


def test_quaternion_normalize():
    quaternion = np.array(
        [[2.0], [1.0], [-2.0], [3.0]],
        dtype=float,
    )

    normalized = utils.quaternion_normalize(quaternion)

    np.testing.assert_allclose(
        normalized,
        quaternion / np.linalg.norm(quaternion),
        rtol=RTOL,
        atol=ATOL,
    )


def test_quaternion_conjugate():
    quaternion = np.array(
        [[0.5], [0.1], [-0.2], [0.3]],
        dtype=float,
    )

    expected = np.array(
        [[0.5], [-0.1], [0.2], [-0.3]],
        dtype=float,
    )

    np.testing.assert_allclose(
        utils.quaternion_conjugate(quaternion),
        expected,
        rtol=RTOL,
        atol=ATOL,
    )


def test_quaternion_multiply_identity():
    quaternion = utils.euler_to_quaternion(
        0.2,
        -0.3,
        0.7,
    )
    identity = np.array(
        [[1.0], [0.0], [0.0], [0.0]],
        dtype=float,
    )

    assert_same_rotation(
        utils.quaternion_multiply(quaternion, identity),
        quaternion,
    )
    assert_same_rotation(
        utils.quaternion_multiply(identity, quaternion),
        quaternion,
    )


def test_dcm_to_quaternion_roundtrip():
    dcm = utils.body_to_eci_euler(
        0.4,
        -0.2,
        1.1,
    )

    quaternion = utils.dcm_to_quaternion(dcm)
    recovered_dcm = utils.body_to_eci_quaternion(quaternion)

    assert_quaternion_is_unit(quaternion)
    np.testing.assert_allclose(
        recovered_dcm,
        dcm,
        rtol=RTOL,
        atol=ATOL,
    )


# ===========================================================================
# ECI / ECEF / NED quaternions
# ===========================================================================

def test_eci_ecef_quaternion_matches_dcm():
    time_s = 1234.0

    q_eci_to_ecef = utils.eci_to_ecef_quaternion(time_s)
    q_ecef_to_eci = utils.ecef_to_eci_quaternion(time_s)

    assert_quaternion_is_unit(q_eci_to_ecef)
    assert_quaternion_is_unit(q_ecef_to_eci)

    assert_same_rotation(
        q_ecef_to_eci,
        utils.quaternion_conjugate(q_eci_to_ecef),
    )

    np.testing.assert_allclose(
        utils.body_to_eci_quaternion(q_eci_to_ecef),
        utils.eci_to_ecef(utils.earth_rate__rad_s * time_s),
        rtol=RTOL,
        atol=ATOL,
    )


def test_ecef_ned_quaternion_matches_dcm():
    latitude_rad = 0.5
    longitude_rad = -0.8

    q_ecef_to_ned = utils.ecef_to_ned_quaternion(
        latitude_rad,
        longitude_rad,
    )
    q_ned_to_ecef = utils.ned_to_ecef_quaternion(
        latitude_rad,
        longitude_rad,
    )

    assert_same_rotation(
        q_ecef_to_ned,
        utils.dcm_to_quaternion(
            utils.ecef_to_ned_matrix(
                latitude_rad,
                longitude_rad,
            )
        ),
    )
    assert_same_rotation(
        q_ned_to_ecef,
        utils.quaternion_conjugate(q_ecef_to_ned),
    )


def test_eci_ned_quaternion_inverse_pair():
    latitude_rad = 0.4
    longitude_rad = -0.9
    time_s = 400.0

    q_eci_to_ned = utils.eci_to_ned_quaternion(
        latitude_rad,
        longitude_rad,
        time_s,
    )
    q_ned_to_eci = utils.ned_to_eci_quaternion(
        latitude_rad,
        longitude_rad,
        time_s,
    )

    assert_same_rotation(
        q_ned_to_eci,
        utils.quaternion_conjugate(q_eci_to_ned),
    )


def test_body_ned_quaternion_inverse_pair():
    body_quaternion = utils.euler_to_quaternion(
        0.2,
        -0.3,
        0.8,
    )

    latitude_rad = 0.4
    longitude_rad = -0.5
    time_s = 200.0

    q_ned_from_body = utils.body_to_ned_quaternion(
        body_quaternion,
        latitude_rad,
        longitude_rad,
        time_s,
    )
    q_body_from_ned = utils.ned_to_body_quaternion(
        body_quaternion,
        latitude_rad,
        longitude_rad,
        time_s,
    )

    assert_same_rotation(
        q_body_from_ned,
        utils.quaternion_conjugate(q_ned_from_body),
    )


# ===========================================================================
# Attitude extraction
# ===========================================================================

def test_eci_to_enu_attitude_returns_unit_quaternion():
    result = utils.eci_to_enu_attitude(
        0.2,
        -0.3,
        0.7,
        np.deg2rad(20.0),
        np.deg2rad(30.0),
        100.0,
    )

    quaternion, roll, pitch, yaw = result

    assert_quaternion_is_unit(quaternion)
    assert np.isfinite([roll, pitch, yaw]).all()


def test_eci_to_ned_attitude_returns_unit_quaternion():
    body_quaternion = utils.euler_to_quaternion(
        0.1,
        -0.2,
        0.4,
    )

    result = utils.eci_to_ned_attitude(
        body_quaternion,
        np.deg2rad(25.0),
        np.deg2rad(-30.0),
        150.0,
    )

    quaternion = np.array(result).reshape(4, 1)
    assert_quaternion_is_unit(quaternion)


def test_continuous_euler_angles_normal_case():
    expected_angles = np.array([0.2, -0.3, 0.7])

    quaternion = utils.euler_to_quaternion(*expected_angles)

    actual, protected = utils.continuous_euler_angles(
        quaternion,
        np.zeros(3),
        False,
    )

    np.testing.assert_allclose(
        actual,
        expected_angles,
        rtol=RTOL,
        atol=RTOL,
    )
    assert protected is False


def test_continuous_euler_angles_enables_gimbal_lock_protection():
    roll_rad = 0.7
    pitch_rad = np.deg2rad(89.0)
    yaw_rad = -0.4

    quaternion = utils.euler_to_quaternion(
        roll_rad,
        pitch_rad,
        yaw_rad,
    )

    previous = np.array([0.2, pitch_rad, 0.1])

    angles, protected = utils.continuous_euler_angles(
        quaternion,
        previous,
        False,
    )

    assert protected is True
    np.testing.assert_allclose(
        angles[0],
        previous[0],
        rtol=RTOL,
        atol=RTOL,
    )
    np.testing.assert_allclose(
        angles[1],
        pitch_rad,
        atol=1e-6,
    )


def test_continuous_euler_angles_disables_gimbal_lock_protection():
    quaternion = utils.euler_to_quaternion(
        0.4,
        np.deg2rad(60.0),
        -0.2,
    )

    angles, protected = utils.continuous_euler_angles(
        quaternion,
        np.zeros(3),
        True,
    )

    assert protected is False
    assert np.all(np.isfinite(angles))


def test_compute_euler_angles_ned():
    latitude_rad = np.deg2rad(25.0)
    longitude_rad = np.deg2rad(-35.0)
    altitude_m = 1000.0
    time_s = 250.0

    position_eci = utils.geodetic_to_eci(
        latitude_rad,
        longitude_rad,
        altitude_m,
        time_s,
    )

    body_quaternion = utils.euler_to_quaternion(
        0.1,
        -0.2,
        0.3,
    )

    state_vector = np.zeros((7, 1))
    state_vector[0:3, 0] = position_eci
    state_vector[3:7, 0] = body_quaternion[:, 0]

    angles, protected = utils.compute_euler_angles_ned(
        state_vector,
        time_s,
        np.zeros(3),
        False,
    )

    assert angles.shape == (3,)
    assert np.all(np.isfinite(angles))
    assert isinstance(protected, (bool, np.bool_))


# ===========================================================================
# Geodetic / geocentric latitude
# ===========================================================================

@pytest.mark.parametrize(
    "latitude_rad,altitude_m",
    [
        (0.0, 0.0),
        (0.3, 1000.0),
        (-0.8, 5000.0),
    ],
)
def test_geodetic_geocentric_latitude_roundtrip(
    latitude_rad,
    altitude_m,
):
    x_ecef_m, y_ecef_m, z_ecef_m = utils.geodetic_to_ecef(
        latitude_rad,
        0.4,
        altitude_m,
    )

    geocentric_latitude_rad = np.arctan2(
        z_ecef_m,
        np.sqrt(x_ecef_m**2 + y_ecef_m**2),
    )

    recovered = utils.geocentric_to_geodetic_latitude(
        geocentric_latitude_rad,
        x_ecef_m,
        y_ecef_m,
        z_ecef_m,
    )

    np.testing.assert_allclose(
        recovered,
        latitude_rad,
        rtol=1e-6,
        atol=1e-6,
    )


# ===========================================================================
# Cross-product matrices
# ===========================================================================

def test_skew_matrix_represents_cross_product():
    vector = np.array(
        [[1.0], [-2.0], [3.0]],
    )
    other = np.array(
        [[4.0], [5.0], [-6.0]],
    )

    skew = utils.skew_matrix(vector)

    np.testing.assert_allclose(
        skew @ other,
        np.cross(vector.ravel(), other.ravel()).reshape(3, 1),
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        skew + skew.T,
        np.zeros((3, 3)),
        rtol=RTOL,
        atol=ATOL,
    )


def test_skew_matrix_angular_velocity_is_skew_symmetric():
    angular_velocity = np.array(
        [[0.1], [-0.2], [0.3]],
    )

    omega = utils.skew_matrix_angular_velocity(
        angular_velocity,
    )

    assert omega.shape == (4, 4)
    np.testing.assert_allclose(
        omega + omega.T,
        np.zeros((4, 4)),
        rtol=RTOL,
        atol=ATOL,
    )


# ===========================================================================
# Utility functions
# ===========================================================================

def test_integrate_trapezoidal_rule():
    result = utils.integrate(
        new_value=6.0,
        old_value=2.0,
        state=10.0,
        step_s=0.5,
    )

    np.testing.assert_allclose(
        result,
        12.0,
        rtol=RTOL,
        atol=ATOL,
    )


@pytest.mark.parametrize(
    "angle_rad,expected",
    [
        (0.0, 0.0),
        (np.pi, -np.pi),
        (-np.pi, -np.pi),
        (3.0 * np.pi, -np.pi),
        (-3.0 * np.pi, -np.pi),
        (0.5 * np.pi, 0.5 * np.pi),
    ],
)
def test_wrap_to_pi(angle_rad, expected):
    np.testing.assert_allclose(
        utils.wrap_to_pi(angle_rad),
        expected,
        rtol=RTOL,
        atol=ATOL,
    )


def test_initialize_interpolator1d(tmp_path):
    data_file = tmp_path / "data.csv"
    data_file.write_text(
        "0;0\n"
        "1;10\n"
        "2;20\n",
        encoding="utf-8",
    )

    interpolator = utils.initialize_interpolator1d(data_file)

    np.testing.assert_allclose(
        interpolator(0.5),
        5.0,
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        interpolator(3.0),
        30.0,
        rtol=RTOL,
        atol=ATOL,
    )


def test_initialize_interpolator1d_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        utils.initialize_interpolator1d(
            tmp_path / "does_not_exist.csv",
        )


def test_initialize_interpolator1d_requires_two_columns(tmp_path):
    data_file = tmp_path / "invalid.csv"
    data_file.write_text(
        "0\n"
        "1\n",
        encoding="utf-8",
    )

    with pytest.raises(IndexError):
        utils.initialize_interpolator1d(data_file)


def test_clean_pycache_without_flag(tmp_path):
    pycache = tmp_path / "a" / "__pycache__"
    pycache.mkdir(parents=True)
    marker = pycache / "marker.pyc"
    marker.write_bytes(b"test")

    utils.clean_pycache(tmp_path, flag=False)

    assert pycache.exists()
    assert marker.exists()


def test_clean_pycache_with_flag(tmp_path):
    pycache_1 = tmp_path / "a" / "__pycache__"
    pycache_2 = tmp_path / "b" / "c" / "__pycache__"

    pycache_1.mkdir(parents=True)
    pycache_2.mkdir(parents=True)

    utils.clean_pycache(tmp_path, flag=True)

    assert not pycache_1.exists()
    assert not pycache_2.exists()


# ===========================================================================
# TransferFunction
# ===========================================================================

def test_transfer_function_initialization_and_state_space():
    numerator = np.array([[1.0]])
    denominator = np.array([[1.0, 2.0]])

    tf = utils.TransferFunction(
        numerator,
        denominator,
        sampling_time_s=0.1,
    )

    assert tf.m == 1
    assert tf.n == 2
    assert tf.state.shape == (1, 1)

    np.testing.assert_allclose(
        tf.A,
        [[-2.0]],
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        tf.B,
        [[1.0]],
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        tf.C,
        [[1.0]],
        rtol=RTOL,
        atol=ATOL,
    )
    assert tf.D == 0.0


def test_transfer_function_normalizes_denominator():
    numerator = np.array([[2.0]])
    denominator = np.array([[2.0, 4.0]])

    tf = utils.TransferFunction(
        numerator,
        denominator,
        sampling_time_s=0.1,
    )

    np.testing.assert_allclose(
        tf.denominator,
        [[1.0, 2.0]],
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        tf.numerator,
        [[1.0]],
        rtol=RTOL,
        atol=ATOL,
    )


def test_transfer_function_direct_feedthrough():
    numerator = np.array([[2.0, 3.0]])
    denominator = np.array([[1.0, 4.0]])

    tf = utils.TransferFunction(
        numerator,
        denominator,
        sampling_time_s=0.1,
    )

    assert tf.D == 2.0

    np.testing.assert_allclose(
        tf.C,
        [[-5.0]],
        rtol=RTOL,
        atol=ATOL,
    )


def test_transfer_function_state_derivative():
    tf = utils.TransferFunction(
        np.array([[1.0]]),
        np.array([[1.0, 2.0]]),
        sampling_time_s=0.1,
    )

    state = np.array([[3.0]])

    np.testing.assert_allclose(
        tf._state_derivative(state, 4.0),
        [[-2.0]],
        rtol=RTOL,
        atol=ATOL,
    )


def test_transfer_function_output():
    tf = utils.TransferFunction(
        np.array([[2.0, 3.0]]),
        np.array([[1.0, 4.0]]),
        sampling_time_s=0.1,
    )
    tf.state[:] = 5.0

    # y = Cx + Du = (-5)*5 + 2*3
    np.testing.assert_allclose(
        tf._output(3.0),
        -19.0,
        rtol=RTOL,
        atol=ATOL,
    )


def test_transfer_function_rk4_first_order_system():
    # H(s) = 1 / (s + 1)
    # For a unit step and x(0)=0:
    # x(T) = 1 - exp(-T)
    sampling_time_s = 0.1

    tf = utils.TransferFunction(
        np.array([[1.0]]),
        np.array([[1.0, 1.0]]),
        sampling_time_s=sampling_time_s,
    )

    tf._integrate(1.0)

    expected = 1.0 - np.exp(-sampling_time_s)

    np.testing.assert_allclose(
        tf.state[0, 0],
        expected,
        rtol=1e-6,
        atol=1e-8,
    )


def test_transfer_function_set_Ts():
    tf = utils.TransferFunction(
        np.array([[1.0]]),
        np.array([[1.0, 1.0]]),
        sampling_time_s=0.1,
    )

    tf.set_Ts(0.02)

    assert tf.sampling_time_s == 0.02


def test_transfer_function_step():
    tf = utils.TransferFunction(
        np.array([[1.0]]),
        np.array([[1.0, 1.0]]),
        sampling_time_s=0.1,
    )

    output = tf.step(
        input_value=1.0,
        numerator=np.array([[1.0]]),
        denominator=np.array([[1.0, 1.0]]),
    )

    expected = 1.0 - np.exp(-0.1)

    np.testing.assert_allclose(
        output,
        expected,
        rtol=1e-6,
        atol=1e-8,
    )


# ===========================================================================
# Smoothing / Kalman filter
# ===========================================================================

def test_smoothing_initialization():
    filter_ = utils.Smoothing(time_step_s=0.2)

    assert filter_.x is None
    assert filter_.initialized is False

    np.testing.assert_allclose(
        filter_.F,
        [[1.0, 0.2], [0.0, 1.0]],
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        filter_.H,
        [[1.0, 0.0]],
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        filter_.P,
        np.eye(2),
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        filter_.Q,
        1e-3 * np.eye(2),
        rtol=RTOL,
        atol=ATOL,
    )
    assert filter_.R == 1.0


def test_smoothing_set_initial_state():
    filter_ = utils.Smoothing(time_step_s=0.1)

    filter_.set_initial_state(5.0)

    np.testing.assert_allclose(
        filter_.x,
        [[5.0], [0.0]],
        rtol=RTOL,
        atol=ATOL,
    )
    assert filter_.initialized is True


def test_smoothing_first_measurement():
    filter_ = utils.Smoothing(time_step_s=0.1)
    filter_.set_initial_state(0.0)

    result = filter_.smooth(10.0)

    # With x^- = 0 and P^- = FPF' + Q, the update is deterministic.
    x_prediction = np.array([[0.0], [0.0]])
    p_prediction = (
        filter_.F @ np.eye(2) @ filter_.F.T
        + 1e-3 * np.eye(2)
    )
    innovation = np.array([[10.0]])
    innovation_covariance = (
        filter_.H @ p_prediction @ filter_.H.T + 1.0
    )
    kalman_gain = (
        p_prediction
        @ filter_.H.T
        @ np.linalg.inv(innovation_covariance)
    )
    expected_state = (
        x_prediction + kalman_gain @ innovation
    )

    np.testing.assert_allclose(
        result,
        expected_state[0, 0],
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        filter_.x,
        expected_state,
        rtol=RTOL,
        atol=ATOL,
    )


