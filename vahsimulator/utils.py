# 3rd party libraries
import numpy as np
from numpy.linalg import inv, norm
from numpy import cos, sin, atan2, asin, sqrt, copysign, pi
import pandas as pd
from scipy.interpolate import interp1d
import numba
import shutil
from pathlib import Path

# VAHSimulator library
from .parameters import earth_rate__rad_s, a, e, f
from . import performance_decorator
from .launching_reference import LaunchReference


@numba.njit(cache=True)
def eci_to_ecef(greenwich_sidereal_time_rad):
    """
    Compute the rotation matrix from ECI to ECEF coordinates.

    The transformation accounts for the Earth's rotation about its
    spin axis through the Greenwich Sidereal Time (GST). The resulting
    matrix transforms a vector expressed in the Earth-Centered Inertial
    (ECI) frame into the Earth-Centered Earth-Fixed (ECEF) frame.

    Parameters
    ----------
    greenwich_sidereal_time_rad : float
        Greenwich Sidereal Time (GST), in radians.

    Returns
    -------
    numpy.ndarray
        A 3x3 direction cosine matrix (DCM), dimensionless, that
        transforms a vector from the ECI frame to the ECEF frame.

        If ``vector_eci`` is expressed in ECI coordinates, the
        corresponding ECEF vector is obtained as::

            vector_ecef = dcm_eci_to_ecef @ vector_eci

    Notes
    -----
    The transformation is a rotation about the Earth's Z-axis:

    .. math::

        \\mathbf{C}_{ECEF \\leftarrow ECI} =
        \\begin{bmatrix}
        \\cos(\\theta) & \\sin(\\theta) & 0 \\\\
        -\\sin(\\theta) & \\cos(\\theta) & 0 \\\\
        0 & 0 & 1
        \\end{bmatrix}

    where :math:`\\theta` is the Greenwich Sidereal Time.

    The matrix is dimensionless because it represents a coordinate
    rotation.

    See Also
    --------
    ecef_to_eci : Compute the inverse transformation from ECEF to ECI.

    References
    ----------
    .. _vallado:
    
    Vallado, D. A., *Fundamentals of Astrodynamics and Applications*.
    """
    cosine_gst = np.cos(greenwich_sidereal_time_rad)
    sine_gst = np.sin(greenwich_sidereal_time_rad)

    dcm_eci_to_ecef = np.array(
        [
            [cosine_gst, sine_gst, 0.0],
            [-sine_gst, cosine_gst, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    return dcm_eci_to_ecef


@numba.njit(cache=True)
def ecef_to_eci(greenwich_sidereal_time_rad):
    """
    Compute the rotation matrix from ECEF to ECI coordinates.

    The transformation is the inverse of the ECI-to-ECEF rotation.
    Since the transformation matrix is a proper orthogonal rotation
    matrix, its inverse is equal to its transpose.

    Parameters
    ----------
    greenwich_sidereal_time_rad : float
        Greenwich Sidereal Time (GST), in radians.

    Returns
    -------
    numpy.ndarray
        A 3x3 direction cosine matrix (DCM), dimensionless, that
        transforms a vector from the ECEF frame to the ECI frame.

        If ``vector_ecef`` is expressed in ECEF coordinates, the
        corresponding ECI vector is obtained as::

            vector_eci = dcm_ecef_to_eci @ vector_ecef

    Notes
    -----
    The ECEF-to-ECI transformation is the transpose of the
    ECI-to-ECEF transformation:

    .. math::

        \\mathbf{C}_{ECI \\leftarrow ECEF} =
        \\mathbf{C}_{ECEF \\leftarrow ECI}^{T}

    resulting in:

    .. math::

        \\mathbf{C}_{ECI \\leftarrow ECEF} =
        \\begin{bmatrix}
        \\cos(\\theta) & -\\sin(\\theta) & 0 \\\\
        \\sin(\\theta) & \\cos(\\theta) & 0 \\\\
        0 & 0 & 1
        \\end{bmatrix}

    where :math:`\\theta` is the Greenwich Sidereal Time.

    The matrix is dimensionless because it represents a coordinate
    rotation.

    See Also
    --------
    eci_to_ecef : Compute the inverse transformation from ECI to ECEF.

    References
    ----------
    .. _vallado:
    
    Vallado, D. A., *Fundamentals of Astrodynamics and Applications*.
    """
    dcm_eci_to_ecef = eci_to_ecef(greenwich_sidereal_time_rad)
    dcm_ecef_to_eci = np.transpose(dcm_eci_to_ecef)

    return dcm_ecef_to_eci


def ecef_to_vehicle(latitude_rad, longitude_rad, altitude_m):
    """
    Compute the rotation matrix from ECEF to the local vehicle frame.

    The local vehicle frame is defined at the vehicle's geodetic position.
    The supplied geodetic latitude is first converted to geocentric
    latitude before constructing the direction cosine matrix (DCM).

    Parameters
    ----------
    latitude_rad : float
        Geodetic latitude of the vehicle, in radians.
    longitude_rad : float
        Longitude of the vehicle, in radians.
    altitude_m : float
        Geodetic altitude of the vehicle above the reference ellipsoid,
        in meters.

    Returns
    -------
    numpy.ndarray
        A 3x3 dimensionless direction cosine matrix (DCM) that transforms
        a vector from the ECEF frame to the local vehicle frame.

        If ``vector_ecef`` is expressed in ECEF coordinates, the
        corresponding vehicle-frame vector is obtained as::

            vector_vehicle = dcm_ecef_to_vehicle @ vector_ecef

    Notes
    -----
    The supplied geodetic latitude is converted to geocentric latitude
    before constructing the rotation matrix:

    .. math::

        \\phi_c = f(\\phi_g, h)

    where :math:`\\phi_g` is the geodetic latitude,
    :math:`\\phi_c` is the geocentric latitude, and :math:`h` is the
    altitude.

    The resulting DCM is:

    .. math::

        \\mathbf{C}_{V \\leftarrow E} =
        \\begin{bmatrix}
        -\\sin(\\phi_c)\\cos(\\lambda) &
        -\\sin(\\phi_c)\\sin(\\lambda) &
        \\cos(\\phi_c) \\\\
        -\\sin(\\lambda) &
        \\cos(\\lambda) &
        0 \\\\
        -\\cos(\\phi_c)\\cos(\\lambda) &
        -\\cos(\\phi_c)\\sin(\\lambda) &
        -\\sin(\\phi_c)
        \\end{bmatrix}

    where :math:`\\lambda` is the longitude.

    The DCM is dimensionless because it represents a coordinate-frame
    rotation.

    See Also
    --------
    vehicle_to_ecef : Compute the inverse vehicle-to-ECEF transformation.
    geodetic_to_geocentric_latitude : Convert geodetic latitude to
        geocentric latitude.
    """
    geocentric_latitude_rad = geodetic_to_geocentric_latitude(
        latitude_rad,
        altitude_m,
    )

    sine_latitude = sin(geocentric_latitude_rad)
    cosine_latitude = cos(geocentric_latitude_rad)
    sine_longitude = sin(longitude_rad)
    cosine_longitude = cos(longitude_rad)

    dcm_ecef_to_vehicle = np.array(
        [
            [
                -sine_latitude * cosine_longitude,
                -sine_latitude * sine_longitude,
                cosine_latitude,
            ],
            [
                -sine_longitude,
                cosine_longitude,
                0.0,
            ],
            [
                -cosine_latitude * cosine_longitude,
                -cosine_latitude * sine_longitude,
                -sine_latitude,
            ],
        ],
        dtype=np.float64,
    )

    return dcm_ecef_to_vehicle


def vehicle_to_ecef(latitude_rad, longitude_rad, altitude_m):
    """
    Compute the rotation matrix from the local vehicle frame to ECEF.

    This transformation is the inverse of the ECEF-to-vehicle
    transformation. Since the transformation matrix is a proper
    orthogonal rotation matrix, its inverse is equal to its transpose.

    Parameters
    ----------
    latitude_rad : float
        Geodetic latitude of the vehicle, in radians.
    longitude_rad : float
        Longitude of the vehicle, in radians.
    altitude_m : float
        Geodetic altitude of the vehicle above the reference ellipsoid,
        in meters.

    Returns
    -------
    numpy.ndarray
        A 3x3 dimensionless direction cosine matrix (DCM) that transforms
        a vector from the local vehicle frame to the ECEF frame.

        If ``vector_vehicle`` is expressed in vehicle coordinates, the
        corresponding ECEF vector is obtained as::

            vector_ecef = dcm_vehicle_to_ecef @ vector_vehicle

    Notes
    -----
    The transformation is obtained from the transpose of the
    ECEF-to-vehicle DCM:

    .. math::

        \\mathbf{C}_{E \\leftarrow V}
        =
        \\mathbf{C}_{V \\leftarrow E}^{T}

    The DCM is dimensionless because it represents a coordinate-frame
    rotation.

    See Also
    --------
    ecef_to_vehicle : Compute the inverse ECEF-to-vehicle transformation.
    """
    dcm_ecef_to_vehicle = ecef_to_vehicle(
        latitude_rad,
        longitude_rad,
        altitude_m,
    )

    dcm_vehicle_to_ecef = np.transpose(dcm_ecef_to_vehicle)

    return dcm_vehicle_to_ecef


def ecef_to_ned(
    x_ecef_m,
    y_ecef_m,
    z_ecef_m,
    reference_latitude_rad,
    reference_longitude_rad,
    reference_altitude_m,
):
    """
    Convert an ECEF position to local North-East-Down coordinates.

    The input ECEF position is expressed relative to a reference
    geodetic position. The reference point is first converted to ECEF,
    after which the relative ECEF displacement is transformed into the
    local North-East-Down (NED) frame.

    Parameters
    ----------
    x_ecef_m : float
        X-coordinate of the position in the ECEF frame, in meters.
    y_ecef_m : float
        Y-coordinate of the position in the ECEF frame, in meters.
    z_ecef_m : float
        Z-coordinate of the position in the ECEF frame, in meters.
    reference_latitude_rad : float
        Geodetic latitude of the NED reference point, in radians.
    reference_longitude_rad : float
        Longitude of the NED reference point, in radians.
    reference_altitude_m : float
        Geodetic altitude of the NED reference point above the reference
        ellipsoid, in meters.

    Returns
    -------
    north_m : float
        North displacement from the reference point, in meters.
    east_m : float
        East displacement from the reference point, in meters.
    down_m : float
        Down displacement from the reference point, in meters.

    Notes
    -----
    The relative ECEF position is computed as:

    .. math::

        \\Delta \\mathbf{r}_{ECEF}
        =
        \\mathbf{r}_{ECEF}
        -
        \\mathbf{r}_{ECEF,0}

    and transformed into NED coordinates according to:

    .. math::

        \\mathbf{r}_{NED}
        =
        \\mathbf{C}_{NED \\leftarrow ECEF}
        \\Delta \\mathbf{r}_{ECEF}

    All position quantities are expressed in meters.

    See Also
    --------
    geodetic_to_ecef : Convert geodetic coordinates to ECEF coordinates.
    ecef_to_ned_matrix : Compute the ECEF-to-NED rotation matrix.
    """
    reference_x_ecef_m, reference_y_ecef_m, reference_z_ecef_m = (
        geodetic_to_ecef(
            reference_latitude_rad,
            reference_longitude_rad,
            reference_altitude_m,
        )
    )

    delta_position_ecef_m = np.array(
        [
            [x_ecef_m - reference_x_ecef_m],
            [y_ecef_m - reference_y_ecef_m],
            [z_ecef_m - reference_z_ecef_m],
        ],
        dtype=np.float64,
    )

    dcm_ecef_to_ned = ecef_to_ned_matrix(
        reference_latitude_rad,
        reference_longitude_rad,
    )

    position_ned_m = np.matmul(
        dcm_ecef_to_ned,
        delta_position_ecef_m,
    )

    north_m = position_ned_m[0, 0]
    east_m = position_ned_m[1, 0]
    down_m = position_ned_m[2, 0]

    return north_m, east_m, down_m


def ecef_to_enu_matrix(latitude_rad, longitude_rad):
    """
    Compute the rotation matrix from ECEF to ENU coordinates.

    The East-North-Up (ENU) frame is a local tangent-plane coordinate
    system centered at the specified geographic location. Its axes are
    defined as follows:

    * X-axis: East
    * Y-axis: North
    * Z-axis: Up

    Parameters
    ----------
    latitude_rad : float
        Geodetic latitude of the reference point, in radians.
    longitude_rad : float
        Longitude of the reference point, in radians.

    Returns
    -------
    numpy.ndarray
        A 3x3 dimensionless direction cosine matrix (DCM) that
        transforms a vector from ECEF coordinates to ENU coordinates.

        If ``vector_ecef`` is expressed in ECEF coordinates::

            vector_enu = dcm_ecef_to_enu @ vector_ecef

    Notes
    -----
    The transformation matrix is:

    .. math::

        \\mathbf{C}_{ENU \\leftarrow ECEF} =
        \\begin{bmatrix}
        -\\sin(\\lambda) &
        \\cos(\\lambda) &
        0 \\\\
        -\\cos(\\lambda)\\sin(\\phi) &
        -\\sin(\\lambda)\\sin(\\phi) &
        \\cos(\\phi) \\\\
        \\cos(\\lambda)\\cos(\\phi) &
        \\sin(\\lambda)\\cos(\\phi) &
        \\sin(\\phi)
        \\end{bmatrix}

    where :math:`\\phi` is latitude and :math:`\\lambda` is longitude.

    The DCM is dimensionless because it represents a coordinate-frame
    rotation.

    See Also
    --------
    enu_to_ecef_matrix : Compute the inverse ENU-to-ECEF transformation.
    """
    sine_latitude = sin(latitude_rad)
    cosine_latitude = cos(latitude_rad)
    sine_longitude = sin(longitude_rad)
    cosine_longitude = cos(longitude_rad)

    dcm_ecef_to_enu = np.array(
        [
            [
                -sine_longitude,
                cosine_longitude,
                0.0,
            ],
            [
                -cosine_longitude * sine_latitude,
                -sine_longitude * sine_latitude,
                cosine_latitude,
            ],
            [
                cosine_longitude * cosine_latitude,
                sine_longitude * cosine_latitude,
                sine_latitude,
            ],
        ],
        dtype=np.float64,
    )

    return dcm_ecef_to_enu


def enu_to_ecef_matrix(latitude_rad, longitude_rad):
    """
    Compute the rotation matrix from ENU to ECEF coordinates.

    The transformation is the inverse of the ECEF-to-ENU
    transformation. Since the transformation matrix is orthogonal,
    its inverse is equal to its transpose.

    Parameters
    ----------
    latitude_rad : float
        Geodetic latitude of the reference point, in radians.
    longitude_rad : float
        Longitude of the reference point, in radians.

    Returns
    -------
    numpy.ndarray
        A 3x3 dimensionless direction cosine matrix (DCM) that
        transforms a vector from ENU coordinates to ECEF coordinates.

        If ``vector_enu`` is expressed in ENU coordinates::

            vector_ecef = dcm_enu_to_ecef @ vector_enu

    Notes
    -----
    The transformation is obtained from the transpose of the
    ECEF-to-ENU DCM:

    .. math::

        \\mathbf{C}_{ECEF \\leftarrow ENU}
        =
        \\mathbf{C}_{ENU \\leftarrow ECEF}^{T}

    The DCM is dimensionless because it represents a coordinate-frame
    rotation.

    See Also
    --------
    ecef_to_enu_matrix : Compute the inverse ECEF-to-ENU transformation.
    """
    dcm_ecef_to_enu = ecef_to_enu_matrix(
        latitude_rad,
        longitude_rad,
    )

    dcm_enu_to_ecef = np.transpose(dcm_ecef_to_enu)

    return dcm_enu_to_ecef


@numba.njit(cache=True)
def ecef_to_ned_matrix(latitude_rad, longitude_rad):
    """
    Compute the rotation matrix from ECEF to NED coordinates.

    The North-East-Down (NED) frame is a local navigation coordinate
    system centered at the specified geographic location. Its axes are
    defined as follows:

    * X-axis: North
    * Y-axis: East
    * Z-axis: Down

    Parameters
    ----------
    latitude_rad : float
        Latitude of the reference point, in radians.
    longitude_rad : float
        Longitude of the reference point, in radians.

    Returns
    -------
    numpy.ndarray
        A 3x3 dimensionless direction cosine matrix (DCM) that
        transforms a vector from ECEF coordinates to NED coordinates.

        If ``vector_ecef`` is expressed in ECEF coordinates::

            vector_ned = dcm_ecef_to_ned @ vector_ecef

    Notes
    -----
    The transformation matrix is:

    .. math::

        \\mathbf{C}_{NED \\leftarrow ECEF} =
        \\begin{bmatrix}
        -\\sin(\\phi)\\cos(\\lambda) &
        -\\sin(\\phi)\\sin(\\lambda) &
        \\cos(\\phi) \\\\
        -\\sin(\\lambda) &
        \\cos(\\lambda) &
        0 \\\\
        -\\cos(\\phi)\\cos(\\lambda) &
        -\\cos(\\phi)\\sin(\\lambda) &
        -\\sin(\\phi)
        \\end{bmatrix}

    where :math:`\\phi` is latitude and :math:`\\lambda` is longitude.

    The DCM is dimensionless because it represents a coordinate-frame
    rotation.
    """
    sine_latitude = sin(latitude_rad)
    cosine_latitude = cos(latitude_rad)
    sine_longitude = sin(longitude_rad)
    cosine_longitude = cos(longitude_rad)

    dcm_ecef_to_ned = np.array(
        [
            [
                -sine_latitude * cosine_longitude,
                -sine_latitude * sine_longitude,
                cosine_latitude,
            ],
            [
                -sine_longitude,
                cosine_longitude,
                0.0,
            ],
            [
                -cosine_latitude * cosine_longitude,
                -cosine_latitude * sine_longitude,
                -sine_latitude,
            ],
        ],
        dtype=np.float64,
    )

    return dcm_ecef_to_ned


@numba.njit(cache=True)
def ned_to_ecef_matrix(latitude_rad, longitude_rad):
    """
    Compute the rotation matrix from NED to ECEF coordinates.

    The transformation is the inverse of the ECEF-to-NED
    transformation. Since the transformation matrix is orthogonal,
    its inverse is equal to its transpose.

    Parameters
    ----------
    latitude_rad : float
        Latitude of the reference point, in radians.
    longitude_rad : float
        Longitude of the reference point, in radians.

    Returns
    -------
    numpy.ndarray
        A 3x3 dimensionless direction cosine matrix (DCM) that
        transforms a vector from NED coordinates to ECEF coordinates.

        If ``vector_ned`` is expressed in NED coordinates::

            vector_ecef = dcm_ned_to_ecef @ vector_ned

    Notes
    -----
    The transformation is obtained from the transpose of the
    ECEF-to-NED DCM:

    .. math::

        \\mathbf{C}_{ECEF \\leftarrow NED}
        =
        \\mathbf{C}_{NED \\leftarrow ECEF}^{T}

    The DCM is dimensionless because it represents a coordinate-frame
    rotation.

    See Also
    --------
    ecef_to_ned_matrix : Compute the inverse ECEF-to-NED transformation.
    """
    dcm_ecef_to_ned = ecef_to_ned_matrix(
        latitude_rad,
        longitude_rad,
    )

    dcm_ned_to_ecef = np.transpose(dcm_ecef_to_ned)

    return dcm_ned_to_ecef


def enu_to_eci(
    east_m,
    north_m,
    up_m,
    reference_latitude_rad,
    reference_longitude_rad,
    reference_altitude_m,
    time_s,
):
    """
    Convert a position from a local ENU frame to ECI coordinates.

    The ENU position is first converted to ECEF coordinates using the
    specified geodetic reference point. The resulting ECEF position is
    then converted to geodetic coordinates and finally to ECI
    coordinates at the specified time.

    Parameters
    ----------
    east_m : float
        East displacement from the ENU reference point, in meters.
    north_m : float
        North displacement from the ENU reference point, in meters.
    up_m : float
        Up displacement from the ENU reference point, in meters.
    reference_latitude_rad : float
        Geodetic latitude of the ENU frame origin, in radians.
    reference_longitude_rad : float
        Longitude of the ENU frame origin, in radians.
    reference_altitude_m : float
        Geodetic altitude of the ENU frame origin above the reference
        ellipsoid, in meters.
    time_s : float
        Time associated with the ECI coordinate transformation, in
        seconds.

    Returns
    -------
    numpy.ndarray
        A 3x1 array containing the ECI position:

        .. math::

            \\mathbf{r}_{ECI} =
            \\begin{bmatrix}
            x_{ECI} \\\\
            y_{ECI} \\\\
            z_{ECI}
            \\end{bmatrix}

        with all position components expressed in meters.

    Notes
    -----
    The transformation chain is:

    .. math::

        ENU \\rightarrow ECEF \\rightarrow Geodetic \\rightarrow ECI

    The ENU coordinates represent a displacement from the specified
    reference point, while the returned ECI coordinates represent the
    corresponding absolute position.

    See Also
    --------
    enu_to_ecef : Convert ENU coordinates to ECEF coordinates.
    geodetic_to_eci : Convert geodetic coordinates to ECI coordinates.
    """
    x_ecef_m, y_ecef_m, z_ecef_m = enu_to_ecef(
        east_m,
        north_m,
        up_m,
        reference_latitude_rad,
        reference_longitude_rad,
        reference_altitude_m,
    )

    latitude_rad, longitude_rad, altitude_m = ecef_to_geodetic(
        x_ecef_m,
        y_ecef_m,
        z_ecef_m,
    )

    x_eci_m, y_eci_m, z_eci_m = geodetic_to_eci(
        latitude_rad,
        longitude_rad,
        altitude_m,
        time_s,
    )

    return np.array(
        [[x_eci_m], [y_eci_m], [z_eci_m]],
        dtype=np.float64,
    )


def eci_to_enu(
    x_eci_m,
    y_eci_m,
    z_eci_m,
    reference_latitude_rad,
    reference_longitude_rad,
    reference_altitude_m,
    time_s,
):
    """
    Convert an ECI position to local ENU coordinates.

    The ECI position is first converted to geodetic coordinates at the
    specified time and then to ECEF coordinates. The resulting ECEF
    position is finally expressed as a displacement in the local
    East-North-Up (ENU) frame centered at the specified reference point.

    Parameters
    ----------
    x_eci_m : float
        X-coordinate of the position in the ECI frame, in meters.
    y_eci_m : float
        Y-coordinate of the position in the ECI frame, in meters.
    z_eci_m : float
        Z-coordinate of the position in the ECI frame, in meters.
    reference_latitude_rad : float
        Geodetic latitude of the ENU frame origin, in radians.
    reference_longitude_rad : float
        Longitude of the ENU frame origin, in radians.
    reference_altitude_m : float
        Geodetic altitude of the ENU frame origin above the reference
        ellipsoid, in meters.
    time_s : float
        Time associated with the ECI coordinate transformation, in
        seconds.

    Returns
    -------
    numpy.ndarray
        A 3x1 array containing the local ENU position:

        .. math::

            \\mathbf{r}_{ENU} =
            \\begin{bmatrix}
            E \\\\
            N \\\\
            U
            \\end{bmatrix}

        where all components are expressed in meters.

    Notes
    -----
    The transformation chain is:

    .. math::

        ECI \\rightarrow Geodetic \\rightarrow ECEF \\rightarrow ENU

    The returned coordinates are relative to the specified ENU reference
    point.

    See Also
    --------
    ecef_to_enu : Convert ECEF coordinates to local ENU coordinates.
    enu_to_eci : Compute the inverse position transformation.
    """
    latitude_rad, longitude_rad, altitude_m = eci_to_geodetic(
        x_eci_m,
        y_eci_m,
        z_eci_m,
        time_s,
    )

    x_ecef_m, y_ecef_m, z_ecef_m = geodetic_to_ecef(
        latitude_rad,
        longitude_rad,
        altitude_m,
    )

    east_m, north_m, up_m = ecef_to_enu(
        x_ecef_m,
        y_ecef_m,
        z_ecef_m,
        reference_latitude_rad,
        reference_longitude_rad,
        reference_altitude_m,
    )

    return np.array(
        [[east_m], [north_m], [up_m]],
        dtype=np.float64,
    )


def eci_to_enu_attitude(
    roll_rad,
    pitch_rad,
    yaw_rad,
    reference_latitude_rad,
    reference_longitude_rad,
    time_s,
):
    """
    Transform body attitude from ECI-referenced coordinates to ENU.

    The input Euler angles define the body attitude relative to the ECI
    frame. The corresponding body-to-ECI rotation matrix is transformed
    successively into the ECEF and local ENU frames.

    Parameters
    ----------
    roll_rad : float
        Body roll angle relative to the ECI frame, in radians.
    pitch_rad : float
        Body pitch angle relative to the ECI frame, in radians.
    yaw_rad : float
        Body yaw angle relative to the ECI frame, in radians.
    reference_latitude_rad : float
        Latitude defining the local ENU frame, in radians.
    reference_longitude_rad : float
        Longitude defining the local ENU frame, in radians.
    time_s : float
        Time associated with the ECI-to-ECEF transformation, in seconds.

    Returns
    -------
    quaternion_enu : numpy.ndarray
        Normalized quaternion representing the body attitude relative to
        the ENU frame. The quaternion is dimensionless.
    roll_enu_rad : float
        Roll angle of the body relative to the ENU frame, in radians.
    pitch_enu_rad : float
        Pitch angle of the body relative to the ENU frame, in radians.
    yaw_enu_rad : float
        Yaw angle of the body relative to the ENU frame, in radians.

    Notes
    -----
    The rotation chain implemented by this function is:

    .. math::

        \\mathbf{C}_{ENU \\leftarrow B}
        =
        \\mathbf{C}_{ENU \\leftarrow ECEF}
        \\mathbf{C}_{ECEF \\leftarrow ECI}
        \\mathbf{C}_{ECI \\leftarrow B}

    where:

    * :math:`B` is the body frame.
    * :math:`ECI` is the Earth-Centered Inertial frame.
    * :math:`ECEF` is the Earth-Centered Earth-Fixed frame.
    * :math:`ENU` is the local East-North-Up frame.

    The ECI-to-ECEF rotation is evaluated using:

    .. math::

        \\theta = \\omega_E t

    where :math:`\\omega_E` is ``earth_rate__rad_s`` and :math:`t` is
    ``time_s``.

    Consequently, this implementation implicitly assumes that the
    ECI-to-ECEF rotation angle is zero at ``time_s = 0``. No additional
    Greenwich sidereal angle or initial Earth orientation angle is
    included in this function.

    The ENU Euler angles are extracted using the rotation sequence
    implemented by ``body_to_eci_euler`` and ``euler_to_quaternion``.

    Notes
    -----
    The clipping of the matrix element used by ``arcsin`` protects
    against small floating-point errors that could otherwise produce a
    value slightly outside the valid interval ``[-1, 1]``.
    """
    dcm_eci_from_body = body_to_eci_euler(
        roll_rad,
        pitch_rad,
        yaw_rad,
    )

    dcm_ecef_from_eci = eci_to_ecef(
        earth_rate__rad_s * time_s,
    )

    dcm_enu_from_ecef = ecef_to_enu_matrix(
        reference_latitude_rad,
        reference_longitude_rad,
    )

    dcm_enu_from_body = np.matmul(
        dcm_enu_from_ecef,
        np.matmul(
            dcm_ecef_from_eci,
            dcm_eci_from_body,
        ),
    )

    roll_enu_rad = np.arctan2(
        dcm_enu_from_body[2, 1],
        dcm_enu_from_body[2, 2],
    )

    pitch_matrix_element = dcm_enu_from_body[2, 0]

    if abs(pitch_matrix_element) > 1.0:
        pitch_matrix_element = np.sign(pitch_matrix_element)

    pitch_enu_rad = -np.arcsin(pitch_matrix_element)

    yaw_enu_rad = np.arctan2(
        dcm_enu_from_body[1, 0],
        dcm_enu_from_body[0, 0],
    )

    quaternion_enu = euler_to_quaternion(
        roll_enu_rad,
        pitch_enu_rad,
        yaw_enu_rad,
    )

    quaternion_enu = quaternion_enu / norm(quaternion_enu)

    return (
        quaternion_enu,
        roll_enu_rad,
        pitch_enu_rad,
        yaw_enu_rad,
    )


def eci_to_ned(
    x_eci_m,
    y_eci_m,
    z_eci_m,
    reference_latitude_rad,
    reference_longitude_rad,
    reference_altitude_m,
    time_s,
):
    """
    Convert an ECI position to local NED coordinates.

    The ECI position is converted to geodetic and then ECEF coordinates
    at the specified time. The resulting ECEF position is expressed as
    a displacement relative to the specified reference point in the
    local North-East-Down (NED) frame.

    Parameters
    ----------
    x_eci_m : float
        X-coordinate of the position in the ECI frame, in meters.
    y_eci_m : float
        Y-coordinate of the position in the ECI frame, in meters.
    z_eci_m : float
        Z-coordinate of the position in the ECI frame, in meters.
    reference_latitude_rad : float
        Geodetic latitude of the NED reference point, in radians.
    reference_longitude_rad : float
        Longitude of the NED reference point, in radians.
    reference_altitude_m : float
        Geodetic altitude of the NED reference point above the reference
        ellipsoid, in meters.
    time_s : float
        Time associated with the ECI coordinate transformation, in
        seconds.

    Returns
    -------
    numpy.ndarray
        A 1D array containing the local NED displacement:

        .. math::

            \\mathbf{r}_{NED} =
            \\begin{bmatrix}
            N & E & D
            \\end{bmatrix}^{T}

        with all components expressed in meters.

    Notes
    -----
    The transformation chain is:

    .. math::

        ECI \\rightarrow Geodetic \\rightarrow ECEF \\rightarrow NED

    The NED coordinates are relative to the supplied reference point.

    See Also
    --------
    ecef_to_ned : Convert ECEF position to local NED coordinates.
    eci_to_enu : Convert ECI position to local ENU coordinates.
    """
    latitude_rad, longitude_rad, altitude_m = eci_to_geodetic(
        x_eci_m,
        y_eci_m,
        z_eci_m,
        time_s,
    )

    x_ecef_m, y_ecef_m, z_ecef_m = geodetic_to_ecef(
        latitude_rad,
        longitude_rad,
        altitude_m,
    )

    north_m, east_m, down_m = ecef_to_ned(
        x_ecef_m,
        y_ecef_m,
        z_ecef_m,
        reference_latitude_rad,
        reference_longitude_rad,
        reference_altitude_m,
    )

    return np.array(
        [north_m, east_m, down_m],
        dtype=np.float64,
    )


def eci_to_ned_velocity(
    velocity_x_eci_m_s,
    velocity_y_eci_m_s,
    velocity_z_eci_m_s,
    latitude_rad,
    longitude_rad,
    time_s,
):
    """
    Rotate an ECI velocity vector into the local NED frame.

    The input velocity vector is first rotated from ECI to ECEF using
    the Earth rotation angle corresponding to ``time_s``. It is then
    rotated from ECEF to the local NED frame defined by the supplied
    latitude and longitude.

    Parameters
    ----------
    velocity_x_eci_m_s : float
        X-component of the velocity vector in ECI coordinates, in m/s.
    velocity_y_eci_m_s : float
        Y-component of the velocity vector in ECI coordinates, in m/s.
    velocity_z_eci_m_s : float
        Z-component of the velocity vector in ECI coordinates, in m/s.
    latitude_rad : float
        Latitude defining the local NED frame, in radians.
    longitude_rad : float
        Longitude defining the local NED frame, in radians.
    time_s : float
        Time associated with the ECI-to-ECEF rotation, in seconds.

    Returns
    -------
    numpy.ndarray
        A 1D array containing the NED components of the rotated velocity
        vector:

        .. math::

            \\mathbf{v}_{NED} =
            \\begin{bmatrix}
            v_N & v_E & v_D
            \\end{bmatrix}^{T}

        in meters per second.

    Notes
    -----
    The implemented transformation is:

    .. math::

        \\mathbf{v}_{ECEF}
        =
        \\mathbf{C}_{ECEF \\leftarrow ECI}
        \\mathbf{v}_{ECI}

    followed by:

    .. math::

        \\mathbf{v}_{NED}
        =
        \\mathbf{C}_{NED \\leftarrow ECEF}
        \\mathbf{v}_{ECEF}

    Therefore:

    .. math::

        \\mathbf{v}_{NED}
        =
        \\mathbf{C}_{NED \\leftarrow ECEF}
        \\mathbf{C}_{ECEF \\leftarrow ECI}
        \\mathbf{v}_{ECI}

    This function performs a coordinate rotation of the supplied
    velocity vector. It does **not** account for the translational
    velocity of the rotating ECEF frame or the motion of the local
    NED reference frame.

    Consequently, if the intended quantity is the physical velocity
    relative to a ground-fixed NED frame, additional Earth-rotation
    and reference-frame-motion terms may be required.

    The ECI-to-ECEF rotation uses:

    .. math::

        \\theta = \\omega_E t

    where :math:`\\omega_E` is ``earth_rate__rad_s``.
    """
    dcm_ecef_from_eci = eci_to_ecef(
        earth_rate__rad_s * time_s,
    )

    velocity_eci_m_s = np.array(
        [
            [velocity_x_eci_m_s],
            [velocity_y_eci_m_s],
            [velocity_z_eci_m_s],
        ],
        dtype=np.float64,
    )

    velocity_ecef_m_s = np.matmul(
        dcm_ecef_from_eci,
        velocity_eci_m_s,
    )

    dcm_ned_from_ecef = ecef_to_ned_matrix(
        latitude_rad,
        longitude_rad,
    )

    velocity_ned_m_s = np.matmul(
        dcm_ned_from_ecef,
        velocity_ecef_m_s,
    )

    return np.array(
        [
            velocity_ned_m_s[0, 0],
            velocity_ned_m_s[1, 0],
            velocity_ned_m_s[2, 0],
        ],
        dtype=np.float64,
    )


def ecef_to_enu(
    x_ecef_m,
    y_ecef_m,
    z_ecef_m,
    reference_latitude_rad,
    reference_longitude_rad,
    reference_altitude_m,
):
    """
    Convert an ECEF position to local ENU coordinates.

    The ENU frame is a local tangent-plane coordinate system centered
    at the specified reference point. The input ECEF position is first
    expressed as a displacement from that reference point and then
    rotated into the East-North-Up frame.

    Parameters
    ----------
    x_ecef_m : float
        X-coordinate of the position in ECEF coordinates, in meters.
    y_ecef_m : float
        Y-coordinate of the position in ECEF coordinates, in meters.
    z_ecef_m : float
        Z-coordinate of the position in ECEF coordinates, in meters.
    reference_latitude_rad : float
        Geodetic latitude of the ENU reference point, in radians.
    reference_longitude_rad : float
        Longitude of the ENU reference point, in radians.
    reference_altitude_m : float
        Geodetic altitude of the ENU reference point above the reference
        ellipsoid, in meters.

    Returns
    -------
    east_m : float
        East displacement from the reference point, in meters.
    north_m : float
        North displacement from the reference point, in meters.
    up_m : float
        Up displacement from the reference point, in meters.

    Notes
    -----
    The ECEF displacement relative to the reference point is:

    .. math::

        \\Delta \\mathbf{r}_{ECEF}
        =
        \\mathbf{r}_{ECEF}
        -
        \\mathbf{r}_{ECEF,0}

    The local ENU coordinates are then obtained through:

    .. math::

        \\mathbf{r}_{ENU}
        =
        \\mathbf{C}_{ENU \\leftarrow ECEF}
        \\Delta \\mathbf{r}_{ECEF}

    The reference point is defined using the supplied geodetic
    latitude, longitude, and altitude.

    All position and displacement quantities are expressed in meters.
    """
    reference_x_ecef_m, reference_y_ecef_m, reference_z_ecef_m = (
        geodetic_to_ecef(
            reference_latitude_rad,
            reference_longitude_rad,
            reference_altitude_m,
        )
    )

    delta_x_ecef_m = x_ecef_m - reference_x_ecef_m
    delta_y_ecef_m = y_ecef_m - reference_y_ecef_m
    delta_z_ecef_m = z_ecef_m - reference_z_ecef_m

    sine_latitude = sin(reference_latitude_rad)
    cosine_latitude = cos(reference_latitude_rad)
    sine_longitude = sin(reference_longitude_rad)
    cosine_longitude = cos(reference_longitude_rad)

    east_m = (
        -sine_longitude * delta_x_ecef_m
        + cosine_longitude * delta_y_ecef_m
    )

    north_m = (
        -cosine_longitude * sine_latitude * delta_x_ecef_m
        -sine_longitude * sine_latitude * delta_y_ecef_m
        + cosine_latitude * delta_z_ecef_m
    )

    up_m = (
        cosine_latitude * cosine_longitude * delta_x_ecef_m
        + cosine_latitude * sine_longitude * delta_y_ecef_m
        + sine_latitude * delta_z_ecef_m
    )

    return east_m, north_m, up_m


def enu_to_ecef(
    east_m,
    north_m,
    up_m,
    reference_latitude_rad,
    reference_longitude_rad,
    reference_altitude_m,
):
    """
    Convert local ENU coordinates to ECEF coordinates.

    The ENU coordinates are interpreted as displacements from the
    specified reference point. The displacement is rotated into the
    ECEF frame and added to the ECEF coordinates of the reference point.

    Parameters
    ----------
    east_m : float
        East displacement from the reference point, in meters.
    north_m : float
        North displacement from the reference point, in meters.
    up_m : float
        Up displacement from the reference point, in meters.
    reference_latitude_rad : float
        Geodetic latitude of the ENU reference point, in radians.
    reference_longitude_rad : float
        Longitude of the ENU reference point, in radians.
    reference_altitude_m : float
        Geodetic altitude of the ENU reference point above the reference
        ellipsoid, in meters.

    Returns
    -------
    x_ecef_m : float
        X-coordinate of the resulting ECEF position, in meters.
    y_ecef_m : float
        Y-coordinate of the resulting ECEF position, in meters.
    z_ecef_m : float
        Z-coordinate of the resulting ECEF position, in meters.

    Notes
    -----
    The ENU displacement is transformed according to:

    .. math::

        \\Delta \\mathbf{r}_{ECEF}
        =
        \\mathbf{C}_{ECEF \\leftarrow ENU}
        \\mathbf{r}_{ENU}

    The resulting absolute ECEF position is then:

    .. math::

        \\mathbf{r}_{ECEF}
        =
        \\mathbf{r}_{ECEF,0}
        +
        \\Delta \\mathbf{r}_{ECEF}

    All position and displacement quantities are expressed in meters.

    See Also
    --------
    ecef_to_enu : Compute the inverse ECEF-to-ENU transformation.
    """
    reference_x_ecef_m, reference_y_ecef_m, reference_z_ecef_m = (
        geodetic_to_ecef(
            reference_latitude_rad,
            reference_longitude_rad,
            reference_altitude_m,
        )
    )

    delta_x_ecef_m = (
        -sin(reference_longitude_rad) * east_m
        - cos(reference_longitude_rad)
        * sin(reference_latitude_rad)
        * north_m
        + cos(reference_latitude_rad)
        * cos(reference_longitude_rad)
        * up_m
    )

    delta_y_ecef_m = (
        cos(reference_longitude_rad) * east_m
        - sin(reference_latitude_rad)
        * sin(reference_longitude_rad)
        * north_m
        + cos(reference_latitude_rad)
        * sin(reference_longitude_rad)
        * up_m
    )

    delta_z_ecef_m = (
        cos(reference_latitude_rad) * north_m
        + sin(reference_latitude_rad) * up_m
    )

    x_ecef_m = reference_x_ecef_m + delta_x_ecef_m
    y_ecef_m = reference_y_ecef_m + delta_y_ecef_m
    z_ecef_m = reference_z_ecef_m + delta_z_ecef_m

    return x_ecef_m, y_ecef_m, z_ecef_m


@numba.njit(cache=True)
def body_to_eci_euler(roll_rad, pitch_rad, yaw_rad):
    """
    Compute the body-to-ECI direction cosine matrix from 3-2-1 Euler angles.

    The Euler-angle convention implemented by this function corresponds to
    a Z-Y-X (3-2-1) sequence.

    Parameters
    ----------
    roll_rad : float
        Roll angle, in radians.
    pitch_rad : float
        Pitch angle, in radians.
    yaw_rad : float
        Yaw angle, in radians.

    Returns
    -------
    numpy.ndarray
        A 3x3 dimensionless direction cosine matrix (DCM) that transforms
        a vector from the body frame to the ECI frame:

        .. math::

            \\mathbf{v}_{ECI}
            =
            \\mathbf{C}_{ECI \\leftarrow B}
            \\mathbf{v}_{B}

    Notes
    -----
    The implemented rotation matrix is:

    .. math::

        \\mathbf{C}_{ECI \\leftarrow B} =
        \\begin{bmatrix}
        c_\\theta c_\\psi &
        s_\\phi s_\\theta c_\\psi - c_\\phi s_\\psi &
        c_\\phi s_\\theta c_\\psi + s_\\phi s_\\psi \\\\
        c_\\theta s_\\psi &
        s_\\phi s_\\theta s_\\psi + c_\\phi c_\\psi &
        c_\\phi s_\\theta s_\\psi - s_\\phi c_\\psi \\\\
        -s_\\theta &
        s_\\phi c_\\theta &
        c_\\phi c_\\theta
        \\end{bmatrix}

    where:

    * :math:`\\phi` is roll,
    * :math:`\\theta` is pitch,
    * :math:`\\psi` is yaw.

    The DCM is dimensionless because it represents a pure coordinate
    rotation.

    See Also
    --------
    eci_to_body_euler : Compute the inverse ECI-to-body DCM.
    euler_to_quaternion : Convert the same Euler angles to a quaternion.
    """
    sine_roll = sin(roll_rad)
    cosine_roll = cos(roll_rad)
    sine_pitch = sin(pitch_rad)
    cosine_pitch = cos(pitch_rad)
    sine_yaw = sin(yaw_rad)
    cosine_yaw = cos(yaw_rad)

    dcm_eci_from_body = np.array(
        [
            [
                cosine_pitch * cosine_yaw,
                sine_roll * sine_pitch * cosine_yaw
                - cosine_roll * sine_yaw,
                cosine_roll * sine_pitch * cosine_yaw
                + sine_roll * sine_yaw,
            ],
            [
                cosine_pitch * sine_yaw,
                sine_roll * sine_pitch * sine_yaw
                + cosine_roll * cosine_yaw,
                cosine_roll * sine_pitch * sine_yaw
                - sine_roll * cosine_yaw,
            ],
            [
                -sine_pitch,
                sine_roll * cosine_pitch,
                cosine_roll * cosine_pitch,
            ],
        ],
        dtype=np.float64,
    )

    return dcm_eci_from_body


def eci_to_body_euler(roll_rad, pitch_rad, yaw_rad):
    """
    Compute the ECI-to-body direction cosine matrix from 3-2-1 Euler angles.

    This function computes the inverse of the body-to-ECI DCM. Since the
    DCM is orthogonal, the inverse is equal to its transpose.

    Parameters
    ----------
    roll_rad : float
        Roll angle, in radians.
    pitch_rad : float
        Pitch angle, in radians.
    yaw_rad : float
        Yaw angle, in radians.

    Returns
    -------
    numpy.ndarray
        A 3x3 dimensionless DCM that transforms a vector from ECI
        coordinates to body coordinates:

        .. math::

            \\mathbf{v}_{B}
            =
            \\mathbf{C}_{B \\leftarrow ECI}
            \\mathbf{v}_{ECI}

    Notes
    -----
    The transformation is:

    .. math::

        \\mathbf{C}_{B \\leftarrow ECI}
        =
        \\mathbf{C}_{ECI \\leftarrow B}^{T}

    See Also
    --------
    body_to_eci_euler : Compute the inverse body-to-ECI DCM.
    """
    dcm_eci_from_body = body_to_eci_euler(
        roll_rad,
        pitch_rad,
        yaw_rad,
    )

    dcm_body_from_eci = np.transpose(dcm_eci_from_body)

    return dcm_body_from_eci


@numba.njit(cache=True)
def body_to_eci_quaternion(quaternion_body_eci):
    """
    Convert a body-to-ECI quaternion into a body-to-ECI DCM.

    Parameters
    ----------
    quaternion_body_eci : numpy.ndarray, shape (4, 1)
        Quaternion representing the body-to-ECI rotation, using the
        scalar-first convention ``[w, x, y, z]``. The quaternion is
        dimensionless.

    Returns
    -------
    numpy.ndarray
        A 3x3 dimensionless DCM representing the same body-to-ECI
        rotation:

        .. math::

            \\mathbf{v}_{ECI}
            =
            \\mathbf{C}_{ECI \\leftarrow B}
            \\mathbf{v}_{B}

    Notes
    -----
    The quaternion is assumed to use the scalar-first ordering:

    .. math::

        \\mathbf{q}_{ECI \\leftarrow B}
        =
        \\begin{bmatrix}
        w & x & y & z
        \\end{bmatrix}^{T}

    The input quaternion is not normalized inside this function. For a
    unit quaternion, the resulting matrix is a proper rotation matrix.

    See Also
    --------
    eci_to_body_quaternion : Compute the inverse DCM from the same quaternion.
    dcm_to_quaternion : Convert a DCM back to a quaternion.
    quaternion_normalize : Normalize a quaternion.
    """
    quaternion_w = quaternion_body_eci[0, 0]
    quaternion_x = quaternion_body_eci[1, 0]
    quaternion_y = quaternion_body_eci[2, 0]
    quaternion_z = quaternion_body_eci[3, 0]

    dcm_eci_from_body = np.array(
        [
            [
                quaternion_w**2 + quaternion_x**2
                - quaternion_y**2 - quaternion_z**2,
                2.0 * (
                    quaternion_x * quaternion_y
                    - quaternion_w * quaternion_z
                ),
                2.0 * (
                    quaternion_x * quaternion_z
                    + quaternion_w * quaternion_y
                ),
            ],
            [
                2.0 * (
                    quaternion_x * quaternion_y
                    + quaternion_w * quaternion_z
                ),
                quaternion_w**2 - quaternion_x**2
                + quaternion_y**2 - quaternion_z**2,
                2.0 * (
                    quaternion_y * quaternion_z
                    - quaternion_w * quaternion_x
                ),
            ],
            [
                2.0 * (
                    quaternion_x * quaternion_z
                    - quaternion_w * quaternion_y
                ),
                2.0 * (
                    quaternion_y * quaternion_z
                    + quaternion_w * quaternion_x
                ),
                quaternion_w**2 - quaternion_x**2
                - quaternion_y**2 + quaternion_z**2,
            ],
        ],
        dtype=np.float64,
    )

    return dcm_eci_from_body


@numba.njit(cache=True)
def eci_to_body_quaternion(quaternion_body_eci):
    """
    Compute the inverse DCM from a body-to-ECI quaternion.

    Despite the function name, the implemented operation does not return
    a quaternion. It converts the input quaternion into a 3x3 body-to-ECI
    DCM and returns its transpose, corresponding to the ECI-to-body DCM.

    Parameters
    ----------
    quaternion_body_eci : numpy.ndarray, shape (4, 1)
        Quaternion representing the body-to-ECI rotation, using the
        scalar-first convention ``[w, x, y, z]``.

    Returns
    -------
    numpy.ndarray
        A 3x3 dimensionless DCM representing the ECI-to-body rotation:

        .. math::

            \\mathbf{C}_{B \\leftarrow ECI}
            =
            \\mathbf{C}_{ECI \\leftarrow B}^{T}

    Notes
    -----
    The function name is therefore potentially misleading: the returned
    quantity is a DCM, not a quaternion.

    See Also
    --------
    body_to_eci_quaternion : Convert a body-to-ECI quaternion to a DCM.
    """
    dcm_eci_from_body = body_to_eci_quaternion(
        quaternion_body_eci,
    )

    dcm_body_from_eci = np.transpose(dcm_eci_from_body)

    return dcm_body_from_eci


def eci_to_nav(latitude_rad, longitude_rad, azimuth_rad):
    """
    Compute the ECI-to-NAV direction cosine matrix.

    The NAV frame is parameterized by latitude, longitude, and azimuth.
    The exact axis definitions are those encoded by the implemented
    rotation matrix.

    Parameters
    ----------
    latitude_rad : float
        Latitude used to define the local navigation frame, in radians.
    longitude_rad : float
        Longitude used to define the local navigation frame, in radians.
    azimuth_rad : float
        Azimuth angle defining the local NAV orientation about the
        local vertical, in radians.

    Returns
    -------
    numpy.ndarray
        A 3x3 dimensionless DCM that transforms vectors from ECI
        coordinates to the NAV frame:

        .. math::

            \\mathbf{v}_{NAV}
            =
            \\mathbf{C}_{NAV \\leftarrow ECI}
            \\mathbf{v}_{ECI}

    Notes
    -----
    The function directly implements the NAV-frame orientation from
    ``latitude_rad``, ``longitude_rad``, and ``azimuth_rad``.

    The DCM is dimensionless because it represents a coordinate rotation.

    See Also
    --------
    nav_to_eci : Compute the inverse NAV-to-ECI transformation.
    """
    sine_latitude = sin(latitude_rad)
    cosine_latitude = cos(latitude_rad)
    sine_longitude = sin(longitude_rad)
    cosine_longitude = cos(longitude_rad)
    sine_azimuth = sin(azimuth_rad)
    cosine_azimuth = cos(azimuth_rad)

    dcm_nav_from_eci = np.array(
        [
            [
                cosine_latitude * cosine_longitude,
                cosine_latitude * sine_longitude,
                sine_latitude,
            ],
            [
                sine_latitude * cosine_longitude * sine_azimuth
                - sine_longitude * cosine_azimuth,
                cosine_longitude * cosine_azimuth
                + sine_latitude * sine_longitude * sine_azimuth,
                -cosine_latitude * sine_azimuth,
            ],
            [
                -sine_longitude * sine_azimuth
                - sine_latitude * cosine_longitude * cosine_azimuth,
                cosine_longitude * sine_azimuth
                - sine_latitude * sine_longitude * cosine_azimuth,
                cosine_latitude * cosine_azimuth,
            ],
        ],
        dtype=np.float64,
    )

    return dcm_nav_from_eci


def nav_to_eci(latitude_rad, longitude_rad, azimuth_rad):
    """
    Compute the NAV-to-ECI direction cosine matrix.

    This transformation is the inverse of ``eci_to_nav``.

    Parameters
    ----------
    latitude_rad : float
        Latitude used to define the local navigation frame, in radians.
    longitude_rad : float
        Longitude used to define the local navigation frame, in radians.
    azimuth_rad : float
        Azimuth angle defining the local NAV orientation, in radians.

    Returns
    -------
    numpy.ndarray
        A 3x3 dimensionless DCM that transforms vectors from NAV
        coordinates to ECI coordinates:

        .. math::

            \\mathbf{v}_{ECI}
            =
            \\mathbf{C}_{ECI \\leftarrow NAV}
            \\mathbf{v}_{NAV}

    Notes
    -----
    The inverse transformation is obtained from:

    .. math::

        \\mathbf{C}_{ECI \\leftarrow NAV}
        =
        \\mathbf{C}_{NAV \\leftarrow ECI}^{T}

    See Also
    --------
    eci_to_nav : Compute the inverse ECI-to-NAV transformation.
    """
    dcm_nav_from_eci = eci_to_nav(
        latitude_rad,
        longitude_rad,
        azimuth_rad,
    )

    dcm_eci_from_nav = np.transpose(dcm_nav_from_eci)

    return dcm_eci_from_nav


def polar_to_cartesian(
    magnitude,
    azimuth_rad,
    elevation_rad,
):
    """
    Convert spherical-style polar coordinates to Cartesian coordinates.

    Parameters
    ----------
    magnitude : float
        Radial magnitude or distance. The unit is inherited by the
        returned Cartesian coordinates.
    azimuth_rad : float
        Azimuth angle measured in the XY plane, in radians.
    elevation_rad : float
        Elevation angle measured from the XY plane, in radians.

    Returns
    -------
    numpy.ndarray
        A 3x1 Cartesian coordinate vector:

        .. math::

            \\mathbf{r} =
            \\begin{bmatrix}
            x \\\\
            y \\\\
            z
            \\end{bmatrix}

        with the same length unit as ``magnitude``.

    Notes
    -----
    The implemented convention is:

    .. math::

        x = r\\cos(e)\\cos(a)

    .. math::

        y = r\\cos(e)\\sin(a)

    .. math::

        z = -r\\sin(e)

    Therefore, positive elevation produces a negative Cartesian Z
    component. This sign convention should be preserved consistently
    with the frame definition used by the caller.

    See Also
    --------
    """
    x_cartesian = (
        magnitude
        * cos(elevation_rad)
        * cos(azimuth_rad)
    )

    y_cartesian = (
        magnitude
        * cos(elevation_rad)
        * sin(azimuth_rad)
    )

    z_cartesian = (
        -magnitude
        * sin(elevation_rad)
    )

    return np.array(
        [
            [x_cartesian],
            [y_cartesian],
            [z_cartesian],
        ],
        dtype=np.float64,
    )


@numba.njit(cache=True)
def ecef_to_geodetic(
    x_ecef_m,
    y_ecef_m,
    z_ecef_m,
):
    """
    Convert ECEF Cartesian coordinates to WGS-84 geodetic coordinates.

    The conversion uses the algorithm described by Osen (2017).

    Parameters
    ----------
    x_ecef_m : float
        ECEF X-coordinate, in meters.
    y_ecef_m : float
        ECEF Y-coordinate, in meters.
    z_ecef_m : float
        ECEF Z-coordinate, in meters.

    Returns
    -------
    latitude_rad : float
        Geodetic latitude, in radians.
    longitude_rad : float
        Geodetic longitude, in radians.
    altitude_m : float
        Geodetic altitude above the WGS-84 reference ellipsoid, in meters.

    Notes
    -----
    If the input does not satisfy the validity conditions required by
    the implemented algorithm, the function returns:

    .. code-block:: python

        (np.nan, np.nan, np.nan)

    The implementation uses WGS-84 ellipsoid constants expressed in
    SI-compatible units.

    References
    ----------
    Osen, K. (2017), *Accurate Conversion of Earth-Fixed Earth-Centered
    Coordinates to Geodetic Coordinates*.
    """
    wgs84_reciprocal_a_squared = 2.45817225764733181057e-14
    wgs84_ellipsoid_parameter = 3.34718999507065852867e-3
    wgs84_ellipsoid_parameter_squared = 1.12036808631011150655e-5
    wgs84_one_minus_eccentricity_squared = 9.93305620009858682943e-1
    wgs84_one_minus_eccentricity_squared_over_b_m = (
        1.56259921876129741211e-7
    )
    wgs84_minimum_h = 2.25010182030430273673e-14
    reciprocal_cuberoot_two = 7.93700525984099737380e-1

    transverse_radius_squared_m2 = (
        x_ecef_m**2 + y_ecef_m**2
    )

    normalized_equatorial_radius_squared = (
        transverse_radius_squared_m2
        * wgs84_reciprocal_a_squared
    )

    normalized_polar_radius_squared = (
        z_ecef_m
        * wgs84_one_minus_eccentricity_squared_over_b_m
    )**2

    parameter_p = (
        normalized_equatorial_radius_squared
        + normalized_polar_radius_squared
        - 4.0 * wgs84_ellipsoid_parameter_squared
    ) / 6.0

    parameter_g = (
        normalized_equatorial_radius_squared
        * normalized_polar_radius_squared
        * wgs84_ellipsoid_parameter_squared
    )

    parameter_h = 2.0 * parameter_p**3 + parameter_g

    if parameter_h < wgs84_minimum_h:
        return np.nan, np.nan, np.nan

    parameter_c = (
        (
            parameter_h
            + parameter_g
            + 2.0 * sqrt(parameter_h * parameter_g)
        )**(1.0 / 3.0)
        * reciprocal_cuberoot_two
    )

    parameter_i = -(
        2.0 * wgs84_ellipsoid_parameter_squared
        + normalized_equatorial_radius_squared
        + normalized_polar_radius_squared
    ) / 2.0

    parameter_p_squared = parameter_p**2

    parameter_beta = (
        parameter_i / 3.0
        - parameter_c
        - parameter_p_squared / parameter_c
    )

    parameter_k = (
        wgs84_ellipsoid_parameter_squared
        * (
            wgs84_ellipsoid_parameter_squared
            - normalized_equatorial_radius_squared
            - normalized_polar_radius_squared
        )
    )

    parameter_t = (
        sqrt(
            sqrt(parameter_beta**2 - parameter_k)
            - (parameter_beta + parameter_i) / 2.0
        )
        - copysign(1.0, normalized_equatorial_radius_squared
                   - normalized_polar_radius_squared)
        * sqrt(abs((parameter_beta - parameter_i) / 2.0))
    )

    polynomial_value = (
        parameter_t**4
        + 2.0 * parameter_i * parameter_t**2
        + 2.0 * wgs84_ellipsoid_parameter
        * (
            normalized_equatorial_radius_squared
            - normalized_polar_radius_squared
        )
        * parameter_t
        + parameter_k
    )

    polynomial_derivative = (
        4.0 * parameter_t**3
        + 4.0 * parameter_i * parameter_t
        + 2.0 * wgs84_ellipsoid_parameter
        * (
            normalized_equatorial_radius_squared
            - normalized_polar_radius_squared
        )
    )

    correction_t = -polynomial_value / polynomial_derivative

    auxiliary_u = (
        parameter_t
        + correction_t
        + wgs84_ellipsoid_parameter
    )

    auxiliary_v = (
        parameter_t
        + correction_t
        - wgs84_ellipsoid_parameter
    )

    transverse_radius_m = sqrt(transverse_radius_squared_m2)

    correction_transverse_m = (
        transverse_radius_m
        * (1.0 - 1.0 / auxiliary_u)
    )

    correction_z_m = (
        z_ecef_m
        * (
            1.0
            - wgs84_one_minus_eccentricity_squared / auxiliary_v
        )
    )

    latitude_rad = atan2(
        z_ecef_m * auxiliary_u,
        transverse_radius_m * auxiliary_v,
    )

    longitude_rad = atan2(
        y_ecef_m,
        x_ecef_m,
    )

    altitude_m = (
        copysign(1.0, auxiliary_u - 1.0)
        * sqrt(
            correction_transverse_m**2
            + correction_z_m**2
        )
    )

    return latitude_rad, longitude_rad, altitude_m


@numba.njit(cache=True)
def geodetic_to_ecef(
    latitude_rad,
    longitude_rad,
    altitude_m,
):
    """
    Convert WGS-84 geodetic coordinates to ECEF Cartesian coordinates.

    Parameters
    ----------
    latitude_rad : float
        Geodetic latitude, in radians.
    longitude_rad : float
        Geodetic longitude, in radians.
    altitude_m : float
        Geodetic altitude above the WGS-84 reference ellipsoid,
        in meters.

    Returns
    -------
    x_ecef_m : float
        ECEF X-coordinate, in meters.
    y_ecef_m : float
        ECEF Y-coordinate, in meters.
    z_ecef_m : float
        ECEF Z-coordinate, in meters.

    Notes
    -----
    The prime-vertical radius of curvature is computed as:

    .. math::

        N =
        \\frac{a}
        {\\sqrt{1-e^2\\sin^2\\phi}}

    where :math:`a` is the WGS-84 semi-major axis,
    :math:`e` is eccentricity, and :math:`\\phi` is geodetic latitude.

    The Cartesian coordinates are then:

    .. math::

        x = (N+h)\\cos\\phi\\cos\\lambda

    .. math::

        y = (N+h)\\cos\\phi\\sin\\lambda

    .. math::

        z = \\left[N(1-e^2)+h\\right]\\sin\\phi

    where :math:`h` is altitude and :math:`\\lambda` is longitude.

    References
    ----------
    Osen, K. (2017), *Accurate Conversion of Earth-Fixed Earth-Centered
    Coordinates to Geodetic Coordinates*.
    """
    wgs84_one_minus_eccentricity_squared = 9.93305620009858682943e-1
    wgs84_a_squared_over_c_m = 7.79540464078689228919e7
    wgs84_b_squared_over_c_squared = 1.48379031586596594555e2

    prime_vertical_radius_m = (
        wgs84_a_squared_over_c_m
        / sqrt(
            cos(latitude_rad)**2
            + wgs84_b_squared_over_c_squared
        )
    )

    projected_distance_m = (
        prime_vertical_radius_m + altitude_m
    ) * cos(latitude_rad)

    x_ecef_m = (
        projected_distance_m
        * cos(longitude_rad)
    )

    y_ecef_m = (
        projected_distance_m
        * sin(longitude_rad)
    )

    z_ecef_m = (
        (
            prime_vertical_radius_m
            * wgs84_one_minus_eccentricity_squared
            + altitude_m
        )
        * sin(latitude_rad)
    )

    return x_ecef_m, y_ecef_m, z_ecef_m


@numba.njit(cache=True)
def eci_to_geodetic(
    x_eci_m,
    y_eci_m,
    z_eci_m,
    time_s,
):
    """
    Convert ECI Cartesian coordinates to geodetic coordinates.

    The ECI position is first rotated into ECEF coordinates using the
    Earth rotation angle associated with ``time_s`` and is then converted
    from ECEF Cartesian coordinates to WGS-84 geodetic coordinates.

    Parameters
    ----------
    x_eci_m : float
        ECI X-coordinate, in meters.
    y_eci_m : float
        ECI Y-coordinate, in meters.
    z_eci_m : float
        ECI Z-coordinate, in meters.
    time_s : float
        Time from the ECI/ECEF reference epoch, in seconds.

    Returns
    -------
    latitude_rad : float
        Geodetic latitude, in radians.
    longitude_rad : float
        Geodetic longitude, in radians.
    altitude_m : float
        Geodetic altitude above the WGS-84 ellipsoid, in meters.

    Notes
    -----
    The implemented Earth rotation angle is:

    .. math::

        \\theta = \\omega_E t

    where ``earth_rate__rad_s`` is the Earth rotation rate.

    The ECI-to-ECEF transformation therefore assumes zero Earth rotation
    angle at ``time_s = 0``.
    """
    dcm_ecef_from_eci = eci_to_ecef(
        earth_rate__rad_s * time_s,
    )

    position_eci_m = np.array(
        [
            [x_eci_m],
            [y_eci_m],
            [z_eci_m],
        ],
        dtype=np.float64,
    )

    position_ecef_m = dcm_ecef_from_eci @ position_eci_m

    latitude_rad, longitude_rad, altitude_m = ecef_to_geodetic(
        position_ecef_m[0, 0],
        position_ecef_m[1, 0],
        position_ecef_m[2, 0],
    )

    return latitude_rad, longitude_rad, altitude_m


@numba.njit(cache=True)
def geodetic_to_eci(
    latitude_rad,
    longitude_rad,
    altitude_m,
    time_s,
):
    """
    Convert geodetic coordinates to ECI Cartesian coordinates.

    The geodetic position is first converted to ECEF coordinates and
    then rotated from ECEF to ECI using the Earth rotation angle
    associated with ``time_s``.

    Parameters
    ----------
    latitude_rad : float
        Geodetic latitude, in radians.
    longitude_rad : float
        Geodetic longitude, in radians.
    altitude_m : float
        Geodetic altitude above the WGS-84 ellipsoid, in meters.
    time_s : float
        Time from the ECI/ECEF reference epoch, in seconds.

    Returns
    -------
    x_eci_m : float
        ECI X-coordinate, in meters.
    y_eci_m : float
        ECI Y-coordinate, in meters.
    z_eci_m : float
        ECI Z-coordinate, in meters.

    Notes
    -----
    The implemented Earth rotation angle is:

    .. math::

        \\theta = \\omega_E t

    where ``earth_rate__rad_s`` is the Earth rotation rate.

    The transformation assumes zero Earth rotation angle at
    ``time_s = 0``.
    """
    x_ecef_m, y_ecef_m, z_ecef_m = geodetic_to_ecef(
        latitude_rad,
        longitude_rad,
        altitude_m,
    )

    dcm_eci_from_ecef = ecef_to_eci(
        earth_rate__rad_s * time_s,
    )

    position_ecef_m = np.array(
        [
            [x_ecef_m],
            [y_ecef_m],
            [z_ecef_m],
        ],
        dtype=np.float64,
    )

    position_eci_m = dcm_eci_from_ecef @ position_ecef_m

    return (
        position_eci_m[0, 0],
        position_eci_m[1, 0],
        position_eci_m[2, 0],
    )


def geodetic_to_geocentric_latitude(
    geodetic_latitude_rad,
    altitude_m,
):
    """
    Convert geodetic latitude to geocentric latitude.

    Geodetic latitude is measured between the equatorial plane and the
    normal to the reference ellipsoid. Geocentric latitude is measured
    between the equatorial plane and the line joining the Earth's center
    to the point of interest.

    Parameters
    ----------
    geodetic_latitude_rad : float
        Geodetic latitude, in radians.
    altitude_m : float
        Altitude above the reference ellipsoid, in meters.

    Returns
    -------
    float
        Geocentric latitude, in radians.

    Notes
    -----
    The conversion is performed using the WGS-84 ellipsoid parameters.

    Unlike the geodetic latitude, geocentric latitude depends on
    altitude because the radial line from the Earth center changes with
    distance from the ellipsoid.
    """
    wgs84_one_minus_eccentricity_squared = (
        9.93305620009858682943e-1
    )

    prime_vertical_radius_m = (
        a
        / sqrt(
            1.0
            - e**2
            * sin(geodetic_latitude_rad)**2
        )
    )

    projected_radius_m = (
        prime_vertical_radius_m + altitude_m
    ) * cos(geodetic_latitude_rad)

    z_position_m = (
        (
            prime_vertical_radius_m
            * wgs84_one_minus_eccentricity_squared
            + altitude_m
        )
        * sin(geodetic_latitude_rad)
    )

    return atan2(
        z_position_m,
        projected_radius_m,
    )


def geocentric_to_geodetic_latitude(
    geocentric_latitude_rad,
    x_position_m,
    y_position_m,
    z_position_m,
):
    """
    Convert geocentric latitude to geodetic latitude.

    The conversion uses the WGS-84 ellipsoid and the supplied Cartesian
    position to reconstruct the radial distance corresponding to the
    given geocentric latitude.

    Parameters
    ----------
    geocentric_latitude_rad : float
        Geocentric latitude, in radians.
    x_position_m : float
        Cartesian X-coordinate of the point, in meters.
    y_position_m : float
        Cartesian Y-coordinate of the point, in meters.
    z_position_m : float
        Cartesian Z-coordinate of the point, in meters.

    Returns
    -------
    float
        Geodetic latitude, in radians.

    Notes
    -----
    The Cartesian coordinates are assumed to belong to the same
    Earth-centered Cartesian frame used to define the geocentric
    latitude.

    The WGS-84 ellipsoid parameters ``a``, ``e``, and ``f`` are taken
    from the surrounding module.
    """
    wgs84_one_minus_eccentricity_squared = (
        9.93305620009858682943e-1
    )

    geocentric_radius_m = sqrt(
        x_position_m**2
        + y_position_m**2
        + z_position_m**2
    )

    horizontal_radius_m = (
        geocentric_radius_m
        * cos(geocentric_latitude_rad)
    )

    vertical_radius_m = (
        geocentric_radius_m
        * sin(geocentric_latitude_rad)
    )

    semi_minor_axis_m = a * (1.0 - f)

    second_eccentricity_squared = (
        e**2
        / wgs84_one_minus_eccentricity_squared
    )

    reduced_latitude_rad = atan2(
        (1.0 - f)
        * sin(geocentric_latitude_rad),
        cos(geocentric_latitude_rad),
    )

    geodetic_latitude_rad = atan2(
        vertical_radius_m
        + semi_minor_axis_m
        * second_eccentricity_squared
        * sin(reduced_latitude_rad)**3,
        horizontal_radius_m
        - a
        * e**2
        * cos(reduced_latitude_rad)**3,
    )

    return geodetic_latitude_rad


@numba.njit(cache=True)
def euler_to_quaternion(
    roll_rad,
    pitch_rad,
    yaw_rad,
):
    """
    Convert 3-2-1 Euler angles to a scalar-first quaternion.

    Parameters
    ----------
    roll_rad : float
        Roll angle, in radians.
    pitch_rad : float
        Pitch angle, in radians.
    yaw_rad : float
        Yaw angle, in radians.

    Returns
    -------
    numpy.ndarray
        A 4x1 dimensionless quaternion in scalar-first
        ``[w, x, y, z]`` format.

    Notes
    -----
    The implemented quaternion corresponds to the 3-2-1 / Z-Y-X
    Euler-angle convention:

    .. math::

        C = R_z(\\psi) R_y(\\theta) R_x(\\phi)

    where :math:`\\phi`, :math:`\\theta`, and :math:`\\psi` are roll,
    pitch, and yaw, respectively.

    See Also
    --------
    quaternion_to_euler_321 : Perform the inverse conversion.
    quaternion_normalize : Normalize the returned quaternion.
    """
    half_roll_rad = roll_rad / 2.0
    half_pitch_rad = pitch_rad / 2.0
    half_yaw_rad = yaw_rad / 2.0

    return np.array(
        [
            [
                cos(half_roll_rad)
                * cos(half_pitch_rad)
                * cos(half_yaw_rad)
                + sin(half_roll_rad)
                * sin(half_pitch_rad)
                * sin(half_yaw_rad)
            ],
            [
                sin(half_roll_rad)
                * cos(half_pitch_rad)
                * cos(half_yaw_rad)
                - cos(half_roll_rad)
                * sin(half_pitch_rad)
                * sin(half_yaw_rad)
            ],
            [
                cos(half_roll_rad)
                * sin(half_pitch_rad)
                * cos(half_yaw_rad)
                + sin(half_roll_rad)
                * cos(half_pitch_rad)
                * sin(half_yaw_rad)
            ],
            [
                cos(half_roll_rad)
                * cos(half_pitch_rad)
                * sin(half_yaw_rad)
                - sin(half_roll_rad)
                * sin(half_pitch_rad)
                * cos(half_yaw_rad)
            ],
        ],
        dtype=np.float64,
    )


def ned_to_body(
    roll_ned_rad,
    pitch_ned_rad,
    yaw_ned_rad,
):
    """
    Compute the NED-to-body direction cosine matrix from 3-2-1 Euler angles.

    Parameters
    ----------
    roll_ned_rad : float or array-like
        Roll angle relative to NED, in radians.
    pitch_ned_rad : float or array-like
        Pitch angle relative to NED, in radians.
    yaw_ned_rad : float or array-like
        Yaw angle relative to NED, in radians.

    Returns
    -------
    numpy.ndarray
        A 3x3 dimensionless DCM that transforms a vector from NED
        coordinates to body coordinates.

        .. math::

            \\mathbf{v}_B
            =
            \\mathbf{C}_{B \\leftarrow NED}
            \\mathbf{v}_{NED}

    Notes
    -----
    Scalar values as well as one-element lists or NumPy arrays are
    accepted by the current implementation.

    See Also
    --------
    body_to_ned : Compute the inverse body-to-NED DCM.
    """
    if isinstance(roll_ned_rad, (list, np.ndarray)):
        roll_ned_rad = roll_ned_rad[0]

    if isinstance(pitch_ned_rad, (list, np.ndarray)):
        pitch_ned_rad = pitch_ned_rad[0]

    if isinstance(yaw_ned_rad, (list, np.ndarray)):
        yaw_ned_rad = yaw_ned_rad[0]

    sine_roll = sin(roll_ned_rad)
    cosine_roll = cos(roll_ned_rad)
    sine_pitch = sin(pitch_ned_rad)
    cosine_pitch = cos(pitch_ned_rad)
    sine_yaw = sin(yaw_ned_rad)
    cosine_yaw = cos(yaw_ned_rad)

    c11 = cosine_pitch * cosine_yaw
    c12 = cosine_pitch * sine_yaw
    c13 = -sine_pitch

    c21 = (
        sine_roll * sine_pitch * cosine_yaw
        - cosine_roll * sine_yaw
    )
    c22 = (
        sine_roll * sine_pitch * sine_yaw
        + cosine_roll * cosine_yaw
    )
    c23 = sine_roll * cosine_pitch

    c31 = (
        cosine_roll * sine_pitch * cosine_yaw
        + sine_roll * sine_yaw
    )
    c32 = (
        cosine_roll * sine_pitch * sine_yaw
        - sine_roll * cosine_yaw
    )
    c33 = cosine_roll * cosine_pitch

    return np.array(
        [
            [c11, c12, c13],
            [c21, c22, c23],
            [c31, c32, c33],
        ],
        dtype=np.float64,
    )


@numba.njit(cache=True)
def body_to_ned(
    roll_ned_rad,
    pitch_ned_rad,
    yaw_ned_rad,
):
    """
    Compute the body-to-NED direction cosine matrix.

    Parameters
    ----------
    roll_ned_rad : float
        Roll angle relative to NED, in radians.
    pitch_ned_rad : float
        Pitch angle relative to NED, in radians.
    yaw_ned_rad : float
        Yaw angle relative to NED, in radians.

    Returns
    -------
    numpy.ndarray
        A 3x3 dimensionless DCM that transforms a vector from body
        coordinates to NED coordinates.

    Notes
    -----
    The inverse transformation is obtained by transposing the
    NED-to-body DCM:

    .. math::

        \\mathbf{C}_{NED \\leftarrow B}
        =
        \\mathbf{C}_{B \\leftarrow NED}^{T}

    See Also
    --------
    ned_to_body : Compute the inverse NED-to-body DCM.
    """
    dcm_body_from_ned = ned_to_body(
        roll_ned_rad,
        pitch_ned_rad,
        yaw_ned_rad,
    )

    dcm_ned_from_body = np.transpose(dcm_body_from_ned)

    return dcm_ned_from_body


@numba.njit(cache=True)
def quaternion_multiply(
    quaternion_a_from_b,
    quaternion_b_from_c,
):
    """
    Compose two frame transformations represented by quaternions.

    Parameters
    ----------
    quaternion_a_from_b : numpy.ndarray, shape (4, 1)
        Quaternion representing the transformation from frame B to
        frame A, using the scalar-first ``[w, x, y, z]`` convention.
    quaternion_b_from_c : numpy.ndarray, shape (4, 1)
        Quaternion representing the transformation from frame C to
        frame B, using the scalar-first ``[w, x, y, z]`` convention.

    Returns
    -------
    numpy.ndarray
        Quaternion representing the composite transformation from
        frame C to frame A:

        .. math::

            q_{A\\leftarrow C}
            =
            q_{A\\leftarrow B}
            \\otimes
            q_{B\\leftarrow C}

        The result uses ``[w, x, y, z]`` ordering.

    Notes
    -----
    Quaternion multiplication is not commutative. The order of the
    arguments therefore matters.

    The input quaternions are not normalized inside this function.
    """
    w_a_b, x_a_b, y_a_b, z_a_b = quaternion_a_from_b.flatten()
    w_b_c, x_b_c, y_b_c, z_b_c = quaternion_b_from_c.flatten()

    return np.array(
        [
            [
                w_a_b * w_b_c
                - x_a_b * x_b_c
                - y_a_b * y_b_c
                - z_a_b * z_b_c
            ],
            [
                w_a_b * x_b_c
                + x_a_b * w_b_c
                + y_a_b * z_b_c
                - z_a_b * y_b_c
            ],
            [
                w_a_b * y_b_c
                - x_a_b * z_b_c
                + y_a_b * w_b_c
                + z_a_b * x_b_c
            ],
            [
                w_a_b * z_b_c
                + x_a_b * y_b_c
                - y_a_b * x_b_c
                + z_a_b * w_b_c
            ],
        ],
        dtype=np.float64,
    )


@numba.njit(cache=True)
def quaternion_normalize(quaternion):
    """
    Normalize a quaternion to unit magnitude.

    Parameters
    ----------
    quaternion : numpy.ndarray, shape (4, 1)
        Quaternion to normalize. The quaternion is dimensionless.

    Returns
    -------
    numpy.ndarray
        Unit-norm quaternion with the same shape as the input.

    Notes
    -----
    The normalization is computed as:

    .. math::

        q_{normalized} = \\frac{q}{\\lVert q \\rVert}
    """
    return quaternion / np.linalg.norm(quaternion)


@numba.njit(cache=True)
def quaternion_conjugate(quaternion):
    """
    Compute the conjugate of a scalar-first quaternion.

    Parameters
    ----------
    quaternion : numpy.ndarray, shape (4, 1)
        Quaternion in ``[w, x, y, z]`` format.

    Returns
    -------
    numpy.ndarray
        Quaternion conjugate:

        .. math::

            q^* =
            \\begin{bmatrix}
            w & -x & -y & -z
            \\end{bmatrix}^{T}
    """
    return np.array(
        [
            [quaternion[0, 0]],
            [-quaternion[1, 0]],
            [-quaternion[2, 0]],
            [-quaternion[3, 0]],
        ],
        dtype=np.float64,
    )


@numba.njit(cache=True)
def axis_angle_quaternion(
    angle_rad,
    axis,
):
    """
    Create a scalar-first quaternion for a principal-axis rotation.

    Parameters
    ----------
    angle_rad : float
        Rotation angle, in radians.
    axis : int
        Principal rotation axis:

        * ``0``: X-axis
        * ``1``: Y-axis
        * ``2``: Z-axis

    Returns
    -------
    numpy.ndarray
        A 4x1 dimensionless quaternion in ``[w, x, y, z]`` format.

    Notes
    -----
    For a rotation of angle :math:`\\theta` about a unit axis
    :math:`\\hat{u}`, the quaternion is:

    .. math::

        q =
        \\begin{bmatrix}
        \\cos(\\theta/2) \\\\
        \\hat{u}_x\\sin(\\theta/2) \\\\
        \\hat{u}_y\\sin(\\theta/2) \\\\
        \\hat{u}_z\\sin(\\theta/2)
        \\end{bmatrix}

    The current implementation supports only the three principal
    coordinate axes.
    """
    half_angle_rad = 0.5 * angle_rad

    quaternion = np.zeros(
        (4, 1),
        dtype=np.float64,
    )

    quaternion[0, 0] = np.float64(
        cos(half_angle_rad)
    )

    quaternion[axis + 1, 0] = np.float64(
        sin(half_angle_rad)
    )

    return quaternion


@numba.njit(cache=True)
def eci_to_ecef_quaternion(time_s):
    """
    Compute the ECI-to-ECEF rotation quaternion.

    Parameters
    ----------
    time_s : float
        Time from the ECI/ECEF reference epoch, in seconds.

    Returns
    -------
    numpy.ndarray
        A normalized 4x1 quaternion in ``[w, x, y, z]`` format
        representing the ECI-to-ECEF rotation.

    Notes
    -----
    The implemented Earth rotation angle is:

    .. math::

        \\theta = \\omega_E t

    where ``earth_rate__rad_s`` is the Earth rotation rate.

    The corresponding ECI-to-ECEF rotation is a rotation of
    ``-theta`` about the Z-axis:

    .. math::

        q_{ECEF\\leftarrow ECI}
        =
        q_z(-\\theta)

    The implementation therefore assumes zero Earth rotation angle
    at ``time_s = 0``.
    """
    earth_rotation_angle_rad = (
        earth_rate__rad_s * time_s
    )

    return quaternion_normalize(
        axis_angle_quaternion(
            -earth_rotation_angle_rad,
            axis=2,
        )
    )


@numba.njit(cache=True)
def ecef_to_eci_quaternion(time_s):
    """
    Compute the ECEF-to-ECI rotation quaternion.

    Parameters
    ----------
    time_s : float
        Time from the ECI/ECEF reference epoch, in seconds.

    Returns
    -------
    numpy.ndarray
        A normalized 4x1 quaternion in ``[w, x, y, z]`` format
        representing the ECEF-to-ECI rotation.

    Notes
    -----
    This is the inverse of ``eci_to_ecef_quaternion`` and is obtained
    through quaternion conjugation:

    .. math::

        q_{ECI\\leftarrow ECEF}
        =
        q_{ECEF\\leftarrow ECI}^{*}

    The corresponding rotation angle is:

    .. math::

        \\theta = \\omega_E t
    """
    return quaternion_conjugate(
        eci_to_ecef_quaternion(time_s)
    )


@numba.njit(cache=True)
def dcm_to_quaternion(dcm):
    """
    Convert a 3x3 DCM to a scalar-first quaternion.

    Parameters
    ----------
    dcm : numpy.ndarray, shape (3, 3)
        Direction cosine matrix satisfying the convention:

        .. math::

            \\mathbf{v}_A
            =
            \\mathbf{C}_{A \\leftarrow B}
            \\mathbf{v}_B

        The resulting quaternion therefore represents the
        B-to-A transformation.

    Returns
    -------
    numpy.ndarray
        A normalized 4x1 quaternion in ``[w, x, y, z]`` format.

    Notes
    -----
    The implementation uses a trace-based branch followed by
    diagonal-element branches to avoid numerical issues when the
    DCM trace is small.

    The resulting quaternion is normalized before being returned.

    Since quaternions have a sign ambiguity, ``q`` and ``-q`` represent
    the same rotation.
    """
    quaternion = np.zeros(
        (4, 1),
        dtype=np.float64,
    )

    dcm_trace = (
        dcm[0, 0]
        + dcm[1, 1]
        + dcm[2, 2]
    )

    if dcm_trace > 0.0:
        scale_factor = 2.0 * np.sqrt(
            dcm_trace + 1.0
        )

        quaternion[0, 0] = 0.25 * scale_factor
        quaternion[1, 0] = (
            dcm[2, 1] - dcm[1, 2]
        ) / scale_factor
        quaternion[2, 0] = (
            dcm[0, 2] - dcm[2, 0]
        ) / scale_factor
        quaternion[3, 0] = (
            dcm[1, 0] - dcm[0, 1]
        ) / scale_factor

    elif dcm[0, 0] > dcm[1, 1] and dcm[0, 0] > dcm[2, 2]:
        scale_factor = 2.0 * np.sqrt(
            1.0
            + dcm[0, 0]
            - dcm[1, 1]
            - dcm[2, 2]
        )

        quaternion[0, 0] = (
            dcm[2, 1] - dcm[1, 2]
        ) / scale_factor
        quaternion[1, 0] = 0.25 * scale_factor
        quaternion[2, 0] = (
            dcm[0, 1] + dcm[1, 0]
        ) / scale_factor
        quaternion[3, 0] = (
            dcm[0, 2] + dcm[2, 0]
        ) / scale_factor

    elif dcm[1, 1] > dcm[2, 2]:
        scale_factor = 2.0 * np.sqrt(
            1.0
            + dcm[1, 1]
            - dcm[0, 0]
            - dcm[2, 2]
        )

        quaternion[0, 0] = (
            dcm[0, 2] - dcm[2, 0]
        ) / scale_factor
        quaternion[1, 0] = (
            dcm[0, 1] + dcm[1, 0]
        ) / scale_factor
        quaternion[2, 0] = 0.25 * scale_factor
        quaternion[3, 0] = (
            dcm[1, 2] + dcm[2, 1]
        ) / scale_factor

    else:
        scale_factor = 2.0 * np.sqrt(
            1.0
            + dcm[2, 2]
            - dcm[0, 0]
            - dcm[1, 1]
        )

        quaternion[0, 0] = (
            dcm[1, 0] - dcm[0, 1]
        ) / scale_factor
        quaternion[1, 0] = (
            dcm[0, 2] + dcm[2, 0]
        ) / scale_factor
        quaternion[2, 0] = (
            dcm[1, 2] + dcm[2, 1]
        ) / scale_factor
        quaternion[3, 0] = 0.25 * scale_factor

    return quaternion_normalize(quaternion)


@numba.njit(cache=True)
def ecef_to_ned_quaternion(
    latitude_rad,
    longitude_rad,
):
    """
    Compute the ECEF-to-NED rotation quaternion.

    Parameters
    ----------
    latitude_rad : float
        Latitude defining the local NED frame, in radians.
    longitude_rad : float
        Longitude defining the local NED frame, in radians.

    Returns
    -------
    numpy.ndarray
        A normalized 4x1 quaternion in ``[w, x, y, z]`` format
        representing the ECEF-to-NED rotation.

    Notes
    -----
    The quaternion is generated directly from the ECEF-to-NED DCM
    using ``dcm_to_quaternion``.

    See Also
    --------
    ecef_to_ned_matrix : Compute the corresponding DCM.
    ned_to_ecef_quaternion : Compute the inverse transformation.
    """
    dcm_ned_from_ecef = ecef_to_ned_matrix(
        latitude_rad,
        longitude_rad,
    )

    return dcm_to_quaternion(
        dcm_ned_from_ecef
    )


@numba.njit(cache=True)
def ned_to_ecef_quaternion(
    latitude_rad,
    longitude_rad,
):
    """
    Compute the NED-to-ECEF rotation quaternion.

    Parameters
    ----------
    latitude_rad : float
        Latitude defining the local NED frame, in radians.
    longitude_rad : float
        Longitude defining the local NED frame, in radians.

    Returns
    -------
    numpy.ndarray
        A normalized 4x1 quaternion in ``[w, x, y, z]`` format
        representing the NED-to-ECEF rotation.

    Notes
    -----
    The transformation is the inverse of the ECEF-to-NED rotation:

    .. math::

        q_{ECEF\\leftarrow NED}
        =
        q_{NED\\leftarrow ECEF}^{*}
    """
    return quaternion_normalize(
        quaternion_conjugate(
            ecef_to_ned_quaternion(
                latitude_rad,
                longitude_rad,
            )
        )
    )


@numba.njit(cache=True)
def eci_to_ned_quaternion(
    latitude_rad,
    longitude_rad,
    time_s,
):
    """
    Compute the ECI-to-NED rotation quaternion.

    Parameters
    ----------
    latitude_rad : float
        Latitude defining the local NED frame, in radians.
    longitude_rad : float
        Longitude defining the local NED frame, in radians.
    time_s : float
        Time from the ECI/ECEF reference epoch, in seconds.

    Returns
    -------
    numpy.ndarray
        A normalized 4x1 quaternion in ``[w, x, y, z]`` format
        representing the ECI-to-NED rotation.

    Notes
    -----
    The transformation is composed according to:

    .. math::

        q_{NED\\leftarrow ECI}
        =
        q_{NED\\leftarrow ECEF}
        \\otimes
        q_{ECEF\\leftarrow ECI}

    The quaternion multiplication order is therefore important.
    """
    quaternion_ned_from_ecef = ecef_to_ned_quaternion(
        latitude_rad,
        longitude_rad,
    )

    quaternion_ecef_from_eci = eci_to_ecef_quaternion(
        time_s,
    )

    return quaternion_normalize(
        quaternion_multiply(
            quaternion_ned_from_ecef,
            quaternion_ecef_from_eci,
        )
    )


@numba.njit(cache=True)
def ned_to_eci_quaternion(
    latitude_rad,
    longitude_rad,
    time_s,
):
    """
    Compute the NED-to-ECI rotation quaternion.

    Parameters
    ----------
    latitude_rad : float
        Latitude defining the local NED frame, in radians.
    longitude_rad : float
        Longitude defining the local NED frame, in radians.
    time_s : float
        Time from the ECI/ECEF reference epoch, in seconds.

    Returns
    -------
    numpy.ndarray
        A normalized 4x1 quaternion in ``[w, x, y, z]`` format
        representing the NED-to-ECI rotation.

    Notes
    -----
    The inverse transformation is obtained through quaternion
    conjugation:

    .. math::

        q_{ECI\\leftarrow NED}
        =
        q_{NED\\leftarrow ECI}^{*}
    """
    return quaternion_normalize(
        quaternion_conjugate(
            eci_to_ned_quaternion(
                latitude_rad,
                longitude_rad,
                time_s,
            )
        )
    )


@numba.njit(cache=True)
def quaternion_to_euler_321(quaternion_ned_from_body):
    """
    Convert a quaternion to 3-2-1 (ZYX) Euler angles.

    Parameters
    ----------
    quaternion_ned_from_body : numpy.ndarray, shape (4, 1)
        Quaternion representing the body-to-NED rotation:

        .. math::

            \\mathbf{v}_{NED}
            =
            \\mathbf{C}_{NED \\leftarrow B}
            \\mathbf{v}_B

        The quaternion uses the scalar-first ``[w, x, y, z]`` convention.

    Returns
    -------
    roll_rad : float
        Roll angle, in radians.
    pitch_rad : float
        Pitch angle, in radians.
    yaw_rad : float
        Yaw angle, in radians.

    Notes
    -----
    The Euler representation follows the 3-2-1 / ZYX convention:

    .. math::

        \\mathbf{C}_{NED \\leftarrow B}
        =
        R_z(\\psi)
        R_y(\\theta)
        R_x(\\phi)

    where:

    * :math:`\\phi` is roll,
    * :math:`\\theta` is pitch,
    * :math:`\\psi` is yaw.

    The pitch angle is constrained to the principal interval:

    .. math::

        -\\frac{\\pi}{2}
        \\leq \\theta
        \\leq \\frac{\\pi}{2}

    The implementation clips the intermediate sine of pitch to
    ``[-1, 1]`` to protect against floating-point round-off.
    """
    quaternion_unit = quaternion_normalize(
        quaternion_ned_from_body
    )

    quaternion_w = quaternion_unit[0, 0]
    quaternion_x = quaternion_unit[1, 0]
    quaternion_y = quaternion_unit[2, 0]
    quaternion_z = quaternion_unit[3, 0]

    roll_rad = atan2(
        2.0 * (
            quaternion_w * quaternion_x
            + quaternion_y * quaternion_z
        ),
        1.0
        - 2.0 * (
            quaternion_x**2
            + quaternion_y**2
        ),
    )

    sine_pitch = 2.0 * (
        quaternion_w * quaternion_y
        - quaternion_z * quaternion_x
    )

    if sine_pitch > 1.0:
        sine_pitch = 1.0
    elif sine_pitch < -1.0:
        sine_pitch = -1.0

    pitch_rad = asin(sine_pitch)

    yaw_rad = atan2(
        2.0 * (
            quaternion_w * quaternion_z
            + quaternion_x * quaternion_y
        ),
        1.0
        - 2.0 * (
            quaternion_y**2
            + quaternion_z**2
        ),
    )

    return roll_rad, pitch_rad, yaw_rad


@numba.njit(cache=True)
def body_to_ned_quaternion(
    quaternion_body_eci,
    latitude_rad,
    longitude_rad,
    time_s,
):
    """
    Transform a body-to-ECI quaternion into a body-to-NED quaternion.

    Parameters
    ----------
    quaternion_body_eci : numpy.ndarray, shape (4, 1)
        Quaternion representing the body-to-ECI rotation:

        .. math::

            q_{ECI\\leftarrow B}

        The quaternion uses the scalar-first ``[w, x, y, z]`` convention.
    latitude_rad : float
        Latitude defining the local NED frame, in radians.
    longitude_rad : float
        Longitude defining the local NED frame, in radians.
    time_s : float
        Time from the ECI/ECEF reference epoch, in seconds.

    Returns
    -------
    numpy.ndarray
        Normalized quaternion representing the body-to-NED rotation:

        .. math::

            q_{NED\\leftarrow B}

    Notes
    -----
    The transformation is composed as:

    .. math::

        q_{NED\\leftarrow B}
        =
        q_{NED\\leftarrow ECI}
        \\otimes
        q_{ECI\\leftarrow B}

    and therefore:

    .. math::

        q_{NED\\leftarrow B}
        =
        q_{NED\\leftarrow ECEF}
        \\otimes
        q_{ECEF\\leftarrow ECI}
        \\otimes
        q_{ECI\\leftarrow B}
    """
    quaternion_ned_from_eci = eci_to_ned_quaternion(
        latitude_rad,
        longitude_rad,
        time_s,
    )

    quaternion_eci_from_body = quaternion_normalize(
        quaternion_body_eci
    )

    quaternion_ned_from_body = quaternion_multiply(
        quaternion_ned_from_eci,
        quaternion_eci_from_body,
    )

    return quaternion_normalize(
        quaternion_ned_from_body
    )


@numba.njit(cache=True)
def ned_to_body_quaternion(
    quaternion_body_eci,
    latitude_rad,
    longitude_rad,
    time_s,
):
    """
    Compute the NED-to-body quaternion corresponding to a body ECI attitude.

    Parameters
    ----------
    quaternion_body_eci : numpy.ndarray, shape (4, 1)
        Quaternion representing the body-to-ECI rotation,
        ``q_{ECI\\leftarrow B}``, in ``[w, x, y, z]`` format.
    latitude_rad : float
        Latitude defining the local NED frame, in radians.
    longitude_rad : float
        Longitude defining the local NED frame, in radians.
    time_s : float
        Time from the ECI/ECEF reference epoch, in seconds.

    Returns
    -------
    numpy.ndarray
        Normalized quaternion representing the NED-to-body rotation:

        .. math::

            q_{B\\leftarrow NED}

    Notes
    -----
    The function first computes ``q_{NED<-B}`` and then takes its
    quaternion conjugate:

    .. math::

        q_{B\\leftarrow NED}
        =
        q_{NED\\leftarrow B}^{*}

    Note that the input remains the original body-to-ECI quaternion.
    """
    return quaternion_normalize(
        quaternion_conjugate(
            body_to_ned_quaternion(
                quaternion_body_eci,
                latitude_rad,
                longitude_rad,
                time_s,
            )
        )
    )


@numba.njit(cache=True)
def eci_to_ned_attitude(
    quaternion_body_eci,
    latitude_rad,
    longitude_rad,
    time_s,
):
    """
    Compute the body-to-NED quaternion components from an ECI attitude.

    Despite the function name ``eci_to_ned_attitude``, the implemented
    function does not return Euler angles. It returns the four
    components of the normalized body-to-NED quaternion.

    Parameters
    ----------
    quaternion_body_eci : numpy.ndarray, shape (4, 1)
        Quaternion representing the body-to-ECI rotation,
        ``q_{ECI\\leftarrow B}``, in scalar-first ``[w, x, y, z]`` format.
    latitude_rad : float
        Latitude defining the local NED frame, in radians.
    longitude_rad : float
        Longitude defining the local NED frame, in radians.
    time_s : float
        Time from the ECI/ECEF reference epoch, in seconds.

    Returns
    -------
    quaternion_w_ned_from_body : numpy.float64
        Scalar component of the normalized body-to-NED quaternion.
    quaternion_x_ned_from_body : numpy.float64
        X component of the normalized body-to-NED quaternion.
    quaternion_y_ned_from_body : numpy.float64
        Y component of the normalized body-to-NED quaternion.
    quaternion_z_ned_from_body : numpy.float64
        Z component of the normalized body-to-NED quaternion.

    Notes
    -----
    The returned components correspond to:

    .. math::

        q_{NED\\leftarrow B}
        =
        [w, x, y, z]^T
    """
    quaternion_ned_from_body = body_to_ned_quaternion(
        quaternion_body_eci,
        latitude_rad,
        longitude_rad,
        time_s,
    )

    return (
        np.float64(quaternion_ned_from_body[0, 0]),
        np.float64(quaternion_ned_from_body[1, 0]),
        np.float64(quaternion_ned_from_body[2, 0]),
        np.float64(quaternion_ned_from_body[3, 0]),
    )


@numba.njit(cache=True)
def skew_matrix(vector):
    """
    Construct the 3x3 skew-symmetric cross-product matrix of a vector.

    Parameters
    ----------
    vector : numpy.ndarray, shape (3, 1)
        Three-dimensional vector. The unit of the resulting matrix
        coefficients is the same as the unit of ``vector``.

    Returns
    -------
    numpy.ndarray
        A 3x3 skew-symmetric matrix:

        .. math::

            [\\mathbf{v}]_\\times =
            \\begin{bmatrix}
            0 & -v_z & v_y \\\\
            v_z & 0 & -v_x \\\\
            -v_y & v_x & 0
            \\end{bmatrix}

    Notes
    -----
    The matrix satisfies:

    .. math::

        [\\mathbf{v}]_\\times \\mathbf{u}
        =
        \\mathbf{v} \\times \\mathbf{u}

    for any three-dimensional vector :math:`\\mathbf{u}`.
    """
    return np.array(
        [
            [0.0, -vector[2, 0], vector[1, 0]],
            [vector[2, 0], 0.0, -vector[0, 0]],
            [-vector[1, 0], vector[0, 0], 0.0],
        ],
        dtype=np.float64,
    )


@numba.njit(cache=True)
def skew_matrix_angular_velocity(angular_velocity):
    """
    Construct the 4x4 skew-symmetric matrix associated with angular velocity.

    Parameters
    ----------
    angular_velocity : numpy.ndarray, shape (3, 1)
        Angular velocity vector, in radians per second.

    Returns
    -------
    numpy.ndarray
        A 4x4 skew-symmetric matrix with coefficients in radians per
        second.

    Notes
    -----
    The matrix is:

    .. math::

        \\mathbf{\\Omega}(\\boldsymbol{\\omega}) =
        \\begin{bmatrix}
        0 & \\omega_x & \\omega_y & \\omega_z \\\\
        -\\omega_x & 0 & -\\omega_z & \\omega_y \\\\
        -\\omega_y & \\omega_z & 0 & -\\omega_x \\\\
        -\\omega_z & -\\omega_y & \\omega_x & 0
        \\end{bmatrix}

    Its sign convention is specific to the quaternion propagation
    convention used by the surrounding attitude-dynamics implementation.
    """
    angular_velocity_x_rad_s = angular_velocity[0, 0]
    angular_velocity_y_rad_s = angular_velocity[1, 0]
    angular_velocity_z_rad_s = angular_velocity[2, 0]

    return np.array(
        [
            [
                0.0,
                angular_velocity_x_rad_s,
                angular_velocity_y_rad_s,
                angular_velocity_z_rad_s,
            ],
            [
                -angular_velocity_x_rad_s,
                0.0,
                -angular_velocity_z_rad_s,
                angular_velocity_y_rad_s,
            ],
            [
                -angular_velocity_y_rad_s,
                angular_velocity_z_rad_s,
                0.0,
                -angular_velocity_x_rad_s,
            ],
            [
                -angular_velocity_z_rad_s,
                -angular_velocity_y_rad_s,
                angular_velocity_x_rad_s,
                0.0,
            ],
        ],
        dtype=np.float64,
    )


def compute_downrange_crossrange(
    latitude_deg,
    longitude_deg,
    altitude_m,
    launching_ref: LaunchReference,
):
    """
    Compute downrange, crossrange, and total ground-plane displacement.

    The target geodetic position is converted to ECEF coordinates and
    then expressed as a local NED displacement from the launch reference.
    This displacement is subsequently rotated by the launch azimuth into
    downrange and crossrange coordinates.

    Parameters
    ----------
    latitude_deg : float
        Target geodetic latitude, in degrees.
    longitude_deg : float
        Target geodetic longitude, in degrees.
    altitude_m : float
        Target geodetic altitude above the reference ellipsoid, in meters.
    launching_ref : LaunchReference
        Launch reference containing at least:

        * ``azimuth``: launch azimuth, in radians.
        * ``lat``: reference latitude, in radians.
        * ``lon``: reference longitude, in radians.
        * ``alt``: reference altitude, in meters.

    Returns
    -------
    downrange_m : float
        Displacement along the launch azimuth, in meters.
    crossrange_m : float
        Lateral displacement perpendicular to the launch azimuth,
        in meters.
    displacement_m : float
        Magnitude of the horizontal displacement in the local
        downrange-crossrange plane, in meters.

    Notes
    -----
    The downrange and crossrange transformation is:

    .. math::

        d = N\\cos\\psi + E\\sin\\psi

    .. math::

        c = -N\\sin\\psi + E\\cos\\psi

    where:

    * :math:`N` is the local North displacement,
    * :math:`E` is the local East displacement,
    * :math:`\\psi` is the launch azimuth.

    The horizontal displacement magnitude is:

    .. math::

        s = \\sqrt{d^2 + c^2}

    Latitude and longitude inputs are the only angular quantities in
    this function expressed in degrees; all internally computed angles
    are in radians.
    """
    latitude_rad = np.deg2rad(latitude_deg)
    longitude_rad = np.deg2rad(longitude_deg)

    launch_azimuth_rad = launching_ref.azimuth
    reference_latitude_rad = launching_ref.lat
    reference_longitude_rad = launching_ref.lon
    reference_altitude_m = launching_ref.alt

    x_ecef_m, y_ecef_m, z_ecef_m = geodetic_to_ecef(
        latitude_rad,
        longitude_rad,
        altitude_m,
    )

    north_m, east_m, _ = ecef_to_ned(
        x_ecef_m,
        y_ecef_m,
        z_ecef_m,
        reference_latitude_rad,
        reference_longitude_rad,
        reference_altitude_m,
    )

    downrange_m = (
        north_m * cos(launch_azimuth_rad)
        + east_m * sin(launch_azimuth_rad)
    )

    crossrange_m = (
        -north_m * sin(launch_azimuth_rad)
        + east_m * cos(launch_azimuth_rad)
    )

    displacement_m = np.sqrt(
        downrange_m**2
        + crossrange_m**2
    )

    return downrange_m, crossrange_m, displacement_m


def integrate(
    new_value,
    old_value,
    state,
    step_s,
):
    """
    Integrate a quantity over one time step using the trapezoidal rule.

    Parameters
    ----------
    new_value : float
        Value of the derivative or integrand at the end of the time
        step. Its unit is the unit of ``state`` per second.
    old_value : float
        Value of the derivative or integrand at the beginning of the
        time step. Its unit is the unit of ``state`` per second.
    state : float
        Previously accumulated state value. Its unit is the integral
        of ``new_value`` and ``old_value`` with respect to time.
    step_s : float
        Integration time step, in seconds.

    Returns
    -------
    float
        Updated integrated state, with the same unit as ``state``.

    Notes
    -----
    The implemented trapezoidal integration is:

    .. math::

        x_{k+1}
        =
        x_k
        +
        \\frac{f_{k+1}+f_k}{2}\\Delta t

    where :math:`\\Delta t` is ``step_s``.
    """
    return state + (
        new_value + old_value
    ) * step_s / 2.0


def initialize_interpolator1d(
    data_path,
    sep=";",
):
    """
    Initialize a one-dimensional linear interpolator from a CSV file.

    The first column of the input file is interpreted as the independent
    variable and the second column as the dependent variable.

    Parameters
    ----------
    data_path : str or path-like
        Path to the input CSV file.
    sep : str, default=";"
        Field delimiter used in the input file.

    Returns
    -------
    scipy.interpolate.interp1d
        Linear interpolation function with extrapolation enabled.

    Raises
    ------
    FileNotFoundError
        If ``data_path`` does not exist.
    pandas.errors.EmptyDataError
        If the file is empty.
    ValueError
        If the loaded data does not contain at least two columns.

    Notes
    -----
    No header row is expected because ``header=None`` is used.

    The interpolator preserves the physical units of the supplied
    independent and dependent variables.
    """
    data = pd.read_csv(
        data_path,
        header=None,
        sep=sep,
    )

    independent_variable = data.iloc[:, 0].values
    dependent_variable = data.iloc[:, 1].values

    interpolator = interp1d(
        independent_variable,
        dependent_variable,
        kind="linear",
        fill_value="extrapolate",
    )

    return interpolator


def clean_pycache(
    main_directory,
    flag=False,
):
    """
    Remove Python ``__pycache__`` directories recursively.

    Parameters
    ----------
    main_directory : pathlib.Path
        Root directory from which ``__pycache__`` directories are
        searched recursively.
    flag : bool, default=False
        If ``True``, matching directories are removed. If ``False``,
        no filesystem changes are performed.

    Notes
    -----
    This function performs filesystem deletion only when ``flag`` is
    explicitly set to ``True``.
    """
    if flag:
        for pycache_directory in main_directory.rglob("__pycache__"):
            if pycache_directory.is_dir():
                shutil.rmtree(pycache_directory)


def wrap_to_pi(angle_rad):
    """
    Wrap an angle to the interval [-pi, pi).

    Parameters
    ----------
    angle_rad : float
        Input angle, in radians.

    Returns
    -------
    float
        Wrapped angle, in radians, constrained to:

        .. math::

            -\\pi \\leq \\theta < \\pi
    """
    return (
        angle_rad + np.pi
    ) % (
        2.0 * np.pi
    ) - np.pi


def continuous_euler_angles(
    quaternion_ned_from_body,
    previous_euler_angles_rad,
    gimbal_lock_protected,
):
    """
    Convert a body-to-NED quaternion to continuous canonical 3-2-1 Euler angles.

    The function computes the principal 3-2-1 Euler solution and then
    applies hysteresis-based gimbal-lock protection. Near the pitch
    singularity, roll is held at its previous value and the observable
    yaw combination is reconstructed from the quaternion.

    Parameters
    ----------
    quaternion_ned_from_body : numpy.ndarray, shape (4, 1)
        Quaternion representing the body-to-NED rotation,
        ``q_{NED<-B}``, using ``[w, x, y, z]`` ordering.

    previous_euler_angles_rad : numpy.ndarray, shape (3,)
        Previously returned Euler angles in the order
        ``[roll, pitch, yaw]``, in radians.

    gimbal_lock_protected : bool
        Current state of the gimbal-lock protection logic.

    Returns
    -------
    euler_angles_rad : numpy.ndarray, shape (3,)
        Continuous Euler angles in the order:

        ``[roll, pitch, yaw]``

        with ranges:

        * Roll: ``[-pi, pi]``
        * Pitch: ``[-pi/2, pi/2]``
        * Yaw: ``[-pi, pi]``

        All angles are in radians.

    gimbal_lock_protected : bool
        Updated state of the gimbal-lock protection logic.

    Notes
    -----
    The protection uses hysteresis with the following thresholds:

    .. math::

        \\delta_{on} = 5^\\circ

    .. math::

        \\delta_{off} = 12^\\circ

    where the distance from gimbal lock is:

    .. math::

        \\delta =
        \\frac{\\pi}{2} - |\\theta|

    Protection is enabled when :math:`\\delta \\leq 5^\\circ` and
    disabled only when :math:`\\delta \\geq 20^\\circ`.

    Near gimbal lock, roll and yaw are not independently observable.
    The implementation therefore preserves the previous roll and
    reconstructs the corresponding observable yaw combination.

    For positive pitch close to +90 degrees, the observable
    combination is:

    .. math::

        \\phi - \\psi

    For negative pitch close to -90 degrees, the observable
    combination is:

    .. math::

        \\phi + \\psi

    The ``previous_euler_angles_rad`` input is therefore relevant only
    to the gimbal-lock handling, with the previous roll specifically
    used during protection.
    """
    protection_onset_angle_rad = np.deg2rad(5.0)
    protection_offset_angle_rad = np.deg2rad(12.0)

    quaternion_w = quaternion_ned_from_body[0, 0]
    quaternion_x = quaternion_ned_from_body[1, 0]
    quaternion_y = quaternion_ned_from_body[2, 0]
    quaternion_z = quaternion_ned_from_body[3, 0]

    previous_roll_rad = previous_euler_angles_rad[0]

    roll_rad, pitch_rad, yaw_rad = quaternion_to_euler_321(
        quaternion_ned_from_body
    )

    roll_rad = wrap_to_pi(roll_rad)
    yaw_rad = wrap_to_pi(yaw_rad)

    pitch_rad = np.clip(
        pitch_rad,
        -0.5 * np.pi,
        0.5 * np.pi,
    )

    distance_from_gimbal_lock_rad = (
        0.5 * np.pi - abs(pitch_rad)
    )

    if gimbal_lock_protected:
        if (
            distance_from_gimbal_lock_rad
            >= protection_offset_angle_rad
        ):
            gimbal_lock_protected = False
    else:
        if (
            distance_from_gimbal_lock_rad
            <= protection_onset_angle_rad
        ):
            gimbal_lock_protected = True

    if gimbal_lock_protected:
        roll_rad = wrap_to_pi(previous_roll_rad)

        if pitch_rad >= 0.0:

            roll_minus_yaw_rad = atan2(
                2.0 * (
                    quaternion_w * quaternion_x
                    - quaternion_y * quaternion_z
                ),
                1.0
                - 2.0 * (
                    quaternion_x**2
                    + quaternion_z**2
                ),
            )

            yaw_rad = (
                roll_rad
                - roll_minus_yaw_rad
            )

        else:

            roll_plus_yaw_rad = atan2(
                -2.0 * (
                    quaternion_w * quaternion_x
                    - quaternion_y * quaternion_z
                ),
                -(
                    1.0
                    - 2.0 * (
                        quaternion_x**2
                        + quaternion_z**2
                    )
                ),
            )

            yaw_rad = (
                roll_plus_yaw_rad
                - roll_rad
            )

        yaw_rad = wrap_to_pi(yaw_rad)

    euler_angles_rad = np.array(
        [
            roll_rad,
            pitch_rad,
            yaw_rad,
        ]
    )

    return (
        euler_angles_rad,
        gimbal_lock_protected
    )


def compute_euler_angles_ned(
    state_vector,
    time_s,
    previous_euler_angles_ned_rad,
    gimbal_lock_protected,
):
    """
    Extract continuous body Euler angles relative to the local NED frame.

    The function extracts the vehicle ECI position and body-to-ECI
    quaternion from the supplied state vector, determines the current
    geodetic latitude and longitude, transforms the attitude into the
    local NED frame, and finally converts the quaternion to continuous
    3-2-1 Euler angles.

    Parameters
    ----------
    state_vector : numpy.ndarray, shape (>= 7, 1)
        Vehicle state vector. The current implementation expects:

        * indices ``0:3``: ECI position ``[x, y, z]`` in meters.
        * indices ``3:7``: body-to-ECI quaternion
          ``[w, x, y, z]``.

    time_s : float
        Time from the ECI/ECEF reference epoch, in seconds.
    previous_euler_angles_ned_rad : numpy.ndarray, shape (3,)
        Previously returned NED Euler angles in
        ``[roll, pitch, yaw]`` order, in radians.
    gimbal_lock_protected : bool
        Previous gimbal-lock protection state.

    Returns
    -------
    euler_angles_ned_rad : numpy.ndarray, shape (3,)
        Continuous body attitude relative to the local NED frame:

        ``[roll, pitch, yaw]``

        in radians.
    gimbal_lock_protected : bool
        Updated gimbal-lock protection state.

    Notes
    -----
    The implemented transformation chain is:

    .. math::

        q_{NED\\leftarrow B}
        =
        q_{NED\\leftarrow ECI}
        \\otimes
        q_{ECI\\leftarrow B}

    The current vehicle latitude and longitude are obtained from the
    first three entries of ``state_vector`` through ``eci_to_geodetic``.

    The intermediate function ``eci_to_ned_attitude`` returns the four
    quaternion components separately; they are then reconstructed into
    a ``(4, 1)`` quaternion before Euler-angle extraction.
    """
    latitude_rad, longitude_rad, _ = eci_to_geodetic(
        state_vector[0, 0],
        state_vector[1, 0],
        state_vector[2, 0],
        time_s,
    )

    quaternion_body_eci = np.array(
        [
            [state_vector[3, 0]],
            [state_vector[4, 0]],
            [state_vector[5, 0]],
            [state_vector[6, 0]],
        ],
        dtype=np.float64,
    )

    (
        quaternion_w_ned_from_body,
        quaternion_x_ned_from_body,
        quaternion_y_ned_from_body,
        quaternion_z_ned_from_body,
    ) = eci_to_ned_attitude(
        quaternion_body_eci,
        latitude_rad,
        longitude_rad,
        time_s,
    )

    quaternion_ned_from_body = np.array(
        [
            [quaternion_w_ned_from_body],
            [quaternion_x_ned_from_body],
            [quaternion_y_ned_from_body],
            [quaternion_z_ned_from_body],
        ],
        dtype=np.float64,
    )

    (
        euler_angles_ned_rad,
        gimbal_lock_protected,
    ) = continuous_euler_angles(
        quaternion_ned_from_body,
        previous_euler_angles_ned_rad,
        gimbal_lock_protected,
    )

    return (
        euler_angles_ned_rad,
        gimbal_lock_protected,
    )

class TransferFunction:
    """
    State-space realization of a continuous-time transfer function.

    This class represents a proper transfer function using a controllable
    canonical-form state-space realization. The continuous-time state
    equations are numerically integrated using a fourth-order
    Runge-Kutta (RK4) method over the configured sampling/integration
    interval.

    The transfer-function coefficients can be updated at every call to
    :meth:`step`, allowing the model dynamics to vary with time.

    Parameters
    ----------
    numerator : numpy.ndarray, shape (1, m)
        Transfer-function numerator coefficients, ordered from highest
        to lowest power of the Laplace variable ``s``.
    denominator : numpy.ndarray, shape (1, n)
        Transfer-function denominator coefficients, ordered from highest
        to lowest power of ``s``.
    sampling_time_s : float
        Integration time step, in seconds.

    Attributes
    ----------
    m : int
        Number of numerator coefficients.
    n : int
        Number of denominator coefficients.
    state : numpy.ndarray, shape (n - 1, 1)
        Continuous-time state vector.
    sampling_time_s : float
        Integration time step, in seconds.
    A : numpy.ndarray, shape (n - 1, n - 1)
        Continuous-time state matrix.
    B : numpy.ndarray, shape (n - 1, 1)
        Continuous-time input matrix.
    C : numpy.ndarray, shape (1, n - 1)
        Output matrix.
    D : float
        Direct feedthrough coefficient.
    numerator : numpy.ndarray
        Current numerator coefficients.
    denominator : numpy.ndarray
        Current denominator coefficients.

    Notes
    -----
    The state-space realization is of the form:

    .. math::

        \\dot{x} = A x + B u

    .. math::

        y = C x + D u

    The state derivative is integrated over ``sampling_time_s`` using
    classical fourth-order Runge-Kutta integration.

    The implementation assumes a proper transfer function, i.e. the
    numerator order does not exceed the denominator order. In addition,
    the denominator must contain at least one coefficient.

    See Also
    --------
    scipy.signal.StateSpace : General state-space representation.
    """

    def __init__(self, numerator, denominator, sampling_time_s):
        """
        Initialize the transfer-function model.

        Parameters
        ----------
        numerator : numpy.ndarray, shape (1, m)
            Transfer-function numerator coefficients.
        denominator : numpy.ndarray, shape (1, n)
            Transfer-function denominator coefficients.
        sampling_time_s : float
            Integration time step, in seconds.
        """
        self.m = numerator.shape[1]
        self.n = denominator.shape[1]

        self.state = np.zeros(
            (self.n - 1, 1)
        )

        self.sampling_time_s = sampling_time_s

        self.A = None
        self.B = None
        self.C = None
        self.D = None

        self._update_state_space(
            numerator,
            denominator,
        )

    def _update_state_space(
        self,
        numerator,
        denominator,
    ):
        """
        Update the continuous-time state-space realization.

        Parameters
        ----------
        numerator : numpy.ndarray, shape (1, m)
            Transfer-function numerator coefficients.
        denominator : numpy.ndarray, shape (1, n)
            Transfer-function denominator coefficients.

        Notes
        -----
        The denominator is normalized so that its leading coefficient
        is equal to one before constructing the canonical-form
        realization.

        The current implementation distinguishes between:

        * equal numerator and denominator lengths, in which case a
          direct feedthrough term ``D`` is present;
        * shorter numerator, in which case ``D = 0``.

        The resulting model is represented as:

        .. math::

            \\dot{x} = A x + B u

        .. math::

            y = C x + D u
        """
        denominator_leading_coefficient = denominator.item(0)

        if denominator_leading_coefficient != 1:
            denominator = (
                denominator
                / denominator_leading_coefficient
            )
            numerator = (
                numerator
                / denominator_leading_coefficient
            )

        self.numerator = numerator
        self.denominator = denominator

        self.A = np.zeros(
            (self.n - 1, self.n - 1)
        )
        self.B = np.zeros(
            (self.n - 1, 1)
        )
        self.C = np.zeros(
            (1, self.n - 1)
        )

        self.B[0, 0] = 1.0

        if self.m == self.n:
            self.D = numerator.item(0)

            for state_index in range(self.n - 1):
                self.C[0, state_index] = (
                    numerator.item(state_index + 1)
                    - self.D
                    * denominator.item(state_index + 1)
                )

            for state_index in range(self.n - 1):
                self.A[0, state_index] = (
                    -denominator.item(state_index + 1)
                )

            for state_index in range(1, self.n - 1):
                self.A[state_index, state_index - 1] = 1.0

        else:
            self.D = 0.0

            for numerator_index in range(self.m):
                self.C[
                    0,
                    self.n - numerator_index - 2,
                ] = numerator.item(
                    self.m - numerator_index - 1
                )

            for state_index in range(self.n - 1):
                self.A[0, state_index] = (
                    -denominator.item(state_index + 1)
                )

            for state_index in range(1, self.n - 1):
                self.A[state_index, state_index - 1] = 1.0

    def _state_derivative(
        self,
        state,
        input_value,
    ):
        """
        Compute the continuous-time state derivative.

        Parameters
        ----------
        state : numpy.ndarray, shape (n - 1, 1)
            Current state vector.
        input_value : float
            Current model input. Its physical unit is application-dependent.

        Returns
        -------
        numpy.ndarray
            State derivative, with units corresponding to
            ``state / second``.

        Notes
        -----
        The derivative is computed according to:

        .. math::

            \\dot{x} = A x + B u
        """
        state_derivative = (
            np.matmul(self.A, state)
            + self.B * input_value
        )

        return state_derivative

    def _output(
        self,
        input_value,
    ):
        """
        Compute the current transfer-function output.

        Parameters
        ----------
        input_value : float
            Current model input. Its physical unit is application-dependent.

        Returns
        -------
        float
            Model output. Its physical unit is application-dependent.

        Notes
        -----
        The output equation is:

        .. math::

            y = C x + D u
        """
        output_value = (
            np.matmul(
                self.C,
                self.state,
            )
            + self.D * input_value
        )

        return output_value.item(0)

    def _integrate(
        self,
        input_value,
    ):
        """
        Advance the state using fourth-order Runge-Kutta integration.

        Parameters
        ----------
        input_value : float
            Model input held constant over the current integration step.

        Notes
        -----
        The RK4 stages are computed over ``sampling_time_s``:

        .. math::

            k_1 = f(x_k, u)

        .. math::

            k_2 =
            f\\left(
                x_k + \\frac{T_s}{2}k_1,
                u
            \\right)

        .. math::

            k_3 =
            f\\left(
                x_k + \\frac{T_s}{2}k_2,
                u
            \\right)

        .. math::

            k_4 =
            f\\left(
                x_k + T_s k_3,
                u
            \\right)

        followed by:

        .. math::

            x_{k+1}
            =
            x_k
            +
            \\frac{T_s}{6}
            \\left(
                k_1 + 2k_2 + 2k_3 + k_4
            \\right)

        where ``Ts`` is the integration step in seconds.
        """
        stage_1 = self._state_derivative(
            self.state,
            input_value,
        )

        stage_2 = self._state_derivative(
            self.state
            + 0.5 * self.sampling_time_s * stage_1,
            input_value,
        )

        stage_3 = self._state_derivative(
            self.state
            + 0.5 * self.sampling_time_s * stage_2,
            input_value,
        )

        stage_4 = self._state_derivative(
            self.state
            + self.sampling_time_s * stage_3,
            input_value,
        )

        self.state += (
            self.sampling_time_s
            / 6.0
            * (
                stage_1
                + 2.0 * stage_2
                + 2.0 * stage_3
                + stage_4
            )
        )

    def set_Ts(
        self,
        sampling_time_s,
    ):
        """
        Set the integration time step.

        Parameters
        ----------
        sampling_time_s : float
            New integration time step, in seconds.
        """
        self.sampling_time_s = sampling_time_s

    @performance_decorator.time_execution_stats
    def step(
        self,
        input_value,
        numerator,
        denominator,
    ):
        """
        Advance the transfer-function model by one integration step.

        Parameters
        ----------
        input_value : float
            Current model input. Its physical unit is application-dependent.
        numerator : numpy.ndarray, shape (1, m)
            Current numerator coefficients.
        denominator : numpy.ndarray, shape (1, n)
            Current denominator coefficients.

        Returns
        -------
        float
            Model output after the state has been advanced by one
            integration step.

        Notes
        -----
        The operation order is:

        #. Update the state-space realization using the supplied
           numerator and denominator.
        #. Integrate the state over one ``sampling_time_s`` interval.
        #. Evaluate the output equation using the updated state.

        The input is assumed constant during the RK4 integration step.
        """
        self._update_state_space(
            numerator,
            denominator,
        )

        self._integrate(
            input_value,
        )

        output_value = self._output(
            input_value,
        )

        return output_value


class Smoothing:
    """
    Constant-velocity Kalman filter for one-dimensional measurement smoothing.

    This class implements a two-state discrete-time Kalman filter with
    a constant-velocity process model. The state consists of a scalar
    position-like quantity and its corresponding rate.

    The measurement model observes only the first state.

    Parameters
    ----------
    time_step_s : float
        Kalman-filter propagation interval, in seconds.

    Attributes
    ----------
    x : numpy.ndarray, shape (2, 1) or None
        State estimate:

        .. math::

            x =
            \\begin{bmatrix}
            p \\\\
            \\dot{p}
            \\end{bmatrix}

        where the physical units depend on the quantity being smoothed.
    initialized : bool
        ``True`` once :meth:`set_initial_state` has been called.
    time_step_s : float
        Filter propagation interval, in seconds.
    F : numpy.ndarray, shape (2, 2)
        Discrete-time state-transition matrix.
    H : numpy.ndarray, shape (1, 2)
        Measurement matrix.
    P : numpy.ndarray, shape (2, 2)
        State-estimation covariance matrix.
    Q : numpy.ndarray, shape (2, 2)
        Process-noise covariance matrix.
    R : float
        Scalar measurement-noise variance.

    Notes
    -----
    The state-transition model is:

    .. math::

        x_k = F x_{k-1} + w_k

    with:

    .. math::

        F =
        \\begin{bmatrix}
        1 & \\Delta t \\\\
        0 & 1
        \\end{bmatrix}

    The measurement model is:

    .. math::

        z_k = H x_k + v_k

    with:

    .. math::

        H =
        \\begin{bmatrix}
        1 & 0
        \\end{bmatrix}

    The process-noise covariance is initialized as ``1e-3 * I`` and
    the measurement-noise variance is initialized to ``1.0``.
    These values are dimension-dependent and should therefore be
    interpreted according to the physical quantity being filtered.
    """

    def __init__(
        self,
        time_step_s,
    ):
        """
        Initialize the Kalman-filter model.

        Parameters
        ----------
        time_step_s : float
            Filter propagation interval, in seconds.
        """
        self.x = None
        self.initialized = False

        self.time_step_s = time_step_s

        self.F = np.array(
            [
                [1.0, self.time_step_s],
                [0.0, 1.0],
            ]
        )

        self.H = np.array(
            [[1.0, 0.0]]
        )

        self.P = np.eye(2)

        self.Q = (
            1e-3
            * np.eye(2)
        )

        self.R = 1.0

    def set_initial_state(
        self,
        initial_measurement,
    ):
        """
        Set the initial state estimate.

        Parameters
        ----------
        initial_measurement : float
            Initial estimate of the position-like state. Its physical
            unit is the same as the first state component.

        Notes
        -----
        The initial rate state is set to zero:

        .. math::

            x_0 =
            \\begin{bmatrix}
            z_0 \\\\
            0
            \\end{bmatrix}
        """
        self.x = np.array(
            [
                [initial_measurement],
                [0.0],
            ]
        )

        self.initialized = True

    def smooth(
        self,
        measurement,
    ):
        """
        Perform one Kalman-filter propagation and measurement update.

        Parameters
        ----------
        measurement : float
            Current scalar measurement. Its physical unit is the same
            as the first state component.

        Returns
        -------
        float
            Updated estimate of the first state component.

        Raises
        ------
        AttributeError
            May occur if the filter has not been initialized using
            :meth:`set_initial_state`.

        Notes
        -----
        The method performs the standard two-stage Kalman-filter cycle.

        Prediction:

        .. math::

            x_k^- = F x_{k-1}

        .. math::

            P_k^- =
            F P_{k-1} F^T + Q

        Measurement innovation:

        .. math::

            d_k =
            z_k - Hx_k^-

        Innovation covariance:

        .. math::

            S_k =
            H P_k^- H^T + R

        Kalman gain:

        .. math::

            K_k =
            P_k^- H^T S_k^{-1}

        State update:

        .. math::

            x_k =
            x_k^-
            +
            K_k d_k

        Covariance update:

        .. math::

            P_k =
            (I - K_k H)P_k^-

        The method returns only the updated first state component.
        """
        # ----------------------------------------------------------
        # Prediction
        # ----------------------------------------------------------

        self.x = np.matmul(
            self.F,
            self.x,
        )

        self.P = np.add(
            np.matmul(
                self.F,
                np.matmul(
                    self.P,
                    np.transpose(self.F),
                ),
            ),
            self.Q,
        )

        # ----------------------------------------------------------
        # Measurement update
        # ----------------------------------------------------------

        measurement_array = np.array(
            [[measurement]]
        )

        innovation = np.add(
            measurement_array,
            -np.matmul(
                self.H,
                self.x,
            ),
        )

        innovation_covariance = np.add(
            np.matmul(
                self.H,
                np.matmul(
                    self.P,
                    np.transpose(self.H),
                ),
            ),
            self.R,
        )

        inverse_innovation_covariance = inv(
            innovation_covariance
        )

        kalman_gain = np.matmul(
            self.P,
            np.matmul(
                np.transpose(self.H),
                inverse_innovation_covariance,
            ),
        )

        self.x = np.add(
            self.x,
            np.matmul(
                kalman_gain,
                innovation,
            ),
        )

        self.P = np.matmul(
            np.add(
                np.eye(2),
                -np.matmul(
                    kalman_gain,
                    self.H,
                ),
            ),
            self.P,
        )

        return self.x[0, 0]

