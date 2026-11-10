import os
import re
import urllib.parse
import urllib.request
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import astropy.constants as const
import astropy.units as u
from angutils.angutils import projplane, calcang, rotMat, calcDCM
from astropy.coordinates import (
    SkyCoord,
    CartesianDifferential,
    CartesianRepresentation,
    get_body_barycentric,
    BarycentricMeanEcliptic,
)
from astropy.time import Time
import scipy.optimize
from scipy.interpolate import CubicSpline

SIAF_FILE = os.path.join(
    os.path.dirname(__file__), "data", "roman_siaf_inflight_20261021_CGI.xml"
)
HORIZONS_API = "https://ssd.jpl.nasa.gov/api/horizons.api"


# find L2 location
f = (
    lambda x, mustar: x
    - (1 - mustar) * (x + mustar) / np.abs(x + mustar) ** 3
    - mustar * (x - 1 + mustar) / np.abs(x - 1 + mustar) ** 3
)
mustar_sunearth = ((1 * u.Mearth) / (1 * u.Mearth + 1 * u.Msun)).decompose().value
fsunearth = lambda x: f(x, mustar_sunearth)  # noqa
L2loc = scipy.optimize.fsolve(fsunearth, 1)[0]


def getSunPositions(ts):
    """Retrieve the barycentric position of the sun for given observing times

    Args:
        ts (astropy.time.Time):
            Observation time(s) - can be an array of times

    Returns:
        numpy.ndarray(float):
            3xn array of sun barycentric positions where n is the size of ts

    """

    # Get sun position
    sun = SkyCoord(
        get_body_barycentric("Sun", ts), frame="icrs", obstime=ts
    ).transform_to(BarycentricMeanEcliptic)

    # sun barycentric cartesian coordinates
    r_sun_G = sun.cartesian.xyz

    return r_sun_G


def getL2Positions(ts):
    """Retrieve the barycentric position of L2 for given observing times

    Args:
        ts (astropy.time.Time):
            Observation time(s) - can be an array of times

    Returns:
        numpy.ndarray(float):
            3xn array of approximate L2 barycentric positions where n is the size of ts

    """
    earth = SkyCoord(
        get_body_barycentric("Earth", ts), frame="icrs", obstime=ts
    ).transform_to(BarycentricMeanEcliptic)
    r_L2_G = L2loc * earth.cartesian.xyz

    return r_L2_G


def getHorizonsVectors(command, center, start, stop, step):
    """Query the JPL Horizons API for a table of geometric ICRF position vectors

    Requests the plain-text table: the CSV table that astroquery requests is
    returned without vector components by the current Horizons API.

    Args:
        command (str):
            Horizons target body ID (e.g. '-211' for Roman, '10' for the Sun)
        center (str):
            Horizons center (e.g. '500@0' for the solar system barycenter)
        start (str):
            Start time (TDB)
        stop (str):
            Stop time (TDB)
        step (str):
            Table step size (e.g. '1d', '1h')

    Returns:
        tuple:
            jd (numpy.ndarray(float)): Julian dates (TDB) of the table rows;
            xyz (numpy.ndarray(float)): 3xn array of positions (AU)

    """
    params = {
        "format": "text",
        "COMMAND": f"'{command}'",
        "CENTER": f"'{center}'",
        "EPHEM_TYPE": "VECTORS",
        "START_TIME": f"'{start}'",
        "STOP_TIME": f"'{stop}'",
        "STEP_SIZE": f"'{step}'",
        "REF_PLANE": "FRAME",
        "REF_SYSTEM": "ICRF",
        "VEC_TABLE": "1",
        "VEC_CORR": "NONE",
        "OUT_UNITS": "AU-D",
        "CSV_FORMAT": "NO",
        "OBJ_DATA": "NO",
    }
    url = f"{HORIZONS_API}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=60) as response:
        text = response.read().decode()
    if "$$SOE" not in text:
        raise ValueError(f"No Horizons ephemeris returned:\n{text[-1000:]}")
    table = text.split("$$SOE")[1].split("$$EOE")[0]
    jd = np.array(re.findall(r"^\s*(\d+\.\d+) = A\.D\.", table, flags=re.M), dtype=float)
    xyz = np.array(
        re.findall(r"X =\s*(\S+)\s+Y =\s*(\S+)\s+Z =\s*(\S+)", table), dtype=float
    ).T

    return jd, xyz


def getRomanPositions(ts, step="1d"):
    """Retrieve the barycentric position of Roman from the JPL Horizons ephemeris

    Falls back to getL2Positions, with a warning, if the query fails (no network,
    or ts outside the Horizons coverage).

    Args:
        ts (astropy.time.Time):
            Observation time(s) - can be an array of times
        step (str):
            Sampling of the Horizons table that is interpolated to ts

    Returns:
        astropy.units.Quantity(numpy.ndarray(float)):
            3xn array of Roman barycentric positions where n is the size of ts

    """
    jd = np.atleast_1d(ts.tdb.jd)
    start = Time(np.floor(jd.min() - 0.5) - 1.5, format="jd", scale="tdb").iso
    stop = Time(np.ceil(jd.max() - 0.5) + 2.5, format="jd", scale="tdb").iso
    try:
        jd_table, xyz_table = getHorizonsVectors("-211", "500@0", start, stop, step)
    except Exception as err:
        warnings.warn(f"Horizons query failed ({err}); using the L2 approximation")
        return getL2Positions(ts)
    xyz = CubicSpline(jd_table, xyz_table, axis=1)(jd)
    roman = SkyCoord(CartesianRepresentation(xyz * u.AU), frame="icrs").transform_to(
        BarycentricMeanEcliptic
    )

    return roman.cartesian.xyz[:, 0] if ts.isscalar else roman.cartesian.xyz


def getL2Velocities(ts):
    """Barycentric velocity of the approximate L2 point (getL2Positions) for given times

    Args:
        ts (astropy.time.Time):
            Observation time(s) - can be an array of times

    Returns:
        astropy.units.Quantity(numpy.ndarray(float)):
            3xn array of L2 barycentric velocities (km/s) where n is the size of ts

    """
    dt = 1 * u.min
    v = (getL2Positions(ts + dt) - getL2Positions(ts - dt)) / (2 * dt)

    return v.to(u.km / u.s)


def getRomanVelocities(ts, step="1d"):
    """Retrieve the barycentric velocity of Roman from the JPL Horizons ephemeris

    The velocity is the derivative of the cubic spline used by getRomanPositions.
    Falls back to getL2Velocities, with a warning, if the query fails.

    Args:
        ts (astropy.time.Time):
            Observation time(s) - can be an array of times
        step (str):
            Sampling of the Horizons table that is interpolated to ts

    Returns:
        astropy.units.Quantity(numpy.ndarray(float)):
            3xn array of Roman barycentric velocities (km/s) where n is the size of ts

    """
    jd = np.atleast_1d(ts.tdb.jd)
    start = Time(np.floor(jd.min() - 0.5) - 1.5, format="jd", scale="tdb").iso
    stop = Time(np.ceil(jd.max() - 0.5) + 2.5, format="jd", scale="tdb").iso
    try:
        jd_table, xyz_table = getHorizonsVectors("-211", "500@0", start, stop, step)
    except Exception as err:
        warnings.warn(f"Horizons query failed ({err}); using the L2 approximation")
        return getL2Velocities(ts)
    spline = CubicSpline(jd_table, xyz_table, axis=1)
    xyz = CartesianRepresentation(spline(jd) * u.AU)
    vxyz = CartesianDifferential(spline(jd, 1) * u.AU / u.day)
    roman = SkyCoord(xyz.with_differentials(vxyz), frame="icrs").transform_to(
        BarycentricMeanEcliptic
    )
    v = roman.velocity.d_xyz.to(u.km / u.s)

    return v[:, 0] if ts.isscalar else v


def applyAberration(rhat, v_obs_G, inverse=False):
    """Apply (or remove) the aberration of light due to the observatory velocity

    Uses the exact (special-relativistic) aberration formula. The apparent
    direction of a source is displaced toward the observatory velocity by up to
    v/c (20.6 arcsec for 30 km/s).

    Args:
        rhat (numpy.ndarray(float)):
            3xn array of unit vectors toward sources, in the frame of v_obs_G:
            catalog (rest-frame) directions, or apparent directions if inverse is True
        v_obs_G (astropy.units.Quantity(numpy.ndarray(float))):
            Observatory barycentric velocity, 3xn (or 3) array
        inverse (bool):
            If True, convert apparent directions to catalog directions

    Returns:
        numpy.ndarray(float):
            3xn array of apparent (or, if inverse is True, catalog) unit vectors

    """
    beta = (v_obs_G / const.c).decompose().value
    if inverse:
        beta = -beta
    if beta.ndim == 1:
        beta = beta[:, None]
    gamma = 1 / np.sqrt(1 - np.sum(beta**2, axis=0))
    s_beta = np.sum(rhat * beta, axis=0)
    r = (rhat / gamma + beta + gamma / (1 + gamma) * s_beta * beta) / (1 + s_beta)

    return r / np.linalg.norm(r, axis=0)


def removeAberration(B_C_I, v_obs_G, aperture=None):
    """Express a physical (apparent-frame) attitude in the aberration-free catalog frame

    Rotates each attitude by the smallest rotation that moves the apparent direction
    of the reference point (the aperture, or b_1) to its catalog direction. This is
    the convention of the WFI pointing metadata and of the observation plans, whereas
    the attitude quaternion and the CGI header pointing keywords are apparent.

    Args:
        B_C_I (numpy.ndarray(float)):
            3x3xn attitude matrices from calcRomanAngles(..., v_obs_G=v_obs_G)
        v_obs_G (astropy.units.Quantity(numpy.ndarray(float))):
            Observatory barycentric velocity, 3xn array
        aperture (dict, optional):
            Aperture from getSIAFAperture() used as the reference point. If None,
            b_1 (V1) is used.

    Returns:
        numpy.ndarray(float):
            3x3xn attitude matrices in the catalog frame

    """
    u_ref = np.array([1.0, 0.0, 0.0]) if aperture is None else aperture["u_B"]
    rhat_app = np.einsum("ijt,i->jt", B_C_I, u_ref)
    rhat_cat = applyAberration(rhat_app, v_obs_G, inverse=True)
    B_C_I_cat = np.empty_like(B_C_I)
    for t in range(B_C_I.shape[2]):
        n = np.cross(rhat_app[:, t], rhat_cat[:, t])
        angle = np.arctan2(np.linalg.norm(n), rhat_app[:, t] @ rhat_cat[:, t])
        rot = calcDCM(n, angle) if angle > 0 else np.eye(3)  # rotates rhat_app onto rhat_cat
        B_C_I_cat[:, :, t] = B_C_I[:, :, t] @ rot.T

    return B_C_I_cat


def getSIAFAperture(aper_name="CGI_CEN", siaf_file=SIAF_FILE):
    """Read an aperture reference point and orientation from the Roman SIAF

    Args:
        aper_name (str):
            SIAF aperture name. Defaults to CGI_CEN (EXCAM).
        siaf_file (str):
            Path to the SIAF XML file.

    Returns:
        dict:
            V2Ref, V3Ref (arcsec), V3IdlYAngle (deg), VIdlParity, and u_B, the unit
            vector to the aperture reference point in the body (b_1, b_2, b_3) frame.

    """
    for entry in ET.parse(siaf_file).getroot().iter("SiafEntry"):
        if entry.findtext("AperName") == aper_name:
            aperture = {
                k: float(entry.findtext(k))
                for k in ("V2Ref", "V3Ref", "V3IdlYAngle", "VIdlParity")
            }
            v2 = np.radians(aperture["V2Ref"] / 3600)
            v3 = np.radians(aperture["V3Ref"] / 3600)
            aperture["u_B"] = np.array(
                [np.cos(v2) * np.cos(v3), np.sin(v2) * np.cos(v3), np.sin(v3)]
            )
            return aperture
    raise ValueError(f"Aperture {aper_name} not found in {siaf_file}")


def calcRomanAngles(target, ts, r_obs_G, r_sun_G=None, aperture=None, v_obs_G=None):
    """Compute Roman's pointing and sun angles for a particular target

    Args:
        target (astropy.coordinates.SkyCoord):
            Target coordinates
        ts (astropy.time.Time):
            Observation time(s) - can be an array of times
        r_obs_G (astropy.unitsQuantity(numpy.ndarray(float))):
            Observatory position wrt solar system barycenter for each observation time.
            Should have dimension 3xn where n is the size of ts.
        r_sun_G (astropy.unitsQuantity(numpy.ndarray(float)), optional):
            Sun position wrt solar system barycenter for each observation time.
            Should have dimension 3xn where n is the size of ts. If None, will be
            computed automatically
        aperture (dict, optional):
            Aperture from getSIAFAperture(). If given, the nominal attitude places the
            aperture reference point, instead of b_1, on the target. If None, b_1 is
            pointed at the target.
        v_obs_G (astropy.units.Quantity(numpy.ndarray(float)), optional):
            Observatory barycentric velocity for each observation time (3xn), e.g.
            from getRomanVelocities. If given, the target and Sun directions are
            aberrated, so B_C_I is the physical attitude in the apparent frame (the
            frame of the attitude quaternion) and sun_ang is the apparent Sun angle.
            Use removeAberration to express B_C_I in the catalog frame.

    Returns:
        tuple:
            sun_ang, yaw, pitch, B_C_I. sun_ang, yaw and pitch refer to b_1 (V1),
            where the field of regard limits apply.

    """

    # get coords of the sun, if needed
    if r_sun_G is None:
        r_sun_G = getSunPositions(ts)

    # sun position and unit vector wrt observatory
    r_sun_obs = r_sun_G - r_obs_G
    rhat_sun_obs = (r_sun_obs / np.linalg.norm(r_sun_obs, axis=0)).value

    # update target position and compute position and unit vector wrt observatory
    r_target_G = target.apply_space_motion(new_obstime=ts).cartesian.xyz
    r_target_obs = r_target_G - r_obs_G
    rhat_target_obs = (r_target_obs / np.linalg.norm(r_target_obs, axis=0)).value

    # apparent directions as seen from the moving observatory
    if v_obs_G is not None:
        rhat_sun_obs = applyAberration(rhat_sun_obs, v_obs_G)
        r_sun_obs = rhat_sun_obs * np.linalg.norm(r_sun_obs, axis=0)
        rhat_target_obs = applyAberration(rhat_target_obs, v_obs_G)

    # compute angle between sun and target vectors
    sun_ang = (
        np.arccos([np.dot(x, y) for x, y in zip(rhat_sun_obs.T, rhat_target_obs.T)])
        * u.rad
    )

    # define inertial basis vectors
    # e1 = np.array([1, 0, 0])
    e2 = np.array([0, 1, 0])
    e3 = np.array([0, 0, 1])

    # align b_3 with -r_obs/sun (equivalently r_sun/obs)
    # first a rotation about b_2 by the angle between b_3 and the projection
    # of the sun/obs vector onto the e1/e3 plane

    # projection of sun/obs vector onto e1/e3 plane:
    r_sun_obs_proj1 = projplane(r_sun_obs, e2)
    rhat_sun_obs_proj1 = (
        r_sun_obs_proj1 / np.linalg.norm(r_sun_obs_proj1, axis=0)
    ).value
    ang1 = np.array([calcang(x, e3, e2) for x in rhat_sun_obs_proj1.T])
    B_C_I = np.dstack(
        [rotMat(2, -a) for a in ang1]
    )  # DCM between inertial and body frames

    # second a rotation about b_1 by the angle between the new b_3 and r_sun/obs
    b_3 = B_C_I[2, :, :].T
    b_1 = B_C_I[0, :, :].T
    ang2 = np.array([calcang(x, b3, b1) for x, b3, b1 in zip(rhat_sun_obs.T, b_3, b_1)])

    B_C_I_sun = np.dstack(
        [np.matmul(rotMat(1, -a), B_C_I[:, :, j]) for j, a in enumerate(ang2)]
    )

    # now we wish to align b_1 to the pointing direction with yaw, pitch, roll
    # (b_3, b_2, b_1). The pointing direction is the target or, with an aperture, the
    # direction that places the aperture reference point on the target, which is found
    # by iteration.
    rhat_point = rhat_target_obs
    for _ in range(10):
        # projection of pointing vector onto b1/e2 plane:
        rhat_point_proj1 = np.hstack(
            [
                projplane(np.array(rhat_point[:, j], ndmin=2).T, B_C_I_sun[2, :, j].T)
                for j in range(len(ts))
            ]
        )
        rhat_point_proj1 = rhat_point_proj1 / np.linalg.norm(rhat_point_proj1, axis=0)

        # yaw is angle between projection and b_1
        b_1 = B_C_I_sun[0, :, :].T
        b_3 = B_C_I_sun[2, :, :].T
        yaw = -np.array(
            [calcang(x, b1, b3) for x, b1, b3 in zip(rhat_point_proj1.T, b_1, b_3)]
        )

        B_C_I = np.dstack(
            [np.matmul(rotMat(3, a), B_C_I_sun[:, :, j]) for j, a in enumerate(yaw)]
        )

        # next we pitch! rotate about b_2 by the angle between b_1 and final look vector
        b_1 = B_C_I[0, :, :].T
        b_2 = B_C_I[1, :, :].T
        pitch = -np.array(
            [calcang(x, b1, b2) for x, b1, b2 in zip(rhat_point.T, b_1, b_2)]
        )

        B_C_I = np.dstack(
            [np.matmul(rotMat(2, a), B_C_I[:, :, j]) for j, a in enumerate(pitch)]
        )

        if aperture is None:
            break

        # offset of the aperture reference point (inertial direction B_C_I^T u_B)
        # from the target; shift the pointing direction by the same amount
        dr = rhat_target_obs - np.einsum("ijt,i->jt", B_C_I, aperture["u_B"])
        if np.max(np.linalg.norm(dr, axis=0)) < 1e-10:
            break
        rhat_point = (rhat_point + dr) / np.linalg.norm(rhat_point + dr, axis=0)

    # with an aperture, the sun angle (field of regard constraint) applies to b_1
    if aperture is not None:
        sun_ang = np.arccos(np.sum(rhat_sun_obs * rhat_point, axis=0)) * u.rad

    return sun_ang, yaw * u.rad, pitch * u.rad, B_C_I


def applyRollAngle(B_C_I, roll_angles, aperture=None):
    """Apply roll angles to the spacecraft body-centered inertial frame

    Args:
        B_C_I (numpy.ndarray(float)):
            Matrix of spacecraft body-centered unit vectors in the inertial reference
            frame. Should have dimension 3x3xn where n is the number of time steps.
            The last axis represents time. Typically computed by calcRomanAngles().
        roll_angles (numpy.ndarray(float) or astropy.units.Quantity):
            Roll angles to apply at each time step. Should have length n matching
            the time dimension of B_C_I. Roll is rotation about the b_1 body axis.
        aperture (dict, optional):
            Aperture from getSIAFAperture(). If given, roll about the aperture
            reference point direction u_B instead of b_1, so the target stays on the
            aperture.

    Returns:
        numpy.ndarray(float):
            Updated 3x3xn matrix of body-centered unit vectors after applying roll

    """
    if aperture is None:
        dcms = [rotMat(1, a) for a in roll_angles]
    else:
        # calcDCM returns the DCM in the opposite sense to rotMat; with -a,
        # calcDCM([1, 0, 0], -a) equals rotMat(1, a)
        dcms = [
            calcDCM(aperture["u_B"], -a) for a in u.Quantity(roll_angles, u.rad).value
        ]
    B_C_I_roll = np.dstack(
        [np.matmul(dcm, B_C_I[:, :, j]) for j, dcm in enumerate(dcms)]
    )

    return B_C_I_roll


def getRomanPositionAngle(B_C_I, aperture=None):
    """Compute the position angle of the Roman observatory +Z axis (b_3 body vector)
       with respect to celestial North, projected onto the instrument focal plane.

    Args:
        B_C_I (numpy.ndarray(float)):
            Matrix of spacecraft body-centered unit vectors in the inertial reference
            frame. Should have dimension 3x3xn where n is the number of time steps.
            The last axis represents time. Typically computed by calcRomanAngles().
        aperture (dict, optional):
            Aperture from getSIAFAperture(). If given, the position angle is the local
            PA_V3 at the aperture reference point, measured about its direction
            instead of b_1.

    Returns:
        astropy.units.Quantity(numpy.ndarray(float)):
            Array of position angles at each time. Position angle is measured
            counter-clockwise from celestial North to the observatory +Z (b_3 body
            vector) axis.

    """
    b_1 = B_C_I[0, :, :]
    b_3 = B_C_I[2, :, :]

    celestial_north = SkyCoord(ra=0 * u.deg, dec=90 * u.deg, frame="icrs").transform_to(
        BarycentricMeanEcliptic
    )
    rhat_north = celestial_north.cartesian.xyz.value

    PA_Z = []
    for t in range(b_3.shape[1]):
        # with an aperture, b_3 projected onto the focal plane at the aperture is the
        # local +V3 direction there
        axis = b_1[:, t] if aperture is None else B_C_I[:, :, t].T @ aperture["u_B"]
        north_proj = projplane(rhat_north.reshape(3, 1), axis)
        b_3_proj = projplane(b_3[:, t].reshape(3, 1), axis)
        PA_Z.append(calcang(b_3_proj, north_proj, axis))

    return np.array(PA_Z) * u.rad


def getEXCAMPositionAngle(B_C_I, aperture=None):
    """Return the EXCAM position angle on sky

    Args:
        B_C_I (numpy.ndarray(float)):
            Matrix of spacecraft body-centered unit vectors in the inertial reference
            frame. Should have dimension 3x3xn where n is the number of time steps.
            The last axis represents time. Typically computed by calcRomanAngles().
        aperture (dict, optional):
            Aperture from getSIAFAperture(). If given, PA_APER = PA_V3 (local, at the
            aperture) + V3IdlYAngle + 180 deg, since CGI LOS_Y is EXCAM -Y. If None,
            the nominal offset of 150 deg from PA_V3 is used.

    Returns:
        astropy.units.Quantity(numpy.ndarray(float)):
            Array of position angles at each time.

    """
    if aperture is None:
        return getRomanPositionAngle(B_C_I) + 150 * u.deg

    return getRomanPositionAngle(B_C_I, aperture) + (aperture["V3IdlYAngle"] + 180) * u.deg
