import numpy as np
import astropy.units as u
from keplertools.angutils import projplane, calcang, rotMat
from astropy.coordinates import (
    SkyCoord,
    get_body_barycentric,
    BarycentricMeanEcliptic,
)
import scipy.optimize


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
            3xn array of sun barycentric positions where n is the size of ts

    """
    earth = SkyCoord(
        get_body_barycentric("Earth", ts), frame="icrs", obstime=ts
    ).transform_to(BarycentricMeanEcliptic)
    r_L2_G = L2loc * earth.cartesian.xyz

    return r_L2_G


def calcRomanAngles(target, ts, r_obs_G, r_sun_G=None):
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

    Returns:
        tuple:
            sun_ang, yaw, pitch, B_C_I

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

    B_C_I = np.dstack(
        [np.matmul(rotMat(1, -a), B_C_I[:, :, j]) for j, a in enumerate(ang2)]
    )

    # now we wish to align b_1 to r_star/obs with yaw, pitch, roll (b_3, b_2, b_1)
    # projection of star/obs vector onto b1/e2 plane:
    r_target_obs_proj1 = np.hstack(
        [
            projplane(np.array(r_target_obs[:, j], ndmin=2).T, B_C_I[2, :, j].T)
            for j in range(len(ts))
        ]
    )
    rhat_target_obs_proj1 = r_target_obs_proj1 / np.linalg.norm(
        r_target_obs_proj1, axis=0
    )

    # yaw is angle between projection and b_1
    b_1 = B_C_I[0, :, :].T
    b_3 = B_C_I[2, :, :].T
    yaw = -np.array(
        [calcang(x, b1, b3) for x, b1, b3 in zip(rhat_target_obs_proj1.T, b_1, b_3)]
    )

    B_C_I = np.dstack(
        [np.matmul(rotMat(3, a), B_C_I[:, :, j]) for j, a in enumerate(yaw)]
    )

    # next we pitch! rotate about b_2 by the angle between b_1 and final look vector
    b_1 = B_C_I[0, :, :].T
    b_2 = B_C_I[1, :, :].T
    pitch = -np.array(
        [calcang(x, b1, b2) for x, b1, b2 in zip(rhat_target_obs.T, b_1, b_2)]
    )

    B_C_I = np.dstack(
        [np.matmul(rotMat(2, a), B_C_I[:, :, j]) for j, a in enumerate(pitch)]
    )

    return sun_ang, yaw * u.rad, pitch * u.rad, B_C_I
