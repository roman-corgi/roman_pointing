import numpy as np
import astropy.units as u
from astropy.time import Time
from astroquery.simbad import Simbad
from astroquery.jplhorizons import Horizons
from astropy.coordinates import (
    SkyCoord,
    Distance,
    get_body_barycentric,
    BarycentricMeanEcliptic,
)
from angutils.angutils import projplane, calcang, rotMat
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib

%matplotlib inline

# Query SIMBAD and convert target coordinates 
def get_target_coords(target_names):
    """Query SIMBAD and convert results to BarycentricMeanEcliptic."""
    simbad = Simbad()
    simbad.add_votable_fields("pmra", "pmdec", "plx_value", "rvz_radvel")

    coords = {}

    for name in target_names:
        res = simbad.query_object(name)
        if res is None:
            print(f"SIMBAD could not find {name}")
            continue

        c_icrs = SkyCoord(
            res["ra"].value.data[0],
            res["dec"].value.data[0],
            unit=(res["ra"].unit, res["dec"].unit),
            frame="icrs",
            distance=Distance(parallax=res["plx_value"].value.data[0] *
                              res["plx_value"].unit),
            pm_ra_cosdec=res["pmra"].value.data[0] * res["pmra"].unit,
            pm_dec=res["pmdec"].value.data[0] * res["pmdec"].unit,
            radial_velocity=res["rvz_radvel"].value.data[0] * res["rvz_radvel"].unit,
            equinox="J2000",
            obstime="J2000"
        ).transform_to(BarycentricMeanEcliptic)

        coords[name] = c_icrs

    return coords

# Find L2 location 
f = (
    lambda x, mustar: x
    - (1 - mustar) * (x + mustar) / np.abs(x + mustar) ** 3
    - mustar * (x - 1 + mustar) / np.abs(x - 1 + mustar) ** 3
)
mustar_sunearth = ((1 * u.Mearth) / (1 * u.Mearth + 1 * u.Msun)).decompose().value
fsunearth = lambda x: f(x, mustar_sunearth)  # noqa
L2loc = scipy.optimize.fsolve(fsunearth, 1)[0]

#Functions for Sun and L2 positions, and Roman angles 
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

#Compute functions for Roman angles and keepout 

def compute_roman_angles(coord, start_date, days, time_step):
    """
    Compute Roman spacecraft sun angle, yaw, and pitch for a single target.
    """
    t0 = Time(start_date, format="isot", scale="utc")
    ts = t0 + np.arange(0, days, time_step) * u.d

    sun_ang, yaw, pitch, BCI = calcRomanAngles(
        coord,
        ts,
        getL2Positions(ts)
    )

    return ts, sun_ang, yaw, pitch



def compute_keepout(coords_dict, start_date, days, time_step,
                    min_sun=54, max_sun=126):
    """
    Compute solar keepout for multiple targets.
    """
    solar_angles = {}
    keepout = {}
    ts_global = None

    for name, coord in coords_dict.items():
        ts, sun_ang, yaw, pitch = compute_roman_angles(
            coord, start_date, days, time_step
        )

        solar_angles[name] = sun_ang
        keepout[name] = (sun_ang > min_sun * u.deg) & (sun_ang < max_sun * u.deg)

        if ts_global is None:
            ts_global = ts

    return ts_global, keepout, solar_angles

# Plotting function for Solar Angle, Plot Pitch, and KeepOut 
def plot_solar_angle(ts, solar_angles_dict):
    fig, ax = plt.subplots(figsize=(10, 5))

    for name, sun in solar_angles_dict.items():
        ax.plot(range(len(ts)), sun.to(u.deg), label=name)

    ax.set_xlabel(f"Time after {ts[0].value} (days)")
    ax.set_ylabel("Solar Angle (deg)")
    ax.axhline(54, color="k", ls="--")
    ax.axhline(126, color="k", ls="--")
    ax.set_title("Solar Angle vs Time")
    ax.legend()

    return fig, ax


def plot_pitch(ts, pitch_dict):
    fig, ax = plt.subplots(figsize=(10, 5))

    for name, pitch in pitch_dict.items():
        ax.plot(range(len(ts)), pitch.to(u.deg), label=name)

    ax.set_xlabel(f"Time after {ts[0].value} (days)")
    ax.set_ylabel("Pitch Angle (deg)")
    ax.set_title("Pitch Angle vs Time")
    ax.legend()

    return fig, ax


def plot_keepout(keepout_dict, ts):
    names = list(keepout_dict.keys())
    M = len(names)

    komap = np.vstack([keepout_dict[n] for n in names])

    fig, ax = plt.subplots(figsize=(12, 1.3 * M + 3))
    cmap = matplotlib.colors.ListedColormap(["black", "green"])

    p = ax.pcolor(np.arange(komap.shape[1]), np.arange(M), komap, cmap=cmap)

    ax.set_yticks(np.arange(M))
    ax.set_yticklabels(names)
    ax.set_xlabel("Time Index")

    cbar = plt.colorbar(p, ticks=[0.25, 0.75])
    cbar.ax.set_yticklabels(["Unavailable", "Available"])

    ax.set_title(f"Roman Keepout Map\n{ts[0].iso} → {ts[-1].iso}")

    return fig, ax
