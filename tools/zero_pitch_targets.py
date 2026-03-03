"""
Command line tool to find objects with zero pitch at a given time

Usage:
  python zero_pitch_targets.py -t 2026-11-17T00:00:00.0 -b 2.0

"""

import argparse

from roman_pointing.roman_pointing import (
    calcRomanAngles,
    getL2Positions,
    getSunPositions,
)
from astropy.table import Column
from astropy.time import Time
from astroquery.simbad import Simbad
import numpy as np
import astropy.units as u
from angutils.angutils import projplane, calcang, rotMat
from astroquery.simbad import Simbad
from astropy.coordinates import (
    SkyCoord,
    Distance,
    get_body_barycentric,
    BarycentricMeanEcliptic,
)

def get_zero_pitch_azel_from_yaw(ts, yaw):
    """
    Return the azimuth + elevation for a given time and yaw at zero pitch

    Args:
        ts (astropy.time.Time):
            Observation time to evaluate at
        yaw (numpy.ndarray(float)):
            Yaw value to use, can be an array of values

    Returns:
        tuple:
            az, el (both numpy.ndarray(float) of the same size as yaw)

    """
    # assume observatory is at exactly L2
    r_obs_G = getL2Positions(ts)

    # get sun position and unit vector wrt observatory
    r_sun_G = getSunPositions(ts)
    r_sun_obs = r_sun_G - r_obs_G
    rhat_sun_obs = (r_sun_obs / np.linalg.norm(r_sun_obs, axis=0)).value

    # define inertial basis vectors
    # e1 = np.array([1, 0, 0])
    e2 = np.array([0, 1, 0])
    e3 = np.array([0, 0, 1])

    # align b_3 with -r_obs/sun (equivalently r_sun/obs)
    # first a rotation about b_2 by the angle between b_3 and the projection
    # of the sun/obs vector onto the e1/e3 plane

    # projection of sun/obs vector onto e1/e3 plane:
    r_sun_obs_proj1 = projplane(r_sun_obs, e2)
    rhat_sun_obs_proj1 = (r_sun_obs_proj1 / np.linalg.norm(r_sun_obs_proj1,
                                                           axis=0)).value
    ang1 = calcang(rhat_sun_obs_proj1, e3, e2)
    B_C_I = rotMat(2, -ang1)  # DCM between inertial and body frames

    # second a rotation about b_1 by the angle between the new b_3 and
    # r_sun/obs
    b_3 = B_C_I[2]
    b_1 = B_C_I[0]
    ang2 = calcang(rhat_sun_obs, b_3, b_1)
    B_C_I = np.matmul(rotMat(1, -ang2), B_C_I)

    # consistency check - at this point b_3 should be rhat_sun_obs
    assert np.max(np.abs(np.array(B_C_I[2], ndmin=2).T - rhat_sun_obs)) \
        < np.spacing(2 * np.pi)

    # we're interested in the full range of yaws - for now, brute force this
    yaw = np.linspace(0, 2 * np.pi, 100)
    rhat_targ_obs = np.vstack([np.matmul(rotMat(3, a), B_C_I)[0] for a in yaw])

    # decompose to spherical coords as if this was barycentric
    az = np.atan2(rhat_targ_obs[:, 1], rhat_targ_obs[:, 0])
    el = np.asin(rhat_targ_obs[:, 2])

    return az, el



if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        prog="zero_pitch_targets.py",
        description="Find objects with zero pitch at a given time",
    )
    ap.add_argument('-t', '--time', type=str, help="UTC timestamp to evaluate at, in ISO format")
    ap.add_argument('-b', '--bound', type=float, help="Permitted +/- tolerance on pitch=0, in degrees")
    ap.add_argument('-o', '--output', default=None, help="File to write output to.  If not supplied, prints only to stdout")
    ap.add_argument('--vmag', type=float, default=5.0, help="Lower bound on V-magnitude, defaults to 5.0 if not supplied")
    ap.add_argument('--quiet', action="store_true", help="If this flag is present, does not print to stdout")
    args = ap.parse_args()

    if "T" in args.time:
        f = "isot"
    else:
        f = "iso"
    t0 = Time([args.time], format=f, scale="utc")
    bound = args.bound*u.deg

    # Use the tolerance bound to select the yaw sampling + search radius.
    # Since we need to make a band and only have circles, we'll oversize the
    # circles we pull from simbad enough that they exactly cover the band, and
    # then do a second cut to remove stragglers from the bumps where the
    # circles are oversized
    radius = (bound*2/np.sqrt(3)).to(u.rad)
    npts = np.ceil((2*np.pi*u.rad)/radius) # round up so we don't lose pts at the end
    yaw = np.arange(npts)*radius

    az, el = get_zero_pitch_azel_from_yaw(t0, yaw)

    Simbad.add_votable_fields(
        "ra",
        "dec",
        "flux",
        "pmra",
        "pmdec",
        "plx_value",
        "rvz_radvel",
    )

    # let's see what we can find, cutting on Vmag
    goodres = Simbad.query_region(
        SkyCoord(
            az,
            el,
            unit=(u.rad, u.rad),
            frame="barycentricmeanecliptic"
        ),
        radius=radius,
        criteria=f"flux.filter = 'V' AND flux < {float(args.vmag)}",
    )
    nrows = len(goodres)
    goodres.add_column(Column(name='pitch', data=[None]*nrows, unit=u.deg))
    goodres.add_column(Column(name='yaw', data=[None]*nrows, unit=u.deg))

    discard = []
    for index in range(nrows):
        target = SkyCoord(
            goodres[index]["ra"],
            goodres[index]["dec"],
            unit=(goodres["ra"].unit, goodres["dec"].unit),
            frame="icrs",
            distance=Distance(
                parallax=goodres[index]["plx_value"]
                *goodres["plx_value"].unit),
            pm_ra_cosdec=goodres[index]["pmra"]*goodres["pmra"].unit,
            pm_dec=goodres[index]["pmdec"]*goodres["pmdec"].unit,
            radial_velocity=goodres[index]["rvz_radvel"]*goodres["rvz_radvel"].unit,
            equinox="J2000",
            obstime="J2000",
        ).transform_to(BarycentricMeanEcliptic)

        _, yaw, pitch, _ = calcRomanAngles(
            target,
            t0,
            getL2Positions(t0)
        )
        goodres[index]["pitch"] = pitch.to_value(u.deg)[0]
        goodres[index]["yaw"] = yaw.to_value(u.deg)[0]
        if np.abs(pitch) > bound or np.isnan(pitch) or np.isnan(yaw):
            discard.append(index)

    goodres.remove_rows(discard)

    if not args.quiet:
        print(goodres)

    if args.output is not None:
        goodres.write(args.output, format='ascii.csv', overwrite=True)
