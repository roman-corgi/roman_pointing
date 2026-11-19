"""Tests of the aperture and ephemeris options of the planning tools (offline: L2 positions)."""

import astropy.units as u
import numpy as np
from astropy.coordinates import BarycentricMeanEcliptic, SkyCoord
from astropy.time import Time

from roman_pointing.roman_excam_planner import calculate_roman_EXCAMangles, compute_angles
from roman_pointing.roman_observability import compute_roman_angles
from roman_pointing.roman_pointing import calcRomanAngles, getL2Positions, getSIAFAperture

TARGET = SkyCoord(ra=164.8667 * u.deg, dec=40.4303 * u.deg, distance=14 * u.pc, pm_ra_cosdec=0 * u.mas / u.yr,
                  pm_dec=0 * u.mas / u.yr, radial_velocity=0 * u.km / u.s, frame="icrs",
                  obstime=Time("J2000")).transform_to(BarycentricMeanEcliptic)
TS = Time("2027-06-01T00:00:00", scale="utc") + np.array([0, 10, 20]) * u.day


def test_observability_boresight_l2():
    # aperture=None, ephemeris="L2" reproduces the previous implementation
    ts, sun_ang, yaw, pitch = compute_roman_angles(TARGET, "2027-06-01T00:00:00", 30, 10, aperture=None, ephemeris="L2")
    ref = calcRomanAngles(TARGET, ts, getL2Positions(ts))
    for a, b in zip((sun_ang, yaw, pitch), ref[:3]):
        np.testing.assert_allclose(a.value, b.value, rtol=0, atol=0)


def test_observability_aperture():
    aperture = getSIAFAperture()
    ts, sun_ang, _, _ = compute_roman_angles(TARGET, "2027-06-01T00:00:00", 30, 10, ephemeris="L2")
    ref = calcRomanAngles(TARGET, ts, getL2Positions(ts), aperture=aperture)
    np.testing.assert_allclose(sun_ang.value, ref[0].value, rtol=0, atol=0)


def test_excam_planner_aperture():
    aperture = getSIAFAperture()
    angles = calculate_roman_EXCAMangles(TARGET, TS, ephemeris="L2")
    # PA_APER = PA_V3 + V3IdlYAngle + 180 deg at the aperture
    pa_aper, pa_v3 = angles["PA_EXCAM_Y"].to_value(u.deg), angles["PA_Z"].to_value(u.deg)
    d = (pa_aper - pa_v3 - (aperture["V3IdlYAngle"] + 180)) % 360
    assert np.all(np.minimum(d, 360 - d) < 1e-9)
    # a roll about the aperture changes PA_APER by exactly -roll
    rolled = compute_angles(TARGET, TS, 10.0, ephemeris="L2")
    d = (rolled["PA_EXCAM_Y_plusroll"].to_value(u.deg) - rolled["PA_EXCAM_Y"].to_value(u.deg) + 10 + 180) % 360 - 180
    assert np.max(np.abs(d)) < 1e-9
