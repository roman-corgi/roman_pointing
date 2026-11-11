"""Tests of getRomanPositions (JPL Horizons trajectory of Roman). Skipped when Horizons is unreachable."""

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import BarycentricMeanEcliptic, CartesianRepresentation, SkyCoord
from astropy.time import Time
from astroquery.jplhorizons import Horizons

from roman_pointing.field_rotation import pa_v3_drift
from roman_pointing.roman_pointing import (
    genRomanOrbitInterpolant,
    getL2Positions,
    getRomanPositions,
    getRomanVelocities,
    getSIAFAperture,
    getSunPositions,
    queryRomanOrbit,
    queryRomanOrbitSpan,
)

T0 = Time("2026-10-26T00:00:00", scale="tdb")


@pytest.fixture(scope="module", autouse=True)
def horizons():
    try:
        queryRomanOrbit(T0)
    except Exception as err:
        pytest.skip(f"JPL Horizons not reachable: {err}")


def to_ecliptic(vectors_table):
    """Convert the ICRF positions of a Horizons vectors table to BarycentricMeanEcliptic.

    Args:
        vectors_table (astropy.table.Table):
            Horizons vectors table (refplane="earth", AU)

    Returns:
        astropy.units.Quantity:
            3xn positions (AU)
    """
    xyz = np.vstack([np.asarray(vectors_table[c]) for c in "xyz"]) * u.AU
    return SkyCoord(CartesianRepresentation(xyz), frame="icrs").transform_to(
        BarycentricMeanEcliptic).cartesian.xyz


def test_distance_from_l2():
    ts = Time("2026-10-01T00:00:00") + np.arange(0, 720, 30) * u.day
    dist = np.linalg.norm(getRomanPositions(ts) - getL2Positions(ts), axis=0).to_value(u.km)
    assert np.all((dist > 2e5) & (dist < 9e5))
    assert getRomanPositions(ts[0]).shape == (3,)


def test_interpolation():
    ts = T0 + np.arange(0, 48) * u.hour
    hourly = Horizons(id="-211", location="@0", epochs={"start": "2026-10-26 00:00", "stop": "2026-10-27 23:00",
                                                        "step": "1h"}).vectors(refplane="earth", cache=False)
    np.testing.assert_allclose(hourly["datetime_jd"].data, ts.jd)
    dist = np.linalg.norm(getRomanPositions(ts) - to_ecliptic(hourly), axis=0).to_value(u.km)
    assert dist.max() < 1


def test_orbit_interpolant():
    table = queryRomanOrbit(T0 + np.arange(0, 3) * u.day)
    spline = genRomanOrbitInterpolant(table)
    nodes = np.vstack([np.asarray(table[c]) for c in "xyz"]).T
    np.testing.assert_allclose(spline(Time(table["datetime_jd"].data, format="jd", scale="tdb").mjd), nodes, rtol=0, atol=1e-15)
    # a table in km gives the same interpolant in AU
    table_km = table.copy()
    for c in "xyz":
        table_km[c] = (table[c].data * u.AU).to(u.km)
    np.testing.assert_allclose(genRomanOrbitInterpolant(table_km)(T0.tdb.mjd), spline(T0.tdb.mjd), rtol=1e-12)


def test_sun_direction():
    ts = T0 + np.arange(0, 48, 6) * u.hour
    sun = Horizons(id="10", location="500@-211", epochs={"start": "2026-10-26 00:00", "stop": "2026-10-27 18:00",
                                                         "step": "6h"}).vectors(refplane="earth", cache=False)
    sun_horizons = to_ecliptic(sun).value
    sun_model = (getSunPositions(ts) - getRomanPositions(ts)).value
    cos_ang = np.sum(sun_horizons * sun_model, axis=0) / np.linalg.norm(sun_horizons, axis=0) / np.linalg.norm(
        sun_model, axis=0)
    assert np.degrees(np.arccos(np.minimum(cos_ang, 1))).max() * 3600 < 1


@pytest.mark.filterwarnings("ignore:ERFA function")
def test_outside_coverage():
    ts = Time("2040-06-01T00:00:00") + np.arange(3) * u.day
    with pytest.warns(UserWarning, match="Horizons query failed"):
        r = getRomanPositions(ts)
    np.testing.assert_allclose(r, getL2Positions(ts))


@pytest.mark.filterwarnings("ignore:ERFA function")
def test_partial_coverage():
    # times before and after the Horizons coverage get L2, the others Horizons
    ts = Time(["2026-01-01T00:00:00", "2027-06-01T00:00:00", "2040-01-01T00:00:00"])
    with pytest.warns(UserWarning, match="2 of 3 times are outside the Horizons table"):
        r = getRomanPositions(ts)
    # same orbit to < 1.5 km (the interpolation of a different table span)
    np.testing.assert_allclose(r[:, 1].value, getRomanPositions(ts[1]).value, rtol=0, atol=1e-8)
    np.testing.assert_allclose(r[:, [0, 2]], getL2Positions(ts[[0, 2]]), rtol=0, atol=1e-15)
    with pytest.warns(UserWarning, match="outside the Horizons table"):
        v = getRomanVelocities(ts)
    speed = np.linalg.norm(v, axis=0).to_value(u.km / u.s)
    assert np.all((speed > 29) & (speed < 31.5))


def test_query_cache():
    ts = T0 + np.arange(0, 5) * u.day
    getRomanPositions(ts)
    hits = queryRomanOrbitSpan.cache_info().hits
    getRomanPositions(ts)
    assert queryRomanOrbitSpan.cache_info().hits == hits + 1


def test_pa_v3_drift_ephemeris():
    args = (150.72, 73.79, Time("2026-10-26T19:00:00"), np.linspace(-1, 1, 5), None)
    # the default ephemeris is Horizons
    np.testing.assert_allclose(pa_v3_drift(*args)["s"], pa_v3_drift(*args, ephemeris="horizons")["s"], rtol=0, atol=0)
    horizons = pa_v3_drift(*args, ephemeris="horizons")
    default = pa_v3_drift(*args, ephemeris="L2")
    # differences of a few tenths of a degree at most (docs/attitude_model/README.md, section 4.1)
    assert 0 < abs(horizons["pa_mid"] - default["pa_mid"]) < 1
    assert 0 < np.max(np.abs(horizons["sun_ang"] - default["sun_ang"])) < 1


def test_roman_velocity():
    ts = T0 + np.arange(0, 48, 6) * u.hour
    v = getRomanVelocities(ts)
    v_fd = ((getRomanPositions(ts + 1 * u.min) - getRomanPositions(ts - 1 * u.min)) / (2 * u.min)).to(u.km / u.s)
    np.testing.assert_allclose(v.to_value(u.km / u.s), v_fd.value, rtol=0, atol=1e-3)
    speed = np.linalg.norm(v, axis=0).to_value(u.km / u.s)
    assert np.all((speed > 29) & (speed < 31.5))
    assert getRomanVelocities(ts[0]).shape == (3,)


@pytest.mark.parametrize("name, radec, t_plan, v3pa_t", [
    ("CAR-55.1", (80.552486, -69.513382), "2026-10-18T04:44:00", 122.989),
    ("CAR-55.4", (80.552486, -69.513382), "2026-10-22T07:03:46", 127.041),
    ("CAR-193.1", (229.2745, 71.82388889), "2026-11-01T04:57:00", 193.115),
])
def test_opup_planned_pa(name, radec, t_plan, v3pa_t):
    # OPUP "CGI Fixed v3pa@T" (catalog frame) at the visit-slew start, reproduced by the full model:
    # CGI aperture on the target, Horizons position, velocity aberration of target and Sun
    aperture = getSIAFAperture()
    d = pa_v3_drift(*radec, Time(t_plan, scale="utc"), np.array([-0.01, 0, 0.01]), aperture, "horizons", aberration=True)
    assert abs((v3pa_t - d["pa_mid"] + 180) % 360 - 180) < 0.005
