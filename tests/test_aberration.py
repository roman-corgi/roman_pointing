"""Tests of the velocity-aberration option of calcRomanAngles, applyAberration and removeAberration (offline: L2
velocities)."""

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import BarycentricMeanEcliptic, SkyCoord
from astropy.time import Time

from roman_pointing.field_rotation import pa_v3_drift
from roman_pointing.roman_pointing import (
    applyAberration,
    calcRomanAngles,
    getL2Positions,
    getL2Velocities,
    getRomanPositionAngle,
    getSIAFAperture,
    getSunPositions,
    removeAberration,
)

TS = Time("2027-03-01T00:00:00") + np.array([0, 1.5]) * u.day
TARGETS = [(53.2327, -9.4583), (219.9, -60.8), (229.2745, 71.8239), (80.5525, -69.5134)]


def make_target(ra, dec):
    """Distant target without proper motion, in BarycentricMeanEcliptic.

    Args:
        ra (float):
            Right ascension (deg, ICRS)
        dec (float):
            Declination (deg, ICRS)

    Returns:
        astropy.coordinates.SkyCoord:
            Target coordinates
    """
    return SkyCoord(ra=ra * u.deg, dec=dec * u.deg, distance=1e6 * u.pc, pm_ra_cosdec=0 * u.mas / u.yr,
                    pm_dec=0 * u.mas / u.yr, radial_velocity=0 * u.km / u.s, frame="icrs",
                    obstime=Time("J2000")).transform_to(BarycentricMeanEcliptic)


def unit_vectors(target, ts):
    """Unit vectors from L2 to the target and to the Sun.

    Args:
        target (astropy.coordinates.SkyCoord):
            Target coordinates
        ts (astropy.time.Time):
            Observation times

    Returns:
        tuple:
            rhat_target, rhat_sun (3xn numpy.ndarray)
    """
    r_obs = getL2Positions(ts)
    r_target = target.cartesian.xyz[:, None] - r_obs
    r_sun = getSunPositions(ts) - r_obs
    return (r_target / np.linalg.norm(r_target, axis=0)).value, (r_sun / np.linalg.norm(r_sun, axis=0)).value


def pa_by_transport(B_C_I, v_obs_G, u_ref):
    """Catalog-frame PA of b_3 at the reference point, by removing the aberration from two nearby points.

    Args:
        B_C_I (numpy.ndarray):
            3x3 apparent-frame attitude
        v_obs_G (astropy.units.Quantity):
            Observatory velocity (3)
        u_ref (numpy.ndarray):
            Reference point in the body frame

    Returns:
        float:
            Position angle (deg), measured as in getRomanPositionAngle
    """
    p_app = B_C_I.T @ u_ref
    q_app = p_app + 1e-6 * B_C_I[2]
    p_cat = applyAberration(p_app[:, None], v_obs_G, inverse=True)[:, 0]
    q_cat = applyAberration((q_app / np.linalg.norm(q_app))[:, None], v_obs_G, inverse=True)[:, 0]
    # an attitude with b_1 at p_cat and b_3 along the transported direction gives the PA about p_cat
    b3 = q_cat - (q_cat @ p_cat) * p_cat
    b3 /= np.linalg.norm(b3)
    B = np.array([p_cat, np.cross(b3, p_cat), b3])[:, :, None]
    return getRomanPositionAngle(B).to_value(u.deg)[0]


@pytest.fixture(scope="module")
def aperture():
    return getSIAFAperture()


@pytest.fixture(scope="module")
def velocity():
    return getL2Velocities(TS)


def test_l2_velocity(velocity):
    speed = np.linalg.norm(velocity, axis=0).to_value(u.km / u.s)
    assert np.all((speed > 29) & (speed < 31.5))


def test_apply_aberration(velocity):
    rng = np.random.default_rng(1)
    rhat = rng.normal(size=(3, 2))
    rhat /= np.linalg.norm(rhat, axis=0)
    app = applyAberration(rhat, velocity)
    beta = (velocity / (299792.458 * u.km / u.s)).decompose().value
    vhat = beta / np.linalg.norm(beta, axis=0)
    shift = np.arccos(np.clip(np.sum(app * rhat, axis=0), -1, 1))
    sin_theta = np.linalg.norm(np.cross(rhat.T, vhat.T), axis=1)
    np.testing.assert_allclose(shift, np.linalg.norm(beta, axis=0) * sin_theta, rtol=2e-4)
    assert np.all(np.sum(app * vhat, axis=0) > np.sum(rhat * vhat, axis=0))  # displaced toward the velocity
    np.testing.assert_allclose(applyAberration(app, velocity, inverse=True), rhat, rtol=0, atol=1e-14)


@pytest.mark.parametrize("radec", TARGETS)
def test_zero_velocity_unchanged(aperture, radec):
    target = make_target(*radec)
    ref = calcRomanAngles(target, TS, getL2Positions(TS), aperture=aperture)
    zero = calcRomanAngles(target, TS, getL2Positions(TS), aperture=aperture, v_obs_G=np.zeros((3, 2)) * u.km / u.s)
    for a, b in zip(ref, zero):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0, atol=1e-12)


@pytest.mark.parametrize("radec", TARGETS)
def test_apparent_attitude(aperture, velocity, radec):
    target = make_target(*radec)
    sun_ang, _, _, B_C_I = calcRomanAngles(target, TS, getL2Positions(TS), aperture=aperture, v_obs_G=velocity)
    rhat_target, rhat_sun = unit_vectors(target, TS)
    t_app, s_app = applyAberration(rhat_target, velocity), applyAberration(rhat_sun, velocity)
    # the aperture is on the apparent target, and the apparent Sun is in the b_1-b_3 plane
    assert np.max(np.linalg.norm(np.einsum("ijt,i->jt", B_C_I, aperture["u_B"]) - t_app, axis=0)) < 1e-9
    assert np.max(np.abs(np.einsum("jt,jt->t", B_C_I[1], s_app))) < 1e-12
    np.testing.assert_allclose(np.cos(sun_ang.value), np.einsum("jt,jt->t", B_C_I[0], s_app), rtol=0, atol=1e-12)


@pytest.mark.parametrize("radec", TARGETS)
def test_remove_aberration(aperture, velocity, radec):
    target = make_target(*radec)
    B_C_I = calcRomanAngles(target, TS, getL2Positions(TS), aperture=aperture, v_obs_G=velocity)[3]
    B_C_I_cat = removeAberration(B_C_I, velocity, aperture)
    rhat_target, _ = unit_vectors(target, TS)
    # the catalog-frame attitude has the aperture on the catalog target
    assert np.max(np.linalg.norm(np.einsum("ijt,i->jt", B_C_I_cat, aperture["u_B"]) - rhat_target, axis=0)) < 1e-9
    # its PA agrees with removing the aberration from two nearby points
    pa_cat = getRomanPositionAngle(B_C_I_cat, aperture).to_value(u.deg)
    for j in range(len(TS)):
        pa_ref = pa_by_transport(B_C_I[:, :, j], velocity[:, j], aperture["u_B"])
        assert abs((pa_cat[j] - pa_ref + 180) % 360 - 180) < 1e-6
    # apparent and catalog PAs differ by the aberration rotation, at most ~20.6" x tan|Dec|
    d = (getRomanPositionAngle(B_C_I, aperture).to_value(u.deg) - pa_cat + 180) % 360 - 180
    assert np.all(np.abs(d) <= 20.7 / 3600 * np.tan(np.radians(abs(radec[1]) + 1)) + 1e-6)


def test_pa_v3_drift_aberration(aperture):
    args = (229.2745, 71.8239, Time("2026-11-01T04:57:00"), np.linspace(-0.5, 0.5, 5), aperture, "L2")
    plain, aberr = pa_v3_drift(*args), pa_v3_drift(*args, aberration=True)
    d = (aberr["pa_mid"] - plain["pa_mid"] + 180) % 360 - 180
    # for 11 UMi on this date the effect is about the PA_V3 rate times the Sun's 20.5" aberration lag (~8 min)
    assert 0.002 < abs(d) < 0.02
    np.testing.assert_allclose(aberr["rate"], plain["rate"], rtol=0.02)
