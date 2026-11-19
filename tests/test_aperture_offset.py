"""Tests of the SIAF aperture offset in calcRomanAngles, applyRollAngle, getRomanPositionAngle and
getEXCAMPositionAngle."""

import os
import warnings
import xml.etree.ElementTree as ET

import astropy.units as u
import numpy as np
import pytest
from angutils.angutils import rotMat
from astropy.coordinates import BarycentricMeanEcliptic, SkyCoord
from astropy.time import Time

from roman_pointing.field_rotation import (
    ROLL_MAX,
    ROLL_MAX_INSTRUMENT,
    ROLL_MAX_OBSERVATORY,
    SUN_ANG_MAX,
    SUN_ANG_MIN,
    pa_v3_drift,
)
from roman_pointing.roman_pointing import (
    SIAF_FILE,
    applyRollAngle,
    calcRomanAngles,
    getEXCAMPositionAngle,
    getL2Positions,
    getRomanPositionAngle,
    getRomanRollAngle,
    getSIAFAperture,
    getSunPositions,
)

TS = Time("2027-03-01T00:00:00") + np.array([0, 1.5]) * u.day

# outputs of the main branch (no aperture offset) at TS: sun angle, yaw, pitch, PA_V3, PA_V3 after rolls of
# (+10, -5) deg, and EXCAM PA after those rolls (deg)
REFERENCE = {
    (53.2327, -9.4583): [
        [70.960758579816, 69.651444112988], [60.527256962258, 60.261006128811],
        [-19.039241420184, -20.348555887012], [-95.155687693017, -94.365606053456],
        [-105.155687693017, -89.365606053456], [44.844312306983, 60.634393946544]],
    (219.9, -60.8): [
        [97.591278627026, 98.690711828242], [-46.969572913427, -46.822335075242],
        [7.591278627026, 8.690711828242], [121.506204753427, 122.545705003482],
        [111.506204753427, 127.545705003482], [261.506204753427, 277.545705003482]],
    (150.7, 73.8): [
        [113.799541198628, 113.147445539763], [154.633264266556, 154.043799316265],
        [23.799541198628, 23.147445539763], [-11.528299867311, -13.008126682605],
        [-21.528299867311, -8.008126682605], [128.471700132689, 141.991873317395]],
}


def make_target(ra, dec):
    """Target at 10 pc without proper motion, in BarycentricMeanEcliptic.

    Args:
        ra (float or numpy.ndarray):
            Right ascension (deg, ICRS)
        dec (float or numpy.ndarray):
            Declination (deg, ICRS)

    Returns:
        astropy.coordinates.SkyCoord:
            Target coordinates
    """
    return SkyCoord(ra=ra * u.deg, dec=dec * u.deg, distance=10 * u.pc, pm_ra_cosdec=0 * u.mas / u.yr,
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


def wrap(angle):
    """Wrap an angle to [-180, 180) deg.

    Args:
        angle (astropy.units.Quantity):
            Angle

    Returns:
        numpy.ndarray:
            Wrapped angle (deg)
    """
    return (angle.to_value(u.deg) + 180) % 360 - 180


@pytest.fixture(scope="module")
def aperture():
    return getSIAFAperture()


@pytest.fixture(scope="module")
def for_targets():
    """Random targets inside the field of regard at TS[0], with aperture attitudes."""
    rng = np.random.default_rng(20261021)
    ra = rng.uniform(0, 360, 400)
    dec = np.degrees(np.arcsin(rng.uniform(-1, 1, 400)))
    targets = []
    for r, d in zip(ra, dec):
        target = make_target(r, d)
        sun_ang = calcRomanAngles(target, TS[:1], getL2Positions(TS[:1]))[0].to_value(u.deg)[0]
        if SUN_ANG_MIN + 1 < sun_ang < SUN_ANG_MAX - 1:
            targets.append(target)
        if len(targets) == 25:
            break
    return targets


@pytest.mark.parametrize("radec", REFERENCE)
def test_default_matches_main(radec):
    target = make_target(*radec)
    sun_ang, yaw, pitch, B_C_I = calcRomanAngles(target, TS, getL2Positions(TS))
    B_C_I_roll = applyRollAngle(B_C_I, [10, -5] * u.deg)
    out = [sun_ang, yaw, pitch, getRomanPositionAngle(B_C_I), getRomanPositionAngle(B_C_I_roll),
           getEXCAMPositionAngle(B_C_I_roll)]
    for value, ref in zip(out, REFERENCE[radec]):
        np.testing.assert_allclose(value.to_value(u.deg), ref, rtol=0, atol=1e-9)


def test_siaf_extract(aperture):
    # the packaged extract holds only the parameters roman_pointing uses
    for entry in ET.parse(SIAF_FILE).getroot().iter("SiafEntry"):
        assert [e.tag for e in entry] == ["AperName", "V2Ref", "V3Ref", "V3IdlYAngle"]
    assert aperture["V2Ref"] == -1104.214 and aperture["V3Ref"] == -840.945
    # V3IdlYAngle near -30 deg (SIAF releases from 20260928); older releases give ~240 deg, which would make
    # PA_APER = PA_V3 + V3IdlYAngle + 180 deg wrong by 90 deg. The wide tolerance allows later calibration updates.
    assert abs(aperture["V3IdlYAngle"] + 30) < 10
    assert np.isclose(np.linalg.norm(aperture["u_B"]), 1, rtol=0, atol=1e-15)
    with pytest.raises(ValueError):
        getSIAFAperture("WFI01_FULL")


def test_aperture_on_target(aperture, for_targets):
    v3ref = (aperture["V3Ref"] * u.arcsec).to_value(u.deg)
    for target in for_targets:
        sun_ang_v1, _, _, B_C_I = calcRomanAngles(target, TS, getL2Positions(TS), aperture=aperture)
        sun_ang = calcRomanAngles(target, TS, getL2Positions(TS))[0]
        rhat_target, rhat_sun = unit_vectors(target, TS)
        rhat_aper = np.einsum("ijt,i->jt", B_C_I, aperture["u_B"])
        assert np.max(np.linalg.norm(rhat_aper - rhat_target, axis=0)) < 1e-9
        # nominal attitude: the Sun is in the b_1-b_3 plane
        assert np.max(np.abs(np.einsum("jt,jt->t", B_C_I[1], rhat_sun))) < 1e-12
        # the Sun angle refers to V1, displaced from the target by V3Ref along the Sun great circle
        np.testing.assert_allclose((sun_ang_v1 - sun_ang).to_value(u.deg), v3ref, rtol=0, atol=1e-3)
        np.testing.assert_allclose(np.arccos(np.einsum("jt,jt->t", B_C_I[0], rhat_sun)), sun_ang_v1.value,
                                   rtol=0, atol=1e-12)


def test_aperture_iteration_fallback(aperture, for_targets):
    target = for_targets[0]
    # the CGI aperture converges without a warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        calcRomanAngles(target, TS, getL2Positions(TS), aperture=aperture)
    # an aperture 80 deg from V1 does not converge; the attitude falls back to V1 on the target
    v2 = np.radians(80.0)
    far = dict(aperture, V2Ref=80.0 * 3600, V3Ref=0.0, u_B=np.array([np.cos(v2), np.sin(v2), 0.0]))
    with pytest.warns(UserWarning, match="did not converge"):
        sun_ang, yaw, pitch, B_C_I = calcRomanAngles(target, TS, getL2Positions(TS), aperture=far)
    sun_ang_v1, yaw_v1, pitch_v1, B_C_I_v1 = calcRomanAngles(target, TS, getL2Positions(TS))
    np.testing.assert_allclose(B_C_I, B_C_I_v1, rtol=0, atol=1e-12)
    for value, ref in ((sun_ang, sun_ang_v1), (yaw, yaw_v1), (pitch, pitch_v1)):
        np.testing.assert_allclose(value.to_value(u.rad), ref.to_value(u.rad), rtol=0, atol=1e-12)

def test_local_pa_change(aperture, for_targets):
    v2 = np.radians(aperture["V2Ref"] / 3600)
    v3 = np.radians(aperture["V3Ref"] / 3600)
    for target in for_targets:
        sun_ang, _, _, B_C_I = calcRomanAngles(target, TS, getL2Positions(TS))
        B_C_I_aper = calcRomanAngles(target, TS, getL2Positions(TS), aperture=aperture)[3]
        dpa = wrap(getRomanPositionAngle(B_C_I_aper, aperture) - getRomanPositionAngle(B_C_I))
        expected = np.degrees(v2 / np.tan(sun_ang.value) - v2 * v3)
        np.testing.assert_allclose(dpa, expected, rtol=0, atol=1e-5)


def test_pa_v1_equals_default_without_offset(aperture, for_targets):
    # an aperture at V1 (u_B = e_1) reproduces the default attitude, roll and PA
    v1 = dict(aperture, V2Ref=0.0, V3Ref=0.0, u_B=np.array([1.0, 0.0, 0.0]))
    target = for_targets[0]
    B_C_I = calcRomanAngles(target, TS, getL2Positions(TS))[3]
    B_C_I_v1 = calcRomanAngles(target, TS, getL2Positions(TS), aperture=v1)[3]
    np.testing.assert_allclose(B_C_I_v1, B_C_I, rtol=0, atol=1e-12)
    roll = [12, -7] * u.deg
    np.testing.assert_allclose(applyRollAngle(B_C_I, roll, v1), applyRollAngle(B_C_I, roll), rtol=0, atol=1e-15)
    np.testing.assert_allclose(getRomanPositionAngle(B_C_I, v1), getRomanPositionAngle(B_C_I), rtol=0, atol=1e-12)
    # bare floats are radians, as for rotMat
    np.testing.assert_allclose(applyRollAngle(B_C_I, [0.2, -0.1], v1)[:, :, 0], rotMat(1, 0.2) @ B_C_I[:, :, 0],
                               rtol=0, atol=1e-15)


def test_roll_about_aperture(aperture, for_targets):
    roll = [15, -15] * u.deg
    for target in for_targets:
        B_C_I = calcRomanAngles(target, TS, getL2Positions(TS), aperture=aperture)[3]
        B_C_I_roll = applyRollAngle(B_C_I, roll, aperture)
        rhat_target, _ = unit_vectors(target, TS)
        rhat_aper = np.einsum("ijt,i->jt", B_C_I_roll, aperture["u_B"])
        assert np.max(np.linalg.norm(rhat_aper - rhat_target, axis=0)) < 1e-9
        dpa = wrap(getRomanPositionAngle(B_C_I_roll, aperture) - getRomanPositionAngle(B_C_I, aperture))
        np.testing.assert_allclose(dpa, -roll.to_value(u.deg), rtol=0, atol=1e-9)


def test_excam_pa(aperture, for_targets):
    B_C_I = calcRomanAngles(for_targets[0], TS, getL2Positions(TS), aperture=aperture)[3]
    diff = getEXCAMPositionAngle(B_C_I, aperture) - getRomanPositionAngle(B_C_I, aperture)
    np.testing.assert_allclose(diff.to_value(u.deg), aperture["V3IdlYAngle"] + 180, rtol=0, atol=1e-12)


def test_pa_v3_drift_aperture(aperture):
    default = pa_v3_drift(53.2327, -9.4583, Time("2027-01-20T12:00:00"), np.linspace(-1, 1, 5), None, "L2")
    offset = pa_v3_drift(53.2327, -9.4583, Time("2027-01-20T12:00:00"), np.linspace(-1, 1, 5), aperture, "L2")
    # an aperture name is equivalent to the aperture
    by_name = pa_v3_drift(53.2327, -9.4583, Time("2027-01-20T12:00:00"), np.linspace(-1, 1, 5), "CGI_CEN", "L2")
    np.testing.assert_allclose(by_name["s"], offset["s"], rtol=0, atol=0)
    # PA_V3 changes by about V2Ref / tan(eps); the Sun angle of V1 by V3Ref
    v2 = np.radians(aperture["V2Ref"] / 3600)
    expected = np.degrees(v2 / np.tan(np.radians(default["sun_ang"][2])))
    assert abs(offset["pa_mid"] - default["pa_mid"] - expected) < 2e-3
    np.testing.assert_allclose(offset["sun_ang"] - default["sun_ang"], aperture["V3Ref"] / 3600, atol=1e-3)
    with pytest.raises(ValueError):
        pa_v3_drift(53.2327, -9.4583, Time("2027-01-20T12:00:00"), np.linspace(-1, 1, 5), ephemeris="SOC")


def test_roll_angle_about_v1(aperture, for_targets):
    for target in for_targets:
        sun_ang, _, _, B_C_I = calcRomanAngles(target, TS, getL2Positions(TS), aperture=aperture)
        _, rhat_sun = unit_vectors(target, TS)
        # nominal attitude: no roll about V1
        np.testing.assert_allclose(getRomanRollAngle(B_C_I, rhat_sun).to_value(u.deg), 0, rtol=0, atol=1e-9)
        # a roll about V1 gives exactly that roll angle, in the applyRollAngle sign convention
        B_C_I_v1 = applyRollAngle(B_C_I, [12, -7] * u.deg)
        np.testing.assert_allclose(getRomanRollAngle(B_C_I_v1, rhat_sun).to_value(u.deg), [12, -7], rtol=0,
                                   atol=1e-9)
        # a 15 deg roll about the aperture is within 0.05 deg of 15 deg about V1: more for eps < 90 deg, less above
        roll_v1 = getRomanRollAngle(applyRollAngle(B_C_I, [15, -15] * u.deg, aperture), rhat_sun).to_value(u.deg)
        eps = sun_ang.to_value(u.deg)[0]
        assert np.all(np.abs(np.abs(roll_v1) - 15) < 0.05)
        if abs(eps - 90) > 2:
            assert np.all(np.sign(np.abs(roll_v1) - 15) == np.sign(90 - eps))
        # the aperture roll that gives +/-15 deg about V1, by the iteration of plan_roll_limits.py
        for sign in (1, -1):
            a = np.full(TS.size, sign * 15.0)
            for _ in range(3):
                a += sign * 15 - getRomanRollAngle(applyRollAngle(B_C_I, a * u.deg, aperture), rhat_sun).to_value(u.deg)
            roll_v1 = getRomanRollAngle(applyRollAngle(B_C_I, a * u.deg, aperture), rhat_sun).to_value(u.deg)
            np.testing.assert_allclose(roll_v1, sign * 15, rtol=0, atol=1e-8)  # error 0.05 deg / 300**3
    # pa_v3_drift returns the nominal attitude and the Sun direction it was solved with (physical and apparent with
    # aberration), so their roll about V1 is 0
    for aberration in (False, True):
        drift = pa_v3_drift(53.2327, -9.4583, Time("2027-01-20T12:00:00"), np.linspace(-1, 1, 5), aperture, "L2",
                            aberration)
        np.testing.assert_allclose(getRomanRollAngle(drift["B_C_I"], drift["rhat_sun"]).to_value(u.deg), 0, rtol=0,
                                   atol=1e-9)

def test_compound_roll_limit(aperture, for_targets):
    # the boresight model uses the stricter of the two limits, which apply to the same axis when V1 is on the target
    assert ROLL_MAX == min(ROLL_MAX_OBSERVATORY, ROLL_MAX_INSTRUMENT)
    for target in for_targets:
        sun_ang, _, _, B_C_I = calcRomanAngles(target, TS, getL2Positions(TS), aperture=aperture)
        _, rhat_sun = unit_vectors(target, TS)
        eps = sun_ang.to_value(u.deg)
        for sign in (1, -1):
            # aperture roll for the observatory limit about V1, then the stricter limit as in plan_roll_limits.py
            a = np.full(TS.size, sign * ROLL_MAX_OBSERVATORY)
            for _ in range(3):
                a += sign * ROLL_MAX_OBSERVATORY - getRomanRollAngle(applyRollAngle(B_C_I, a * u.deg, aperture),
                                                                     rhat_sun).to_value(u.deg)
            limit = sign * np.minimum(sign * a, ROLL_MAX_INSTRUMENT)
            roll_v1 = getRomanRollAngle(applyRollAngle(B_C_I, limit * u.deg, aperture), rhat_sun).to_value(u.deg)
            # both limits hold, and one of them is reached
            assert np.all(np.abs(limit) <= ROLL_MAX_INSTRUMENT + 1e-12)
            assert np.all(np.abs(roll_v1) <= ROLL_MAX_OBSERVATORY + 1e-8)
            # sunward of quadrature the observatory limit about V1 is the stricter one, beyond it the aperture limit
            sunward, antisunward = eps < 88, eps > 92
            np.testing.assert_allclose(roll_v1[sunward], sign * ROLL_MAX_OBSERVATORY, rtol=0, atol=1e-8)
            np.testing.assert_allclose(limit[antisunward], sign * ROLL_MAX_INSTRUMENT, rtol=0, atol=1e-12)

def test_pysiaf_conventions(aperture, for_targets):
    pysiaf = pytest.importorskip("pysiaf")
    from pysiaf.utils.rotations import attitude, posangle

    # for Roman, pysiaf ignores filename unless basepath is also given
    siaf = pysiaf.Siaf("Roman", basepath=os.path.dirname(SIAF_FILE), filename=os.path.basename(SIAF_FILE))
    siaf_aperture = siaf["CGI_CEN"]
    assert siaf_aperture.V3IdlYAngle == aperture["V3IdlYAngle"]
    for target in for_targets[:5]:
        B_C_I = calcRomanAngles(target, TS[:1], getL2Positions(TS[:1]), aperture=aperture)[3]
        icrs = target.transform_to("icrs")
        pa_local = getRomanPositionAngle(B_C_I, aperture).to_value(u.deg)[0]
        att = attitude(aperture["V2Ref"], aperture["V3Ref"], icrs.ra.deg, icrs.dec.deg, pa_local)
        # PA_V3 at V1 from pysiaf equals the PA measured about b_1
        pa_v1 = getRomanPositionAngle(B_C_I).to_value(u.deg)[0]
        assert abs((posangle(att, 0, 0) - pa_v1 + 180) % 360 - 180) < 1e-3
