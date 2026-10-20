"""Field rotation of the Roman nominal-roll attitude.

PA_V3 is the position angle of the observatory +Z body axis (V3), the body vector
normal to the sunshield, measured east of north. In the nominal attitude +Z is the Sun
direction projected onto the sky at the target, so PA_V3 is the position angle of the
Sun as seen from the target. With target ecliptic coordinates (lam, beta), solar
ecliptic longitude lam_sun(t), and Delta = lam_sun - lam:

    theta(t)    = atan2(sin Delta, -sin(beta) cos Delta)   (PA from ecliptic north)
    PA_V3(t)    = theta(t) + eta(lam, beta)                (eta constant per target)
    cos eps     = cos(beta) cos Delta                      (eps = Sun-target angle)
    dPA_V3/dt   = -(dlam_sun/dt) sin(beta) / sin^2(eps)

PA_EXCAM_Y ≈ PA_V3 + 150 deg (nominal offset, to be replaced by the calibrated SIAF
value) and any roll offset shift PA_V3 by a constant, so they rotate at the same rate.
See docs/field_rotation_rate/README.md.
"""

import astropy.units as u
import erfa
import numpy as np
from astropy.coordinates import BarycentricMeanEcliptic, SkyCoord
from astropy.time import Time

from roman_pointing.roman_pointing import (
    calcRomanAngles,
    getL2Positions,
    getRomanPositionAngle,
)

SUN_RATE = 0.98565  # mean rate of solar ecliptic longitude (deg/day)
# J2000 obliquity of the ecliptic, IAU 2006 (the model of BarycentricMeanEcliptic) (deg)
OBLIQUITY = np.degrees(erfa.obl06(2451545.0, 0.0))
SUN_ANG_MIN = 54.0  # field of regard limits on Sun-target angle (deg)
SUN_ANG_MAX = 126.0
ROLL_MAX = 15.0  # allowed off-nominal roll (deg)


def pa_from_ecliptic_north(delta, beta):
    """Position angle (E of ecliptic N) of the Sun's direction as seen from the target.

    Args:
        delta (numpy.ndarray or float):
            Solar ecliptic longitude minus target ecliptic longitude (deg)
        beta (numpy.ndarray or float):
            Target ecliptic latitude (deg)

    Returns:
        numpy.ndarray or float:
            Position angle theta (deg), wrapped to (-180, 180]
    """
    d = np.radians(delta)
    b = np.radians(beta)
    return np.degrees(np.arctan2(np.sin(d), -np.sin(b) * np.cos(d)))


def pa_v3_rate(delta, beta, sun_rate=SUN_RATE):
    """Analytic rate of change of PA_V3 (and PA_EXCAM_Y).

    Args:
        delta (numpy.ndarray or float):
            Solar ecliptic longitude minus target ecliptic longitude (deg)
        beta (numpy.ndarray or float):
            Target ecliptic latitude (deg)
        sun_rate (numpy.ndarray or float):
            Rate of change of solar ecliptic longitude (deg/day)

    Returns:
        numpy.ndarray or float:
            dPA_V3/dt (deg/day)
    """
    d = np.radians(delta)
    b = np.radians(beta)
    sin2_eps = 1 - np.cos(b) ** 2 * np.cos(d) ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        return -sun_rate * np.sin(b) / sin2_eps


def sun_longitude_doy(doy, year):
    """Solar mean ecliptic longitude vs. day of year (circular-orbit approximation).

    The Sun's geocentric mean longitude is the Earth's heliocentric mean longitude
    (IERS 2003, erfa.fae03) plus 180 deg, referred to the J2000 ecliptic and equinox as
    in BarycentricMeanEcliptic. It omits the equation of center (up to ±1.9 deg); use
    getSunPositions for accurate positions.

    Args:
        doy (numpy.ndarray or float):
            Day of year (1.0 = Jan 1, 0h UTC)
        year (int):
            Calendar year

    Returns:
        numpy.ndarray or float:
            lam_sun (deg); its rate is 0.98561 deg/day
    """
    t = (Time(f"{year}-01-01T00:00:00", scale="utc") + (np.asarray(doy) - 1) * u.day).tt
    T = (t.jd1 - 2451545.0 + t.jd2) / 36525  # Julian centuries (TT) since J2000.0
    return (np.degrees(erfa.fae03(T)) + 180) % 360


def ecliptic_north_offset(lam, beta):
    """Constant offset eta such that PA_V3 = theta + eta (PA of ecliptic N from celestial N).

    Args:
        lam (float):
            Target ecliptic longitude (deg)
        beta (float):
            Target ecliptic latitude (deg)

    Returns:
        float:
            eta (deg)
    """
    l = np.radians(lam)
    b = np.radians(beta)
    i = np.radians(OBLIQUITY)
    # PA of the celestial pole measured from ecliptic north is -eta
    return -np.degrees(
        np.arctan2(
            np.sin(i) * np.cos(l),
            np.cos(b) * np.cos(i) - np.sin(b) * np.sin(i) * np.sin(l),
        )
    )


def for_window(beta):
    """Duration and PA_V3 swing of one field-of-regard visibility window.

    Valid for |beta| < SUN_ANG_MIN; targets at higher latitude are continuously visible.

    Args:
        beta (numpy.ndarray or float):
            Target ecliptic latitude (deg)

    Returns:
        tuple:
            delta_entry (deg), duration (days), PA_V3 swing magnitude (deg)
    """
    b = np.radians(np.abs(beta))
    c0 = np.cos(np.radians(SUN_ANG_MIN))
    with np.errstate(invalid="ignore", divide="ignore"):
        delta_entry = np.degrees(np.arccos(c0 / np.cos(b)))
        swing = 2 * np.degrees(
            np.arctan(np.sin(b) * c0 / np.sqrt(np.cos(b) ** 2 - c0**2))
        )
    duration = (180 - 2 * delta_entry) / SUN_RATE
    return delta_entry, duration, swing


def pa_swing_in_window(beta, ndays, placement="worst"):
    """PA_V3 swing over a scheduling window of fixed length lying entirely within the FOR.

    Args:
        beta (float):
            Target ecliptic latitude (deg)
        ndays (float):
            Length of the scheduling window (days)
        placement (str):
            'worst' maximizes the swing over all window placements inside the FOR;
            'quadrature' centers the window on Delta = 90 deg.

    Returns:
        float:
            PA_V3 swing magnitude (deg), NaN if no placement fits inside the FOR
    """
    span = SUN_RATE * ndays
    if placement == "quadrature":
        starts = np.array([90 - span / 2])
    else:
        starts = np.arange(0, 180, 0.1)

    # window must stay inside the FOR at every point; accumulate the PA change in steps
    b = np.radians(beta)
    cos_for = np.cos(np.radians(SUN_ANG_MIN)) + 1e-12
    inside = np.abs(np.cos(b) * np.cos(np.radians(starts))) <= cos_for
    dtheta = np.zeros(starts.size)
    fracs = np.linspace(0, 1, max(int(span / 2), 40) + 1)
    for f0, f1 in zip(fracs[:-1], fracs[1:]):
        cos_eps = np.cos(b) * np.cos(np.radians(starts + f1 * span))
        inside &= np.abs(cos_eps) <= cos_for
        step = pa_from_ecliptic_north(
            starts + f1 * span, beta
        ) - pa_from_ecliptic_north(starts + f0 * span, beta)
        dtheta += (step + 180) % 360 - 180
    if not inside.any():
        return np.nan
    return np.abs(dtheta[inside]).max()


def fixed_pa_window(beta, placement="worst", max_swing=2 * ROLL_MAX):
    """Longest window inside the FOR over which the nominal PA_V3 changes by at most max_swing.

    With the default max_swing this is the longest window over which one PA_V3 is
    reachable with ±ROLL_MAX.

    Args:
        beta (float):
            Target ecliptic latitude (deg)
        placement (str):
            'worst' or 'quadrature', as in pa_swing_in_window()
        max_swing (float):
            Allowed change in nominal PA_V3 over the window (deg)

    Returns:
        float:
            Window length (days); for |beta| < 54 deg capped by the FOR window length
    """
    best = np.nan
    for ndays in np.arange(0.5, 366, 0.5):
        swing = pa_swing_in_window(beta, ndays, placement)
        if not swing <= max_swing:  # also stops when the window no longer fits (NaN)
            break
        best = ndays
    return best


def pa_v3_drift(ra, dec, t_mid, dt):
    """Change of the nominal PA_V3 relative to t_mid for a specific target.

    Computed with calcRomanAngles and getRomanPositionAngle (solar ephemeris, observatory
    at L2). Proper motion and parallax are neglected.

    Args:
        ra (float):
            Target right ascension (deg, ICRS)
        dec (float):
            Target declination (deg, ICRS)
        t_mid (astropy.time.Time):
            Reference time, e.g. the planned middle of a visit
        dt (numpy.ndarray):
            Times relative to t_mid (days); must include 0

    Returns:
        dict:
            s: PA_V3(t) - PA_V3(t_mid) at dt (deg), sun_ang: Sun angle at dt (deg),
            rate: dPA_V3/dt at t_mid (deg/day), beta: ecliptic latitude (deg)
    """
    ts = t_mid + dt * u.day
    target = SkyCoord(
        ra=ra * u.deg,
        dec=dec * u.deg,
        distance=10 * u.pc,
        pm_ra_cosdec=0 * u.mas / u.yr,
        pm_dec=0 * u.mas / u.yr,
        radial_velocity=0 * u.km / u.s,
        frame="icrs",
        obstime=Time("J2000"),
    ).transform_to(BarycentricMeanEcliptic)
    sun_ang, _, _, B_C_I = calcRomanAngles(target, ts, getL2Positions(ts))
    pa = np.degrees(np.unwrap(getRomanPositionAngle(B_C_I).value))
    return {
        "s": pa - np.interp(0, dt, pa),
        "sun_ang": sun_ang.to_value(u.deg),
        "rate": np.interp(0, dt, np.gradient(pa, dt)),
        "beta": target.lat.to_value(u.deg),
    }
