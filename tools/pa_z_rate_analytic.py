"""
Analytic model for the rate of change of the Roman nominal-roll position angle
(PA_Z, and equivalently PA_EXCAM_Y = PA_Z + 150 deg) as a function of target
ecliptic latitude and time of year. Validates the model against
roman_pointing.calcRomanAngles / getRomanPositionAngle and makes figures.

In the nominal attitude, +Z (b_3) is the Sun direction projected onto the sky plane
at the target, so PA_Z is the position angle of the Sun as seen from the target.
With target ecliptic coordinates (lam, beta), solar ecliptic longitude lam_sun(t),
and Delta = lam_sun - lam:

    theta(t)    = atan2(sin Delta, -sin(beta) cos Delta)   (PA from ecliptic north)
    PA_Z(t)     = theta(t) + eta(lam, beta)                (eta constant per target)
    cos eps     = cos(beta) cos Delta                      (eps = Sun-target angle)
    dPA_Z/dt    = -(dlam_sun/dt) sin(beta) / sin^2(eps)

Usage:
  python tools/pa_z_rate_analytic.py -o docs/pa_z_rate_analytic

"""

import argparse
import os

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import BarycentricMeanEcliptic, SkyCoord
from astropy.time import Time
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

from roman_pointing.roman_pointing import (
    applyRollAngle,
    calcRomanAngles,
    getL2Positions,
    getRomanPositionAngle,
    getSunPositions,
)

SUN_RATE = 0.98565  # mean rate of solar ecliptic longitude (deg/day)
OBLIQUITY = 23.4393  # J2000 obliquity of the ecliptic (deg)
SUN_ANG_MIN = 54.0  # field of regard limits on Sun-target angle (deg)
SUN_ANG_MAX = 126.0
ROLL_MAX = 15.0  # allowed off-nominal roll (deg)

# chart chrome and palette (dataviz reference palette)
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
KEEPOUT = "#f0efec"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6",
             "#1c5cab", "#184f95", "#104281", "#0d366b"]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "lines.linewidth": 2,
    "font.size": 10,
})


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


def pa_z_rate(delta, beta, sun_rate=SUN_RATE):
    """Analytic rate of change of PA_Z (and PA_EXCAM_Y).

    Args:
        delta (numpy.ndarray or float):
            Solar ecliptic longitude minus target ecliptic longitude (deg)
        beta (numpy.ndarray or float):
            Target ecliptic latitude (deg)
        sun_rate (numpy.ndarray or float):
            Rate of change of solar ecliptic longitude (deg/day)

    Returns:
        numpy.ndarray or float:
            dPA_Z/dt (deg/day)
    """
    d = np.radians(delta)
    b = np.radians(beta)
    sin2_eps = 1 - np.cos(b) ** 2 * np.cos(d) ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        return -sun_rate * np.sin(b) / sin2_eps


def sun_longitude_doy(doy):
    """Solar ecliptic longitude vs. day of year, circular-orbit approximation.

    Args:
        doy (numpy.ndarray or float):
            Day of year (1.0 = Jan 1, 0h UT)

    Returns:
        numpy.ndarray or float:
            lam_sun (deg); its rate is the constant SUN_RATE
    """
    return (280.46 + SUN_RATE * (doy - 1)) % 360


def ecliptic_north_offset(lam, beta):
    """Constant offset eta such that PA_Z = theta + eta (PA of ecliptic N from celestial N).

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
    return -np.degrees(np.arctan2(np.sin(i) * np.cos(l),
                                  np.cos(b) * np.cos(i) - np.sin(b) * np.sin(i) * np.sin(l)))


def for_window(beta):
    """Duration and PA_Z swing of one field-of-regard visibility window.

    Valid for |beta| < SUN_ANG_MIN; targets at higher latitude are continuously visible.

    Args:
        beta (numpy.ndarray or float):
            Target ecliptic latitude (deg)

    Returns:
        tuple:
            delta_entry (deg), duration (days), PA_Z swing magnitude (deg)
    """
    b = np.radians(np.abs(beta))
    c0 = np.cos(np.radians(SUN_ANG_MIN))
    with np.errstate(invalid="ignore", divide="ignore"):
        delta_entry = np.degrees(np.arccos(c0 / np.cos(b)))
        swing = 2 * np.degrees(np.arctan(np.sin(b) * c0 / np.sqrt(np.cos(b) ** 2 - c0 ** 2)))
    duration = (180 - 2 * delta_entry) / SUN_RATE
    return delta_entry, duration, swing


def pa_swing_in_window(beta, ndays, placement="worst"):
    """PA_Z swing over a scheduling window of fixed length lying entirely within the FOR.

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
            PA_Z swing magnitude (deg), NaN if no placement fits inside the FOR
    """
    span = SUN_RATE * ndays
    if placement == "quadrature":
        starts = np.array([90 - span / 2])
    else:
        starts = np.arange(0, 180, 0.1)

    # window must stay inside the FOR at every point; accumulate the PA change in small steps
    b = np.radians(beta)
    inside = np.abs(np.cos(b) * np.cos(np.radians(starts))) <= np.cos(np.radians(SUN_ANG_MIN)) + 1e-12
    dtheta = np.zeros(starts.size)
    fracs = np.linspace(0, 1, max(int(span / 2), 40) + 1)
    for f0, f1 in zip(fracs[:-1], fracs[1:]):
        cos_eps = np.cos(b) * np.cos(np.radians(starts + f1 * span))
        inside &= np.abs(cos_eps) <= np.cos(np.radians(SUN_ANG_MIN)) + 1e-12
        step = (pa_from_ecliptic_north(starts + f1 * span, beta)
                - pa_from_ecliptic_north(starts + f0 * span, beta))
        dtheta += (step + 180) % 360 - 180
    if not inside.any():
        return np.nan
    return np.abs(dtheta[inside]).max()


def fixed_pa_window(beta, placement="worst", max_swing=2 * ROLL_MAX):
    """Longest window (inside the FOR) over which the nominal PA_Z changes by at most max_swing.

    With the default max_swing this is the longest window over which one PA_Z is reachable
    with ±ROLL_MAX.

    Args:
        beta (float):
            Target ecliptic latitude (deg)
        placement (str):
            'worst' or 'quadrature', as in pa_swing_in_window()
        max_swing (float):
            Allowed change in nominal PA_Z over the window (deg)

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


def validate(year=2027):
    """Compare the analytic model with roman_pointing over one year.

    Args:
        year (int):
            Calendar year to evaluate

    Returns:
        dict:
            doy, list of per-target result dicts, and summary residual statistics
    """
    doy = np.arange(1, 366, 0.5)
    ts = Time(f"{year}-01-01T00:00:00", scale="utc") + (doy - 1) * u.day
    r_obs_G = getL2Positions(ts)
    r_sun_G = getSunPositions(ts)
    r_sun_obs = (r_sun_G - r_obs_G).value
    lam_sun_eph = np.degrees(np.unwrap(np.arctan2(r_sun_obs[1], r_sun_obs[0])))
    lam_sun_doy = sun_longitude_doy(doy)
    dlam = (lam_sun_doy - lam_sun_eph + 180) % 360 - 180
    print(f"Solar longitude: circular model - ephemeris = {dlam.min():.2f} to {dlam.max():.2f} deg")
    print(f"Solar longitude rate: {np.gradient(lam_sun_eph, doy).min():.4f} to "
          f"{np.gradient(lam_sun_eph, doy).max():.4f} deg/day")

    targets = [(40, -80), (100, -45), (160, -20), (220, 5), (280, 30), (340, 60), (190, 85)]
    results = []
    print(f"{'lam':>5} {'beta':>5} {'max|PA err|':>12} {'max|rate err| eph':>18} "
          f"{'max|rate err| DOY':>18} {'PA sense chk':>13}")
    for lam, beta in targets:
        target = SkyCoord(lon=lam * u.deg, lat=beta * u.deg, distance=10 * u.pc,
                          pm_lon_coslat=0 * u.mas / u.yr, pm_lat=0 * u.mas / u.yr,
                          radial_velocity=0 * u.km / u.s,
                          frame=BarycentricMeanEcliptic, obstime=Time("J2000"))
        sun_ang, _, _, B_C_I = calcRomanAngles(target, ts, r_obs_G, r_sun_G)
        pa_num = np.degrees(np.unwrap(getRomanPositionAngle(B_C_I).value))
        rate_num = np.gradient(pa_num, doy)
        in_for = (sun_ang.to_value(u.deg) > SUN_ANG_MIN) & (sun_ang.to_value(u.deg) < SUN_ANG_MAX)

        # analytic PA_Z using the ephemeris solar longitude (tests the geometry alone)
        pa_an = pa_from_ecliptic_north(lam_sun_eph - lam, beta) + ecliptic_north_offset(lam, beta)
        pa_err = (pa_an - pa_num + 180) % 360 - 180
        rate_an_eph = pa_z_rate(lam_sun_eph - lam, beta, np.gradient(lam_sun_eph, doy))
        # fully analytic rate vs. day of year (circular orbit)
        rate_an_doy = pa_z_rate(lam_sun_doy - lam, beta)

        # PA_Z is measured E of N: compare with astropy PA of the Sun seen from the target
        j = np.flatnonzero(in_for)[0]
        sun_dir = SkyCoord(*(r_sun_obs[:, j] * u.AU), representation_type="cartesian",
                           frame=BarycentricMeanEcliptic)
        pa_astropy = target.icrs.position_angle(sun_dir.icrs).to_value(u.deg)
        sense = (pa_astropy - pa_num[j] + 180) % 360 - 180

        print(f"{lam:5.0f} {beta:5.0f} {np.abs(pa_err[in_for]).max():12.4f} "
              f"{np.abs(rate_an_eph - rate_num)[in_for].max():18.5f} "
              f"{np.abs(rate_an_doy - rate_num)[in_for].max():18.5f} {sense:13.4f}")
        results.append({"lam": lam, "beta": beta, "in_for": in_for, "rate_num": rate_num,
                        "rate_an_doy": rate_an_doy, "rate_an_eph": rate_an_eph})

    # roll offsets PA_Z one-for-one, so it does not change the rate
    target = SkyCoord(lon=100 * u.deg, lat=30 * u.deg, frame=BarycentricMeanEcliptic,
                      distance=10 * u.pc, pm_lon_coslat=0 * u.mas / u.yr, pm_lat=0 * u.mas / u.yr,
                      radial_velocity=0 * u.km / u.s, obstime=Time("J2000"))
    _, _, _, B_C_I = calcRomanAngles(target, ts[:10], r_obs_G[:, :10], r_sun_G[:, :10])
    pa0 = getRomanPositionAngle(B_C_I).to_value(u.deg)
    pa_roll = getRomanPositionAngle(applyRollAngle(B_C_I, [10 * u.deg] * 10)).to_value(u.deg)
    print(f"PA_Z change for +10 deg roll: {((pa_roll - pa0 + 180) % 360 - 180).mean():.4f} deg")

    return {"doy": doy, "results": results}


def plot_rate_vs_time(outdir):
    """Plot dPA_Z/dt vs. days after solar conjunction, and PA_Z change vs. days from quadrature.

    Args:
        outdir (str):
            Output directory
    """
    betas = [5, 15, 30, 45, 54, 70, 90]
    colors = BLUE_RAMP[2:]
    days = np.linspace(0, 365.25, 4001)
    delta = SUN_RATE * days

    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.35, 1]})
    for beta, c in zip(betas, colors):
        rate = np.abs(pa_z_rate(delta, beta))
        cos_eps = np.cos(np.radians(beta)) * np.cos(np.radians(delta))
        in_for = np.abs(cos_eps) <= np.cos(np.radians(SUN_ANG_MIN))
        axs[0].plot(days, np.where(in_for, rate, np.nan), color=c, label=f"β = {beta}°")
        axs[0].plot(days, np.where(in_for, np.nan, rate), color=c, lw=0.8, ls=":")
        k = 1300 if beta == 70 else 1000
        axs[0].annotate(f"{beta}°", (days[k], rate[k]), xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, color=INK2)

        # PA_Z change relative to quadrature (Delta = 90 deg), first window
        dq = np.linspace(-91, 91, 1001)
        dtheta = pa_from_ecliptic_north(90 + SUN_RATE * dq, beta) - pa_from_ecliptic_north(90, beta)
        dtheta = (dtheta + 180) % 360 - 180
        cos_eps = np.cos(np.radians(beta)) * np.cos(np.radians(90 + SUN_RATE * dq))
        in_for = np.abs(cos_eps) <= np.cos(np.radians(SUN_ANG_MIN))
        axs[1].plot(dq, np.where(in_for, dtheta, np.nan), color=c)
        if beta < SUN_ANG_MIN:
            k = np.flatnonzero(in_for)[-1]
            axs[1].annotate(f"{beta}°", (dq[k], dtheta[k]), xytext=(4, 0), textcoords="offset points",
                            va="center", fontsize=8.5, color=INK2)
    axs[1].text(-85, -60, "darkest lines: 54°, 70°, 90°\n(continuous viewing)", fontsize=8.5,
                color=INK2, va="top")

    axs[0].set_ylim(0, 2.0)
    axs[0].set_xlim(0, 365.25)
    axs[0].set_xticks(np.arange(0, 366, 30.44 * 1.5))
    axs[0].set_xticklabels([f"{x:.0f}" for x in np.arange(0, 366, 30.44 * 1.5)])
    for x, lab in [(0, "conj."), (365.25 / 4, "quad."), (365.25 / 2, "opp."), (3 * 365.25 / 4, "quad.")]:
        axs[0].axvline(x, color=AXIS, lw=0.8, zorder=0)
        axs[0].text(x, 2.0, lab, ha="center", va="bottom", fontsize=8.5, color=MUTED)
    axs[0].set_xlabel("Days after solar conjunction (λ☉ = λ_target)")
    axs[0].set_ylabel("|dPA_Z / dt|  (deg/day)")
    axs[0].set_title("Rate of image rotation through the year", pad=16)
    axs[0].text(0.01, 0.97, "solid: in field of regard (54°–126° from Sun)\n"
                "dotted: Sun keep-out (not schedulable)",
                transform=axs[0].transAxes, va="top", fontsize=8.5, color=INK2)
    axs[0].legend(loc="upper right", ncol=2, title="ecliptic latitude |β|", title_fontsize=9)

    axs[1].axhspan(-ROLL_MAX, ROLL_MAX, color=BLUE_RAMP[0], alpha=0.6, lw=0, zorder=0)
    axs[1].text(-88, -ROLL_MAX - 2, "±15° off-nominal roll reach\nabout the quadrature PA",
                fontsize=8.5, color=INK2, va="top")
    axs[1].set_xlim(-92, 100)
    axs[1].set_ylim(-100, 100)
    axs[1].set_xlabel("Days from quadrature (Δ = λ☉ − λ = 90°)")
    axs[1].set_ylabel("ΔPA_Z relative to quadrature (deg)")
    axs[1].set_title("Accumulated PA_Z change, β > 0  (sign flips for β < 0)", pad=16)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "pa_z_rate_vs_time.png"), dpi=150)
    plt.close(fig)


def plot_rate_map(outdir):
    """Map of |dPA_Z/dt| vs. days after solar conjunction and ecliptic latitude.

    Args:
        outdir (str):
            Output directory
    """
    days = np.linspace(0, 365.25 / 2, 731)
    betas = np.linspace(0, 90, 361)
    D, B = np.meshgrid(days, betas)
    rate = np.abs(pa_z_rate(SUN_RATE * D, B))
    cos_eps = np.cos(np.radians(B)) * np.cos(np.radians(SUN_RATE * D))
    rate = np.where(np.abs(cos_eps) <= np.cos(np.radians(SUN_ANG_MIN)), rate, np.nan)

    cmap = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)
    levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25]
    fig, ax = plt.subplots(figsize=(9, 5.8))
    ax.set_facecolor(KEEPOUT)
    ax.grid(False)
    cf = ax.contourf(D, B, rate, levels=levels, cmap=cmap)
    cs = ax.contour(D, B, rate, levels=[0.1, 0.25, 0.5, 0.75, 1.0], colors=INK, linewidths=0.8)
    for txt in ax.clabel(cs, fmt=lambda r: f"{r:g}°/d  ({2 * ROLL_MAX / r:.0f} d)", fontsize=8,
                         colors=INK):
        txt.set_bbox(dict(facecolor=SURFACE, alpha=0.8, lw=0, pad=1))
    for eps in [SUN_ANG_MIN, SUN_ANG_MAX]:
        b = np.linspace(0, eps if eps < 90 else 180 - eps, 400)
        d = np.degrees(np.arccos(np.cos(np.radians(eps)) / np.cos(np.radians(b)))) / SUN_RATE
        ax.plot(d, b, color=INK, lw=1.2)
    ax.axhline(SUN_ANG_MIN, color=INK, lw=0.8, ls="--")
    ax.text(91, SUN_ANG_MIN + 1.5, "|β| ≥ 54°: continuous viewing", ha="center", va="bottom",
            fontsize=8.5, color=INK, bbox=dict(facecolor=SURFACE, alpha=0.8, lw=0, pad=1))
    ax.text(8, 22, "Sun keep-out\n(ε < 54°)", fontsize=9, color=INK2)
    ax.text(175, 22, "Anti-Sun\nkeep-out\n(ε > 126°)", fontsize=9, color=INK2, ha="right")
    ax.set_xlabel("Days after solar conjunction (pattern repeats for days 183–365)")
    ax.set_ylabel("Ecliptic latitude |β| (deg)")
    ax.set_title("|dPA_Z/dt| within the field of regard\n"
                 "contour labels: rate and days a fixed PA_Z stays within ±15° roll", fontsize=10.5)
    cb = fig.colorbar(cf, ax=ax, pad=0.02)
    cb.set_label("|dPA_Z/dt| (deg/day)", color=INK2)
    cb.outline.set_edgecolor(AXIS)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "pa_z_rate_map.png"), dpi=150)
    plt.close(fig)


def plot_latitude_summary(outdir):
    """Summary curves vs. ecliptic latitude: rates, window length and PA swing, scheduling margins.

    Args:
        outdir (str):
            Output directory
    """
    beta = np.linspace(0.01, 90, 901)
    s54 = np.sin(np.radians(SUN_ANG_MIN))
    sb = np.sin(np.radians(beta))
    cvz = beta >= SUN_ANG_MIN
    _, duration, swing = for_window(beta)
    rate_quad = SUN_RATE * sb
    rate_max = np.where(cvz, SUN_RATE / sb, SUN_RATE * sb / s54 ** 2)
    rate_mean = np.where(cvz, SUN_RATE, swing / duration)

    fig, axs = plt.subplots(2, 3, figsize=(15, 8.6))
    ax = axs[0, 0]
    ax.plot(beta, rate_max, color=SERIES[1], label="maximum (FOR edge, or Δ = 0°/180° for |β| ≥ 54°)")
    ax.plot(beta, rate_mean, color=SERIES[2], label="mean over window (360°/yr for |β| ≥ 54°)")
    ax.plot(beta, rate_quad, color=SERIES[0], label="minimum (quadrature, ε = 90°)")
    ax.set_ylabel("|dPA_Z/dt| (deg/day)")
    ax.set_title("(a) Rate of PA_Z change within the FOR")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylim(0, 1.45)

    ax = axs[0, 1]
    ax.plot(beta[~cvz], duration[~cvz], color=SERIES[0])
    ax.set_ylabel("Window duration (days)")
    ax.set_title("(b) Length of each FOR window (2 per year)")
    ax.text(60, 105, "continuous\nviewing", fontsize=9, color=INK2)
    ax.set_ylim(0, 190)

    ax = axs[0, 2]
    ax.plot(beta[~cvz], swing[~cvz], color=SERIES[0])
    ax.set_ylabel("|ΔPA_Z| across window (deg)")
    ax.set_title("(c) Total PA_Z rotation across one FOR window")
    ax.text(60, 105, "360° per year", fontsize=9, color=INK2)
    ax.set_ylim(0, 190)

    ndays_list = [14, 30, 60, 90]
    b_grid = np.arange(0, 90.1, 0.5)
    colors = [BLUE_RAMP[2], BLUE_RAMP[4], BLUE_RAMP[6], BLUE_RAMP[8]]
    for col, placement, title in [(1, "worst", "(e) Required PA_Z range: worst-case window"),
                                  (2, "quadrature", "(f) Required PA_Z range: window centered on quadrature")]:
        ax = axs[1, col]
        for ndays, c in zip(ndays_list, colors):
            sw = np.array([pa_swing_in_window(b, ndays, placement) for b in b_grid])
            width = np.maximum(sw - 2 * ROLL_MAX, 0)
            ax.plot(b_grid, width, color=c, label=f"{ndays}-day window", lw=3 if ndays == 14 else 2)
        ax.text(1, 52, "14-day: 0 at all β\n90-day: starts where the FOR\nwindow is ≥ 90 days long",
                fontsize=8.5, color=INK2, va="top")
        ax.set_ylabel("Min. PA_Z range width (deg)")
        ax.set_title(title, fontsize=10.5)
        ax.legend(loc="upper left", fontsize=8.5)
        ax.set_ylim(0, 100)

    ax = axs[1, 0]
    for placement, c, lab in [("worst", SERIES[1], "worst-case placement in FOR"),
                              ("quadrature", SERIES[0], "centered on quadrature")]:
        tmax = [fixed_pa_window(b, placement) for b in b_grid]
        ax.plot(b_grid, tmax, color=c, label=lab)
    ax.text(1, 95, "|β| ≲ 20°: whole FOR window\n(swing < 30°)", fontsize=8.5, color=INK2)
    ax.set_ylabel("Max. window length (days)")
    ax.set_title("(d) Scheduling window for a single fixed PA_Z (±15° roll)", fontsize=10.5)
    ax.legend(loc="upper right", fontsize=8.5)
    ax.set_ylim(0, 200)

    for ax in axs.flat:
        ax.axvline(SUN_ANG_MIN, color=AXIS, lw=0.8, ls="--", zorder=0)
        ax.set_xlim(0, 90)
        ax.set_xticks(np.arange(0, 91, 15))
        ax.set_xlabel("Ecliptic latitude |β| (deg)")
    fig.suptitle(f"PA_Z rate and scheduling margins vs. ecliptic latitude "
                 f"(mean solar rate {SUN_RATE}°/day; up to +3.4% near perihelion in early January)",
                 fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "pa_z_rate_vs_latitude.png"), dpi=150)
    plt.close(fig)


def plot_geometry_diagram(outdir):
    """Diagram of the Sun-target geometry that sets the PA_Z rate, with rate map in (eps, beta).

    Args:
        outdir (str):
            Output directory
    """
    vec = lambda lon, lat: np.array([np.cos(np.radians(lat)) * np.cos(np.radians(lon)),
                                     np.cos(np.radians(lat)) * np.sin(np.radians(lon)),
                                     np.sin(np.radians(lat))])

    # 3D sketch: Roman at the center of the celestial sphere, viewed obliquely from outside
    view = vec(-15, 20)
    pole = np.array([0, 0, 1.0])
    up = pole - pole.dot(view) * view
    up /= np.linalg.norm(up)
    right = np.cross(up, view)
    proj = lambda p: np.array([np.dot(p, right), np.dot(p, up)])

    def arc(a, b, n=100):
        omega = np.arccos(np.clip(np.dot(a, b), -1, 1))
        f = np.linspace(0, 1, n)[:, None]
        return (np.sin((1 - f) * omega) * a + np.sin(f * omega) * b) / np.sin(omega)

    def tangent(p, q):
        t = q - np.dot(q, p) * p
        return t / np.linalg.norm(t)

    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.25, 0.62, 0.62], height_ratios=[1.25, 1])
    ax = fig.add_subplot(gs[:, 0])
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, color=AXIS, lw=1))

    # ecliptic plane through the observatory
    ecl = np.array([vec(l, 0) for l in np.linspace(0, 360, 721)])
    xy = np.array([proj(p) for p in ecl])
    front = ecl @ view > 0
    ax.fill(xy[:, 0], xy[:, 1], color=KEEPOUT, lw=0, zorder=0)
    ax.plot(np.where(front, xy[:, 0], np.nan), np.where(front, xy[:, 1], np.nan), color=MUTED, lw=1.5)
    ax.plot(np.where(~front, xy[:, 0], np.nan), np.where(~front, xy[:, 1], np.nan), color=MUTED,
            lw=0.8, ls=":")
    ax.text(*proj(0.75 * vec(-100, 0)), "ecliptic plane", color=INK2, fontsize=10, ha="center",
            va="center")

    lam_t, beta_t, delta_main = -20.0, 45.0, 60.0
    T = vec(lam_t, beta_t)
    F = vec(lam_t, 0)  # foot of the target's ecliptic meridian
    K = pole
    S_main = vec(lam_t + delta_main, 0)
    O = np.zeros(2)

    # lines of sight from Roman
    for p, c, ls in [(K, INK2, "--"), (T, INK, "-"), (S_main, INK, "-")]:
        ax.plot(*np.c_[O, proj(p)], color=c, lw=1.4 if ls == "-" else 1, ls=ls, zorder=2)
    ax.plot(0, 0, "s", ms=10, color=INK, zorder=6)
    ax.text(-0.04, -0.05, "Roman (at L2)", fontsize=10.5, color=INK, va="top", ha="right")

    # eps: the angle at Roman between the lines of sight to the target and to the Sun
    pts = np.array([proj(0.45 * p) for p in arc(T, S_main)])
    ax.fill(np.r_[0, pts[:, 0]], np.r_[0, pts[:, 1]], color=BLUE_RAMP[1], alpha=0.7, lw=0, zorder=1)
    ax.plot(pts[:, 0], pts[:, 1], color=INK, lw=1.5, zorder=3)
    ax.text(*proj(0.3 * arc(T, S_main)[50]), "ε", fontsize=16, color=INK, ha="center", va="center",
            zorder=4)

    # beta along the target's meridian, Delta along the ecliptic (both measured on the sphere)
    for a_pt, b_pt, lab, off in [(F, T, "β", [-0.07, 0.0]), (F, S_main, "Δ = λ☉ − λ", [0.0, -0.06])]:
        pts = np.array([proj(p) for p in arc(a_pt, b_pt)])
        ax.plot(pts[:, 0], pts[:, 1], color=INK2, lw=2.2, zorder=2)
        ax.text(*pts[50] + off, lab, fontsize=12, color=INK, ha="center", va="center")
    pts = np.array([proj(p) for p in arc(T, K)])
    ax.plot(pts[:, 0], pts[:, 1], color=MUTED, lw=0.9, ls=":")

    # +Z on the sky at T follows the great circle toward the Sun
    pts = np.array([proj(p) for p in arc(T, S_main)])
    ax.plot(pts[:, 0], pts[:, 1], color=SERIES[0], lw=2.2)
    ax.text(*pts[38] + [0.02, 0.09], "great circle\nto the Sun",
            fontsize=9.5, color=SERIES[0], va="bottom", ha="left")

    ax.plot(*proj(K), "o", ms=7, color=INK, zorder=5)
    ax.text(*proj(K) + [0.03, 0.03], "ecliptic north pole", fontsize=10, color=INK)
    ax.plot(*proj(T), "*", ms=18, color=SERIES[1], mec=SURFACE, mew=1, zorder=6)
    ax.text(*proj(T) + [-0.05, 0.04], "target\n(λ, β)", fontsize=10.5, color=INK, ha="right")
    ax.plot(*proj(S_main), "o", ms=13, color=SERIES[3], mec=SURFACE, mew=1.5, zorder=6)
    ax.text(*proj(S_main) + [0.0, -0.06], "Sun (λ☉, 0)", fontsize=10.5, color=INK, ha="center", va="top")

    # at T: ecliptic north and +Z (toward the Sun) on the sky, and the angle theta between them
    pT = proj(T)
    tK, tS = proj(tangent(T, K)), proj(tangent(T, S_main))
    ax.annotate("", xy=pT + 0.22 * tK / np.linalg.norm(tK), xytext=pT,
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.5))
    ax.annotate("", xy=pT + 0.25 * tS / np.linalg.norm(tS), xytext=pT,
                arrowprops=dict(arrowstyle="-|>", color=SERIES[0], lw=2.2))
    ax.text(*(pT + 0.24 * tK / np.linalg.norm(tK)) + [-0.02, 0.02], "ecl. N", color=INK2,
            fontsize=9, ha="right")
    ax.text(*(pT + 0.27 * tS / np.linalg.norm(tS)) + [0.0, 0.03], "+Z", color=SERIES[0], fontsize=11)
    a0, a1 = np.arctan2(tK[1], tK[0]), np.arctan2(tS[1], tS[0])
    ang = np.linspace(a0, a1 - 2 * np.pi if a1 - a0 > np.pi else a1, 40)
    ax.plot(pT[0] + 0.1 * np.cos(ang), pT[1] + 0.1 * np.sin(ang), color=INK, lw=1)
    ax.text(pT[0] + 0.15 * np.cos(ang[20]), pT[1] + 0.15 * np.sin(ang[20]), "θ", fontsize=13,
            color=INK, ha="center", va="center")

    # decompose the Sun's motion along the ecliptic into along-arc and cross-arc parts
    lam_s = np.radians(lam_t + delta_main)
    v = np.array([-np.sin(lam_s), np.cos(lam_s), 0])
    along = -tangent(S_main, T)  # away from T
    cross = np.cross(S_main, along)
    pS = proj(S_main)
    scale = 0.45
    v_perp = np.dot(v, cross) * cross
    ax.annotate("", xy=pS + scale * proj(v), xytext=pS,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=2), annotation_clip=False)
    ax.annotate("", xy=pS + scale * proj(v_perp), xytext=pS,
                arrowprops=dict(arrowstyle="-|>", color=SERIES[1], lw=2), annotation_clip=False)
    ax.plot(*np.c_[pS + scale * proj(v_perp), pS + scale * proj(v)], color=MUTED, lw=0.8, ls=":")
    ax.text(*(pS + scale * proj(v)) + [0.02, -0.02], "Sun's motion\nλ̇☉ ≈ 0.986°/day",
            fontsize=10, color=INK, va="top")
    ax.text(*(pS + scale * proj(v_perp)) + [0.08, 0.16], "part across the\ntarget–Sun arc:\nλ̇☉ sin β / sin ε",
            fontsize=10, color=SERIES[1], va="bottom",
            bbox=dict(facecolor=SURFACE, alpha=0.85, lw=0, pad=1))

    ax.text(-1.05, -1.08,
            "ε is the angle at Roman between the lines of sight to the target and the Sun; it equals the\n"
            "length of the great-circle arc from target to Sun, which sets the +Z direction on the sky.\n"
            "As the Sun moves along the ecliptic, only its motion across that arc swings the arc,\n"
            "and seen from the target (a distance ε away) the arc turns at",
            fontsize=10, color=INK2, va="top")
    ax.text(-1.05, -1.38, "dθ/dt = −λ̇☉ sin β / sin²ε", fontsize=14, color=INK, va="top")
    ax.text(-1.05, -1.51, "Sketch viewed from outside the sphere, so sky directions at the target appear\n"
            "mirrored relative to Roman's view (see panel c). Example: β = 45°, Δ = 60°, ε = 69°.",
            fontsize=8.5, color=MUTED, va="top")
    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-1.68, 1.12)
    ax.set_title("(a) Geometry: Roman at the center of the celestial sphere", loc="left")

    # rate map in (eps, beta)
    ax = fig.add_subplot(gs[0, 1:])
    eps = np.linspace(0, 180, 721)
    beta = np.linspace(0, 90, 361)
    E, Bm = np.meshgrid(eps, beta)
    rate = SUN_RATE * np.sin(np.radians(Bm)) / np.sin(np.radians(E)) ** 2
    rate = np.where((E >= Bm) & (E <= 180 - Bm), rate, np.nan)
    cmap = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)
    levels = np.arange(0, 1.31, 0.1)
    ax.grid(False)
    ax.set_facecolor(KEEPOUT)
    cf = ax.contourf(E, Bm, rate, levels=levels, cmap=cmap, extend="max")
    cs = ax.contour(E, Bm, rate, levels=[0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 2], colors=INK, linewidths=0.7)
    for txt in ax.clabel(cs, fmt="%g", fontsize=8, colors=INK):
        txt.set_bbox(dict(facecolor=SURFACE, alpha=0.8, lw=0, pad=0.5))
    ax.axvspan(0, SUN_ANG_MIN, color=SURFACE, alpha=0.55, lw=0)
    ax.axvspan(SUN_ANG_MAX, 180, color=SURFACE, alpha=0.55, lw=0)
    for x in [SUN_ANG_MIN, SUN_ANG_MAX]:
        ax.axvline(x, color=INK, lw=1.2)
    ax.text(90, 91, "field of regard", ha="center", va="bottom", fontsize=9, color=INK)
    ax.text(27, 91, "keep-out", ha="center", va="bottom", fontsize=9, color=INK2)
    ax.text(153, 91, "keep-out", ha="center", va="bottom", fontsize=9, color=INK2)
    ax.text(14, 62, "ε < |β|\nimpossible", fontsize=8.5, color=INK2, ha="center")
    ax.text(166, 62, "ε > 180° − |β|\nimpossible", fontsize=8.5, color=INK2, ha="center")
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 90)
    ax.set_xticks(np.arange(0, 181, 30))
    ax.set_xlabel("Sun–target angle ε (deg)")
    ax.set_ylabel("Ecliptic latitude |β| (deg)")
    ax.set_title("(b) |dPA_Z/dt| = λ̇☉ |sin β| / sin²ε  (deg/day); each target moves\n"
                 "horizontally between ε = |β| and 180° − |β| once per half year", loc="left",
                 fontsize=10, pad=18)
    cb = fig.colorbar(cf, ax=ax, pad=0.02, ticks=[0, 0.25, 0.5, 0.75, 1.0, 1.25])
    cb.outline.set_edgecolor(AXIS)
    cb.set_label("deg/day", color=INK2)

    # dials: +Z direction every 5 days over 60 days centered on quadrature
    sub = gs[1, 1:].subgridspec(1, 3, wspace=0.05)
    for k, b in enumerate([10, 35, 70]):
        ax = fig.add_subplot(sub[0, k])
        ax.set_aspect("equal")
        ax.axis("off")
        days = np.arange(-30, 30.1, 5)
        th = pa_from_ecliptic_north(90 + SUN_RATE * days, b)
        th0 = pa_from_ecliptic_north(90, b)
        w = np.radians(np.linspace(th0 - ROLL_MAX, th0 + ROLL_MAX, 50))
        # sky convention: ecliptic north up, ecliptic east left
        ax.fill(np.r_[0, -np.sin(w)], np.r_[0, np.cos(w)], color=BLUE_RAMP[0], lw=0)
        ax.add_patch(plt.Circle((0, 0), 1, fill=False, color=AXIS, lw=0.8))
        cols = LinearSegmentedColormap.from_list("t", [BLUE_RAMP[2], BLUE_RAMP[8]])(np.linspace(0, 1, days.size))
        for t, c in zip(th, cols):
            ax.plot([0, -0.92 * np.sin(np.radians(t))], [0, 0.92 * np.cos(np.radians(t))], color=c, lw=1.6)
        ax.text(0, 1.08, "N_ecl", ha="center", fontsize=8, color=MUTED)
        ax.text(-1.08, 0, "E", ha="right", va="center", fontsize=8, color=MUTED)
        swing = abs(((th[-1] - th[0]) + 180) % 360 - 180)
        ax.text(0, -1.22, f"β = {b}°: {swing:.0f}° in 60 days", ha="center", fontsize=10, color=INK)
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.45, 1.3)
        if k == 0:
            ax.set_title("(c) +Z direction on the sky as seen by Roman, every 5 days for ±30 d\n"
                         "around quadrature (light→dark = earlier→later; wedge = ±15° roll reach)",
                         loc="left", fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "pa_z_rate_geometry_diagram.png"), dpi=150)
    plt.close(fig)


def roll_offset_limits(beta, day_mid, early, late):
    """Allowed requested roll offsets when CPGS fixes the sky PA at the planned middle of a visit.

    A requested roll offset A becomes an absolute sky PA, P = PA_nom(t_mid) + A. Executing at time t then
    needs a roll offset A - [PA_nom(t) - PA_nom(t_mid)], which must stay within ±ROLL_MAX from the earliest
    possible visit start to the latest possible visit end.

    Args:
        beta (float or numpy.ndarray):
            Target ecliptic latitude (deg)
        day_mid (float or numpy.ndarray):
            Planned middle of the visit, days after solar conjunction
        early (float):
            Days from the planned middle back to the earliest possible start (D/2 + allowed early shift)
        late (float):
            Days from the planned middle to the latest possible end (D/2 + allowed late shift)

    Returns:
        tuple:
            a_min, a_max (deg, PA sense): allowed range of requested roll offsets
    """
    day_mid, beta = np.broadcast_arrays(np.asarray(day_mid, dtype=float), np.asarray(beta, dtype=float))
    t = np.linspace(-early, late, 61).reshape((-1,) + (1,) * day_mid.ndim)
    s = (pa_from_ecliptic_north(SUN_RATE * (day_mid + t), beta)
         - pa_from_ecliptic_north(SUN_RATE * day_mid, beta))
    s = (s + 180) % 360 - 180
    return -ROLL_MAX + s.max(axis=0), ROLL_MAX + s.min(axis=0)


def max_symmetric_roll(beta, halfspan, placement="worst"):
    """Largest |roll| usable for both signs, for a visit whose span reaches halfspan days either side of mid-visit.

    Args:
        beta (float):
            Target ecliptic latitude (deg)
        halfspan (float):
            H = D/2 + U: half the visit duration plus the allowed start shift (days)
        placement (str):
            'worst' takes the most restrictive mid-visit date with the whole span inside the FOR;
            'quadrature' puts mid-visit at Delta = 90 deg.

    Returns:
        float:
            Maximum requested |roll| (deg); NaN if the span never fits inside the FOR
    """
    if placement == "quadrature":
        days = np.array([90 / SUN_RATE])
    else:
        days = np.arange(0, 365.25 / 2, 0.25)
    lo, hi = roll_offset_limits(beta, days, halfspan, halfspan)
    inside = np.ones(days.size, dtype=bool)
    for f in (-halfspan, 0, halfspan):
        cos_eps = np.cos(np.radians(beta)) * np.cos(np.radians(SUN_RATE * (days + f)))
        inside &= np.abs(cos_eps) <= np.cos(np.radians(SUN_ANG_MIN)) + 1e-12
    if not inside.any():
        return np.nan
    return np.minimum(hi, -lo)[inside].min()


def roll_limits_for_target(ra, dec, start, duration, early=0.0, late=0.0):
    """Allowed requested roll offsets for a specific target and visit, using roman_pointing numerics.

    Args:
        ra (float):
            Target right ascension (deg, ICRS)
        dec (float):
            Target declination (deg, ICRS)
        start (str):
            Requested visit start time (ISO format, UTC)
        duration (float):
            Visit duration D (days)
        early (float):
            Days the start may move earlier than requested
        late (float):
            Days the start may move later than requested

    Returns:
        dict:
            a_min, a_max (deg, PA sense), beta (deg), Sun angle range (deg), PA_Z rate at mid-visit (deg/day)
    """
    t_mid = Time(start, scale="utc") + duration / 2 * u.day
    dt = np.linspace(-(duration / 2 + early), duration / 2 + late, 241)
    ts = t_mid + dt * u.day
    target = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, distance=10 * u.pc, pm_ra_cosdec=0 * u.mas / u.yr,
                      pm_dec=0 * u.mas / u.yr, radial_velocity=0 * u.km / u.s, frame="icrs",
                      obstime=Time("J2000")).transform_to(BarycentricMeanEcliptic)
    sun_ang, _, _, B_C_I = calcRomanAngles(target, ts, getL2Positions(ts))
    pa = np.degrees(np.unwrap(getRomanPositionAngle(B_C_I).value))
    s = pa - np.interp(0, dt, pa)
    return {"a_min": -ROLL_MAX + s.max(), "a_max": ROLL_MAX + s.min(), "beta": target.lat.to_value(u.deg),
            "sun_ang": (sun_ang.to_value(u.deg).min(), sun_ang.to_value(u.deg).max()),
            "rate": np.interp(0, dt, np.gradient(pa, dt))}


def plot_roll_limits_example(outdir):
    """Example: needed roll vs. execution time when the sky PA is fixed at the planned mid-visit.

    Args:
        outdir (str):
            Output directory
    """
    beta, day_mid, duration, buffer = 45.0, 60.0, 2.0, 3.0
    offsets = {"A": -13.0, "B": +13.0}
    half = duration / 2
    t = np.linspace(-half - buffer, half + buffer, 400)
    s = (pa_from_ecliptic_north(SUN_RATE * (day_mid + t), beta)
         - pa_from_ecliptic_north(SUN_RATE * day_mid, beta))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5.8), gridspec_kw={"width_ratios": [1.5, 1]})
    crossings = []
    ax.axvspan(-half - buffer, half + buffer, color=KEEPOUT, lw=0, zorder=0)
    ax.axvspan(-half, half, color=BLUE_RAMP[0], lw=0, zorder=0)
    ax.text(0, 17.2, f"requested visit\n(D = {duration:.0f} d)", ha="center", fontsize=9, color=INK2)
    ax.text(-half - buffer + 0.1, 17.2, f"±{buffer:.0f} d buffer\n(earliest start → latest end)",
            fontsize=9, color=INK2)
    for x in [-ROLL_MAX, ROLL_MAX]:
        ax.axhline(x, color=INK, lw=1.2)
    for (name, off), c in zip(offsets.items(), [SERIES[0], SERIES[1]]):
        need = off - s
        ax.plot(t, need, color=c, label=f"roll {name}: requested {off:+.0f}°")
        bad = np.abs(need) > ROLL_MAX
        ax.plot(t, np.where(bad, need, np.nan), color=c, lw=5, alpha=0.35, solid_capstyle="butt")
        k = np.flatnonzero(np.diff(bad.astype(int)) != 0)
        for j in k:
            crossings.append(t[j])
            ax.plot(t[j], need[j], "o", ms=7, color=c, mec=SURFACE, mew=1.2, zorder=5)
            ax.annotate(f"{t[j]:+.1f} d", (t[j], need[j]), xytext=(0, 10 if off > 0 else -16),
                        textcoords="offset points", ha="center", fontsize=9, color=INK)
    early_ok = max([x for x in crossings if x < 0], default=t[0]) + half
    late_ok = min([x for x in crossings if x > 0], default=t[-1]) - half
    ax.text(0, 2.5, f"Allowed shift of the start: {early_ok:+.1f} to {late_ok:+.1f} d\n"
            f"(vs. the ±{buffer:.0f} d buffer)", ha="center", fontsize=10, color=INK,
            bbox=dict(facecolor=SURFACE, alpha=0.9, lw=0.5, edgecolor=AXIS, pad=4))
    ax.set_xlim(t[0] - 0.3, t[-1] + 0.3)
    ax.set_ylim(-19, 19.5)
    ax.set_xlabel("Execution time relative to the requested mid-visit (days)")
    ax.set_ylabel("Roll needed to hold the CPGS sky PA (deg, PA sense)")
    ax.set_title(f"(a) β = {beta:.0f}°, mid-visit {day_mid:.0f} d after conjunction "
                 f"(|rate| ≈ {abs(pa_z_rate(SUN_RATE * day_mid, beta)):.2f}°/d): thick = beyond ±15°",
                 fontsize=10.5, loc="left")
    ax.legend(loc="center right", fontsize=9)

    buffers = np.linspace(0, 3, 61)
    for dur, c in zip([0.5, 1, 2, 3], [BLUE_RAMP[2], BLUE_RAMP[4], BLUE_RAMP[6], BLUE_RAMP[8]]):
        lims = [roll_offset_limits(beta, day_mid, dur / 2 + b, dur / 2 + b) for b in buffers]
        sym = [min(hi, -lo) for lo, hi in lims]
        ax2.plot(buffers, sym, color=c, label=f"D = {dur:g} d")
    ax2.axhline(abs(offsets["A"]), color=MUTED, lw=1, ls="--")
    ax2.text(0.05, abs(offsets["A"]) + 0.15, "requested ±13°", fontsize=8.5, color=INK2)
    ax2.set_xlim(0, 3)
    ax2.set_ylim(9, 15.2)
    ax2.set_xlabel("Allowed shift of the start, ±U (days)")
    ax2.set_ylabel("Maximum requested |roll| (deg)")
    ax2.set_title("(b) Same target and date: maximum |roll| vs. buffer\nand visit duration",
                  fontsize=10.5, loc="left")
    ax2.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "roll_limits_example.png"), dpi=150)
    plt.close(fig)


def plot_roll_limits_vs_latitude(outdir):
    """Maximum requested |roll| vs. ecliptic latitude for several half-spans H = D/2 + U.

    Args:
        outdir (str):
            Output directory
    """
    b_grid = np.arange(0, 90.1, 0.5)
    halfspans = [1, 2, 3, 4, 5]
    colors = [BLUE_RAMP[2], BLUE_RAMP[4], BLUE_RAMP[5], BLUE_RAMP[7], BLUE_RAMP[8]]

    fig, axs = plt.subplots(1, 3, figsize=(16, 5.4))
    for h, c in zip(halfspans, colors):
        for ax, placement in zip(axs[:2], ["worst", "quadrature"]):
            ax.plot(b_grid, [max_symmetric_roll(b, h, placement) for b in b_grid], color=c, label=f"H = {h} d")
    for ax, where in zip(axs[:2], ["(a) worst-case date in the FOR", "(b) mid-visit at quadrature"]):
        ax.set_ylim(7, 15.3)
        ax.set_ylabel("Maximum requested |roll| (deg)")
        ax.set_title(f"{where}\nH = D/2 + U (half visit + allowed start shift)", fontsize=10.5)
        ax.legend(loc="lower left", fontsize=9)

    sb = np.sin(np.radians(np.maximum(b_grid, 0.01)))
    rate_worst = np.where(b_grid >= SUN_ANG_MIN, SUN_RATE / sb, SUN_RATE * sb / np.sin(np.radians(SUN_ANG_MIN)) ** 2)
    axs[2].plot(b_grid, 1 / rate_worst, color=SERIES[1], label="worst-case date (fastest rotation)")
    axs[2].plot(b_grid, 1 / (SUN_RATE * sb), color=SERIES[0], label="at quadrature")
    axs[2].set_ylim(0, 6)
    axs[2].set_ylabel("Days of start-time slack per 1° less |roll|")
    axs[2].set_title("(c) Slack bought by reducing |roll|: 1/|rate|\n(each 1° buys this many days, "
                     "split early/late)", fontsize=10.5)
    axs[2].legend(loc="upper right", fontsize=9)
    for ax in axs:
        ax.axvline(SUN_ANG_MIN, color=AXIS, lw=0.8, ls="--", zorder=0)
        ax.set_xlim(0, 90)
        ax.set_xticks(np.arange(0, 91, 15))
        ax.set_xlabel("Ecliptic latitude |β| (deg)")
    fig.suptitle("Maximum requested roll offset when CPGS fixes the sky PA at mid-visit "
                 "(|roll| ≤ 15° − |rate| × H)", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "roll_limits_vs_latitude.png"), dpi=150)
    plt.close(fig)


def plot_roll_limits_map(outdir):
    """Map of maximum requested |roll| vs. mid-visit date and ecliptic latitude.

    Args:
        outdir (str):
            Output directory
    """
    days = np.linspace(0, 365.25 / 2, 367)
    betas = np.linspace(0, 90, 181)
    D, B = np.meshgrid(days, betas)
    cos_for = np.cos(np.radians(SUN_ANG_MIN)) + 1e-12
    cmap = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP[::-1])

    fig, axs = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
    for ax, h in zip(axs, [2, 4]):
        lo, hi = roll_offset_limits(B, D, h, h)
        sym = np.minimum(hi, -lo)
        inside = np.ones(D.shape, dtype=bool)
        for f in (-h, 0, h):
            inside &= np.abs(np.cos(np.radians(B)) * np.cos(np.radians(SUN_RATE * (D + f)))) <= cos_for
        sym = np.where(inside, sym, np.nan)
        ax.set_facecolor(KEEPOUT)
        ax.grid(False)
        cf = ax.contourf(D, B, sym, levels=np.arange(8, 15.01, 0.5), cmap=cmap, extend="min")
        cs = ax.contour(D, B, sym, levels=[10, 11, 12, 13, 14], colors=INK, linewidths=0.8)
        for txt in ax.clabel(cs, fmt="%g°", fontsize=8, colors=INK):
            txt.set_bbox(dict(facecolor=SURFACE, alpha=0.8, lw=0, pad=1))
        ax.axhline(SUN_ANG_MIN, color=INK, lw=0.8, ls="--")
        ax.set_xlabel("Requested mid-visit: days after solar conjunction")
        ax.set_title(f"H = D/2 + U = {h} d  (e.g. D = 2 d with ±{h - 1} d buffer)", fontsize=10.5)
        ax.text(8, 22, "keep-out", fontsize=9, color=INK2)
        ax.text(175, 22, "keep-out", fontsize=9, color=INK2, ha="right")
    axs[0].set_ylabel("Ecliptic latitude |β| (deg)")
    cb = fig.colorbar(cf, ax=axs, pad=0.02)
    cb.set_label("Maximum requested |roll| (deg)", color=INK2)
    cb.outline.set_edgecolor(AXIS)
    fig.suptitle("Maximum requested roll offset that keeps the CPGS mid-visit PA reachable over the visit and "
                 "buffer (pattern repeats for days 183–365)", fontsize=11, color=INK)
    fig.savefig(os.path.join(outdir, "roll_limits_map.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_validation(val, outdir):
    """Plot analytic vs. roman_pointing PA_Z rates over one year.

    Args:
        val (dict):
            Output of validate()
        outdir (str):
            Output directory
    """
    doy = val["doy"]
    fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    geom_err = 0
    for res, c in zip(val["results"], BLUE_RAMP[2:]):
        m = res["in_for"]
        lab = f"λ = {res['lam']}°, β = {res['beta']}°"
        axs[0].plot(doy, np.where(m, res["rate_an_doy"], np.nan), color=c, label=lab)
        axs[0].plot(doy[m][::8], res["rate_num"][m][::8], "o", ms=4, mfc="none", mec=INK2, mew=0.8)
        axs[1].plot(doy, np.where(m, res["rate_an_doy"] - res["rate_num"], np.nan), color=c, lw=1.2)
        geom_err = max(geom_err, np.abs(res["rate_an_eph"] - res["rate_num"])[m].max())
    axs[0].set_ylabel("dPA_Z/dt (deg/day)")
    axs[0].set_title("Analytic, circular-orbit model (lines) vs. roman_pointing (circles), "
                     "2027, in-FOR times only", fontsize=10.5)
    axs[0].legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(1.01, 1))
    axs[1].set_title(f"Residual is set by Earth's orbital eccentricity; with the ephemeris solar "
                     f"longitude it is ≤ {geom_err:.3f} deg/day", fontsize=9.5, color=INK2)
    axs[1].set_ylabel("Analytic − numeric\n(deg/day)")
    axs[1].set_xlabel("Day of year 2027")
    axs[1].set_xlim(1, 366)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "pa_z_rate_validation.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="pa_z_rate_analytic.py",
                                 description="Analytic PA_Z rate model: validation and figures")
    ap.add_argument("-o", "--outdir", default="docs/pa_z_rate_analytic", help="Output directory for figures")
    ap.add_argument("--year", type=int, default=2027, help="Year used for validation against roman_pointing")
    ap.add_argument("--ra", type=float, help="Roll-limit calculator: target RA (deg, ICRS)")
    ap.add_argument("--dec", type=float, help="Roll-limit calculator: target Dec (deg, ICRS)")
    ap.add_argument("--start", type=str, help="Roll-limit calculator: requested visit start (ISO, UTC)")
    ap.add_argument("--duration", type=float, default=1.0, help="Roll-limit calculator: visit duration (days)")
    ap.add_argument("--early", type=float, default=0.0, help="Days the start may move earlier than requested")
    ap.add_argument("--late", type=float, default=0.0, help="Days the start may move later than requested")
    args = ap.parse_args()

    if args.ra is not None:
        res = roll_limits_for_target(args.ra, args.dec, args.start, args.duration, args.early, args.late)
        print(f"Target beta = {res['beta']:.2f} deg, Sun angle {res['sun_ang'][0]:.1f}-{res['sun_ang'][1]:.1f} deg, "
              f"dPA_Z/dt at mid-visit = {res['rate']:+.3f} deg/day")
        print(f"Allowed requested roll offsets (PA sense): {res['a_min']:+.2f} to {res['a_max']:+.2f} deg; "
              f"symmetric limit +/-{min(res['a_max'], -res['a_min']):.2f} deg")
        if res["sun_ang"][0] < SUN_ANG_MIN or res["sun_ang"][1] > SUN_ANG_MAX:
            print("WARNING: part of the visit span is outside the 54-126 deg field of regard")
        raise SystemExit

    os.makedirs(args.outdir, exist_ok=True)
    val = validate(args.year)
    plot_validation(val, args.outdir)
    plot_rate_vs_time(args.outdir)
    plot_rate_map(args.outdir)
    plot_latitude_summary(args.outdir)
    plot_geometry_diagram(args.outdir)
    plot_roll_limits_example(args.outdir)
    plot_roll_limits_vs_latitude(args.outdir)
    plot_roll_limits_map(args.outdir)

    # guideline table: maximum requested |roll| with the PA fixed at mid-visit, H = D/2 + U
    print(f"\n{'|beta|':>6} {'rate worst':>10} {'rate quad':>9} | max |roll| worst / quad for H = "
          f"{'1 d':>13} {'2 d':>13} {'3 d':>13} {'4 d':>13} | {'slack d/deg worst/quad':>22}")
    for b in [5, 10, 20, 30, 40, 45, 50, 54, 60, 70, 80, 90]:
        sb = np.sin(np.radians(b))
        rmax = SUN_RATE / sb if b >= SUN_ANG_MIN else SUN_RATE * sb / np.sin(np.radians(SUN_ANG_MIN)) ** 2
        cols = [f"{max_symmetric_roll(b, h):5.1f} / {max_symmetric_roll(b, h, 'quadrature'):5.1f}"
                for h in (1, 2, 3, 4)]
        print(f"{b:6.0f} {rmax:10.3f} {SUN_RATE * sb:9.3f} | {'':>31}" + " ".join(f"{c:>13}" for c in cols)
              + f" | {1 / rmax:9.2f} / {1 / (SUN_RATE * sb):<9.2f}")

    # guideline table: fixed PA
    print(f"\n{'|beta|':>6} {'rate quad':>10} {'rate max':>9} {'window d':>9} {'swing':>7} "
          f"{'fixed-PA d (worst/quad)':>24} {'W30 worst':>10} {'W60 worst':>10}")
    for b in [5, 10, 15, 20, 30, 40, 45, 50, 54, 60, 70, 80, 90]:
        sb = np.sin(np.radians(b))
        rmax = SUN_RATE / sb if b >= SUN_ANG_MIN else SUN_RATE * sb / np.sin(np.radians(SUN_ANG_MIN)) ** 2
        _, dur, sw = for_window(b)
        tw = fixed_pa_window(b, "worst")
        tq = fixed_pa_window(b, "quadrature")
        w30 = max(pa_swing_in_window(b, 30) - 2 * ROLL_MAX, 0)
        w60 = max(pa_swing_in_window(b, 60) - 2 * ROLL_MAX, 0)
        print(f"{b:6.0f} {SUN_RATE * sb:10.3f} {rmax:9.3f} {dur:9.1f} {sw:7.1f} "
              f"{tw:11.1f} / {tq:<10.1f} {w30:10.1f} {w60:10.1f}")
