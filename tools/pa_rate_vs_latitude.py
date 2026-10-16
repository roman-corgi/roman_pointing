"""Seasonal PA_Z drift rate versus day of year and target ecliptic latitude for Roman / EXCAM.

Makes the figures and tables in docs/pa_rate_vs_latitude/ and, unless --no-validate is given,
checks the analytical expressions against the roman_pointing attitude solver (offline, built-in
ephemeris). Run from the repository root:

    python tools/pa_rate_vs_latitude.py

Angles are in degrees and rates in deg/day unless noted. Delta denotes the solar ecliptic
longitude minus the target ecliptic longitude.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle, FancyArrowPatch

OUTDIR = Path(__file__).resolve().parents[1] / "docs" / "pa_rate_vs_latitude"
N_SUN = 0.9856474  # mean rate of solar ecliptic longitude (deg/day)
ECC = 0.0167  # eccentricity of Earth's orbit
N_SUN_MAX = N_SUN * (1 + ECC) ** 2 / (1 - ECC**2) ** 1.5  # solar longitude rate at perihelion
SUN_MIN, SUN_MAX = 54.0, 126.0  # Roman solar-angle limits
ROLL = 15.0  # off-nominal roll allowance (+/- deg)
# +1 if getRomanPositionAngle increases from north toward east; confirmed by validate()
PA_EAST = 1

# Example coronagraph targets (ICRS J2000 RA, Dec in deg) spanning ecliptic latitude
TARGETS = {
    "Regulus": (152.0929, 11.9672),
    "eps Eri": (53.2327, -9.4583),
    "47 UMa": (164.8666, 40.4303),
    "14 Her": (242.6013, 43.8176),
    "alf Eri": (24.4285, -57.2367),
    "bet Pic": (86.8212, -51.0665),
}

# Chart palette: ordinal blue ramp, categorical slots, ink and chrome
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
         "#184f95", "#104281", "#0d366b"]
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
RATE_CMAP = LinearSegmentedColormap.from_list("rate", ["#f0efec"] + BLUES[1:])
SIGNED_CMAP = LinearSegmentedColormap.from_list(
    "signed", ["#7a1f1f", "#e34948", "#f4b4b3", "#f0efec", "#9ec5f4", "#2a78d6", "#0d366b"])

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10.5, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.facecolor": "white", "savefig.dpi": 160,
})


def sun_longitude(doy, year=2027):
    """Low-precision solar ecliptic longitude (J2000 equinox) and its rate.

    Uses the Astronomical Almanac low-precision formulae (accurate to ~0.01 deg in longitude).

    Args:
        doy (float or numpy.ndarray): Day of year; 1.0 is January 1, 0h UT.
        year (int): Calendar year (1901-2099).

    Returns:
        tuple: Solar ecliptic longitude (deg) and its rate (deg/day).
    """
    d = 365 * (year - 2000) + (year - 1997) // 4 + np.asarray(doy, dtype=float) - 1.5  # days from J2000.0
    g = np.deg2rad(357.528 + 0.9856003 * d)  # mean anomaly
    lon = 280.460 + 0.9856474 * d + 1.915 * np.sin(g) + 0.020 * np.sin(2 * g)
    lon -= 1.39697 * d / 36525  # precession: equinox of date to J2000
    rate = 0.9856474 - 1.39697 / 36525 + np.deg2rad(0.9856003) * (1.915 * np.cos(g) + 0.040 * np.cos(2 * g))
    return lon % 360, rate


def sun_angle(beta, delta):
    """Sun-target angle epsilon (deg), from cos(epsilon) = cos(beta) cos(delta)."""
    return np.rad2deg(np.arccos(np.cos(np.deg2rad(beta)) * np.cos(np.deg2rad(delta))))


def observable(beta, delta):
    """True where the Sun-target angle lies strictly inside the 54-126 deg limits."""
    cos_eps = np.cos(np.deg2rad(beta)) * np.cos(np.deg2rad(delta))
    return np.abs(cos_eps) < np.cos(np.deg2rad(SUN_MIN))


def pa_rate(beta, delta, lon_rate=N_SUN):
    """Nominal-roll dPA_Z/dt from the analytical expression.

    Args:
        beta (float or numpy.ndarray): Target ecliptic latitude (deg).
        delta (float or numpy.ndarray): Solar minus target ecliptic longitude (deg).
        lon_rate (float or numpy.ndarray): Rate of solar ecliptic longitude (deg/day).

    Returns:
        numpy.ndarray: dPA_Z/dt (deg/day); NaN where the Sun lies along the line of sight.
    """
    b, d = np.deg2rad(beta), np.deg2rad(delta)
    sin2_eps = np.sin(d) ** 2 + np.sin(b) ** 2 * np.cos(d) ** 2  # = 1 - cos^2(beta) cos^2(delta)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = -PA_EAST * lon_rate * np.sin(b) / sin2_eps
    return np.where(sin2_eps > 1e-12, rate, np.nan)


def pa_drift(beta, delta1, delta2):
    """Unsigned nominal PA_Z change between two Sun longitudes within one observing season.

    Closed form for the angle between the projected Sun directions at the two dates.

    Args:
        beta (float or numpy.ndarray): Target ecliptic latitude (deg).
        delta1, delta2 (float or numpy.ndarray): Solar minus target longitude at window start and end (deg).

    Returns:
        numpy.ndarray: |PA_Z(delta2) - PA_Z(delta1)| (deg), valid for changes below 180 deg.
    """
    b, d1, d2 = np.deg2rad(beta), np.deg2rad(delta1), np.deg2rad(delta2)
    cross = np.sin(b) * np.sin(d2 - d1)
    dot = np.sin(d1) * np.sin(d2) + np.sin(b) ** 2 * np.cos(d1) * np.cos(d2)
    return np.rad2deg(np.abs(np.arctan2(cross, dot)))


def season_days(abs_beta):
    """Length (days) of each continuous observing season; a full year for |beta| >= 54 deg."""
    ratio = np.cos(np.deg2rad(SUN_MIN)) / np.cos(np.deg2rad(abs_beta))
    edge = np.rad2deg(np.arccos(np.clip(ratio, -1, 1)))  # delta at the season start
    return np.where(ratio < 1, (180 - 2 * edge) / N_SUN, 365.25)


def window_drift(abs_beta, width, worst_case=True):
    """Nominal PA_Z change over a window of `width` days inside one observing season.

    Args:
        abs_beta (float or numpy.ndarray): Absolute target ecliptic latitude (deg).
        width (float or numpy.ndarray): Window full width (days).
        worst_case (bool): If True, place the window where the drift is largest (against a
            keepout edge for |beta| < 54 deg, centered on solar conjunction in longitude
            otherwise) and use the perihelion solar rate. If False, center the window on
            quadrature and use the mean solar rate.

    Returns:
        numpy.ndarray: PA_Z change (deg); NaN if the window is longer than the season.
    """
    abs_beta, width = np.broadcast_arrays(np.asarray(abs_beta, float), np.asarray(width, float))
    if not worst_case:
        half = N_SUN * width / 2
        drift = pa_drift(abs_beta, 90 - half, 90 + half)
    else:
        span = N_SUN_MAX * width
        ratio = np.cos(np.deg2rad(SUN_MIN)) / np.cos(np.deg2rad(abs_beta))
        edge = np.rad2deg(np.arccos(np.clip(ratio, -1, 1)))
        drift = np.where(ratio < 1, pa_drift(abs_beta, edge, edge + span),
                         pa_drift(abs_beta, -span / 2, span / 2))
    return np.where(width <= season_days(abs_beta) + 1e-9, drift, np.nan)


def max_window(abs_beta, allowed_drift, worst_case=True):
    """Longest window (days) whose nominal PA_Z change does not exceed `allowed_drift` (deg)."""
    widths = np.arange(0, 365.25, 0.05)
    drift = window_drift(np.asarray(abs_beta, float)[:, None], widths[None, :], worst_case)
    ok = np.nan_to_num(drift, nan=np.inf) <= allowed_drift
    first_bad = np.where(ok.all(axis=1), len(widths) - 1, np.argmin(ok, axis=1) - 1)
    return widths[np.maximum(first_bad, 0)]


def validate():
    """Compare the analytical rate with finite differences of the roman_pointing solver over 2027.

    Returns:
        tuple: Summary dict of agreement metrics, and per-target time series for plotting.
    """
    import astropy.units as u
    from astropy.coordinates import BarycentricMeanEcliptic, SkyCoord
    from astropy.time import Time
    from astropy.utils import iers
    from roman_pointing.roman_pointing import (
        applyRollAngle, calcRomanAngles, getEXCAMPositionAngle, getL2Positions,
        getRomanPositionAngle, getSunPositions,
    )

    iers.conf.auto_download = False
    step = 0.25
    doy = 1 + np.arange(0, 365, step)
    ts = Time("2027-01-01T00:00:00", scale="utc") + (doy - 1) * u.day
    r_obs = getL2Positions(ts)
    r_sun = getSunPositions(ts)

    # Sun direction from the observatory, its ecliptic longitude, and their time derivatives
    s = (r_sun - r_obs).to_value(u.au)
    s /= np.linalg.norm(s, axis=0)
    s_dot = np.gradient(s, step, axis=1)  # rad/day
    sun_lon = np.rad2deg(np.unwrap(np.arctan2(s[1], s[0])))
    sun_lon_rate = np.gradient(sun_lon, step)
    model_lon, model_rate = sun_longitude(doy, 2027)
    summary = {
        "year": 2027, "step_days": step,
        "sun_lon_model_minus_ephemeris_max_abs_deg": float(np.max(np.abs((model_lon - sun_lon + 180) % 360 - 180))),
        "sun_lon_rate_range_deg_day": [float(sun_lon_rate.min()), float(sun_lon_rate.max())],
        "sun_ecliptic_latitude_max_abs_deg": float(np.rad2deg(np.max(np.abs(np.arcsin(s[2]))))),
        "targets": {},
    }

    series = {}
    for name, (ra, dec) in TARGETS.items():
        target = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, distance=10 * u.pc,
                          pm_ra_cosdec=0 * u.mas / u.yr, pm_dec=0 * u.mas / u.yr,
                          radial_velocity=0 * u.km / u.s, frame="icrs", obstime="J2000",
                          ).transform_to(BarycentricMeanEcliptic)
        lon, beta = target.lon.deg, target.lat.deg
        sun_ang, _, _, bci = calcRomanAngles(target, ts, r_obs, r_sun)
        pa_z = getRomanPositionAngle(bci).to_value(u.deg)
        pa_excam = getEXCAMPositionAngle(bci).to_value(u.deg)
        rate_solver = np.gradient(np.rad2deg(np.unwrap(np.deg2rad(pa_z))), step)
        obs = (sun_ang.to_value(u.deg) > SUN_MIN) & (sun_ang.to_value(u.deg) < SUN_MAX)
        obs[[0, -1]] = False  # one-sided differences at the ends of the year

        # Analytical rate: (1) general vector form, (2) planar form with ephemeris Sun,
        # (3) planar form with the day-of-year solar longitude formula
        n = target.cartesian.xyz.value / np.linalg.norm(target.cartesian.xyz.value)
        n_dot_s = n @ s
        rate_vector = -PA_EAST * np.rad2deg(n @ np.cross(s.T, s_dot.T).T) / (1 - n_dot_s**2)
        rate_planar = pa_rate(beta, sun_lon - lon, sun_lon_rate)
        rate_doy = pa_rate(beta, model_lon - lon, model_rate)
        rate_mean = pa_rate(beta, model_lon - lon, N_SUN)

        # +Z must point to the Sun side, EXCAM +Y must sit 150 deg from +Z, and the roll sign
        b3_dot_s = np.einsum("ij,ij->j", bci[2], s)
        excam_offset = (pa_excam - pa_z) % 360
        pa_rolled = getRomanPositionAngle(applyRollAngle(bci[:, :, :8], np.full(8, 10.0) * u.deg)).to_value(u.deg)
        roll_change = (pa_rolled - pa_z[:8] + 180) % 360 - 180

        def max_err(model):
            return float(np.max(np.abs(rate_solver[obs] - model[obs])))

        summary["targets"][name] = {
            "ecl_lon_deg": float(lon), "ecl_lat_deg": float(beta),
            "observable_fraction": float(obs.mean()),
            "solver_rate_abs_range_deg_day": [float(np.min(np.abs(rate_solver[obs]))),
                                              float(np.max(np.abs(rate_solver[obs])))],
            "max_err_vector_formula_deg_day": max_err(rate_vector),
            "max_err_planar_formula_ephemeris_sun_deg_day": max_err(rate_planar),
            "max_err_planar_formula_doy_sun_deg_day": max_err(rate_doy),
            "max_err_planar_formula_mean_sun_rate_deg_day": max_err(rate_mean),
            "max_err_planar_formula_opposite_pa_sign_deg_day": max_err(-rate_planar),
            "min_b3_dot_sun": float(b3_dot_s.min()),
            "excam_minus_pa_z_deg": [float(excam_offset.min()), float(excam_offset.max())],
            "pa_z_change_for_plus10deg_roll": [float(roll_change.min()), float(roll_change.max())],
        }
        series[name] = {"beta": beta, "doy": doy, "obs": obs, "pa_excam": pa_excam % 360,
                        "rate_solver": rate_solver, "rate_doy": rate_doy}
    return summary, series


def plot_rate_vs_phase(outdir):
    """Figure 1: |dPA_Z/dt| and Sun angle versus days from quadrature for several latitudes."""
    days = np.linspace(-92, 92, 3681)
    delta = 90 + N_SUN * days
    betas = [0, 5, 10, 20, 30, 45, 54, 60, 75, 90]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8.2), sharex=True, height_ratios=[1.5, 1],
                                   layout="constrained")
    for beta, color in zip(betas, BLUES):
        rate = np.abs(pa_rate(beta, delta))
        obs = observable(beta, delta)
        ax1.plot(days, np.where(obs, np.nan, rate), color=color, lw=1, ls=":")
        ax1.plot(days, np.where(obs, rate, np.nan), color=color, lw=2, label=f"|β| = {beta}°")
        eps = sun_angle(beta, delta)
        ax2.plot(days, np.where(obs, eps, np.nan), color=color, lw=2)
        ax2.plot(days, np.where(obs, np.nan, eps), color=color, lw=1, ls=":")
    ax1.set(ylim=(0, 1.6), ylabel="|dPA_Z/dt|  (deg/day)",
            title="Nominal-roll PA_Z drift rate through one observing season")
    ax1.axhline(N_SUN / np.sin(np.deg2rad(SUN_MIN)), color=MUTED, lw=1, ls="--")
    ax1.text(-90, 1.245, "max. observable rate 1.22°/day (|β| = 54°, ε = 54° or 126°)", color=INK2, fontsize=9)
    ax1.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), title="Ecliptic latitude")
    for lo, hi in [(0, SUN_MIN), (SUN_MAX, 180)]:
        ax2.axhspan(lo, hi, color=GRID, alpha=0.7, lw=0)
    ax2.text(-90, 20, "Sun-angle keepout (ε < 54°)", color=INK2, fontsize=9)
    ax2.text(-90, 150, "keepout (ε > 126°)", color=INK2, fontsize=9)
    ax2.set(ylim=(0, 180), yticks=[0, 54, 90, 126, 180], ylabel="Sun-target angle ε (deg)",
            xlabel="Days from quadrature (Sun 90° in ecliptic longitude from the target)",
            xticks=np.arange(-90, 91, 15), xlim=(-92, 92))
    fig.text(0.01, -0.035,
             "Solid: observable (54° < ε < 126°); dotted: keepout. Mean solar motion 0.9856°/day. "
             "The pattern repeats every half year; day −91 is solar conjunction in longitude, day +91 opposition.\n"
             "Quadratures fall near DOY ≈ 80 + (λ* ± 90°)/0.9856 (mod 365). "
             "PA_Z decreases for northern targets (β > 0) and increases for southern targets; EXCAM +Y has the same rate.",
             fontsize=8.5, color=INK2)
    fig.savefig(outdir / "fig1_rate_vs_season.png", bbox_inches="tight")
    plt.close(fig)


def plot_rate_map(outdir):
    """Figure 2: signed dPA_Z/dt versus latitude and Sun-target longitude difference."""
    delta = np.linspace(0, 360, 1441)
    beta = np.linspace(-90, 90, 721)
    dd, bb = np.meshgrid(delta, beta)
    rate = pa_rate(bb, dd)
    obs = observable(bb, dd)
    fig, ax = plt.subplots(figsize=(10.5, 6.2), layout="constrained")
    ax.set_facecolor("white")
    ax.grid(False)
    mesh = ax.pcolormesh(delta, beta, np.ma.masked_where(~obs, rate), cmap=SIGNED_CMAP,
                         vmin=-1.25, vmax=1.25, shading="auto", rasterized=True)
    ax.contourf(delta, beta, (~obs).astype(float), levels=[0.5, 1.5], colors="none", hatches=["////"])
    ax.contour(delta, beta, sun_angle(bb, dd), levels=[SUN_MIN, SUN_MAX], colors=INK2, linewidths=1)
    lines = ax.contour(delta, beta, np.where(obs, np.abs(rate), np.nan),
                       levels=[0.2, 0.4, 0.6, 0.8, 1.0, 1.2], colors=INK, linewidths=0.6)
    ax.clabel(lines, fmt="%.1f", fontsize=8)
    for x in [14, 180, 346]:
        ax.text(x, 3, "keepout", ha="center", fontsize=9, color=INK2,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    ax.set(xlim=(0, 360), ylim=(-90, 90), xticks=np.arange(0, 361, 45), yticks=np.arange(-90, 91, 30),
           xlabel="Solar minus target ecliptic longitude, Δλ = λ☉ − λ* (deg)   [≈ 0.9856°/day]",
           ylabel="Target ecliptic latitude β (deg)",
           title="Signed nominal-roll dPA_Z/dt (deg/day); hatched = Sun-angle keepout")
    fig.colorbar(mesh, ax=ax, label="dPA_Z/dt (deg/day)", extend="both")
    fig.text(0.01, -0.06,
             "Day of year ≈ DOY₀ + Δλ/0.9856, where DOY₀ ≈ 80 + λ*/0.9856 (mod 365) is when the Sun passes the target's "
             "ecliptic longitude.\nContours: |dPA_Z/dt| in deg/day. Δλ = 90° and 270° are quadratures (slowest drift).",
             fontsize=8.5, color=INK2)
    fig.savefig(outdir / "fig2_rate_map.png", bbox_inches="tight")
    plt.close(fig)


def plot_targets(outdir, series):
    """Figure 3: EXCAM PA and dPA_Z/dt versus 2027 day of year for example targets."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.4), sharex=True, layout="constrained")
    for (name, ser), color in zip(series.items(), CAT):
        obs, doy = ser["obs"], ser["doy"]
        pa = np.where(obs, ser["pa_excam"], np.nan)
        pa[1:][np.abs(np.diff(ser["pa_excam"])) > 180] = np.nan  # break lines at 0/360 wraps
        label = f"{name} (β = {ser['beta']:+.1f}°)"
        ax1.plot(doy, pa, color=color, lw=2, label=label)
        ax2.plot(doy, np.where(obs, ser["rate_solver"], np.nan), color=color, lw=2.2, label=label)
        ax2.plot(doy, np.where(obs, ser["rate_doy"], np.nan), color=INK, lw=0.8, ls=(0, (3, 2)))
    ax2.plot([], [], color=INK, lw=0.8, ls=(0, (3, 2)), label="analytical formula, DOY Sun model")
    ax1.set(ylim=(0, 360), yticks=np.arange(0, 361, 60), ylabel="Nominal EXCAM +Y PA (deg)",
            title="2027 example targets: roman_pointing solver at nominal roll, observable dates only")
    ax1.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax2.axhline(0, color=AXIS, lw=1)
    ax2.set(xlim=(1, 366), xticks=[1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335],
            xticklabels=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
            ylabel="dPA_Z/dt (deg/day)", xlabel="Date in 2027 (day of year)", ylim=(-1.35, 1.35))
    ax2.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=9)
    fig.savefig(outdir / "fig3_targets_2027.png", bbox_inches="tight")
    plt.close(fig)


def plot_latitude_envelope(outdir):
    """Figure 4: rate envelope, window drift, and longest fixed-PA window versus |beta|."""
    absb = np.linspace(0, 90, 361)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), layout="constrained")

    ax = axes[0]
    sb = np.sin(np.deg2rad(absb))
    rmax = N_SUN * sb / np.maximum(np.sin(np.deg2rad(SUN_MIN)) ** 2, sb**2)
    ax.fill_between(absb, N_SUN * sb, rmax, color=BLUES[1], alpha=0.6, lw=0, label="range over observable dates")
    ax.plot(absb, rmax, color=BLUES[7], lw=2, label="maximum (at keepout edge or Δλ = 0°/180°)")
    ax.plot(absb, N_SUN * sb, color=BLUES[4], lw=2, label="minimum (quadrature)")
    ax.plot(absb, rmax * N_SUN_MAX / N_SUN, color=BLUES[7], lw=1, ls="--", label="maximum at perihelion speed")
    ax.set(xlim=(0, 90), ylim=(0, 1.4), xticks=[0, 15, 30, 45, 54, 60, 75, 90],
           xlabel="|β| (deg)", ylabel="|dPA_Z/dt| (deg/day)", title="(a) Drift-rate envelope")
    ax.legend(loc="upper left", fontsize=8.5)

    ax = axes[1]
    for width, color in zip([7, 14, 28, 56], [BLUES[2], BLUES[4], BLUES[6], BLUES[9]]):
        ax.plot(absb, window_drift(absb, width, True), color=color, lw=2, label=f"{width} d, worst case")
        ax.plot(absb, window_drift(absb, width, False), color=color, lw=1.3, ls="--", label=f"{width} d, quadrature-centered")
    ax.axhline(2 * ROLL, color=INK2, lw=1)
    ax.text(1, 2 * ROLL + 1, "30°: no fixed PA reachable with ±15° roll", fontsize=8.5, color=INK2)
    ax.set(xlim=(0, 90), ylim=(0, 70), xticks=[0, 15, 30, 45, 54, 60, 75, 90], xlabel="|β| (deg)",
           ylabel="Nominal PA_Z change over window, D (deg)", title="(b) PA drift across an execution window")
    ax.legend(loc="upper left", fontsize=8, ncol=2)

    ax = axes[2]
    ax.plot(absb, np.minimum(season_days(absb), 200), color=MUTED, lw=1.5, label="season length (cap)")
    for allowed, color in zip([30, 20, 10], [BLUES[9], BLUES[6], BLUES[3]]):
        margin = ROLL - allowed / 2
        ax.plot(absb, max_window(absb, allowed, True), color=color, lw=2,
                label=f"D ≤ {allowed}° (±{margin:g}° spare), worst case")
        ax.plot(absb, max_window(absb, allowed, False), color=color, lw=1.3, ls="--",
                label=f"D ≤ {allowed}°, quadrature-centered")
    ax.set(xlim=(0, 90), ylim=(0, 200), xticks=[0, 15, 30, 45, 54, 60, 75, 90], xlabel="|β| (deg)",
           ylabel="Longest execution window (days)", title="(c) Longest window with a common fixed PA_Z")
    ax.legend(loc="upper right", fontsize=8)
    fig.text(0.01, -0.07,
             "Worst case: window placed where drift is fastest within one observing season, at perihelion solar speed "
             "(1.0187°/day). Quadrature-centered: mean speed (0.9856°/day).\n"
             "A single PA_Z is reachable on every date of a window if D ≤ 2 × 15° − 2m, where m is any extra roll "
             "or science PA tolerance reserved; 'spare' is the margin left when the PA is centered on the drift range.",
             fontsize=8.5, color=INK2)
    fig.savefig(outdir / "fig4_latitude_envelope.png", bbox_inches="tight")
    plt.close(fig)


def plot_geometry_diagram(outdir):
    """Figure 5: projected-Sun geometry diagram and |rate| map versus Sun angle and latitude."""
    norm = Normalize(0, 1.25)
    r_keep = np.sin(np.deg2rad(SUN_MIN))
    fig = plt.figure(figsize=(15.5, 11.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.82], hspace=0.28, wspace=0.12,
                          left=0.04, right=0.93, top=0.90, bottom=0.06)
    fig.suptitle("Why the image orientation rotates: the Sun's projection on the sky around the target",
                 fontsize=16, weight="bold", x=0.485)
    fig.text(0.485, 0.925,
             "Sky plane centered on the target, ecliptic north up, east left (looking out). "
             "Nominal +Z points from the target toward the projected Sun.",
             ha="center", fontsize=11, color=INK2)

    full = np.deg2rad(np.linspace(0, 360, 721))
    for col, beta in enumerate([20, 54, 80]):
        ax = fig.add_subplot(gs[0, col])
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set(xlim=(-1.3, 1.3), ylim=(-1.3, 1.3))
        sb = np.sin(np.deg2rad(beta))
        ax.add_patch(Circle((0, 0), r_keep, facecolor=GRID, edgecolor=AXIS, lw=0.8, hatch="///", zorder=0))
        ax.add_patch(Circle((0, 0), 1, fill=False, edgecolor=AXIS, lw=0.8, ls=":", zorder=0))
        ax.plot(-np.sin(full), -sb * np.cos(full), color=MUTED, lw=1.2, zorder=1)

        # Sun positions every 7 days; x = -E (east left), y = N
        delta = np.arange(0, 365.25, 7) * N_SUN
        x, y = -np.sin(np.deg2rad(delta)), -sb * np.cos(np.deg2rad(delta))
        obs = observable(beta, delta)
        rate = np.abs(pa_rate(beta, delta))
        for xi, yi, ri in zip(x[obs], y[obs], rate[obs]):
            ax.plot([0, xi], [0, yi], color=RATE_CMAP(norm(ri)), lw=1.3, zorder=2)
        ax.scatter(x[obs], y[obs], c=rate[obs], cmap=RATE_CMAP, norm=norm, s=46,
                   edgecolor="white", lw=0.8, zorder=3)
        ax.scatter(x[~obs], y[~obs], facecolor="white", edgecolor=MUTED, s=26, lw=0.9, zorder=3)
        ax.scatter([0], [0], marker="*", s=260, color="#eda100", edgecolor=INK, lw=0.6, zorder=5)

        # Direction of the Sun's annual motion along the projected ecliptic
        d0, d1 = np.deg2rad(118), np.deg2rad(132)
        ax.add_patch(FancyArrowPatch((-np.sin(d0) * 1.09, -sb * np.cos(d0) * 1.09 - 0.02),
                                     (-np.sin(d1) * 1.09, -sb * np.cos(d1) * 1.09 - 0.02),
                                     arrowstyle="-|>", mutation_scale=14, color=INK2, lw=1.2,
                                     connectionstyle="arc3,rad=-0.15"))
        # Label one projected radius r = sin(epsilon)
        dq = np.deg2rad(63)
        ax.annotate("r = sin ε", (-np.sin(dq) * 0.55, -sb * np.cos(dq) * 0.55), xytext=(-0.25, -1.22),
                    fontsize=9.5, color=INK2, ha="center",
                    arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
        ax.text(0, -sb - 0.08 if beta < 60 else -sb - 0.1, "ε < 90°", ha="center", va="top", fontsize=9, color=INK2)
        ax.text(0, sb + 0.06, "ε > 90°", ha="center", va="bottom", fontsize=9, color=INK2)
        ax.text(1.27, 1.22, "N", ha="center", fontsize=9, color=INK2)
        ax.annotate("", (1.27, 1.2), (1.27, 0.95), arrowprops={"arrowstyle": "-|>", "color": INK2, "lw": 1})
        ax.annotate("", (1.02, 0.95), (1.27, 0.95), arrowprops={"arrowstyle": "-|>", "color": INK2, "lw": 1})
        ax.text(0.97, 0.95, "E", ha="right", va="center", fontsize=9, color=INK2)

        lo, hi = np.nanmin(rate[obs]), np.nanmax(rate[obs])
        cover = "observable all year" if beta >= SUN_MIN else f"two {season_days(beta):.0f}-day seasons"
        ax.set_title(f"β = {beta}°: projected ecliptic is an ellipse,\nsemi-axes 1 and sin β = {sb:.2f}",
                     fontsize=11.5, color=INK, pad=4)
        ax.text(0, -1.42, f"|dPA_Z/dt| = {lo:.2f}–{hi:.2f}°/day; {cover}", ha="center", fontsize=10.5, color=INK)
    fig.text(0.04, 0.475,
             "Dots: Sun position at 7-day steps (colored = observable, open = keepout). Hatched disk r < sin 54° = 0.81: "
             "Sun-angle keepout (ε < 54° or ε > 126°).  Arrow: direction of the Sun's apparent annual motion.",
             fontsize=9.5, color=INK2)

    ax = fig.add_subplot(gs[1, 0:2])
    ax.grid(False)
    eps = np.linspace(0.25, 179.75, 720)
    absb = np.linspace(0.125, 89.875, 360)
    ee, bb = np.meshgrid(eps, absb)
    rate = N_SUN * np.sin(np.deg2rad(bb)) / np.sin(np.deg2rad(ee)) ** 2
    reachable = (ee >= bb) & (ee <= 180 - bb)
    mesh = ax.pcolormesh(eps, absb, np.ma.masked_where(~reachable, rate), cmap=RATE_CMAP, norm=norm,
                         shading="auto", rasterized=True)
    ax.contourf(eps, absb, (~reachable).astype(float), levels=[0.5, 1.5], colors=["#f7f7f5"])
    ax.contourf(eps, absb, ((ee < SUN_MIN) | (ee > SUN_MAX)).astype(float), levels=[0.5, 1.5],
                colors="none", hatches=["///"])
    for x in [SUN_MIN, SUN_MAX]:
        ax.axvline(x, color=INK2, lw=1)
    lines = ax.contour(eps, absb, np.where(reachable, rate, np.nan), levels=[0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
                       colors=INK, linewidths=0.6)
    ax.clabel(lines, fmt="%.1f°/d", fontsize=8)
    for beta in [20, 54, 80]:
        ax.plot([beta, 180 - beta], [beta, beta], color=INK, lw=1.4, ls=(0, (4, 2)))
        ax.text(90, beta + 1.5, f"β = {beta}° yearly track", ha="center", fontsize=8.5, color=INK)
    ax.scatter([SUN_MIN, SUN_MAX], [SUN_MIN, SUN_MIN], marker="o", s=40, color=INK, zorder=5)
    ax.annotate("fastest observable:\n1.22°/day at |β| = ε = 54°", (SUN_MIN, SUN_MIN), xytext=(4, 68),
                fontsize=9, arrowprops={"arrowstyle": "->", "color": INK2})
    ax.text(20, 8, "Sun-angle\nkeepout", ha="center", fontsize=9, color=INK2,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    ax.text(160, 8, "keepout", ha="center", fontsize=9, color=INK2,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    ax.text(28, 80, "not reachable:\nε ≥ |β| always", ha="center", fontsize=9, color=MUTED)
    ax.set(xlim=(0, 180), ylim=(0, 90), xticks=[0, 30, 54, 90, 126, 150, 180], yticks=np.arange(0, 91, 15),
           xlabel="Sun-target angle ε (deg)", ylabel="Target |β| (deg)",
           title="|dPA_Z/dt| = λ̇☉ sin|β| / sin²ε  (deg/day, λ̇☉ = 0.9856°/day)")
    cax = fig.add_axes([0.945, 0.08, 0.012, 0.78])
    fig.colorbar(mesh, cax=cax, label="|dPA_Z/dt| (deg/day)", extend="max")

    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    notes = [
        ("Geometry", "The +Z axis follows the Sun's projection p onto the sky.\n"
                     "Its length is r = |p| = sin ε. Over a year p traces\n"
                     "an ellipse whose flattening is set by β."),
        ("Equal areas in equal times", "p sweeps area at the constant rate ½ λ̇☉ sin|β|,\n"
                                       "while area = ½ r² ΔPA. Hence\n"
                                       "|dPA_Z/dt| = λ̇☉ sin|β| / sin²ε."),
        ("Latitude", "At fixed ε, the rate scales with sin|β|: zero on the\n"
                     "ecliptic, λ̇☉/sin²ε near the pole."),
        ("Sun angle", "At fixed β, the rate is slowest at ε = 90° (quadrature)\n"
                      "and grows as 1/sin²ε toward the keepout edges.\n"
                      "Keepout caps it at λ̇☉ sin|β| / sin²54°."),
        ("Sense", "Northern targets: PA_Z decreases with time.\nSouthern targets: PA_Z increases."),
    ]
    yy = 0.98
    for head, body in notes:
        ax.text(0.0, yy, head, fontsize=11, weight="bold", va="top", transform=ax.transAxes)
        ax.text(0.0, yy - 0.055, body, fontsize=9.8, va="top", color=INK2, transform=ax.transAxes, linespacing=1.35)
        yy -= 0.07 + 0.052 * (body.count("\n") + 1)
    fig.savefig(outdir / "fig5_geometry_diagram.png", bbox_inches="tight")
    plt.close(fig)


def write_table(outdir):
    """Write the per-latitude summary as CSV and as a Markdown table; return the Markdown."""
    absb = np.array([0, 5, 10, 15, 20, 30, 40, 45, 50, 54, 60, 70, 80, 90], dtype=float)
    sb = np.sin(np.deg2rad(absb))
    rows = {
        "abs_beta_deg": absb,
        "season_days": season_days(absb),
        "rate_min_deg_day": N_SUN * sb,
        "rate_max_obs_deg_day": N_SUN * sb / np.maximum(np.sin(np.deg2rad(SUN_MIN)) ** 2, sb**2),
        "drift_14d_quad_deg": window_drift(absb, 14, False),
        "drift_14d_worst_deg": window_drift(absb, 14, True),
        "drift_28d_quad_deg": window_drift(absb, 28, False),
        "drift_28d_worst_deg": window_drift(absb, 28, True),
        "max_window_D30_worst_days": max_window(absb, 30, True),
        "max_window_D30_quad_days": max_window(absb, 30, False),
        "max_window_D20_worst_days": max_window(absb, 20, True),
    }
    with open(outdir / "pa_rate_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(rows.keys())
        writer.writerows(np.round(np.column_stack(list(rows.values())), 4))
    header = ("| \\|β\\| | Season (d) | Min rate (°/d) | Max obs. rate (°/d) | D, 14 d quad / worst | "
              "D, 28 d quad / worst | Longest window, D ≤ 30°: worst / quad (d) | D ≤ 20°: worst (d) |")
    lines = [header, "|" + "---:|" * 8]
    for i in range(len(absb)):
        season = "all year" if absb[i] >= SUN_MIN else f"{rows['season_days'][i]:.0f}"
        lines.append(
            f"| {absb[i]:.0f}° | {season} | {rows['rate_min_deg_day'][i]:.3f} | {rows['rate_max_obs_deg_day'][i]:.3f} | "
            f"{rows['drift_14d_quad_deg'][i]:.1f}° / {rows['drift_14d_worst_deg'][i]:.1f}° | "
            f"{rows['drift_28d_quad_deg'][i]:.1f}° / {rows['drift_28d_worst_deg'][i]:.1f}° | "
            f"{rows['max_window_D30_worst_days'][i]:.0f} / {rows['max_window_D30_quad_days'][i]:.0f} | "
            f"{rows['max_window_D20_worst_days'][i]:.0f} |")
    table = "\n".join(lines)
    (outdir / "pa_rate_summary.md").write_text(table + "\n")
    return table


def main():
    """Generate figures, tables, and (optionally) the solver validation record."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outdir", type=Path, default=OUTDIR, help="Output directory.")
    parser.add_argument("--no-validate", action="store_true", help="Skip the roman_pointing solver comparison.")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    plot_rate_vs_phase(args.outdir)
    plot_rate_map(args.outdir)
    plot_latitude_envelope(args.outdir)
    plot_geometry_diagram(args.outdir)
    print(write_table(args.outdir))
    if not args.no_validate:
        summary, series = validate()
        (args.outdir / "validation.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        plot_targets(args.outdir, series)


if __name__ == "__main__":
    main()
