"""
Validation and figures for the Roman field rotation model (rate of change of PA_V3, and
equivalently PA_EXCAM_Y ≈ PA_V3 + 150 deg) vs. target ecliptic latitude and time of year.
The model is in roman_pointing.field_rotation; this script validates it against
roman_pointing.calcRomanAngles / getRomanPositionAngle and makes the figures and the
fixed-PA table in docs/field_rotation_rate. Roll-offset planning for specific RDI sequences is
in plan_roll_limits.py.

Usage:
  python tools/field_rotation_rate.py -o docs/field_rotation_rate

"""

import argparse
import os

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import BarycentricMeanEcliptic, SkyCoord
from astropy.time import Time
from matplotlib.colors import LinearSegmentedColormap

from roman_pointing.field_rotation import (
    ROLL_MAX,
    SUN_ANG_MAX,
    SUN_ANG_MIN,
    SUN_RATE,
    ecliptic_north_offset,
    fixed_pa_window,
    for_window,
    pa_from_ecliptic_north,
    pa_swing_in_window,
    pa_v3_rate,
    sun_longitude_doy,
)
from roman_pointing.roman_pointing import (
    applyRollAngle,
    calcRomanAngles,
    getL2Positions,
    getRomanPositionAngle,
    getSunPositions,
)

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
PURPLE = "#6250d6"  # Roman observatory in the geometry diagram

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
    lam_sun_doy = sun_longitude_doy(doy, year)
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

        # analytic PA_V3 using the ephemeris solar longitude (tests the geometry alone)
        pa_an = pa_from_ecliptic_north(lam_sun_eph - lam, beta) + ecliptic_north_offset(lam, beta)
        pa_err = (pa_an - pa_num + 180) % 360 - 180
        rate_an_eph = pa_v3_rate(lam_sun_eph - lam, beta, np.gradient(lam_sun_eph, doy))
        # fully analytic rate vs. day of year (circular orbit)
        rate_an_doy = pa_v3_rate(lam_sun_doy - lam, beta)

        # PA_V3 is measured E of N: compare with astropy PA of the Sun seen from the target
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

    # roll offsets PA_V3 one-for-one, so it does not change the rate
    target = SkyCoord(lon=100 * u.deg, lat=30 * u.deg, frame=BarycentricMeanEcliptic,
                      distance=10 * u.pc, pm_lon_coslat=0 * u.mas / u.yr, pm_lat=0 * u.mas / u.yr,
                      radial_velocity=0 * u.km / u.s, obstime=Time("J2000"))
    _, _, _, B_C_I = calcRomanAngles(target, ts[:10], r_obs_G[:, :10], r_sun_G[:, :10])
    pa0 = getRomanPositionAngle(B_C_I).to_value(u.deg)
    pa_roll = getRomanPositionAngle(applyRollAngle(B_C_I, [10 * u.deg] * 10)).to_value(u.deg)
    print(f"PA_V3 change for +10 deg roll: {((pa_roll - pa0 + 180) % 360 - 180).mean():.4f} deg")

    return {"doy": doy, "results": results}


def plot_rate_vs_time(outdir):
    """Plot dPA_V3/dt vs. days after solar conjunction.

    Args:
        outdir (str):
            Output directory
    """
    betas = [5, 15, 30, 45, 54, 70, 90]
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
    days = np.linspace(0, 365.25, 4001)
    delta = SUN_RATE * days

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    for beta, c in zip(betas, colors):
        rate = np.abs(pa_v3_rate(delta, beta))
        cos_eps = np.cos(np.radians(beta)) * np.cos(np.radians(delta))
        in_for = np.abs(cos_eps) <= np.cos(np.radians(SUN_ANG_MIN))
        ax.plot(days, np.where(in_for, rate, np.nan), color=c, label=f"β = {beta}°")
        ax.plot(days, np.where(in_for, np.nan, rate), color=c, lw=0.8, ls=":")
        k = 1300 if beta == 70 else 1000
        ax.annotate(f"{beta}°", (days[k], rate[k]), xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, color=INK2)

    ax.set_ylim(0, 2.0)
    ax.set_xlim(0, 365.25)
    ax.set_xticks(np.arange(0, 366, 30.44 * 1.5))
    ax.set_xticklabels([f"{x:.0f}" for x in np.arange(0, 366, 30.44 * 1.5)])
    for x, lab in [(0, "conj."), (365.25 / 4, "quad."), (365.25 / 2, "opp."), (3 * 365.25 / 4, "quad.")]:
        ax.axvline(x, color=AXIS, lw=0.8, zorder=0)
        ax.text(x, 2.0, lab, ha="center", va="bottom", fontsize=8.5, color=MUTED)
    ax.set_xlabel("Days after solar conjunction (λ☉ = λ_target)")
    ax.set_ylabel("|dPA$_\\mathrm{V3}$ / dt|  (deg/day)")
    ax.set_title("Rate of image rotation through the year", pad=16)
    ax.text(0.01, 0.97, "solid: in field of regard (54°–126° from Sun)\n"
            "dotted: Sun keep-out (not schedulable)",
            transform=ax.transAxes, va="top", fontsize=8.5, color=INK2)
    ax.legend(loc="upper right", ncol=2, title="ecliptic latitude |β|", title_fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "pa_v3_rate_vs_time.png"), dpi=150)
    plt.close(fig)


def plot_rate_map(outdir):
    """Map of |dPA_V3/dt| vs. days after solar conjunction and ecliptic latitude.

    Args:
        outdir (str):
            Output directory
    """
    days = np.linspace(0, 365.25 / 2, 731)
    betas = np.linspace(0, 90, 361)
    D, B = np.meshgrid(days, betas)
    rate = np.abs(pa_v3_rate(SUN_RATE * D, B))
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
    ax.set_title("|dPA$_\\mathrm{V3}$/dt| within the field of regard\n"
                 "contour labels: rate and days a fixed PA$_\\mathrm{V3}$ stays within ±15° roll", fontsize=10.5)
    cb = fig.colorbar(cf, ax=ax, pad=0.02)
    cb.set_label("|dPA$_\\mathrm{V3}$/dt| (deg/day)", color=INK2)
    cb.outline.set_edgecolor(AXIS)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "pa_v3_rate_map.png"), dpi=150)
    plt.close(fig)


def plot_geometry_diagram(outdir):
    """Diagram of the Sun-target geometry that sets the PA_V3 rate.

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

    fig, ax = plt.subplots(figsize=(8.5, 7.6))
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
    ax.plot(0, 0, "s", ms=10, color=PURPLE, mec=SURFACE, mew=1.2, zorder=6)
    ax.text(-0.04, -0.05, "Roman (at L2)", fontsize=10.5, color=PURPLE, va="top", ha="right")

    # eps: the angle at Roman between the lines of sight to the target and to the Sun
    pts = np.array([proj(0.45 * p) for p in arc(T, S_main)])
    ax.fill(np.r_[0, pts[:, 0]], np.r_[0, pts[:, 1]], color=BLUE_RAMP[1], alpha=0.7, lw=0, zorder=1)
    ax.plot(pts[:, 0], pts[:, 1], color=INK, lw=1.5, zorder=3)
    ax.text(*proj(0.3 * arc(T, S_main)[50]), "ε", fontsize=16, color=INK, ha="center", va="center",
            zorder=4)

    # beta along the target's meridian, Delta along the ecliptic (both measured on the sphere)
    for a_pt, b_pt, lab, off in [(F, T, "β", [-0.07, 0.0]), (F, S_main, r"Δ = $\mathregular{\lambda_\odot}$ − λ", [0.0, -0.06])]:
        pts = np.array([proj(p) for p in arc(a_pt, b_pt)])
        ax.plot(pts[:, 0], pts[:, 1], color=INK2, lw=2.2, zorder=2)
        ax.text(*pts[50] + off, lab, fontsize=12, color=INK, ha="center", va="center")
    pts = np.array([proj(p) for p in arc(T, K)])
    ax.plot(pts[:, 0], pts[:, 1], color=MUTED, lw=0.9, ls=":")

    # +V3 on the sky at T follows the great circle toward the Sun
    pts = np.array([proj(p) for p in arc(T, S_main)])
    ax.plot(pts[:, 0], pts[:, 1], color=SERIES[0], lw=2.2)
    ax.text(*pts[38] + [0.02, 0.09], "great circle\nto the Sun",
            fontsize=9.5, color=SERIES[0], va="bottom", ha="left")

    ax.plot(*proj(K), "o", ms=7, color=INK, zorder=5)
    ax.text(*proj(K) + [0.03, 0.03], "ecliptic north pole", fontsize=10, color=INK)
    ax.plot(*proj(T), "*", ms=18, color=SERIES[1], mec=SURFACE, mew=1, zorder=6)
    ax.text(*proj(T) + [-0.05, 0.04], "target\n(λ, β)", fontsize=10.5, color=INK, ha="right")
    ax.plot(*proj(S_main), "o", ms=13, color=SERIES[3], mec=SURFACE, mew=1.5, zorder=6)
    ax.text(*proj(S_main) + [0.0, -0.06], r"Sun ($\mathregular{\lambda_\odot}$, 0)", fontsize=10.5, color=INK, ha="center", va="top")

    # at T: ecliptic north and +V3 (toward the Sun) on the sky, and the angle theta between them
    pT = proj(T)
    tK, tS = proj(tangent(T, K)), proj(tangent(T, S_main))
    ax.annotate("", xy=pT + 0.22 * tK / np.linalg.norm(tK), xytext=pT,
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.5))
    ax.annotate("", xy=pT + 0.25 * tS / np.linalg.norm(tS), xytext=pT,
                arrowprops=dict(arrowstyle="-|>", color=SERIES[0], lw=2.2))
    ax.text(*(pT + 0.24 * tK / np.linalg.norm(tK)) + [-0.02, 0.02], "ecl. N", color=INK2,
            fontsize=9, ha="right")
    ax.text(*(pT + 0.27 * tS / np.linalg.norm(tS)) + [0.0, 0.03], "+V3", color=SERIES[0], fontsize=11)
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
    ax.text(*(pS + scale * proj(v)) + [0.02, -0.02], "Sun's motion\n" r"$\mathregular{\dot{\lambda}_\odot}$ ≈ 0.986°/day",
            fontsize=10, color=INK, va="top")
    ax.text(*(pS + scale * proj(v_perp)) + [0.08, 0.16],
            "part across the\ntarget–Sun arc:\n" r"$\mathregular{\dot{\lambda}_\odot}$ sin β / sin ε",
            fontsize=10, color=SERIES[1], va="bottom",
            bbox=dict(facecolor=SURFACE, alpha=0.85, lw=0, pad=1))

    ax.text(-1.05, -1.08,
            "ε is the angle at Roman between the lines of sight to the target and the Sun; it equals the\n"
            "length of the great-circle arc from target to Sun, which sets the +V3 direction on the sky.\n"
            "As the Sun moves along the ecliptic, only its motion across that arc swings the arc,\n"
            "and seen from the target (a distance ε away) the arc turns at",
            fontsize=10, color=INK2, va="top")
    ax.text(-1.05, -1.38, r"dθ/dt = −$\mathregular{\dot{\lambda}_\odot}$ sin β / sin²ε", fontsize=14, color=INK, va="top")
    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-1.5, 1.12)
    ax.set_title("Geometry: Roman at the center of the celestial sphere", loc="left")

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "pa_v3_rate_geometry_diagram.png"), dpi=150)
    plt.close(fig)


def plot_validation(val, outdir):
    """Plot analytic vs. roman_pointing PA_V3 rates over one year.

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
    axs[0].set_ylabel("dPA$_\\mathrm{V3}$/dt (deg/day)")
    axs[0].set_title("Analytic, circular-orbit model (lines) vs. roman_pointing (circles), "
                     "2027, in-FOR times only", fontsize=10.5)
    axs[0].legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(1.01, 1))
    axs[1].set_title(f"Residual is set by Earth's orbital eccentricity; with the ephemeris solar "
                     f"longitude it is ≤ {geom_err:.3f} deg/day", fontsize=9.5, color=INK2)
    axs[1].set_ylabel("Analytic − numeric\n(deg/day)")
    axs[1].set_xlabel("Day of year 2027")
    axs[1].set_xlim(1, 366)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "pa_v3_rate_validation.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="field_rotation_rate.py",
                                 description="Analytic PA_V3 rate model: validation and figures")
    ap.add_argument("-o", "--outdir", default="docs/field_rotation_rate", help="Output directory for figures")
    ap.add_argument("--year", type=int, default=2027, help="Year used for validation against roman_pointing")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    val = validate(args.year)
    plot_validation(val, args.outdir)
    plot_rate_vs_time(args.outdir)
    plot_rate_map(args.outdir)
    plot_geometry_diagram(args.outdir)

    # guideline tables: fixed PA. The two fixed PAs of an RDI sequence can be at most 2 ROLL_MAX - (PA swing) apart
    spans = (7, 14, 30)
    label = "max PA separation (deg), worst / quad, for span T ="
    print(f"\n{'|beta|':>6} {'rate quad':>10} {'rate max':>9} {'window d':>9} {'swing':>7} "
          f"{'fixed-PA d (worst/quad)':>24} | {label} " + " ".join(f"{f'{T} d':>13}" for T in spans))
    for b in [5, 10, 15, 20, 30, 40, 45, 50, 54, 60, 70, 80, 90]:
        sb = np.sin(np.radians(b))
        rmax = SUN_RATE / sb if b >= SUN_ANG_MIN else SUN_RATE * sb / np.sin(np.radians(SUN_ANG_MIN)) ** 2
        _, dur, sw = for_window(b)
        tw = fixed_pa_window(b, "worst")
        tq = fixed_pa_window(b, "quadrature")
        seps = [f"{2 * ROLL_MAX - pa_swing_in_window(b, T):5.1f} / "
                f"{2 * ROLL_MAX - pa_swing_in_window(b, T, 'quadrature'):5.1f}" for T in spans]
        print(f"{b:6.0f} {SUN_RATE * sb:10.3f} {rmax:9.3f} {dur:9.1f} {sw:7.1f} "
              f"{tw:11.1f} / {tq:<10.1f} | {'':>{len(label)}} " + " ".join(f"{c:>13}" for c in seps))
