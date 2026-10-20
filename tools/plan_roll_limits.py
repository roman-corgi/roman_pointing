"""
Allowed roll offsets, and reachable fixed sky PAs, for a Roman visit when the sky PA is held fixed.

A requested roll offset A_nom becomes an absolute sky PA, P = PA_nom(t_mid) - A_nom (roll offsets follow
roman_pointing.applyRollAngle: a roll of +r changes PA_V3 by -r). Executing at time t then needs a roll offset
A(t) = A_nom + [PA_nom(t) - PA_nom(t_mid)], which must stay within ±ROLL_MAX from the earliest possible visit
start to the latest possible visit end. Equivalently, an exact sky PA P is reachable over that span only if
max PA_nom - ROLL_MAX <= P <= min PA_nom + ROLL_MAX; the two ends are the pair of fixed PAs with the maximum
separation. The rate model is in roman_pointing.field_rotation.

Usage:
  # allowed roll offsets and fixed sky PAs for one target and visit (add --plot for a figure,
  # --pa PA1 PA2 to check two requested sky PAs)
  python tools/plan_roll_limits.py --ra 53.2327 --dec -9.4583 --start 2027-01-19T00:00:00 \\
      --duration 1.5 --early 3 --late 3
  # same, resolving the target by name (CORGIDB, then SIMBAD) instead of --ra/--dec
  python tools/plan_roll_limits.py --target "eps Eri" --start 2027-01-19T00:00:00 \\
      --duration 1.5 --early 3 --late 3
  # generic roll-limit figures, the README worked example, and the guideline table
  python tools/plan_roll_limits.py --figures -o docs/field_rotation_rate

"""

import argparse
import os
import re

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
from matplotlib.colors import LinearSegmentedColormap

from field_rotation_rate import AXIS, BLUE_RAMP, INK, INK2, KEEPOUT, MUTED, SERIES, SURFACE  # chart style
from roman_pointing.field_rotation import (
    ROLL_MAX,
    SUN_ANG_MAX,
    SUN_ANG_MIN,
    SUN_RATE,
    pa_from_ecliptic_north,
    pa_v3_drift,
)
from roman_pointing.roman_observability import get_target_coords

# README worked example: epsilon Eridani, January 2027
EXAMPLE = {"target": "eps Eri", "start": "2027-01-19T00:00:00", "duration": 1.5, "early": 3.0, "late": 3.0,
           "name": "ε Eridani", "offsets": {"A": -13.0, "B": +13.0}, "filename": "roll_limits_example.png"}


def resolve_target(name):
    """Resolve a target name with get_target_coords (CORGIDB, then SIMBAD).

    Args:
        name (str):
            Target name

    Returns:
        tuple:
            ra, dec (deg, ICRS, J2000); proper motion is neglected, as in pa_v3_drift
    """
    coords = get_target_coords([name])
    if name not in coords:
        raise SystemExit(f"Could not resolve target '{name}'")
    icrs = coords[name].icrs
    print(f"Target {name}: RA {icrs.ra.deg:.4f} deg, Dec {icrs.dec.deg:+.4f} deg (ICRS, J2000)")
    return icrs.ra.deg, icrs.dec.deg


def roll_offset_limits(beta, day_mid, early, late):
    """Allowed requested roll offsets from the analytic model (circular orbit).

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
            a_min, a_max (deg, applyRollAngle sense): allowed range of requested roll offsets
    """
    day_mid, beta = np.broadcast_arrays(np.asarray(day_mid, dtype=float), np.asarray(beta, dtype=float))
    t = np.linspace(-early, late, 61).reshape((-1,) + (1,) * day_mid.ndim)
    s = (pa_from_ecliptic_north(SUN_RATE * (day_mid + t), beta)
         - pa_from_ecliptic_north(SUN_RATE * day_mid, beta))
    s = (s + 180) % 360 - 180
    return -ROLL_MAX - s.min(axis=0), ROLL_MAX - s.max(axis=0)


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
    """Allowed requested roll offsets and reachable fixed sky PA_V3 for a specific target and visit.

    Uses roman_pointing with the solar ephemeris. A fixed sky PA P needs a roll PA_nom(t) - P at time t, so it is
    reachable over the span for pa_lo <= P <= pa_hi, with pa_lo = PA_nom(t_mid) - a_max and
    pa_hi = PA_nom(t_mid) - a_min. pa_lo and pa_hi are the pair of fixed PAs with the maximum separation,
    a_max - a_min = 2 ROLL_MAX - (PA swing over the span); there is none if that is negative.

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
            a_min, a_max (deg, applyRollAngle sense); pa_mid, pa_lo, pa_hi (deg, sky PA_V3 in [0, 360));
            separation (deg); beta (deg); Sun angle range (deg); PA_V3 rate at mid-visit (deg/day)
    """
    t_mid = Time(start, scale="utc") + duration / 2 * u.day
    dt = np.linspace(-(duration / 2 + early), duration / 2 + late, 241)
    drift = pa_v3_drift(ra, dec, t_mid, dt)
    s, pa_mid = drift["s"], drift["pa_mid"]
    a_min, a_max = -ROLL_MAX - s.min(), ROLL_MAX - s.max()
    return {"a_min": a_min, "a_max": a_max, "pa_mid": pa_mid, "pa_lo": (pa_mid - a_max) % 360,
            "pa_hi": (pa_mid - a_min) % 360, "separation": a_max - a_min, "beta": drift["beta"],
            "sun_ang": (drift["sun_ang"].min(), drift["sun_ang"].max()), "rate": drift["rate"]}


def plot_target_roll_limits(outdir, ra, dec, start, duration, early=0.0, late=0.0, name=None, offsets=None,
                            filename="roll_limits_target.png"):
    """Needed roll vs. execution time for a specific target and visit, and the maximum |roll| vs. start shift.

    Args:
        outdir (str):
            Output directory
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
        name (str):
            Target name for the plot title; defaults to the coordinates
        offsets (dict):
            Optional requested rolls to draw for illustration, e.g. {"A": -13.0, "B": 13.0}
        filename (str):
            Output file name
    """
    half = duration / 2
    t_start = Time(start, scale="utc")
    t_mid = t_start + half * u.day
    res = roll_limits_for_target(ra, dec, start, duration, early, late)
    a_min, a_max = res["a_min"], res["a_max"]

    # one dense grid covers panel (a) and the start shifts up to 4 d in panel (b)
    durations = sorted({0.5, 1.0, duration, 3.0})
    t_lim = max(half + max(early, late) + 1.5, max(durations) / 2 + 4) + 0.1
    t = np.linspace(-t_lim, t_lim, int(round(200 * t_lim)) + 1)
    s = pa_v3_drift(ra, dec, t_mid, t)["s"]
    t_plot = (-half - early - 1.5, half + late + 1.5)
    in_plot = (t >= t_plot[0]) & (t <= t_plot[1])
    y_lim = max(19.5, ROLL_MAX + np.abs(s[in_plot]).max() + 1)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5.8), gridspec_kw={"width_ratios": [1.5, 1]})
    ax.axvspan(-half - early, half + late, color=KEEPOUT, lw=0, zorder=0)
    ax.axvspan(-half, half, color=BLUE_RAMP[0], lw=0, zorder=0)
    ax.text(0, 0.03, f"requested\nvisit ({duration:g} d)", ha="center", va="bottom", fontsize=9, color=INK2,
            transform=ax.get_xaxis_transform())
    first, last = t_start - early * u.day, t_start + late * u.day
    fmt = "%b %d" if first.strftime("%H:%M") == last.strftime("%H:%M") == "00:00" else "%b %d %H:%M"
    ax.text(-half - early + 0.1, 0.97, f"possible execution span\n(start {first.strftime(fmt)} – "
            f"{last.strftime(fmt)})", va="top", fontsize=9, color=INK2, transform=ax.get_xaxis_transform())
    for x in [-ROLL_MAX, ROLL_MAX]:
        ax.axhline(x, color=INK, lw=1.2)
    in_span = (t >= -half - early) & (t <= half + late)
    for k, (lim, c) in enumerate(zip([a_min, a_max], [SERIES[0], SERIES[1]])):
        if offsets is not None:
            name_k, off = list(offsets.items())[k]
            need = off + s
            ax.plot(t, need, color=c, lw=1.2, alpha=0.6, label=f"roll {name_k} = {off:+.0f}° (illustration)")
            bad = np.abs(need) > ROLL_MAX
            ax.plot(t, np.where(bad, need, np.nan), color=c, lw=5, alpha=0.25, solid_capstyle="butt")
            for j in np.flatnonzero(np.diff(bad.astype(int)) != 0):
                if t_plot[0] <= t[j] <= t_plot[1]:
                    ax.plot(t[j], need[j], "o", ms=6, color=c, mec=SURFACE, mew=1.2, zorder=5)
                    ax.annotate(f"{t[j]:+.1f} d", (t[j], need[j]), xytext=(0, 10 if off > 0 else -16),
                                textcoords="offset points", ha="center", fontsize=9, color=INK2)
        # limiting roll: its needed roll just reaches ±15° at an end of the execution span
        need = lim + s
        ax.plot(t, need, color=c, lw=2.2, ls="--", label=f"limit {lim:+.2f}°")
        j = np.flatnonzero(in_span)[np.argmax(np.abs(need[in_span]))]
        ax.plot(t[j], need[j], "o", ms=8, color=c, mec=INK, mew=1.2, zorder=6)
    ax.text(0, 0.57, f"Allowed requested rolls over the span:\n{a_min:+.2f}° to {a_max:+.2f}°", ha="center",
            fontsize=10, color=INK, transform=ax.get_xaxis_transform(),
            bbox=dict(facecolor=SURFACE, alpha=0.9, lw=0.5, edgecolor=AXIS, pad=4))
    ax.set_xlim(t_plot[0] - 0.2, t_plot[1] + 0.2)
    ax.set_ylim(-y_lim, y_lim)
    ax.set_xlabel(f"Execution time relative to the planned mid-visit, {t_mid.strftime('%Y-%m-%d %H:%M')} UTC (days)")
    ax.set_ylabel("Roll needed to hold the CPGS sky PA (deg)")
    label = name if name else f"RA {ra:.3f}°, Dec {dec:+.3f}°"
    ax.set_title(f"(a) {label} (β = {res['beta']:.1f}°, |rate| ≈ {abs(res['rate']):.2f}°/d): thick = beyond ±15°",
                 fontsize=10.5, loc="left")
    ax.legend(loc="center right", fontsize=9)

    # symmetric limit for a half-span H: the requested roll must absorb max(s) and min(s) over |t| <= H
    shifts = np.linspace(0, 4, 81)
    sym_all = []
    for dur, c in zip(durations, [BLUE_RAMP[2], BLUE_RAMP[4], BLUE_RAMP[6], BLUE_RAMP[8]]):
        sym = [min(ROLL_MAX - s[np.abs(t) <= dur / 2 + b].max(), ROLL_MAX + s[np.abs(t) <= dur / 2 + b].min())
               for b in shifts]
        sym_all += sym
        ax2.plot(shifts, sym, color=c, label=f"D = {dur:g} d")
    if offsets is not None:
        r = abs(list(offsets.values())[0])
        ax2.axhline(r, color=MUTED, lw=1, ls="--")
        ax2.text(0.05, r + 0.15, f"±{r:g}° (rolls {', '.join(offsets)})", fontsize=8.5, color=INK2)
    y_min = min(9, np.floor(min(sym_all)) - 0.5)
    if early == late:
        sym_lim = min(a_max, -a_min)
        ax2.axvline(early, color=MUTED, lw=1, ls=":")
        ax2.plot(early, sym_lim, "o", ms=8, color=BLUE_RAMP[2 + 2 * durations.index(duration)], mec=INK, mew=1.2,
                 zorder=5)
        ax2.annotate(f"±{sym_lim:.2f}°", (early, sym_lim), xytext=(8, 4), textcoords="offset points", fontsize=9,
                     color=INK, bbox=dict(facecolor=SURFACE, alpha=0.9, lw=0, pad=1))
        ax2.text(early + 0.05, y_min + 0.2, f"this visit\n(U = {early:g} d)", fontsize=8.5, color=INK2)
    ax2.set_xlim(0, 4)
    ax2.set_ylim(y_min, 15.2)
    ax2.set_xlabel("Allowed shift of the start, ±U (days)")
    ax2.set_ylabel("Maximum symmetric |roll| (deg)")
    ax2.set_title("(b) Same target and dates: maximum |roll| vs. allowed\nstart shift and visit duration",
                  fontsize=10.5, loc="left")
    ax2.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, filename), dpi=150)
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="plan_roll_limits.py",
                                 description="Allowed roll offsets when CPGS fixes the sky PA at mid-visit")
    ap.add_argument("--ra", type=float, help="Target RA (deg, ICRS)")
    ap.add_argument("--dec", type=float, help="Target Dec (deg, ICRS)")
    ap.add_argument("--target", type=str,
                    help="Target name, resolved with get_target_coords (CORGIDB, then SIMBAD), instead of --ra/--dec")
    ap.add_argument("--start", type=str, help="Requested visit start (ISO, UTC)")
    ap.add_argument("--duration", type=float, default=1.0, help="Visit duration (days)")
    ap.add_argument("--early", type=float, default=0.0, help="Days the start may move earlier than requested")
    ap.add_argument("--late", type=float, default=0.0, help="Days the start may move later than requested")
    ap.add_argument("--pa", type=float, nargs=2, metavar=("PA1", "PA2"),
                    help="Check two requested fixed sky PA_V3 values (deg) against the reachable range")
    ap.add_argument("--name", type=str, help="Target name for the plot title (default: --target)")
    ap.add_argument("--plot", action="store_true", help="Also plot needed roll vs. execution time for this visit")
    ap.add_argument("--figures", action="store_true",
                    help="Make the generic roll-limit figures, the README worked example, and the guideline table")
    ap.add_argument("-o", "--outdir", help="Output directory for figures (default: docs/field_rotation_rate "
                                           "with --figures, otherwise the current directory)")
    args = ap.parse_args()
    outdir = args.outdir or ("docs/field_rotation_rate" if args.figures else ".")

    if args.figures:
        os.makedirs(outdir, exist_ok=True)
        example = dict(EXAMPLE)
        example["ra"], example["dec"] = resolve_target(example.pop("target"))
        plot_target_roll_limits(outdir, **example)
        plot_roll_limits_vs_latitude(outdir)
        plot_roll_limits_map(outdir)

        # guideline table: maximum requested |roll| with the PA fixed at mid-visit, H = D/2 + U
        print(f"{'|beta|':>6} {'rate worst':>10} {'rate quad':>9} | max |roll| worst / quad for H = "
              f"{'1 d':>13} {'2 d':>13} {'3 d':>13} {'4 d':>13} | {'slack d/deg worst/quad':>22}")
        for b in [5, 10, 20, 30, 40, 45, 50, 54, 60, 70, 80, 90]:
            sb = np.sin(np.radians(b))
            rmax = SUN_RATE / sb if b >= SUN_ANG_MIN else SUN_RATE * sb / np.sin(np.radians(SUN_ANG_MIN)) ** 2
            cols = [f"{max_symmetric_roll(b, h):5.1f} / {max_symmetric_roll(b, h, 'quadrature'):5.1f}"
                    for h in (1, 2, 3, 4)]
            print(f"{b:6.0f} {rmax:10.3f} {SUN_RATE * sb:9.3f} | {'':>31}" + " ".join(f"{c:>13}" for c in cols)
                  + f" | {1 / rmax:9.2f} / {1 / (SUN_RATE * sb):<9.2f}")
        raise SystemExit

    if args.target is not None:
        if args.ra is not None or args.dec is not None:
            ap.error("give either --target or --ra/--dec, not both")
        args.ra, args.dec = resolve_target(args.target)
        args.name = args.name or args.target
    if args.ra is None or args.dec is None or args.start is None:
        ap.error("--start and either --target or --ra/--dec are required unless --figures is given")
    res = roll_limits_for_target(args.ra, args.dec, args.start, args.duration, args.early, args.late)
    print(f"Target beta = {res['beta']:.2f} deg, Sun angle {res['sun_ang'][0]:.1f}-{res['sun_ang'][1]:.1f} deg, "
          f"dPA_V3/dt at mid-visit = {res['rate']:+.3f} deg/day")
    print(f"Allowed requested roll offsets (applyRollAngle sense): {res['a_min']:+.2f} to {res['a_max']:+.2f} deg; "
          f"symmetric limit +/-{min(res['a_max'], -res['a_min']):.2f} deg")
    if res["separation"] >= 0:
        print(f"Reachable fixed sky PA_V3 over the span: {res['pa_lo']:.2f} to {res['pa_hi']:.2f} deg; "
              f"maximum-separation pair {res['pa_lo']:.2f} and {res['pa_hi']:.2f} deg "
              f"(separation {res['separation']:.2f} deg)")
    else:
        print(f"No fixed sky PA_V3 is reachable over the whole span: the nominal PA swing exceeds "
              f"{2 * ROLL_MAX:.0f} deg by {-res['separation']:.2f} deg")
    if args.pa is not None:
        for k, p in enumerate(args.pa, 1):
            # needed roll PA_nom(t) - P over the span; its range follows from a_min, a_max
            r_mid = (res["pa_mid"] - p + 180) % 360 - 180
            r_lo, r_hi = r_mid - ROLL_MAX - res["a_min"], r_mid + ROLL_MAX - res["a_max"]
            ok = -ROLL_MAX - 0.005 <= r_lo and r_hi <= ROLL_MAX + 0.005  # tolerate the printed 0.01 deg rounding
            print(f"PA{k} = {p:.2f} deg: needed roll {r_lo:+.2f} to {r_hi:+.2f} deg over the span "
                  f"({'reachable' if ok else 'NOT reachable'})")
    if res["sun_ang"][0] < SUN_ANG_MIN or res["sun_ang"][1] > SUN_ANG_MAX:
        print("WARNING: part of the visit span is outside the 54-126 deg field of regard")
    if args.plot:
        os.makedirs(outdir, exist_ok=True)
        slug = re.sub(r"[^0-9a-z]+", "_", (args.name or "target").lower()).strip("_") or "target"
        filename = f"roll_limits_{slug}.png"
        plot_target_roll_limits(outdir, args.ra, args.dec, args.start, args.duration, args.early, args.late,
                                name=args.name, filename=filename)
        print(f"Figure written to {os.path.join(outdir, filename)}")
