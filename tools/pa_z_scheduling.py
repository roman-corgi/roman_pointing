"""Reproduce the PA_Z seasonal-drift analysis and figures, without network access.

Run from the repository root in an environment with its dependencies installed::

    python tools/pa_z_scheduling.py

Angles supplied to the analytical helpers are in degrees; rates are in deg/day.
These helpers describe a fixed target and a Sun moving in the ecliptic plane.
The scheduling example uses a 14-day window centered on quadrature (plus/minus
7 days), with a plus/minus 15-degree roll allowance and no additional reserve.
"""

import argparse
import csv
from importlib.metadata import version
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


YEAR_DAYS = 365.2422
MEAN_RATE = 360.0 / YEAR_DAYS
MIN_SUN_DEG = 54.0
ROLL_DEG = 15.0
WINDOW_DAYS = 14.0
ROOT = Path(__file__).resolve().parents[1]


def pa_ecliptic_deg(beta_deg, phase_deg):
    """Return wrapped nominal PA from ecliptic north, with phase = Sun lon - target lon."""
    beta, phase = np.deg2rad(beta_deg), np.deg2rad(phase_deg)
    east, north = np.broadcast_arrays(np.sin(phase), -np.sin(beta) * np.cos(phase))
    valid = np.hypot(east, north) > 1e-14
    return np.where(valid, np.rad2deg(np.arctan2(east, north)), np.nan)


def pa_rate_deg_day(beta_deg, phase_deg, sun_rate_deg_day=MEAN_RATE):
    """Return signed nominal PA_Z rate; the same rate applies to EXCAM +Y."""
    beta, phase = np.deg2rad(beta_deg), np.deg2rad(phase_deg)
    # This form avoids cancellation near the ecliptic at conjunction/opposition.
    denominator = np.sin(phase) ** 2 + np.sin(beta) ** 2 * np.cos(phase) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = -sun_rate_deg_day * np.sin(beta) / denominator
    return np.where(denominator > 1e-28, rate, np.nan)


def observable(beta_deg, phase_deg):
    """Apply the repository's symmetric 54--126 degree Sun-angle limits."""
    cosine = np.cos(np.deg2rad(beta_deg)) * np.cos(np.deg2rad(phase_deg))
    return np.abs(cosine) <= np.cos(np.deg2rad(MIN_SUN_DEG)) + 1e-14


def rate_envelope(beta_deg):
    """Return minimum and supremum observable rate magnitudes for circular motion."""
    sine = np.abs(np.sin(np.deg2rad(beta_deg)))
    lower = MEAN_RATE * sine
    denominator = np.maximum(np.sin(np.deg2rad(MIN_SUN_DEG)) ** 2, sine**2)
    return lower, MEAN_RATE * sine / denominator


def quadrature_drift_deg(beta_deg, width_days):
    """Return exact PA span for a window centered on quadrature, shorter than half a year.

    Solar visibility must be checked separately. The plotted 14-day window
    satisfies the Sun-angle limits at every latitude.
    """
    width_days = np.asarray(width_days)
    if np.any((width_days < 0) | (width_days >= YEAR_DAYS / 2)):
        raise ValueError("Window width must lie in [0, half a year).")
    half_phase = np.deg2rad(MEAN_RATE * width_days / 2)
    return 2 * np.rad2deg(
        np.arctan(np.abs(np.sin(np.deg2rad(beta_deg))) * np.tan(half_phase))
    )


def angle_difference_deg(a, b):
    """Return the signed circular difference a - b in [-180, 180) degrees."""
    return (np.asarray(a) - np.asarray(b) + 180) % 360 - 180


def save_figure(fig, output, name, formats=("png", "pdf")):
    """Save a figure in raster and vector formats."""
    for extension in formats:
        fig.savefig(output / f"{name}.{extension}", dpi=190, bbox_inches="tight")
    plt.close(fig)


def make_projection_cartoon(output):
    """Explain the Sun-angle factor using a side view and equal-area sky projections."""
    from matplotlib.patches import Arc, Circle, Polygon, Rectangle

    output.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(14.4, 8.2), facecolor="white")
    dark, gray = "#24364B", "#8793A0"
    blue, orange, gold = "#0072B2", "#D55E00", "#E6A000"
    centers = [0.175, 0.50, 0.825]
    axes = [fig.add_axes([left, 0.38, 0.285, 0.40])
            for left in [0.0325, 0.3575, 0.6825]]
    fig.text(0.5, 0.965, "Why does Sun angle change the image-orientation rate?",
             ha="center", va="top", fontsize=21, weight="bold", color=dark)
    fig.text(0.5, 0.907, "The same target at two dates: ecliptic latitude β = 30°",
             ha="center", fontsize=15, color=dark)
    headings = ["1  Project the Sun direction", "2  At quadrature", "3  Away from quadrature"]
    subtitles = ["View from the side", "Sky plane, looking toward the target",
                 "Same target, same projected ellipse"]
    for ax, x, heading, subtitle in zip(axes, centers, headings, subtitles):
        ax.set_aspect("equal")
        ax.axis("off")
        fig.text(x, 0.849, heading, ha="center", fontsize=14, weight="bold", color=dark)
        fig.text(x, 0.817, subtitle, ha="center", fontsize=11, color=gray)

    # Side view: n is vertical; the sky plane is horizontal and seen edge-on.
    ax = axes[0]
    ax.set(xlim=(-0.65, 3.95), ylim=(-0.90, 3.00))
    beta = 30.0
    scale = 2.0  # Drawing scale for a unit Sun vector.
    epsilon = np.deg2rad(60)
    tip = scale * np.array([np.sin(epsilon), np.cos(epsilon)])

    # Project a true circle in the ecliptic plane into this side view. The
    # observatory lies on the circle at the origin and the Sun is its center.
    # The projected circle appears elliptical because beta is nonzero.
    beta_rad = np.deg2rad(beta)
    phase = np.arccos(np.cos(epsilon) / np.cos(beta_rad))
    sightline = np.array([np.cos(beta_rad), 0, np.sin(beta_rad)])
    sun_direction = np.array([np.cos(phase), np.sin(phase), 0])
    sun_tangent = np.array([-np.sin(phase), np.cos(phase), 0])
    sky_horizontal = (sun_direction - np.cos(epsilon) * sightline) / np.sin(epsilon)
    screen_basis = np.vstack([sky_horizontal, sightline])
    radial_screen = screen_basis @ sun_direction
    tangent_screen = screen_basis @ sun_tangent

    def orbit_screen(angle):
        """Project the circular heliocentric orbit; angle zero is Roman's position."""
        return tip + scale * (-np.cos(angle)[..., None] * radial_screen
                              - np.sin(angle)[..., None] * tangent_screen)

    orbit = orbit_screen(np.linspace(0, 2 * np.pi, 361))
    orbit_color = "#9AAABE"
    ax.plot(orbit[:, 0], orbit[:, 1], color=orbit_color, lw=1.6,
             ls=(0, (5, 3)), zorder=-1)
    ax.annotate("", orbit_screen(np.array(3.10)), orbit_screen(np.array(2.90)),
                 arrowprops={"arrowstyle": "-|>", "lw": 1.8, "color": orbit_color})
    ax.annotate("Circular orbit\n(projected view)",
                 orbit_screen(np.array(3.75)), xytext=(2.75, 1.80),
                 ha="center", va="center", fontsize=9, color="#6A7D93",
                 arrowprops={"arrowstyle": "-", "lw": 0.9, "color": orbit_color})
    ax.add_patch(Rectangle((-0.60, -0.055), 2.80, 0.11,
                            facecolor="#E9EEF3", edgecolor="none", zorder=0))
    ax.plot([-0.60, 2.20], [0, 0], color=gray, lw=1.2, zorder=1)
    ax.annotate("", (0, 2.31), (0, 0),
                 arrowprops={"arrowstyle": "-|>", "lw": 2, "color": dark})
    ax.scatter([0], [2.48], marker="*", s=260, color=gold, edgecolor=dark, lw=0.7)
    ax.text(0.20, 2.60, "Target", fontsize=12, color=dark, va="center")
    ax.text(-0.28, 1.35, "line of sight", fontsize=10, color=dark,
             rotation=90, ha="right", va="center")
    ax.annotate("", tip, (0, 0),
                 arrowprops={"arrowstyle": "-|>", "lw": 2.3, "color": gold})
    ax.plot([tip[0], tip[0]], [0, tip[1]], ls="--", lw=1.5, color=gray)
    ax.plot([tip[0] - 0.13, tip[0] - 0.13, tip[0]], [0, 0.13, 0.13],
             color=gray, lw=1)
    ax.annotate("", (tip[0], 0), (0, 0),
                 arrowprops={"arrowstyle": "-|>", "lw": 3, "color": blue})
    ax.add_patch(Arc((0, 0), 1.10, 1.10, theta1=30, theta2=90, color=dark, lw=1.5))
    ax.text(0.30, 0.59, r"$\epsilon$", fontsize=19, color=dark)
    ax.text(0.95, 0.89, r"unit Sun vector $\hat{s}$", fontsize=10, color=gold,
             rotation=30, ha="center")
    ax.add_patch(Circle(tip, 0.12, facecolor="#FFD36A", edgecolor=gold, lw=1.3, zorder=5))
    for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        direction = np.array([np.cos(angle), np.sin(angle)])
        ends = tip[:, None] + direction[:, None] * np.array([0.16, 0.21])
        ax.plot(ends[0], ends[1], color=gold, lw=1.2)
    ax.text(tip[0], tip[1] + 0.30, "Sun", ha="center", fontsize=12, color=dark)
    # Simple observatory icon at the common origin of all direction vectors.
    ax.add_patch(Rectangle((-0.10, -0.30), 0.20, 0.25,
                            facecolor=dark, edgecolor="white", lw=0.7, zorder=6))
    for x in [-0.37, 0.15]:
        ax.add_patch(Rectangle((x, -0.25), 0.22, 0.13,
                                facecolor="#547C9D", edgecolor="white", lw=0.7, zorder=6))
    ax.text(-0.25, 0.15, "Roman", fontsize=10, ha="right", color=dark)
    ax.text(1.25, -0.22, r"$p\ \parallel$ nominal +Z", fontsize=12, color=blue, ha="center")
    ax.text(1.40, -0.43, "sky plane (edge-on)", fontsize=9.5, color=gray, ha="center")

    # The projected unit-Sun circle is an ellipse with semiaxes 1 and |sin beta|.
    # Equal increments of solar longitude sweep equal areas on this ellipse.
    sine_beta = np.sin(np.deg2rad(beta))
    phases = np.linspace(0, 360, 721)

    def projected_sun(phase_deg):
        """Return east/north components of the unnormalized Sun projection."""
        phase_rad = np.deg2rad(phase_deg)
        return np.column_stack([np.sin(phase_rad), -sine_beta * np.cos(phase_rad)])

    ellipse = projected_sun(phases)
    for ax, x, sun_angle, color in zip(axes[1:], centers[1:], [90, 60], [blue, orange]):
        ax.set(xlim=(-1.35, 1.45), ylim=(-1.08, 1.12))
        ax.plot([-1.12, 1.12], [0, 0], color="#DCE2E8", lw=1, zorder=0)
        ax.plot([0, 0], [-0.70, 0.70], color="#DCE2E8", lw=1, zorder=0)
        ax.plot(ellipse[:, 0], ellipse[:, 1], color=gray, lw=1.8)
        ax.text(0, 0.67, "annual projected Sun path", ha="center", fontsize=10, color=gray)
        center_phase = np.rad2deg(np.arccos(
            np.cos(np.deg2rad(sun_angle)) / np.cos(np.deg2rad(beta))))
        arc_phase = center_phase + np.linspace(-1, 1, 181) * MEAN_RATE * WINDOW_DAYS / 2
        arc = projected_sun(arc_phase)
        sector = np.vstack([[0, 0], arc, [0, 0]])
        ax.add_patch(Polygon(sector, facecolor=color, alpha=0.25, edgecolor="none"))
        ax.plot(arc[:, 0], arc[:, 1], color=color, lw=3.5)
        for endpoint in [arc[0], arc[-1]]:
            ax.plot([0, endpoint[0]], [0, endpoint[1]], color=color, lw=1.2)
            ax.scatter(*endpoint, color=color, s=23, zorder=4)
        p = projected_sun([center_phase])[0]
        ax.annotate("", p, (0, 0),
                     arrowprops={"arrowstyle": "-|>", "color": color, "lw": 2.8})
        r = np.linalg.norm(p)
        label_position = 0.52 * p + 0.16 * np.array([p[1], -p[0]]) / r
        ax.text(*label_position, r"$p$", color=color, fontsize=16, ha="center", va="center")
        ax.scatter([0], [0], s=32, color=dark, zorder=5)
        ax.text(-0.10, -0.14, "boresight", fontsize=9, ha="right", color=dark)
        angle_start, angle_end = np.rad2deg(np.arctan2(arc[[0, -1], 1], arc[[0, -1], 0]))
        ax.add_patch(Arc((0, 0), 0.65, 0.65, theta1=angle_start, theta2=angle_end,
                         color=color, lw=2.2))
        if sun_angle == 90:
            labels = [("−7 d", (1.08, -0.25), arc[0]), ("+7 d", (1.08, 0.24), arc[-1])]
            ax.annotate("", (-1.15, 0.50), (-1.15, 0),
                         arrowprops={"arrowstyle": "<->", "color": gray, "lw": 1})
            ax.text(-1.22, 0.25, r"$|\sin\beta|$", fontsize=10, color=gray,
                     rotation=90, ha="right", va="center")
        else:
            labels = [("−7 d", (1.00, -0.62), arc[0]), ("+7 d", (1.08, -0.10), arc[-1])]
        for label, position, point in labels:
            ax.annotate(label, point, xytext=position, fontsize=10, color=color,
                         arrowprops={"arrowstyle": "-", "color": color, "lw": 0.8})
        drift = abs(angle_difference_deg(pa_ecliptic_deg(beta, arc_phase[-1]),
                                          pa_ecliptic_deg(beta, arc_phase[0])))
        rate = abs(pa_rate_deg_day(beta, center_phase))
        fig.text(x, 0.336, rf"Midpoint: $\epsilon={sun_angle}^\circ$, $r={r:.3f}$",
                 ha="center", fontsize=13, color=dark)
        fig.text(x, 0.280, f"{rate:.3f}°/day", ha="center", fontsize=23, weight="bold", color=color)
        fig.text(x, 0.239, f"PA sweep in 14 days: {drift:.2f}°",
                 ha="center", fontsize=12, color=dark)

    fig.text(centers[0], 0.336, r"Projection length: $r=|p|=\sin\epsilon$",
             ha="center", fontsize=13, color=blue)
    fig.text(centers[0], 0.285, "Nominal +Z follows its direction.", ha="center", fontsize=12, color=dark)
    fig.text(centers[0], 0.239, "The projection is longest at ε = 90°.", ha="center", fontsize=11, color=dark)
    fig.text(0.5, 0.177, "Equal time steps sweep equal shaded areas: a shorter radius needs a wider PA sweep.",
             ha="center", fontsize=13, color=dark)
    fig.text(0.5, 0.101,
             r"$r^2\,|\dot{\mathrm{PA}}_Z|=|\sin\beta|\,\dot{\lambda}_\odot"
             r"\qquad\Longrightarrow\qquad"
             r"|\dot{\mathrm{PA}}_Z|=\dot{\lambda}_\odot\,\frac{|\sin\beta|}{\sin^2\epsilon}$",
             ha="center", fontsize=21, color=dark)
    fig.text(0.5, 0.036,
             "Latitude sets the ellipse's flattening. Sun angle sets the projected radius.\n"
             "Rates at midpoint; circular solar motion; nominal roll. Both 14-day examples meet the 54°–126° Sun-angle limits.",
             ha="center", fontsize=10.5, color=gray, linespacing=1.6)
    save_figure(fig, output, "pa_z_projection_cartoon", formats=("png", "pdf", "svg"))


def make_figures(output):
    """Plot seasonal rates, their latitude envelope, and exact window margins."""
    plt.rcParams.update(
        {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
         "axes.titleweight": "bold", "figure.facecolor": "white"}
    )
    days = np.linspace(0, YEAR_DAYS, 1462)
    phase = 90 + MEAN_RATE * days
    latitudes = [0, 10, 30, 54, 75, 90]
    colors = ["#555555", "#0072B2", "#009E73", "#D55E00", "#CC79A7", "#6B4C9A"]
    fig, axes = plt.subplots(2, 1, figsize=(10.4, 8.8), sharex=True,
                             layout="constrained")
    fig.suptitle("Seasonal image-orientation drift on Roman / EXCAM", fontsize=17)
    for beta, color in zip(latitudes, colors):
        rate = np.abs(pa_rate_deg_day(beta, phase))
        axes[0].plot(days, np.where(observable(beta, phase), rate, np.nan),
                     color=color, lw=2, label=rf"$|\beta|={beta}^\circ$")
    axes[0].set(ylabel="|dPA_Z / dt| (deg/day)", ylim=(-0.035, 1.31))
    axes[0].legend(ncol=6, loc="lower center", bbox_to_anchor=(0.5, 1.01), fontsize=10)
    axes[0].grid(alpha=0.18)
    beta_grid = np.linspace(-90, 90, 721)[:, None]
    rates = pa_rate_deg_day(beta_grid, phase[None, :])
    rates = np.ma.masked_where(~observable(beta_grid, phase[None, :]), rates)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#e4e4e4")
    limit = MEAN_RATE / np.sin(np.deg2rad(MIN_SUN_DEG))
    mesh = axes[1].pcolormesh(days, beta_grid[:, 0], rates, cmap=cmap,
                              vmin=-limit, vmax=limit, shading="auto", rasterized=True)
    axes[1].set(title="Signed rate: opposite directions in the two hemispheres",
                xlabel=r"Days from quadrature ($\lambda_\odot-\lambda_\star=90^\circ$ at day 0)",
                ylabel=r"Ecliptic latitude $\beta$ (deg)",
                yticks=[-90, -60, -30, 0, 30, 60, 90], xlim=(0, YEAR_DAYS),
                xticks=np.arange(0, 361, 60))
    fig.colorbar(mesh, ax=axes[1], label="dPA_Z / dt (deg/day)", pad=0.02)
    fig.supxlabel("Circular year: 365.2422 days. Gray / line gaps: Sun angle outside 54°–126°.\n"
                  "Target longitude shifts the calendar origin; EXCAM +Y has the same rate.",
                  fontsize=10)
    save_figure(fig, output, "pa_z_seasonal_rate")

    doy = np.linspace(1, 365, 1461)
    sun_longitude = MEAN_RATE * (doy - 80)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, sharey=True,
                             layout="constrained")
    fig.suptitle("Day-of-year dependence: longitude sets the seasonal phase", fontsize=16)
    for ax, target_longitude in zip(axes, [0, 90]):
        calendar_phase = sun_longitude - target_longitude
        for beta, color in zip(latitudes, colors):
            rate = np.abs(pa_rate_deg_day(beta, calendar_phase))
            ax.plot(doy, np.where(observable(beta, calendar_phase), rate, np.nan),
                    color=color, lw=2, label=rf"$|\beta|={beta}^\circ$")
        ax.text(0.015, 0.93, rf"Target longitude $\lambda_\star={target_longitude}^\circ$",
                transform=ax.transAxes, va="top", fontsize=11)
        ax.set(ylabel="|dPA_Z / dt| (deg/day)", ylim=(-0.035, 1.42), xlim=(1, 365))
        ax.grid(alpha=0.18)
    axes[0].legend(ncol=6, loc="lower center", bbox_to_anchor=(0.5, 1.01), fontsize=10)
    axes[1].set(xlabel="Day of year", xticks=[1, 60, 120, 180, 240, 300, 365])
    fig.supxlabel("Illustrative circular year: solar longitude is 0° on day 80 and increases by 0.98565°/day.\n"
                  "Line gaps: Sun angle outside 54°–126°. Calendar dates are approximate.", fontsize=10)
    save_figure(fig, output, "pa_z_day_of_year")

    beta = np.linspace(0, 90, 901)
    lower, upper = rate_envelope(beta)
    fig, ax = plt.subplots(figsize=(9.6, 5.8), layout="constrained")
    ax.fill_between(beta, lower, upper, color="#0072B2", alpha=0.15,
                    label="Range over observable dates")
    ax.plot(beta, upper, color="#0072B2", lw=2.5, label="Maximum observable rate")
    ax.plot(beta, lower, color="#009E73", lw=2.5, label="Quadrature rate (minimum)")
    rate_budget = 2 * ROLL_DEG / WINDOW_DAYS
    ax.axhline(rate_budget, color="#666666", ls="--", lw=1.3,
               label=f"{2 * ROLL_DEG:g}° / {WINDOW_DAYS:g} days (two-week rate budget)")
    ax.scatter([54], [limit], color="#0072B2", zorder=4)
    ax.annotate(f"Peak at |β| = 54°\n{limit:.3f}°/day", (54, limit),
                 xytext=(66, 1.27), arrowprops={"arrowstyle": "->", "color": "#444444"})
    ax.set(title="Fastest usable drift occurs near |β| = 54°",
           xlabel=r"Absolute ecliptic latitude $|\beta|$ (deg)",
           ylabel="|dPA_Z / dt| (deg/day)", xlim=(0, 90),
           ylim=(0, max(1.4, 1.12 * rate_budget)),
           xticks=[0, 15, 30, 45, 54, 60, 75, 90])
    ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.84), fontsize=10)
    ax.grid(alpha=0.18)
    fig.supxlabel("Circular orbit; only the 54°–126° solar-angle constraint is applied.\n"
                  "The rate budget is a sufficient bound for a fixed PA, with optimal centering and no roll reserve.",
                  fontsize=10)
    save_figure(fig, output, "pa_z_rate_envelope")

    beta = np.linspace(0, 90, 361)
    drift = quadrature_drift_deg(beta, WINDOW_DAYS)
    halfwidth = ROLL_DEG - drift / 2
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1]})
    axes[0].plot(beta, drift, color="#D55E00", lw=2.5, label="PA drift D")
    axes[0].plot(beta, halfwidth, color="#0072B2", lw=2.5, label="Common half-width H")
    axes[0].scatter([90, 90], [drift[-1], halfwidth[-1]],
                    color=["#D55E00", "#0072B2"], zorder=4)
    axes[0].annotate(f"Maximum drift: {drift[-1]:.2f}°", (90, drift[-1]),
                     xytext=(45, 15.5), color="#D55E00",
                     arrowprops={"arrowstyle": "->", "color": "#D55E00"})
    axes[0].annotate(f"Minimum half-width: ±{halfwidth[-1]:.2f}°", (90, halfwidth[-1]),
                     xytext=(36, 5.2), color="#0072B2",
                     arrowprops={"arrowstyle": "->", "color": "#0072B2"})
    axes[0].set(xlabel=r"Absolute ecliptic latitude $|\beta|$ (deg)",
                ylabel="PA drift / common half-width (deg)",
                xlim=(0, 93), ylim=(0, 17), xticks=[0, 15, 30, 45, 60, 75, 90])
    axes[0].legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.01), fontsize=10)
    axes[0].grid(alpha=0.18)

    half_days = WINDOW_DAYS / 2
    offsets = np.linspace(-half_days, half_days, 421)
    example_beta = 60
    seasonal = -np.rad2deg(np.arctan(np.sin(np.deg2rad(example_beta))
                                     * np.tan(np.deg2rad(MEAN_RATE * offsets))))
    common_half = ROLL_DEG - quadrature_drift_deg(example_beta, WINDOW_DAYS) / 2
    axes[1].fill_between(offsets, seasonal - ROLL_DEG, seasonal + ROLL_DEG,
                         color="#0072B2", alpha=0.12, label="Reachable with ±15° roll")
    axes[1].plot(offsets, seasonal, color="#0072B2", lw=2, label="Nominal PA")
    axes[1].plot(offsets, seasonal - ROLL_DEG, color="#0072B2", lw=0.8)
    axes[1].plot(offsets, seasonal + ROLL_DEG, color="#0072B2", lw=0.8)
    axes[1].axhspan(-common_half, common_half, color="#D55E00", alpha=0.18,
                    label=f"Common PA band: ±{common_half:.2f}°")
    axes[1].axhline(common_half, color="#D55E00", ls="--", lw=1)
    axes[1].axhline(-common_half, color="#D55E00", ls="--", lw=1)
    axes[1].axhline(0, color="#555555", ls=":", lw=1)
    axes[1].set(title=rf"Two-week example: $\beta=+{example_beta}^\circ$",
                xlabel="Days from the window midpoint (quadrature)",
                ylabel="PA_Z − nominal PA_Z at midpoint (deg)",
                xlim=(-half_days, half_days), ylim=(-25, 25),
                xticks=[-half_days, -4, 0, 4, half_days])
    axes[1].legend(loc="lower left", fontsize=9)
    axes[1].grid(alpha=0.15)
    fig.suptitle("Two-week scheduling margin at quadrature", fontsize=17)
    fig.supxlabel("Exact circular-orbit drift; all plotted dates meet the 54°–126° Sun-angle limits.\n"
                  f"Full width W = {WINDOW_DAYS:g} days means ±{half_days:g} days. "
                  "Roll allowance: ±15°. No additional reserve.", fontsize=10)
    save_figure(fig, output, "pa_z_scheduling_margin")
    make_projection_cartoon(output)


def write_table(output):
    """Write numerical rates and example margins used in the analysis note."""
    rows = []
    for beta in [0, 10, 20, 30, 45, 54, 60, 75, 90]:
        lower, upper = rate_envelope(beta)
        drift = quadrature_drift_deg(beta, WINDOW_DAYS)
        rows.append({
            "abs_beta_deg": beta,
            "window_days": WINDOW_DAYS,
            "min_rate_deg_day": float(lower),
            "max_observable_rate_deg_day": float(upper),
            "quadrature_drift_deg": float(drift),
            "common_pa_exists": bool(drift <= 2 * ROLL_DEG),
            "common_halfwidth_deg": float(max(0, ROLL_DEG - drift / 2)),
            "sufficient_full_window_days_from_max_rate": float(30 / upper) if upper else "inf",
        })
    with (output / "pa_z_scheduling_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def validate_against_attitude_code():
    """Check the formula against the repository's rotations and built-in ephemerides."""
    import astropy
    import astropy.units as u
    from astropy.coordinates import BarycentricMeanEcliptic, SkyCoord, solar_system_ephemeris
    from astropy.time import Time
    from astropy.utils import iers

    sys.path.insert(0, str(ROOT / "src"))
    from roman_pointing.roman_pointing import (
        applyRollAngle, calcRomanAngles, getEXCAMPositionAngle,
        getL2Positions, getRomanPositionAngle, getSunPositions,
    )

    iers.conf.auto_download = False
    lon = 37.0
    phases = np.linspace(0, 360, 91, endpoint=False) + 0.37
    times = Time("2026-01-01") + phases / MEAN_RATE * u.day
    north = SkyCoord(ra=0 * u.deg, dec=90 * u.deg, frame="icrs").transform_to(
        BarycentricMeanEcliptic()).cartesian.xyz.value
    errors = {"projection_max_abs": 0.0, "pa_max_abs_deg": 0.0,
              "rate_max_abs_deg_day": 0.0, "roll_max_abs_deg": 0.0,
              "excam_offset_max_abs_deg": 0.0}

    def fixed_target(beta):
        """Construct a distant stationary target, suppressing parallax in the check."""
        return SkyCoord(lon=lon * u.deg, lat=beta * u.deg, distance=1e6 * u.pc,
                        pm_lon_coslat=0 * u.mas / u.yr, pm_lat=0 * u.mas / u.yr,
                        radial_velocity=0 * u.km / u.s, obstime="J2000",
                        frame=BarycentricMeanEcliptic())

    def circular_body(target, phase):
        """Evaluate the actual attitude solver for a synthetic planar Sun."""
        sun_lon = np.deg2rad(lon + phase)
        sun = np.array([np.cos(sun_lon), np.sin(sun_lon), np.zeros_like(sun_lon)])
        return calcRomanAngles(target, times, np.zeros_like(sun) * u.au, sun * u.au)[3]

    count = 0
    for beta in [-90, -75, -54, -30, -10, 0, 10, 30, 54, 75, 90]:
        target = fixed_target(beta)
        body = circular_body(target, phases)
        n = target.cartesian.xyz.value
        n = n / np.linalg.norm(n)
        sun_lon = np.deg2rad(lon + phases)
        s = np.array([np.cos(sun_lon), np.sin(sun_lon), np.zeros_like(sun_lon)])
        projected = s - n[:, None] * (n @ s)
        projected /= np.linalg.norm(projected, axis=0)
        errors["projection_max_abs"] = max(errors["projection_max_abs"],
                                            float(np.max(np.abs(body[2] - projected))))
        equatorial_north = north - n * (n @ north)
        equatorial_north /= np.linalg.norm(equatorial_north)
        east = np.cross(equatorial_north, n)
        expected_pa = np.rad2deg(np.arctan2(east @ projected, equatorial_north @ projected))
        pa = getRomanPositionAngle(body).to_value(u.deg)
        errors["pa_max_abs_deg"] = max(errors["pa_max_abs_deg"],
                                       float(np.max(np.abs(angle_difference_deg(pa, expected_pa)))))
        h = 0.001
        before = getRomanPositionAngle(circular_body(target, phases - MEAN_RATE * h)).to_value(u.deg)
        after = getRomanPositionAngle(circular_body(target, phases + MEAN_RATE * h)).to_value(u.deg)
        fd_rate = angle_difference_deg(after, before) / (2 * h)
        keep = observable(beta, phases)
        errors["rate_max_abs_deg_day"] = max(errors["rate_max_abs_deg_day"],
            float(np.max(np.abs(fd_rate[keep] - pa_rate_deg_day(beta, phases[keep])))))
        rolled = getRomanPositionAngle(applyRollAngle(body, np.full(len(times), 7) * u.deg)).to_value(u.deg)
        errors["roll_max_abs_deg"] = max(errors["roll_max_abs_deg"],
            float(np.max(np.abs(angle_difference_deg(rolled, pa) + 7))))
        excam = getEXCAMPositionAngle(body).to_value(u.deg)
        errors["excam_offset_max_abs_deg"] = max(errors["excam_offset_max_abs_deg"],
            float(np.max(np.abs(angle_difference_deg(excam, pa) - 150))))
        count += np.count_nonzero(keep)

    assert errors["projection_max_abs"] < 1e-11, errors
    assert errors["pa_max_abs_deg"] < 1e-9, errors
    assert errors["rate_max_abs_deg_day"] < 1e-7, errors
    assert errors["roll_max_abs_deg"] < 1e-9, errors
    assert errors["excam_offset_max_abs_deg"] < 1e-9, errors

    # Independent integration check for the finite-window formula.
    from scipy.integrate import quad
    integration_error = 0.0
    for beta in [0, 10, 30, 54, 90]:
        for width in [14, 28, 42, 56, 70]:
            integrated = quad(lambda day: abs(pa_rate_deg_day(beta, 90 + MEAN_RATE * day)),
                              -width / 2, width / 2, epsabs=1e-10)[0]
            integration_error = max(integration_error,
                                    abs(integrated - quadrature_drift_deg(beta, width)))
    assert integration_error < 1e-9, integration_error

    # Compare actual 2026 ephemerides to both the planar and general vector formulas.
    # Distant stationary targets isolate the Sun-motion terms in the derivation.
    dates = Time("2026-01-01") + np.arange(0, 365, 5) * u.day
    h = 0.01
    all_dates = dates[:, None] + np.array([-h, 0, h])[None, :] * u.day
    flat_dates = all_dates.ravel()
    with solar_system_ephemeris.set("builtin"):
        obs = getL2Positions(flat_dates)
        suns = getSunPositions(flat_dates)
        sun = (suns - obs).to_value(u.au).reshape(3, -1, 3)
        sun /= np.linalg.norm(sun, axis=0)
        sun_dot = (sun[:, :, 2] - sun[:, :, 0]) / (2 * h)
        sun_mid = sun[:, :, 1]
        longitude = np.rad2deg(np.arctan2(sun[1], sun[0]))
        longitude_rate = angle_difference_deg(longitude[:, 2], longitude[:, 0]) / (2 * h)
        vector_error, planar_error, circular_error = 0.0, 0.0, 0.0
        for beta in [-75, -30, 0, 30, 75]:
            target = fixed_target(beta)
            sun_angle, _, _, body = calcRomanAngles(target, flat_dates, obs, suns)
            pa = getRomanPositionAngle(body).to_value(u.deg).reshape(-1, 3)
            measured = angle_difference_deg(pa[:, 2], pa[:, 0]) / (2 * h)
            n = target.cartesian.xyz.value
            n /= np.linalg.norm(n)
            vector_rate = np.rad2deg(-(n @ np.cross(sun_mid.T, sun_dot.T).T)
                                    / (1 - (n @ sun_mid) ** 2))
            actual_phase = longitude[:, 1] - lon
            planar = pa_rate_deg_day(beta, actual_phase, longitude_rate)
            circular = pa_rate_deg_day(beta, actual_phase)
            angles = sun_angle.to_value(u.deg).reshape(-1, 3)[:, 1]
            keep = (angles >= 54) & (angles <= 126)
            vector_error = max(vector_error, float(np.max(np.abs(measured[keep] - vector_rate[keep]))))
            planar_error = max(planar_error, float(np.max(np.abs(measured[keep] - planar[keep]))))
            circular_error = max(circular_error, float(np.max(np.abs(measured[keep] - circular[keep]))))
    assert vector_error < 1e-6, vector_error
    return {"numpy_version": np.__version__, "astropy_version": astropy.__version__,
            "angutils_version": version("angutils"),
            "example_window_days": WINDOW_DAYS, "roll_allowance_deg": ROLL_DEG,
            "example_window_center": "quadrature",
            "synthetic_observable_samples": int(count), **errors,
            "quadrature_integral_max_abs_deg": float(integration_error),
            "ephemeris_year": 2026, "ephemeris": "astropy builtin; repository L2 approximation",
            "ephemeris_sun_rate_range_deg_day": [float(longitude_rate.min()), float(longitude_rate.max())],
            "ephemeris_vector_rate_max_abs_error_deg_day": vector_error,
            "ephemeris_planar_rate_max_abs_error_deg_day": planar_error,
            "ephemeris_mean_speed_rate_max_abs_error_deg_day": circular_error}


def main():
    """Generate the figures, numerical table, and validation record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs" / "figures" / "pa_z")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Only make analytical figures; do not import the attitude solver.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_validation:
        validation = validate_against_attitude_code()
        (args.output_dir / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
        print(json.dumps(validation, indent=2))
    make_figures(args.output_dir)
    write_table(args.output_dir)
    print(f"Wrote five PNG/PDF figure pairs, an SVG cartoon, and the numerical table to {args.output_dir}")


if __name__ == "__main__":
    main()
