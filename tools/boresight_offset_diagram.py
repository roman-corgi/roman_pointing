"""
Diagram of how placing the CGI aperture (instead of V1) on the target tilts V3 at the target by
dPA ~ V2Ref / tan(eps), shown on the sky around the target. Figure for docs/attitude_model/README.md.

Usage:
  python tools/boresight_offset_diagram.py -o docs/attitude_model

"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

# chart chrome and palette (same as field_rotation_rate.py)
INK = "#0b0b0b"
INK2 = "#52514e"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
PURPLE = "#6250d6"  # Roman observatory (V1)
BLUE_LIGHT = "#cde2fb"
GREEN_LIGHT = "#d6f1e5"

# roman_siaf_inflight_20261021.xml (arcsec): CGI_CEN reference point, WFI_CEN reference point, and the
# convex hull of the corners of the 18 WFIxx_FULL detectors (V2, V3)
V2REF, V3REF = -1104.214, -840.945
WFI_CEN = np.array([1686.548, -1014.664])
WFI_OUTLINE = np.array([(1811, 579), (373, -1925), (1239, -2427), (1614, -2631), (2062, -2325), (2389, -1954),
                        (2611, -1572), (2851, -1149), (3007, -680), (3045, -139), (2681, 84)], dtype=float)


def plot_boresight_offset_diagram(outdir, eps=60.0, scale=10.0):
    """Sky-view diagram of the V3 tilt caused by the EXCAM aperture offset.

    Args:
        outdir (str):
            Output directory
        eps (float):
            Sun angle of the target (deg) used for the sketch
        scale (float):
            Exaggeration factor applied to the aperture offset
    """
    d2, d3 = np.radians(np.array([V2REF, V3REF]) / 3600) * scale
    fig, ax1 = plt.subplots(figsize=(8.6, 8.0))
    fig.patch.set_facecolor(SURFACE)
    ax1.set_aspect("equal")
    ax1.axis("off")
    ax1.set_facecolor(SURFACE)
    arrow = lambda ax, p, q, c, lw=2.0, style="-|>": ax.annotate(
        "", xy=q, xytext=p, arrowprops=dict(arrowstyle=style, color=c, lw=lw, shrinkA=0, shrinkB=0))

    # sky at the target, gnomonic projection centred on T (great circles are straight lines), drawn in the
    # body frame of the final attitude: x to the right is -V2, y up is +V3. Focal-plane offsets are scaled
    # by `scale`; layout lengths are in units of the scaled CGI offset r. V3 at V1 points along V1's great
    # circle to the Sun, so the Sun lies straight above V1; it is drawn closer than tan(eps) so that the
    # triangle T-N-Sun is visible.
    k = np.radians(1 / 3600) * scale  # plot units per arcsec
    r = np.hypot(d2, d3)
    T = np.zeros(2)
    V1n = np.array([V2REF, -V3REF]) * k  # V1 relative to the aperture (on the target): -V2Ref, -V3Ref
    N = np.array([V1n[0], 0.0])  # where the V2 and V3 parts of the offset meet
    sun = np.array([V1n[0], 4.2 * r])
    gT = (sun - T) / np.linalg.norm(sun - T)  # the target's great circle to the Sun (V3 with V1 on target)
    y_brk = sun[1] - 1.3 * r  # break in the lines to the Sun (distance shortened)

    # WFI imaging footprint, placed by its SIAF offsets from V1
    wfi = V1n + np.c_[-WFI_OUTLINE[:, 0], WFI_OUTLINE[:, 1]] * k
    ax1.fill(wfi[:, 0], wfi[:, 1], facecolor=GREEN_LIGHT, edgecolor=SERIES[2], lw=1.5, zorder=0)
    wc = V1n + np.array([-WFI_CEN[0], WFI_CEN[1]]) * k
    ax1.plot(*wc, "+", ms=9, mew=1.5, color=SERIES[2])
    ax1.text(wc[0], wc[1] - 0.12 * r, "WFI imaging\nfootprint", color=SERIES[2], fontsize=9.5,
             ha="center", va="top")

    # right triangle T-N-Sun, right angle at N
    ax1.fill([T[0], N[0], sun[0]], [T[1], N[1], sun[1]], color=BLUE_LIGHT, alpha=0.6, lw=0, zorder=0)
    sq = 0.15 * r
    ax1.plot([N[0] + sq, N[0] + sq, N[0]], [0, sq, sq], color=INK, lw=1, zorder=2)
    ax1.text(N[0] - 0.06 * r, -0.06 * r, "N", color=INK, fontsize=10, ha="right", va="top")

    # great circles through the Sun: V1's (vertical) and the target's
    for p0 in (T, V1n):
        g = (sun - p0) / np.linalg.norm(sun - p0)
        lo = p0 - (p0[1] + 1.6 * r) / g[1] * g
        b0 = p0 + (y_brk - 0.1 * r - p0[1]) / g[1] * g
        b1 = p0 + (y_brk + 0.1 * r - p0[1]) / g[1] * g
        ax1.plot(*np.c_[lo, b0], color=SERIES[3], lw=1, ls="--", zorder=1)
        ax1.plot(*np.c_[b1, sun], color=SERIES[3], lw=1, ls="--", zorder=1)
        for b in (b0, b1):
            ax1.plot([b[0] - 0.08 * r, b[0] + 0.08 * r], [b[1] - 0.04 * r, b[1] + 0.04 * r], color=INK2, lw=1)
    ax1.plot(*sun, "o", ms=16, color=SERIES[3], mec=SURFACE, mew=1.5, zorder=5)
    ax1.text(sun[0] + 0.15 * r, sun[1], "Sun", color=INK, fontsize=10, va="center")
    ax1.text(V1n[0] - 0.12 * r, 2.6 * r, "great circles\nthrough the Sun", color=INK2, fontsize=9.5, ha="right",
             va="center")
    # labels along the target's great circle: its meaning on the left, its length (hypotenuse) on the right
    rot = np.degrees(np.arctan2(gT[1], gT[0])) - 180
    n_right = np.array([gT[1], -gT[0]])
    ax1.text(*(T + 1.35 * r * gT - 0.13 * r * n_right), "V3 with V1 on target", color=INK2, fontsize=9.5,
             rotation=rot, rotation_mode="anchor", ha="center", va="center")
    ax1.text(*(T + 2.3 * r * gT + 0.13 * r * n_right), "tan ε", color=INK, fontsize=10,
             rotation=rot, rotation_mode="anchor", ha="center", va="center")

    # dPA at the Sun vertex
    a_down, a_t = -np.pi / 2, np.arctan2(*(T - sun)[::-1])
    ang = np.linspace(a_down, a_t, 20)
    ax1.plot(sun[0] + 0.8 * r * np.cos(ang), sun[1] + 0.8 * r * np.sin(ang), color=INK, lw=1.2)
    ax1.text(sun[0] - 0.1 * r, sun[1] - 0.85 * r, "ΔPA", color=INK, fontsize=11, ha="right", va="center")

    # aperture offset from V1, split into its V2 part (across the Sun plane) and V3 part (along V1's great
    # circle to the Sun, a pitch); V1 moves by the reverse of these when the aperture is put on the target
    arrow(ax1, T, N, SERIES[1], 2.4)
    arrow(ax1, N, V1n, SERIES[0], 2.4)
    ax1.text(N[0] / 2, -0.06 * r, "|V2Ref|", color=SERIES[1], fontsize=9.5, ha="center", va="top")
    ax1.text(N[0] - 0.08 * r, 0.3 * V1n[1], "|V3Ref|", color=SERIES[0], fontsize=9.5, ha="right", va="center")

    # V3 is vertical at V1 and at the target; the target's great circle is tilted from it by dPA
    arrow(ax1, V1n, V1n + [0, 1.1 * r], PURPLE, 2.0)
    ax1.text(*(V1n + [0, 1.1 * r]) + [-0.08 * r, 0.0], "+V3", color=PURPLE, fontsize=10, ha="right", va="center")
    arrow(ax1, V1n, V1n + [-0.85 * r, 0], PURPLE, 2.0)
    ax1.text(*(V1n + [-0.85 * r, 0]) + [0.05 * r, 0.06 * r], "+V2", color=PURPLE, fontsize=10, ha="center",
             va="bottom")
    ax1.plot([0, 0], [-1.5 * r, 0], color=PURPLE, lw=1.2, ls="--")
    arrow(ax1, T, [0, 1.2 * r], PURPLE, 2.0)
    ax1.text(0.1 * r, 1.2 * r, "V3 with aperture\non target", color=PURPLE, fontsize=9.5, va="center")
    ang = np.linspace(-np.pi / 2, np.arctan2(-gT[1], -gT[0]), 20)
    ax1.plot(1.3 * r * np.cos(ang), 1.3 * r * np.sin(ang), color=INK, lw=1.2)
    ax1.text(0.3 * r, -1.35 * r, "ΔPA", color=INK, fontsize=11, ha="left", va="center")

    ax1.plot(*V1n, "o", ms=20, mfc=SURFACE, mec=PURPLE, mew=1.8, zorder=6)
    ax1.plot(*V1n, "x", ms=13, color=PURPLE, mew=1.8, zorder=7)
    ax1.text(V1n[0] - 0.12 * r, V1n[1] - 0.05 * r, "V1", color=PURPLE, fontsize=11, ha="right", va="top")
    ax1.plot(*T, "*", ms=18, color=SERIES[1], mec=SURFACE, mew=1, zorder=7)
    ax1.text(0.15 * r, -0.08 * r, "target = CGI\naperture", color=INK, fontsize=10, ha="left", va="top")

    # inset: side view of the projection, where eps is the angle at the observatory
    tan_eps = np.tan(np.radians(eps))
    axi = ax1.inset_axes([0.95 * r, 1.95 * r, 2.6 * r, 1.95 * r], transform=ax1.transData)
    axi.set_aspect("equal")
    axi.axis("off")
    th = np.radians(np.linspace(55, 125, 50))
    axi.plot(np.cos(th), np.sin(th), color=AXIS, lw=1)
    axi.plot([-0.3, tan_eps + 0.35], [1, 1], color=INK2, lw=1)
    axi.plot([0, 0], [0, 1], color=INK2, lw=1)
    axi.plot([0, tan_eps * 1.12], [0, 1.12], color=SERIES[3], lw=1, ls="--")
    axi.plot(tan_eps * 1.12, 1.12, "o", ms=8, color=SERIES[3], mec=SURFACE)
    axi.plot([0, 0.1, 0.1], [0.9, 0.9, 1], color=INK2, lw=0.8)
    ang = np.linspace(np.pi / 2, np.pi / 2 - np.radians(eps), 20)
    axi.plot(0.3 * np.cos(ang), 0.3 * np.sin(ang), color=INK, lw=1)
    axi.text(0.36 * np.cos(ang[10]), 0.36 * np.sin(ang[10]), "ε", fontsize=11, color=INK, ha="left",
             va="bottom")
    axi.plot(0, 0, "s", ms=6, color=PURPLE)
    axi.text(0.08, -0.02, "Roman", color=PURPLE, fontsize=9, va="top")
    axi.plot(0, 1, "*", ms=10, color=SERIES[1], mec=SURFACE)
    axi.text(-0.06, 0.5, "1", color=INK2, fontsize=9, ha="right", va="center")
    axi.text(tan_eps / 2, 1.04, "tan ε", color=INK, fontsize=10, ha="center", va="bottom")
    axi.text(tan_eps + 0.35, 0.95, "projection\nplane", color=INK2, fontsize=8.5, ha="right", va="top")
    axi.text(tan_eps * 1.12 + 0.1, 1.12, "Sun", color=INK2, fontsize=8.5, va="center")
    axi.set_xlim(-0.35, tan_eps + 0.5)
    axi.set_ylim(-0.2, 1.3)
    ax1.text(0.95 * r, 1.9 * r, "Side view: the Sun projects\ntan ε from the target", color=INK2,
             fontsize=9, va="top")

    # relation to the formula
    ax1.text(3.5 * r, sun[1] + 0.3 * r, "sin ΔPA = |V2Ref| / tan ε,\nso  ΔPA ≈ |V2Ref| / tan ε",
             color=INK, fontsize=10.5, ha="right", va="top", multialignment="left", linespacing=1.5,
             bbox=dict(facecolor=SURFACE, edgecolor=AXIS, boxstyle="round,pad=0.5"))

    ax1.set_xlim(-3.25 * r, 3.6 * r)
    ax1.set_ylim(-1.75 * r, sun[1] + 0.4 * r)

    fig.tight_layout()
    path = os.path.join(outdir, "boresight_offset_geometry.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--outdir", default=os.path.join(os.path.dirname(__file__), "..", "docs",
                                                           "attitude_model"),
                    help="Output directory (default: docs/attitude_model)")
    args = ap.parse_args()
    plot_boresight_offset_diagram(args.outdir)
