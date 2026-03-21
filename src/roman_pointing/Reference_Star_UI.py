import base64
from io import BytesIO, StringIO
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from IPython.display import display, HTML
import ipywidgets as widgets

from roman_pointing.roman_observability import (
    get_target_coords,
    compute_roman_angles,
    compute_keepout,
)


from roman_pointing.Reference_Star_Selection_Tool import (
    load_catalog,
    select_ref_star,
    get_observable_windows,
    build_skycoord,
    SUN_MIN,
    SUN_MAX,
    MAX_PITCH_DIFF,
    MAX_IMAG,
    REF_GRADES,
    SORT_MODES,
    SORT_MODE_LABELS,
    BAND_LABEL,
    CATALOG_URL,
    DEFAULT_CACHE_PATH,
    MAX_CACHE_AGE_HOURS,
)

# Maps sort mode to which result table column gets highlighted
SORT_MODE_TO_COLUMN = {
    "valid_days":     "valid_days",
    "closest_mag":    "mag",
    "brightest":      "mag",
    "faintest":       "mag",
    "closest_pitch":  "pitch",
    "farthest_pitch": "pitch",
}

# Table header styles
STYLE_ACTIVE_TH = (
    "padding:7px 10px;border:1px solid #ddd;"
    "background:#e65100;color:white;font-weight:bold"
)
STYLE_NORMAL_TH = (
    "padding:7px 10px;border:1px solid #ddd;"
    "background:#1565c0;color:white"
)
STYLE_ACTIVE_TD = (
    "padding:6px 10px;border:1px solid #ddd;"
    "background:#fff3e0;font-weight:bold"
)
STYLE_NORMAL_TD = "padding:6px 10px;border:1px solid #ddd"

GRADE_COLORS = {"A": "#2e7d32", "B": "#f57c00", "C": "#c62828"}


def fig_to_html(fig):
    """Encode a Matplotlib figure as an inline base64 PNG image tag."""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%"/>'


def html_panel(content, title=""):
    """Wrap HTML content in a styled card panel with an optional heading."""
    header = f"<h3 style='margin:0 0 8px 0;color:#333'>{title}</h3>" if title else ""
    return (
        "<div style='background:#f9f9f9;border:1px solid #ddd;border-radius:6px;"
        "padding:16px;margin-bottom:12px;font-family:monospace;font-size:13px'>"
        f"{header}{content}</div>"
    )


def html_badge(text, color):
    """Render a small inline coloured badge."""
    return (
        f"<span style='background:{color};color:white;border-radius:4px;"
        f"padding:2px 7px;font-size:12px;font-weight:bold'>{text}</span>"
    )


def grade_color(grade):
    """Return the CSS color for a grade letter (A, B, or C)."""
    return GRADE_COLORS.get(grade, "#555")


def mag_header(effective_sort, sci_mag_available):
    """Return the appropriate magnitude column header: 'Mag Diff' or 'Magnitude'."""
    return "Mag Diff" if (effective_sort == "closest_mag" and sci_mag_available) else "Magnitude"


def mag_cell(ref, effective_sort, sci_mag_available):
    """Format the magnitude value for a reference star table cell."""
    if effective_sort == "closest_mag" and sci_mag_available and ref.get("mag_diff") is not None:
        return f"&Delta;{ref['mag_diff']:.2f}"
    if ref.get("mag") is not None:
        return f"{ref['mag']:.2f}"
    return "&mdash;"

def collect_best_refs(result, ref_coords_map):
    """Collect the best reference star and coordinates for each observable window.

    Args:
        result (dict): Return value from select_ref_star.
        ref_coords_map (dict): Map of star name → SkyCoord.

    Returns:
        list of tuple: (window_index, star_name, coord, win_start, win_end)
    """
    out = []
    for i, win in enumerate(result.get("observable_windows", [])):
        best = win.get("best_ref")
        if best is None:
            continue
        coord = ref_coords_map.get(best["reference_star"])
        if coord is None:
            continue
        out.append((i, best["reference_star"], coord, win.get("_win_start"), win.get("_win_end")))
    return out


def plot_solar_angle(ts, sci_solar, result, ref_coords_map):
    """Plot solar angle vs time for the science target and best reference per window."""
    fig, ax = plt.subplots(figsize=(11, 5))
    dates = [datetime.fromisoformat(t.iso) for t in ts]
    ax.plot(dates, sci_solar.to(u.deg).value, lw=2, label="Science target", color="#1565c0")

    colors = plt.cm.tab10.colors
    for idx, (i, name, coord, ws, we) in enumerate(collect_best_refs(result, ref_coords_map)):
        if ws is None or we is None:
            continue
        try:
            dur = we.mjd - ws.mjd
            _, ref_sun, _, _ = compute_roman_angles(coord, ws.isot, dur, time_step=1.0)
            wdates = [datetime.fromisoformat((ws + d * u.day).iso) for d in np.arange(0, dur, 1.0)]
            ax.plot(wdates, ref_sun.to(u.deg).value, lw=1.5, linestyle="--",
                    color=colors[(idx + 1) % len(colors)], label=f"Ref: {name} (Win {i + 1})")
        except Exception:
            pass

    xlim = ax.get_xlim()
    ax.axhline(SUN_MIN, color="k", linestyle="--", lw=1)
    ax.axhline(SUN_MAX, color="k", linestyle="--", lw=1)
    ax.fill_between(xlim, [SUN_MIN] * 2, [0] * 2, hatch="/", color="none", edgecolor="k", alpha=0.25)
    ax.fill_between(xlim, [SUN_MAX] * 2, [180] * 2, hatch="\\", color="none", edgecolor="k", alpha=0.25)
    ax.set_xlabel("Date")
    ax.set_ylabel("Solar Angle (deg)")
    ax.set_title("Solar Angle vs Time\n(dashed = best reference star per window)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    fig.subplots_adjust(right=0.75)
    return fig_to_html(fig)


def plot_pitch_angle(ts, sci_pitch, result, ref_coords_map):
    """Plot pitch angle vs time for the science target and best reference per window."""
    fig, ax = plt.subplots(figsize=(11, 5))
    dates = [datetime.fromisoformat(t.iso) for t in ts]
    ax.plot(dates, sci_pitch.to(u.deg).value, lw=2, label="Science target", color="#1565c0")

    colors = plt.cm.tab10.colors
    for idx, (i, name, coord, ws, we) in enumerate(collect_best_refs(result, ref_coords_map)):
        if ws is None or we is None:
            continue
        try:
            dur = we.mjd - ws.mjd
            _, _, _, ref_p = compute_roman_angles(coord, ws.isot, dur, time_step=1.0)
            wdates = [datetime.fromisoformat((ws + d * u.day).iso) for d in np.arange(0, dur, 1.0)]
            ax.plot(wdates, ref_p.to(u.deg).value, lw=1.5, linestyle="--",
                    color=colors[(idx + 1) % len(colors)], label=f"Ref: {name} (Win {i + 1})")
        except Exception:
            pass

    ax.set_xlabel("Date")
    ax.set_ylabel("Pitch Angle (deg)")
    ax.set_title("Pitch Angle vs Time\n(dashed = best reference star per window)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    fig.subplots_adjust(right=0.75)
    return fig_to_html(fig)


def plot_keepout_map(ts, keepout_arr, sci_name):
    """Plot a colour-coded Roman keepout timeline for the science target."""
    dates = [datetime.fromisoformat(t.iso) for t in ts]
    date_nums = mdates.date2num(dates)
    fig, ax = plt.subplots(figsize=(12, 2.5))
    cmap = matplotlib.colors.ListedColormap(["black", "green"])
    extended = np.append(date_nums, date_nums[-1] + (date_nums[-1] - date_nums[-2]))
    ax.pcolormesh(extended, [0, 1], keepout_arr.reshape(1, -1).astype(int), cmap=cmap, shading="flat")

    transition_indices = np.where(np.diff(keepout_arr.astype(int)) != 0)[0] + 1
    tick_indices = np.unique(np.concatenate(([0], transition_indices, [len(ts) - 1])))
    ax.set_xticks(mdates.date2num([dates[i] for i in tick_indices]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=9)
    ax.set_yticks([0.5])
    ax.set_yticklabels([sci_name])
    ax.set_ylim(0, 1)
    cbar = plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap), ticks=[0.25, 0.75], ax=ax)
    cbar.ax.set_yticklabels(["Unavailable", "Available"])
    ax.set_title(f"Roman Keepout Map — {sci_name}\n{ts[0].iso[:10]} to {ts[-1].iso[:10]}")
    fig.tight_layout()
    return fig_to_html(fig)


def plot_pitch_diff(ts, sci_pitch, result, ref_coords_map, max_pitch_diff=MAX_PITCH_DIFF):
    """Plot pitch angle difference between the science target and best reference per window."""
    fig, ax = plt.subplots(figsize=(11, 5))
    sci_pitch_vals = sci_pitch.to(u.deg).value
    ax.axhline(max_pitch_diff, color="red", linestyle="--", lw=1.5,
               label=f"Max allowed ({max_pitch_diff} deg)")

    colors = plt.cm.tab10.colors
    for idx, (i, name, coord, ws, we) in enumerate(collect_best_refs(result, ref_coords_map)):
        if ws is None or we is None:
            continue
        try:
            dur = we.mjd - ws.mjd
            _, _, _, ref_p = compute_roman_angles(coord, ws.isot, dur, time_step=1.0)
            ref_pitch_vals = ref_p.to(u.deg).value
            start_idx = int(ws.mjd - ts[0].mjd)
            sci_slice = sci_pitch_vals[start_idx: start_idx + len(ref_pitch_vals)]
            min_len = min(len(sci_slice), len(ref_pitch_vals))
            diff = np.abs(sci_slice[:min_len] - ref_pitch_vals[:min_len])
            wdates = [datetime.fromisoformat((ws + d * u.day).iso) for d in np.arange(0, min_len)]
            ax.plot(wdates, diff, lw=1.5, color=colors[(idx + 1) % len(colors)],
                    label=f"Ref: {name} (Win {i + 1})")
        except Exception:
            pass

    ax.set_xlabel("Date")
    ax.set_ylabel("Pitch Angle Difference (deg)")
    ax.set_title("Pitch Angle Difference: Science Target vs Best Reference per Window")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    fig.subplots_adjust(right=0.75)
    return fig_to_html(fig)


def availability_html(result):
    """Build an HTML availability calendar grid for all observable windows.

    Args:
        result (dict): Return value from select_ref_star.

    Returns:
        str: HTML table showing which reference stars are observable per date.
    """
    wins = result.get("observable_windows", [])
    if not wins:
        return "<p style='color:#c62828'>No observable windows found.</p>"

    sections = []
    for i, win in enumerate(wins):
        valid_refs = win.get("valid_refs", [])
        if not valid_refs:
            sections.append(
                f"<h4 style='color:#c62828;margin:12px 0 4px'>Window {i + 1}: "
                f"{win['start']} to {win['end']} — No valid reference stars</h4>"
            )
            continue

        sections.append(
            f"<h4 style='margin:16px 0 6px 0;color:#1565c0'>"
            f"Window {i + 1}: {win['start']} to {win['end']} ({win['duration_days']:.1f} days)</h4>"
        )

        star_names = [r["reference_star"] for r in valid_refs]
        grades = {r["reference_star"]: r.get("grade", "?") for r in valid_refs}
        n_days_map = {r["reference_star"]: r.get("n_valid_days", 0) for r in valid_refs}
        all_dates = sorted({d for r in valid_refs for d in r.get("valid_dates", [])})
        avail_lookup = {r["reference_star"]: set(r.get("valid_dates", [])) for r in valid_refs}

        th_stars = "".join(
            "<th style='padding:5px 8px;border:1px solid #ddd;white-space:nowrap;"
            "background:#1565c0;color:white;font-size:11px'>"
            f"{name}<br><span style='font-weight:normal;font-size:10px'>"
            f"Grade {grades[name]} &middot; {n_days_map[name]}d</span></th>"
            for name in star_names
        )
        header_row = (
            "<tr><th style='padding:5px 8px;border:1px solid #ddd;background:#1565c0;color:white'>Date</th>"
            f"{th_stars}</tr>"
        )

        body_rows = ""
        for date in all_dates:
            cells = "".join(
                (
                    "<td style='padding:4px 8px;border:1px solid #ddd;text-align:center;background:#e8f5e9'>&#10003;</td>"
                    if date in avail_lookup.get(name, set()) else
                    "<td style='padding:4px 8px;border:1px solid #ddd;text-align:center;color:#ccc'>&mdash;</td>"
                )
                for name in star_names
            )
            body_rows += (
                f"<tr><td style='padding:4px 8px;border:1px solid #ddd;"
                f"font-family:monospace;white-space:nowrap'>{date}</td>{cells}</tr>"
            )

        sections.append(
            "<div style='overflow-x:auto;margin-bottom:20px'>"
            "<table style='border-collapse:collapse;font-size:12px'>"
            f"<thead>{header_row}</thead><tbody>{body_rows}</tbody></table></div>"
        )

    return "".join(sections)


def results_html(result, band_label):
    """Build the main reference star results summary and table.

    Args:
        result (dict): Return value from select_ref_star.
        band_label (str): 'V' or 'I', used in the summary header.

    Returns:
        str: HTML string with a parameter grid and per-window results table.
    """
    sci = result["science_target"]
    vis = result["visibility_pct"]
    sort_method = result["sort_method"]
    wins = result["observable_windows"]
    active_grades = result.get("allowed_grades", REF_GRADES)
    sort_mode = result.get("sort_mode", "valid_days")
    effective = result.get("effective_sort", sort_mode)
    sci_mag = result.get("sci_mag")
    sci_mag_available = sci_mag is not None
    max_pitch_diff = result.get("max_pitch_diff", MAX_PITCH_DIFF)
    max_imag = result.get("max_imag")
    band = result.get("band")
    sci_diameter = result.get("sci_diameter")
    sci_diameter_src = result.get("sci_diameter_src", "")

    mag_hdr = mag_header(effective, sci_mag_available)
    sci_mag_str = f"{sci_mag:.2f}" if sci_mag_available else "N/A"
    sci_diam_str = (
        f"{sci_diameter:.4f} mas"
        if sci_diameter is not None
        else "<span style='color:#c62828'>not found</span>"
    )
    if sci_diameter_src:
        sci_diam_str += f" <span style='color:#888;font-size:11px'>[{sci_diameter_src}]</span>"

    active_col = SORT_MODE_TO_COLUMN.get(effective, "")

    band_display = {
        1:    "1 — V-band NFB",
        3:    "3 — I-band Spec",
        4:    "4 — I-band Wide FOV B4",
        "1w": "1w — V-band Wide FOV B1",
    }.get(band, str(band))

    band_uses_imag = band in (3, 4)
    if not band_uses_imag:
        imag_filter_str = "n/a (V-band mode)"
    elif max_imag is not None:
        imag_filter_str = f"&le; {max_imag}"
    else:
        imag_filter_str = "disabled"

    def th(label, col_id):
        style = STYLE_ACTIVE_TH if col_id == active_col else STYLE_NORMAL_TH
        suffix = " &#9660;" if col_id == active_col else ""
        return f"<th style='{style}'>{label}{suffix}</th>"

    def td(value, col_id, extra_style=""):
        style = STYLE_ACTIVE_TD if col_id == active_col else STYLE_NORMAL_TD
        if extra_style:
            style += ";" + extra_style
        return f"<td style='{style}'>{value}</td>"

    def kv(label, value):
        return (
            f"<td style='padding:5px 14px 5px 0;white-space:nowrap;vertical-align:top'>"
            f"<span style='color:#555;font-weight:600;font-size:12px'>{label}</span></td>"
            f"<td style='padding:5px 20px 5px 0;vertical-align:top'>"
            f"<span style='font-size:12px'>{value}</span></td>"
        )

    header = (
        "<div style='margin-bottom:14px'>"
        "<table style='border-collapse:collapse;font-family:monospace'>"
        "<tr>"
        + kv("Science target", f"<b>{sci}</b>")
        + kv("Band", band_display)
        + kv("Contrast", result.get("contrast", "—"))
        + kv("Visibility", f"{vis:.1f}%")
        + "</tr><tr>"
        + kv(f"Science mag", f"{band_label}: {sci_mag_str}")
        + kv("Science diameter", sci_diam_str)
        + kv("Grade column", f"<span style='font-family:monospace;font-size:11px;background:#f0f4f8;color:#1a1a2e;padding:1px 5px;border-radius:3px;border:1px solid #d0d7e2'>{result.get('grade_column', '—')}</span>")
        + kv("Allowed grades", ", ".join(active_grades))
        + "</tr><tr>"
        + kv("Sort mode", SORT_MODE_LABELS.get(sort_mode, sort_mode))
        + kv("Max pitch diff", f"{max_pitch_diff}&deg;")
        + kv("I-mag filter", imag_filter_str)
        + "</tr>"
        "</table></div>"
    )

    table_header = (
        "<tr>"
        + f"<th style='{STYLE_NORMAL_TH}'>Window</th>"
        + f"<th style='{STYLE_NORMAL_TH}'>Start</th>"
        + f"<th style='{STYLE_NORMAL_TH}'>End</th>"
        + f"<th style='{STYLE_NORMAL_TH}'>Duration</th>"
        + f"<th style='{STYLE_NORMAL_TH}'>Reference Star</th>"
        + f"<th style='{STYLE_NORMAL_TH}'>Grade</th>"
        + th(mag_hdr, "mag")
        + th("Valid Days", "valid_days")
        + th("Min Pitch Diff", "pitch")
        + f"<th style='{STYLE_NORMAL_TH}'>Diameter (mas)</th>"
        + "</tr>"
    )

    rows = ""
    for i, win in enumerate(wins):
        valid_refs = win.get("valid_refs", [])
        n_valid = len(valid_refs)

        if not valid_refs:
            rows += (
                f"<tr>"
                f"<td style='{STYLE_NORMAL_TD}'><b>Window {i + 1}</b></td>"
                f"<td style='{STYLE_NORMAL_TD}'>{win['start']}</td>"
                f"<td style='{STYLE_NORMAL_TD}'>{win['end']}</td>"
                f"<td style='{STYLE_NORMAL_TD}'>{win['duration_days']:.1f} d</td>"
                f"<td colspan='6' style='{STYLE_NORMAL_TD};color:#c62828;font-style:italic'>"
                f"No suitable reference stars found</td></tr>"
            )
            continue

        for j, ref in enumerate(valid_refs):
            ref_name = ref["reference_star"]
            grade = ref.get("grade", "—")
            n_days = ref.get("n_valid_days", 0)
            min_pdiff = ref.get("min_pitch_diff")
            ref_diam = ref.get("diameter_mas")
            ref_diam_src = ref.get("diameter_src") or ""

            grade_badge = html_badge(grade, grade_color(grade))
            best_badge = html_badge("&#9733; best", "#1565c0") if j == 0 else ""
            mag_str = mag_cell(ref, effective, sci_mag_available)
            pdiff_str = (
                f"{min_pdiff:.4f}&deg;"
                if min_pdiff is not None and min_pdiff < 999
                else "&mdash;"
            )
            diam_str = (
                f"{ref_diam:.4f}"
                f"<span style='font-size:10px;color:#888;margin-left:3px'>[{ref_diam_src}]</span>"
                if ref_diam is not None
                else "<span style='color:#c62828;font-size:11px'>not found</span>"
            )

            if j == 0:
                win_cell = (
                    f"<td rowspan='{n_valid}' style='{STYLE_NORMAL_TD};vertical-align:top'><b>Window {i + 1}</b></td>"
                    f"<td rowspan='{n_valid}' style='{STYLE_NORMAL_TD};vertical-align:top'>{win['start']}</td>"
                    f"<td rowspan='{n_valid}' style='{STYLE_NORMAL_TD};vertical-align:top'>{win['end']}</td>"
                    f"<td rowspan='{n_valid}' style='{STYLE_NORMAL_TD};vertical-align:top'>{win['duration_days']:.1f} d</td>"
                )
            else:
                win_cell = ""

            rows += (
                f"<tr>{win_cell}"
                f"<td style='{STYLE_NORMAL_TD};font-weight:bold;color:#1b5e20'>{ref_name} {best_badge}</td>"
                f"<td style='{STYLE_NORMAL_TD};text-align:center'>{grade_badge}</td>"
                + td(mag_str, "mag", "text-align:right")
                + td(n_days, "valid_days", "text-align:center")
                + td(pdiff_str, "pitch")
                + f"<td style='{STYLE_NORMAL_TD};text-align:right'>{diam_str}</td>"
                + "</tr>"
            )

    table = (
        "<div style='overflow-x:auto'>"
        "<table style='border-collapse:collapse;width:100%;font-size:13px'>"
        f"<thead>{table_header}</thead><tbody>{rows}</tbody></table></div>"
    )
    return header + table


def make_csv_data_uri(csv_text: str) -> str:
    """Encode a CSV string as a base64 data URI suitable for an anchor href."""
    b64 = base64.b64encode(csv_text.encode("utf-8")).decode("utf-8")
    return f"data:text/csv;base64,{b64}"


def build_avail_csv(result, win, win_idx, sci_name, band, contrast) -> tuple:
    """Serialise one window's availability DataFrame to CSV text.

    Includes a metadata comment header and per-star diameter info.

    Args:
        result (dict): Full result dict from select_ref_star.
        win (dict): Single window entry.
        win_idx (int): Zero-based window index.
        sci_name (str): Science target name.
        band (int or str): Band identifier.
        contrast (str): Contrast level.

    Returns:
        tuple: (csv_text, filename), or (None, None) if no data.
    """
    avail_df = win.get("avail_df")
    if avail_df is None or avail_df.empty:
        return None, None

    safe_name = sci_name.replace(" ", "_").replace("*", "").strip("_")
    filename = f"availability_{safe_name}_band{band}_{contrast}_window{win_idx + 1}.csv"

    buf = StringIO()
    buf.write(f"# Availability Calendar — {sci_name}\n")
    buf.write(f"# Band: {band}  |  Contrast: {contrast}\n")
    buf.write(f"# Window {win_idx + 1}: {win['start']} to {win['end']}  ({win['duration_days']:.1f} days)\n")
    buf.write(f"# Max pitch diff: {result.get('max_pitch_diff', '')} deg\n")
    buf.write(f"# Sort mode: {result.get('sort_mode', '')}\n")

    sci_diam = result.get("sci_diameter")
    sci_diam_src = result.get("sci_diameter_src") or ""
    buf.write(
        "# Science target diameter: "
        + (f"{sci_diam:.4f} mas [{sci_diam_src}]" if sci_diam else "not found")
        + "\n"
    )

    for ref in win.get("valid_refs", []):
        diam = ref.get("diameter_mas")
        diam_src = ref.get("diameter_src") or ""
        diam_note = f"{diam:.4f} mas [{diam_src}]" if diam else "not found"
        buf.write(f"#   {ref['reference_star']}: diameter = {diam_note}\n")

    buf.write("# Values: 1 = observable, 0 = not observable on that date\n")
    avail_df.astype(int).to_csv(buf)
    return buf.getvalue(), filename


def build_pitch_csv(win, win_idx, sci_name, band, contrast) -> tuple:
    """Serialise one window's pitch-angle DataFrame to CSV text.

    Includes a metadata comment header and per-star diameter info.

    Args:
        win (dict): Single window entry.
        win_idx (int): Zero-based window index.
        sci_name (str): Science target name.
        band (int or str): Band identifier.
        contrast (str): Contrast level.

    Returns:
        tuple: (csv_text, filename), or (None, None) if no data.
    """
    pitch_df = win.get("pitch_df")
    if pitch_df is None or pitch_df.empty:
        return None, None

    safe_name = sci_name.replace(" ", "_").replace("*", "").strip("_")
    filename = f"pitch_table_{safe_name}_band{band}_{contrast}_window{win_idx + 1}.csv"

    buf = StringIO()
    buf.write(f"# Pitch Angle Table — {sci_name}\n")
    buf.write(f"# Band: {band}  |  Contrast: {contrast}\n")
    buf.write(f"# Window {win_idx + 1}: {win['start']} to {win['end']}  ({win['duration_days']:.1f} days)\n")
    buf.write("# Values: pitch angle difference in degrees (NaN = solar-angle check failed)\n")

    for ref in win.get("valid_refs", []):
        diam = ref.get("diameter_mas")
        diam_src = ref.get("diameter_src") or ""
        diam_note = f"{diam:.4f} mas [{diam_src}]" if diam else "not found"
        buf.write(f"#   {ref['reference_star']}: diameter = {diam_note}\n")

    pitch_df.to_csv(buf, float_format="%.3f")
    return buf.getvalue(), filename


def build_csv_download_widgets(result, sci_name, band, contrast):
    """Build ipywidgets download buttons for pitch and availability CSVs.

    One card per observable window, each with a Pitch Angle CSV button and
    an Availability CSV button. Nothing downloads until the user clicks.

    Args:
        result (dict): Return value from select_ref_star.
        sci_name (str): Science target name.
        band (int or str): Band identifier.
        contrast (str): Contrast level.

    Returns:
        ipywidgets.VBox or None: Widget tree, or None if no exportable data.
    """
    wins = result.get("observable_windows", [])
    if not wins:
        return None

    rows = []

    for i, win in enumerate(wins):
        if not win.get("valid_refs"):
            continue

        label = widgets.HTML(
            f"<b style='font-family:sans-serif;font-size:13px;color:#1565c0'>Window {i + 1}</b>"
            f"<span style='font-family:sans-serif;font-size:11px;color:#777;margin-left:8px'>"
            f"{win['start']} &rarr; {win['end']} &nbsp;({win['duration_days']:.1f} days)</span>"
        )

        btn_pitch = widgets.Button(
            description="⬇ Pitch Angle CSV",
            button_style="info",
            layout=widgets.Layout(width="200px", height="32px"),
            tooltip="Download pitch-angle difference table for this window",
        )
        pitch_out = widgets.Output()

        def on_pitch_click(_, win_ref=win, idx=i):
            csv_text, filename = build_pitch_csv(win_ref, idx, sci_name, band, contrast)
            if csv_text is None:
                return
            data_uri = make_csv_data_uri(csv_text)
            with pitch_out:
                pitch_out.clear_output()
                display(HTML(
                    f"<a id='dl-pitch-{idx}' href='{data_uri}' download='{filename}' style='display:none'></a>"
                    f"<script>document.getElementById('dl-pitch-{idx}').click();</script>"
                    f"<span style='font-size:11px;color:#2e7d32'>&#10003; Downloading <span style='font-family:monospace;font-size:12px;background:#f0f4f8;color:#1a1a2e;padding:1px 5px;border-radius:3px;border:1px solid #d0d7e2'>{filename}</span></span>"
                ))

        btn_pitch.on_click(on_pitch_click)

        btn_avail = widgets.Button(
            description="⬇ Availability CSV",
            button_style="primary",
            layout=widgets.Layout(width="200px", height="32px"),
            tooltip="Download availability calendar for this window",
        )
        avail_out = widgets.Output()

        def on_avail_click(_, win_ref=win, idx=i):
            csv_text, filename = build_avail_csv(result, win_ref, idx, sci_name, band, contrast)
            if csv_text is None:
                return
            data_uri = make_csv_data_uri(csv_text)
            with avail_out:
                avail_out.clear_output()
                display(HTML(
                    f"<a id='dl-avail-{idx}' href='{data_uri}' download='{filename}' style='display:none'></a>"
                    f"<script>document.getElementById('dl-avail-{idx}').click();</script>"
                    f"<span style='font-size:11px;color:#2e7d32'>&#10003; Downloading <span style='font-family:monospace;font-size:12px;background:#f0f4f8;color:#1a1a2e;padding:1px 5px;border-radius:3px;border:1px solid #d0d7e2'>{filename}</span></span>"
                ))

        btn_avail.on_click(on_avail_click)

        rows.append(widgets.VBox([
            label,
            widgets.HBox(
                [btn_pitch, pitch_out, btn_avail, avail_out],
                layout=widgets.Layout(align_items="center", gap="10px"),
            ),
        ], layout=widgets.Layout(margin="6px 0", padding="8px 12px", border="1px solid #e0e0e0")))

    if not rows:
        return None

    return widgets.VBox(
        [widgets.HTML("<b style='font-family:sans-serif;font-size:13px;color:#333'>&#11123; Download CSVs by window:</b>")] + rows,
        layout=widgets.Layout(margin_bottom="14px"),
    )


class ReferenceStarPickerUI:
    """Jupyter ipywidgets UI for the Roman Reference Star Picker.

    Instantiate and call display() to render the full interface in a notebook.
    The catalog is fetched from CATALOG_URL and cached to disk; no database
    connection is required.

    Args:
        catalog_url (str): Catalog fetch URL. Defaults to CATALOG_URL.
        cache_path: Path override for the on-disk CSV cache.
        max_cache_age_hours (float): Cache staleness threshold in hours.
        force_refresh (bool): Always fetch a fresh catalog on the next run.
    """

    def __init__(
        self,
        catalog_url: str = CATALOG_URL,
        cache_path=None,
        max_cache_age_hours: float = MAX_CACHE_AGE_HOURS,
        force_refresh: bool = False,
    ):
        self.catalog_url = catalog_url
        self.cache_path = cache_path
        self.max_cache_age_hours = max_cache_age_hours
        self.force_refresh_on_next_run = force_refresh
        self.catalog = None

        self.w_target = widgets.Text(
            value="47 Uma",
            description="Science Target:",
            style={"description_width": "140px"},
            layout=widgets.Layout(width="380px"),
            placeholder="SIMBAD-resolvable name",
        )
        self.w_band = widgets.Dropdown(
            options=[
                ("Band 1  — V-band NFB",          1),
                ("Band 1w — V-band Wide FOV B1",  "1w"),
                ("Band 3  — I-band Spec",          3),
                ("Band 4  — I-band Wide FOV B4",   4),
            ],
            value=1,
            description="Band:",
            style={"description_width": "140px"},
            layout=widgets.Layout(width="360px"),
        )
        self.w_contrast = widgets.Dropdown(
            options=[("High contrast", "high"), ("Medium contrast", "med")],
            value="high",
            description="Contrast:",
            style={"description_width": "140px"},
            layout=widgets.Layout(width="280px"),
        )
        self.w_start = widgets.Text(
            value="2026-12-01T00:00:00",
            description="Analysis Start:",
            style={"description_width": "140px"},
            layout=widgets.Layout(width="380px"),
            placeholder="YYYY-MM-DDTHH:MM:SS",
        )
        self.w_days = widgets.BoundedFloatText(
            value=365, min=1, max=3650,
            description="Duration (days):",
            style={"description_width": "140px"},
            layout=widgets.Layout(width="280px"),
        )
        self.w_timestep = widgets.BoundedFloatText(
            value=1.0, min=0.1, max=30.0, step=0.1,
            description="Time step (days):",
            style={"description_width": "140px"},
            layout=widgets.Layout(width="280px"),
        )
        self.w_max_pitch = widgets.BoundedFloatText(
            value=MAX_PITCH_DIFF, min=0.1, max=30.0, step=0.5,
            description="Max pitch diff (°):",
            style={"description_width": "140px"},
            layout=widgets.Layout(width="280px"),
        )
        self.w_max_imag = widgets.BoundedFloatText(
            value=1.5, min=-5.0, max=20.0, step=0.5,
            description="Max I-mag filter:",
            style={"description_width": "140px"},
            layout=widgets.Layout(width="280px"),
            disabled=True,
        )
        self.w_imag_filter_enabled = widgets.Checkbox(
            value=False,
            description="Enable I-mag filter",
            style={"description_width": "160px"},
            layout=widgets.Layout(width="220px"),
            indent=False,
        )
        self.w_imag_note = widgets.HTML(
            value="<span style='font-size:11px;color:#777'>"
                  "Excludes I-band-faint candidates (bands 3 &amp; 4 only)</span>"
        )
        self.w_grades = widgets.SelectMultiple(
            options=[
                ("A — best PSF quality",       "A"),
                ("B — acceptable PSF quality", "B"),
                ("C — marginal PSF quality",   "C"),
            ],
            value=("A", "B", "C"),
            description="Allowed grades:",
            style={"description_width": "140px"},
            layout=widgets.Layout(width="340px", height="82px"),
            rows=3,
        )
        self.w_sort_mode = widgets.Dropdown(
            options=[
                ("Most valid days first (default)",       "valid_days"),
                ("Closest magnitude to science target",   "closest_mag"),
                ("Brightest first (ascending magnitude)", "brightest"),
                ("Faintest first (descending magnitude)", "faintest"),
                ("Smallest pitch angle difference first", "closest_pitch"),
                ("Largest pitch angle difference first",  "farthest_pitch"),
            ],
            value="valid_days",
            description="Sort by:",
            style={"description_width": "140px"},
            layout=widgets.Layout(width="440px"),
        )
        self.w_force_refresh = widgets.Checkbox(
            value=False,
            description="Force catalog refresh",
            style={"description_width": "160px"},
            layout=widgets.Layout(width="260px"),
            indent=False,
        )
        self.w_run = widgets.Button(
            description="Find Reference Stars",
            button_style="primary",
            icon="search",
            layout=widgets.Layout(width="220px", height="36px"),
        )
        self.w_status = widgets.HTML(value="")

        self.out_results   = widgets.Output()
        self.out_solar     = widgets.Output()
        self.out_pitch     = widgets.Output()
        self.out_pitchdiff = widgets.Output()
        self.out_keepout   = widgets.Output()
        self.out_avail     = widgets.Output()
        self.out_downloads = widgets.Output()

        self.tabs = widgets.Tab(children=[
            self.out_results, self.out_solar, self.out_pitch,
            self.out_pitchdiff, self.out_keepout, self.out_avail,
            self.out_downloads,
        ])
        for idx, title in enumerate([
            "Results", "Solar Angle", "Pitch Angle",
            "Pitch Difference", "Keepout Map", "Availability Calendar",
            "⬇ Downloads",
        ]):
            self.tabs.set_title(idx, title)

        def update_imag_state(*_):
            band_is_vband = self.w_band.value in (1, "1w")
            filter_checked = self.w_imag_filter_enabled.value
            self.w_max_imag.disabled = not filter_checked or band_is_vband
            self.w_imag_filter_enabled.disabled = band_is_vband
            if band_is_vband:
                self.w_imag_note.value = (
                    "<span style='font-size:11px;color:#b0b0b0'>"
                    "I-mag filter not applicable for V-band modes (bands 1 &amp; 1w)</span>"
                )
            else:
                self.w_imag_note.value = (
                    "<span style='font-size:11px;color:#777'>"
                    "Excludes I-band-faint candidates (bands 3 &amp; 4 only)</span>"
                )

        self.w_imag_filter_enabled.observe(update_imag_state, names="value")
        self.w_band.observe(update_imag_state, names="value")
        self.w_run.on_click(self.on_run)

    def build_help_accordion(self):
        """Build and return a collapsed help accordion widget."""
        help_html = widgets.HTML(value="""
<div style="font-family:sans-serif;font-size:13px;line-height:1.7;color:#333;padding:4px 8px">

<h3 style="color:#1565c0;margin:0 0 4px 0">Overview</h3>
<p style="margin:0 0 10px 0">
  Finds valid coronagraph reference stars for the Roman Space Telescope.
  For every observable window, candidates are evaluated against two daily constraints:
</p>
<ul style="margin:0 0 10px 0;padding-left:20px">
  <li><b>Solar angle</b> — must be between <b>54&deg;</b> and <b>126&deg;</b>.</li>
  <li><b>Pitch angle difference</b> — must be less than <b>Max pitch diff</b> degrees.</li>
</ul>
<p style="margin:0 0 14px 0">Passing stars are ranked by grade (A &gt; B &gt; C) then by the chosen sort mode.</p>

<hr style="border:none;border-top:1px solid #e0e0e0;margin:10px 0"/>

<h3 style="color:#1565c0;margin:0 0 6px 0">Band &amp; Contrast</h3>
<table style="border-collapse:collapse;font-size:12px;margin-bottom:14px">
  <thead><tr>
    <th style="padding:5px 10px;background:#1565c0;color:white">Band</th>
    <th style="padding:5px 10px;background:#1565c0;color:white">Mode</th>
    <th style="padding:5px 10px;background:#1565c0;color:white">Magnitude</th>
    <th style="padding:5px 10px;background:#1565c0;color:white">High contrast column</th>
    <th style="padding:5px 10px;background:#1565c0;color:white">Med contrast column</th>
  </tr></thead>
  <tbody>
    <tr style="background:#f5f5f5">
      <td style="padding:5px 10px;border:1px solid #ddd;text-align:center"><b>1</b></td>
      <td style="padding:5px 10px;border:1px solid #ddd">NFB Narrow Field</td>
      <td style="padding:5px 10px;border:1px solid #ddd">V-band</td>
      <td style="padding:5px 10px;border:1px solid #ddd"><span style='font-family:monospace;font-size:12px;background:#f0f4f8;color:#1a1a2e;padding:1px 5px;border-radius:3px;border:1px solid #d0d7e2'>st_psfgrade_nfb1_high</span></td>
      <td style="padding:5px 10px;border:1px solid #ddd"><span style='font-family:monospace;font-size:12px;background:#f0f4f8;color:#1a1a2e;padding:1px 5px;border-radius:3px;border:1px solid #d0d7e2'>st_psfgrade_nfb1_med</span></td>
    </tr>
    <tr>
      <td style="padding:5px 10px;border:1px solid #ddd;text-align:center"><b>1w</b></td>
      <td style="padding:5px 10px;border:1px solid #ddd">Wide FOV B1</td>
      <td style="padding:5px 10px;border:1px solid #ddd"><b>V-band</b></td>
      <td style="padding:5px 10px;border:1px solid #ddd"><span style='font-family:monospace;font-size:12px;background:#f0f4f8;color:#1a1a2e;padding:1px 5px;border-radius:3px;border:1px solid #d0d7e2'>st_psfgrade_wfb4_high</span></td>
      <td style="padding:5px 10px;border:1px solid #ddd"><span style='font-family:monospace;font-size:12px;background:#f0f4f8;color:#1a1a2e;padding:1px 5px;border-radius:3px;border:1px solid #d0d7e2'>st_psfgrade_wfb4_med</span></td>
    </tr>
    <tr style="background:#f5f5f5">
      <td style="padding:5px 10px;border:1px solid #ddd;text-align:center"><b>3</b></td>
      <td style="padding:5px 10px;border:1px solid #ddd">Spec Spectroscopy</td>
      <td style="padding:5px 10px;border:1px solid #ddd">I-band</td>
      <td style="padding:5px 10px;border:1px solid #ddd"><span style='font-family:monospace;font-size:12px;background:#f0f4f8;color:#1a1a2e;padding:1px 5px;border-radius:3px;border:1px solid #d0d7e2'>st_psfgrade_specb3_high</span></td>
      <td style="padding:5px 10px;border:1px solid #ddd"><span style='font-family:monospace;font-size:12px;background:#f0f4f8;color:#1a1a2e;padding:1px 5px;border-radius:3px;border:1px solid #d0d7e2'>st_psfgrade_specb3_med</span></td>
    </tr>
    <tr>
      <td style="padding:5px 10px;border:1px solid #ddd;text-align:center"><b>4</b></td>
      <td style="padding:5px 10px;border:1px solid #ddd">Wide FOV B4</td>
      <td style="padding:5px 10px;border:1px solid #ddd">I-band</td>
      <td style="padding:5px 10px;border:1px solid #ddd"><span style='font-family:monospace;font-size:12px;background:#f0f4f8;color:#1a1a2e;padding:1px 5px;border-radius:3px;border:1px solid #d0d7e2'>st_psfgrade_wfb4_high</span></td>
      <td style="padding:5px 10px;border:1px solid #ddd"><span style='font-family:monospace;font-size:12px;background:#f0f4f8;color:#1a1a2e;padding:1px 5px;border-radius:3px;border:1px solid #d0d7e2'>st_psfgrade_wfb4_med</span></td>
    </tr>
  </tbody>
</table>
<p style="margin:0 0 14px 0;font-size:12px;color:#555">
  Bands 1w and 4 share wfb4 grade columns but differ in magnitude band (V vs I).
</p>

<hr style="border:none;border-top:1px solid #e0e0e0;margin:10px 0"/>

<h3 style="color:#1565c0;margin:0 0 6px 0">Sort Modes</h3>
<table style="border-collapse:collapse;width:100%;font-size:12px;margin-bottom:14px">
  <thead><tr>
    <th style="padding:5px 10px;background:#1565c0;color:white;width:22%">Mode</th>
    <th style="padding:5px 10px;background:#1565c0;color:white">Behaviour</th>
    <th style="padding:5px 10px;background:#1565c0;color:white">Best when&hellip;</th>
  </tr></thead>
  <tbody>
    <tr style="background:#f5f5f5">
      <td style="padding:5px 10px;border:1px solid #ddd"><b>valid_days</b></td>
      <td style="padding:5px 10px;border:1px solid #ddd">Most valid days first.</td>
      <td style="padding:5px 10px;border:1px solid #ddd">Maximising scheduling flexibility.</td>
    </tr>
    <tr>
      <td style="padding:5px 10px;border:1px solid #ddd"><b>closest_mag</b></td>
      <td style="padding:5px 10px;border:1px solid #ddd">Smallest |ref &minus; sci| magnitude first.</td>
      <td style="padding:5px 10px;border:1px solid #ddd">Minimising PSF calibration error.</td>
    </tr>
    <tr style="background:#f5f5f5">
      <td style="padding:5px 10px;border:1px solid #ddd"><b>brightest</b></td>
      <td style="padding:5px 10px;border:1px solid #ddd">Lowest magnitude first.</td>
      <td style="padding:5px 10px;border:1px solid #ddd">Maximising SNR.</td>
    </tr>
    <tr>
      <td style="padding:5px 10px;border:1px solid #ddd"><b>faintest</b></td>
      <td style="padding:5px 10px;border:1px solid #ddd">Highest magnitude first.</td>
      <td style="padding:5px 10px;border:1px solid #ddd">Avoiding detector saturation.</td>
    </tr>
    <tr style="background:#f5f5f5">
      <td style="padding:5px 10px;border:1px solid #ddd"><b>closest_pitch</b></td>
      <td style="padding:5px 10px;border:1px solid #ddd">Smallest minimum pitch difference first.</td>
      <td style="padding:5px 10px;border:1px solid #ddd">Minimising telescope slew.</td>
    </tr>
    <tr>
      <td style="padding:5px 10px;border:1px solid #ddd"><b>farthest_pitch</b></td>
      <td style="padding:5px 10px;border:1px solid #ddd">Largest minimum pitch difference first.</td>
      <td style="padding:5px 10px;border:1px solid #ddd">Exploring the pitch range edges.</td>
    </tr>
  </tbody>
</table>

<hr style="border:none;border-top:1px solid #e0e0e0;margin:10px 0"/>

<h3 style="color:#1565c0;margin:0 0 6px 0">CSV Downloads</h3>
<p style="margin:0 0 6px 0">
  In the <b>⬇ Downloads</b> tab, each window has two buttons. Nothing downloads until clicked.
</p>
<table style="border-collapse:collapse;width:100%;font-size:12px;margin-bottom:14px">
  <thead><tr>
    <th style="padding:5px 10px;background:#1565c0;color:white;width:30%">Button</th>
    <th style="padding:5px 10px;background:#1565c0;color:white">Contents</th>
  </tr></thead>
  <tbody>
    <tr style="background:#f5f5f5">
      <td style="padding:5px 10px;border:1px solid #ddd"><b>⬇ Pitch Angle CSV</b></td>
      <td style="padding:5px 10px;border:1px solid #ddd">
        Rows = dates; columns = reference stars; values = pitch difference in degrees.
      </td>
    </tr>
    <tr>
      <td style="padding:5px 10px;border:1px solid #ddd"><b>⬇ Availability CSV</b></td>
      <td style="padding:5px 10px;border:1px solid #ddd">
        Rows = dates; columns = reference stars; values = 1 (observable) or 0.
        Includes diameter info in header comments.
      </td>
    </tr>
  </tbody>
</table>

<hr style="border:none;border-top:1px solid #e0e0e0;margin:10px 0"/>

<h3 style="color:#1565c0;margin:0 0 6px 0">Catalog Cache</h3>
<table style="border-collapse:collapse;width:100%;font-size:12px;margin-bottom:10px">
  <thead><tr>
    <th style="padding:5px 10px;background:#1565c0;color:white;width:30%">Scenario</th>
    <th style="padding:5px 10px;background:#1565c0;color:white">Behaviour</th>
  </tr></thead>
  <tbody>
    <tr style="background:#f5f5f5">
      <td style="padding:5px 10px;border:1px solid #ddd">Cache fresh (&lt; 24 h)</td>
      <td style="padding:5px 10px;border:1px solid #ddd">Loaded from disk — no network request.</td>
    </tr>
    <tr>
      <td style="padding:5px 10px;border:1px solid #ddd">Cache stale or missing</td>
      <td style="padding:5px 10px;border:1px solid #ddd">Fetched fresh and saved.</td>
    </tr>
    <tr style="background:#f5f5f5">
      <td style="padding:5px 10px;border:1px solid #ddd">Fetch fails, stale cache exists</td>
      <td style="padding:5px 10px;border:1px solid #ddd">Warning issued; stale data used.</td>
    </tr>
    <tr>
      <td style="padding:5px 10px;border:1px solid #ddd">Fetch fails, no cache</td>
      <td style="padding:5px 10px;border:1px solid #ddd">RuntimeError raised.</td>
    </tr>
  </tbody>
</table>

</div>
""")
        accordion = widgets.Accordion(children=[help_html])
        accordion.set_title(0, "ℹ️  Instructions & Reference Guide  (click to expand)")
        accordion.selected_index = None
        return accordion

    def display(self):
        """Render the full UI in the current Jupyter cell output."""

        def section_label(text):
            return widgets.HTML(
                f"<div style='font-family:sans-serif;font-size:11px;font-weight:600;"
                f"color:#1565c0;text-transform:uppercase;letter-spacing:0.08em;"
                f"border-bottom:2px solid #1565c0;padding-bottom:3px;"
                f"margin:10px 0 8px 0'>{text}</div>"
            )

        # Harmonise widget widths at render time
        self.w_target.layout.width    = "340px"
        self.w_band.layout.width      = "300px"
        self.w_contrast.layout.width  = "300px"
        self.w_start.layout.width     = "340px"
        self.w_days.layout.width      = "220px"
        self.w_timestep.layout.width  = "220px"
        self.w_max_pitch.layout.width = "220px"
        self.w_max_imag.layout.width  = "220px"
        self.w_sort_mode.layout.width = "420px"
        self.w_grades.layout.width    = "420px"

        title = widgets.HTML(
            "<h2 style='font-family:sans-serif;color:#1565c0;margin:0 0 2px 0'>"
            "&#128301; Reference Star Picker</h2>"
            "<p style='font-family:sans-serif;color:#777;margin:0 0 10px 0;font-size:13px'>"
            "Roman Space Telescope — Coronagraph Reference Star Selection</p>"
        )

        sec1 = widgets.VBox([
            section_label("Target & Instrument"),
            widgets.HBox([self.w_target, self.w_band], layout=widgets.Layout(gap="16px")),
            widgets.HBox([self.w_contrast], layout=widgets.Layout(gap="16px")),
        ])

        sec2 = widgets.VBox([
            section_label("Analysis Parameters"),
            widgets.HBox([self.w_start, self.w_days], layout=widgets.Layout(gap="16px")),
            widgets.HBox([self.w_timestep, self.w_max_pitch], layout=widgets.Layout(gap="16px")),
        ])

        grades_hint = widgets.HTML(
            "<span style='font-size:11px;color:#777;font-family:sans-serif'>"
            "Ctrl/Cmd+click for multiple &nbsp;·&nbsp; order = priority</span>"
        )
        sec3 = widgets.VBox([
            section_label("Ranking & Filters"),
            self.w_sort_mode,
            widgets.HBox([self.w_grades, grades_hint], layout=widgets.Layout(align_items="center", gap="12px")),
            widgets.HBox([self.w_imag_filter_enabled, self.w_max_imag, self.w_imag_note],
                         layout=widgets.Layout(align_items="center", gap="10px")),
        ], layout=widgets.Layout(gap="6px"))

        run_row = widgets.HBox(
            [self.w_run, self.w_force_refresh, self.w_status],
            layout=widgets.Layout(
                align_items="center", gap="16px",
                margin="10px 0 0 0", padding="10px 0 0 0",
                border_top="1px solid #e0e0e0",
            ),
        )

        controls = widgets.VBox(
            [sec1, sec2, sec3, run_row],
            layout=widgets.Layout(
                padding="14px 16px", border="1px solid #ddd",
                border_radius="6px", margin_bottom="12px", gap="4px",
            ),
        )

        display(widgets.VBox([title, self.build_help_accordion(), controls, self.tabs]))

    def set_status(self, msg, color="#555"):
        """Update the status message displayed below the run button."""
        self.w_status.value = f"<span style='color:{color};font-size:13px'>{msg}</span>"

    def on_run(self, btn):
        """Handle a click on the Find Reference Stars button."""
        self.w_run.disabled = True
        for out in [
            self.out_results, self.out_solar, self.out_pitch,
            self.out_pitchdiff, self.out_keepout, self.out_avail,
            self.out_downloads,
        ]:
            out.clear_output()

        try:
            force = self.w_force_refresh.value or self.force_refresh_on_next_run
            if self.catalog is None or force:
                self.set_status("&#9203; Loading catalog...", "#f57c00")
                self.catalog = load_catalog(
                    url=self.catalog_url,
                    cache_path=self.cache_path,
                    max_cache_age_hours=self.max_cache_age_hours,
                    force_refresh=force,
                )
                self.force_refresh_on_next_run = False

            sci_name    = self.w_target.value.strip()
            band        = self.w_band.value
            contrast    = self.w_contrast.value
            start       = self.w_start.value.strip()
            days        = float(self.w_days.value)
            time_step   = float(self.w_timestep.value)
            max_pitch   = float(self.w_max_pitch.value)
            max_imag    = (
                float(self.w_max_imag.value)
                if self.w_imag_filter_enabled.value and band not in (1, "1w")
                else None
            )
            allowed_grades = list(self.w_grades.value)
            sort_mode      = self.w_sort_mode.value
            band_label     = BAND_LABEL.get(band, "?")

            if not allowed_grades:
                self.set_status("&#10060; Select at least one grade (A, B, or C).", "#c62828")
                return

            self.set_status(f"&#9203; Querying SIMBAD for '{sci_name}'...", "#f57c00")
            coords = get_target_coords([sci_name])
            if sci_name not in coords:
                self.set_status(f"&#10060; '{sci_name}' not found in SIMBAD.", "#c62828")
                return
            sci_coord = coords[sci_name]

            self.set_status("&#9203; Computing Roman visibility...", "#f57c00")
            ts, keepout, solar_angles = compute_keepout(
                {sci_name: sci_coord}, start, days, time_step
            )
            sci_keepout = keepout[sci_name]
            sci_solar   = solar_angles[sci_name]
            _, _, _, sci_pitch = compute_roman_angles(sci_coord, start, days, time_step)

            self.set_status("&#9203; Searching for reference stars...", "#f57c00")
            result = select_ref_star(
                sci_name, start, days,
                band=band, contrast=contrast,
                catalog=self.catalog,
                time_step=time_step,
                allowed_grades=allowed_grades,
                sort_mode=sort_mode,
                max_pitch_diff=max_pitch,
                max_imag=max_imag,
            )

            if "error" in result:
                self.set_status(f"&#10060; {result['error']}", "#c62828")
                return

            wins_raw = get_observable_windows(ts, sci_keepout)
            for i, win in enumerate(result["observable_windows"]):
                win["_win_start"] = wins_raw[i][0] if i < len(wins_raw) else None
                win["_win_end"]   = wins_raw[i][1] if i < len(wins_raw) else None

            ref_coords_map = {}
            for _, ref_row in self.catalog.iterrows():
                name = ref_row.get("main_id", "")
                if isinstance(name, str) and name.strip():
                    try:
                        ref_coords_map[name] = build_skycoord(ref_row)
                    except Exception:
                        pass

            self.set_status("&#9203; Rendering results...", "#f57c00")

            with self.out_results:
                display(HTML(html_panel(
                    results_html(result, band_label),
                    title="Reference Star Selection Results",
                )))

            with self.out_solar:
                display(HTML(plot_solar_angle(ts, sci_solar, result, ref_coords_map)))

            with self.out_pitch:
                display(HTML(plot_pitch_angle(ts, sci_pitch, result, ref_coords_map)))

            with self.out_pitchdiff:
                display(HTML(plot_pitch_diff(ts, sci_pitch, result, ref_coords_map, max_pitch_diff=max_pitch)))

            with self.out_keepout:
                display(HTML(plot_keepout_map(ts, sci_keepout, sci_name)))

            with self.out_avail:
                display(HTML(html_panel(
                    availability_html(result),
                    title="Reference Star Availability — Valid Observation Dates per Window",
                )))

            with self.out_downloads:
                dl_widget = build_csv_download_widgets(result, sci_name, band, contrast)
                if dl_widget is not None:
                    display(dl_widget)
                else:
                    display(HTML(
                        "<p style='font-family:sans-serif;color:#888;font-size:13px'>"
                        "No downloadable data — run the tool first.</p>"
                    ))

            vis    = result["visibility_pct"]
            n_wins = len(result["observable_windows"])
            sci_mag = result.get("sci_mag")
            mag_note = f" | sci {band_label}-mag: {sci_mag:.2f}" if sci_mag is not None else ""

            if band in (1, "1w"):
                imag_note = " | I-mag filter: n/a (V-band)"
            elif max_imag is not None:
                imag_note = f" | I-mag &le; {max_imag}"
            else:
                imag_note = " | I-mag filter: off"

            self.set_status(
                f"&#10003; Done — {n_wins} window(s), {vis:.1f}% observable"
                f" | grades: {', '.join(allowed_grades)}"
                f" | sort: {sort_mode}"
                f" | max pitch: {max_pitch}&deg;"
                f"{imag_note}{mag_note}",
                "#2e7d32",
            )

        except Exception as exc:
            self.set_status(f"&#10060; Error: {exc}", "#c62828")
            import traceback
            traceback.print_exc()
        finally:
            self.w_run.disabled = False



def launch(
    catalog_url: str = CATALOG_URL,
    cache_path=None,
    max_cache_age_hours: float = MAX_CACHE_AGE_HOURS,
    force_refresh: bool = False,
):
    """Create and display the ReferenceStarPickerUI in the current notebook cell.

    Args:
        catalog_url (str): Catalog fetch URL.
        cache_path: Path override for the on-disk CSV cache.
        max_cache_age_hours (float): Cache staleness threshold in hours.
        force_refresh (bool): Force a live fetch on the first run.

    Returns:
        ReferenceStarPickerUI: The instantiated UI object.
    """
    ui = ReferenceStarPickerUI(
        catalog_url=catalog_url,
        cache_path=cache_path,
        max_cache_age_hours=max_cache_age_hours,
        force_refresh=force_refresh,
    )
    ui.display()
    return ui