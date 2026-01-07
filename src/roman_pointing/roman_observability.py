import os
import csv
import base64
from datetime import datetime, timedelta
from io import StringIO

import astropy.units as u
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import BarycentricMeanEcliptic, Distance, SkyCoord
from astropy.time import Time
from astroquery.simbad import Simbad
from IPython.display import HTML, clear_output, display
import ipywidgets as widgets
from roman_pointing.roman_pointing import (
    calcRomanAngles,
    getL2Positions,
)


def get_target_coords(target_names):
    """Query SIMBAD for astronomical target coordinates and proper motions.

    Retrieves celestial coordinates, parallax, proper motion, and radial velocity
    data from SIMBAD database for specified astronomical objects. Transforms
    coordinates to Barycentric Mean Ecliptic frame for Roman Space Telescope
    pointing calculations.

    Args:
        target_names (list of str): List of astronomical object names recognizable
            by SIMBAD (e.g., 'Proxima Cen', 'Sirius', 'Betelgeuse'). Target names
            containing 'bulge' will use hardcoded galactic
            bulge coordinates.

    Returns:
        dict: Dictionary mapping target names (str) to astropy SkyCoord objects in
            BarycentricMeanEcliptic frame. Targets not found in SIMBAD are
            excluded from the returned dictionary.

    Note:
        This function prints a message to stdout when a target cannot be found
        in SIMBAD. Special handling exists for galactic bulge targets.
    """
    simbad = Simbad()
    simbad.add_votable_fields("pmra", "pmdec", "plx_value", "rvz_radvel")
    coords = {}

    for name in target_names:
        if "bulge" in name.lower():
            coords[name] = SkyCoord(
                266.41681662, -29.00782497,
                unit=(u.deg, u.deg), frame="icrs",
                distance=8277 * u.pc,
                pm_ra_cosdec=0 * u.mas / u.year,
                pm_dec=0 * u.mas / u.year,
                radial_velocity=0 * u.km / u.s,
                equinox="J2000", obstime="J2000",
            ).transform_to(BarycentricMeanEcliptic)
            continue

        res = simbad.query_object(name)
        if len(res) == 0:
            print(f"SIMBAD could not find {name}. Skipping.")
            continue

        c_icrs = SkyCoord(
            res["ra"].value.data[0], res["dec"].value.data[0],
            unit=(res["ra"].unit, res["dec"].unit), frame="icrs",
            distance=Distance(parallax=res["plx_value"].value.data[0] * res["plx_value"].unit),
            pm_ra_cosdec=res["pmra"].value.data[0] * res["pmra"].unit,
            pm_dec=res["pmdec"].value.data[0] * res["pmdec"].unit,
            radial_velocity=res["rvz_radvel"].value.data[0] * res["rvz_radvel"].unit,
            equinox="J2000", obstime="J2000",
        ).transform_to(BarycentricMeanEcliptic)

        coords[name] = c_icrs

    return coords


def compute_roman_angles(coord, start_date, days, time_step):
    """Calculate Roman Space Telescope pointing angles for a target over time.

    Computes the solar angle, yaw, and pitch angles required for Roman to observe
    a given celestial coordinate over a specified time period, accounting for the
    telescope's position at the Sun-Earth L2 Lagrange point.

    Args:
        coord (astropy.coordinates.SkyCoord): Target celestial coordinates in
            BarycentricMeanEcliptic frame.
        start_date (str): Start date in ISO format (e.g., '2027-01-01T00:00:00').
        days (int or float): Duration of observation window in days.
        time_step (int or float): Time interval between calculations in days.

    Returns:
        tuple: A tuple containing:
            - ts (astropy.time.Time): Array of time values
            - sun_ang (astropy.units.Quantity): Solar angles in degrees
            - yaw (astropy.units.Quantity): Yaw angles in degrees
            - pitch (astropy.units.Quantity): Pitch angles in degrees
    """
    t0 = Time(start_date, format="isot", scale="utc")
    ts = t0 + np.arange(0, days, time_step) * u.d

    sun_ang, yaw, pitch, _ = calcRomanAngles(
        coord,
        ts,
        getL2Positions(ts),
    )

    return ts, sun_ang, yaw, pitch



def compute_keepout(coords_dict, start_date, days, time_step, min_sun=54, max_sun=126):
    """Determine observability windows for multiple targets based on solar constraints.

    Calculates when targets are observable by Roman Space Telescope based on solar
    exclusion angle limits. The allowed solar angle range avoids thermal and stray
    light issues while keeping the solar panels properly oriented (default: 54-126
    degrees).

    Args:
        coords_dict (dict): Dictionary mapping target names (str) to SkyCoord objects.
        start_date (str): Start date in ISO format (e.g., '2027-01-01T00:00:00').
        days (int or float): Duration of observation window in days.
        time_step (int or float): Time interval between calculations in days.
        min_sun (int or float, optional): Minimum allowed solar angle in degrees.
            Default: 54.
        max_sun (int or float, optional): Maximum allowed solar angle in degrees.
            Default: 126.

    Returns:
        tuple: A tuple containing:
            - ts_global (astropy.time.Time): Array of time values
            - keepout (dict): Dictionary mapping target names to boolean arrays
              indicating observability (True = observable, False = in keepout zone)
            - solar_angles (dict): Dictionary mapping target names to solar angle
              arrays in degrees
    """
    solar_angles = {}
    keepout = {}
    ts_global = None

    for name, coord in coords_dict.items():
        ts, sun_ang, _, _ = compute_roman_angles(coord, start_date, days, time_step)
        solar_angles[name] = sun_ang
        keepout[name] = (sun_ang > min_sun * u.deg) & (sun_ang < max_sun * u.deg)
        if ts_global is None:
            ts_global = ts

    return ts_global, keepout, solar_angles


def plot_solar_angle(ts, solar_angles_dict):
    """Plot solar angle evolution for multiple targets over time.
    
    Creates a line plot showing how the Sun-target angle changes over the
    observation period for each target. Includes visual indicators for Roman's
    54° and 126° solar angle constraints (keepout zone boundaries).
    
    Args:
        ts (astropy.time.Time): Array of time samples.
        solar_angles_dict (dict): Maps target names (str) to arrays of solar
            angles (astropy.units.Quantity in degrees).
    
    Returns:
        tuple: Two-element tuple containing:
            - fig (matplotlib.figure.Figure): Figure object
            - ax (matplotlib.axes.Axes): Axes object
    
    Note:
        - Regions below 54° and above 126° are hatched to indicate keepout zones
        - Each target is plotted with a different color and labeled in the legend
        - X-axis labels are auto-formatted and rotated for readability
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    dates = [datetime.fromisoformat(t.iso) for t in ts]

    for name, sun in solar_angles_dict.items():
        ax.plot(dates, sun.to(u.deg), label=name)

    ax.set_xlabel("Date")
    ax.set_ylabel("Solar Angle (deg)")
    ax.axhline(y=54, color='k', linestyle='--', linewidth=1)
    ax.axhline(y=126, color='k', linestyle='--', linewidth=1)

    xlim = ax.get_xlim()
    ax.fill_between(xlim, [54, 54], [0, 0], hatch="/", color="none", edgecolor="k", alpha=0.3)
    ax.fill_between(xlim, [126, 126], [180, 180], hatch="\\", color="none", edgecolor="k", alpha=0.3)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    ax.set_title("Solar Angle vs Time")
    ax.legend(bbox_to_anchor=(1.15, 1), loc="upper left")
    fig.subplots_adjust(right=0.8)

    return fig, ax


def plot_solar_angle_with_windows(ts, solar_angles_dict, keepout_dict):
    """Plot solar angle with vertical lines marking observability transitions.
    
    Creates a detailed plot for a single target showing solar angle evolution
    with red vertical lines indicating exact dates when the target enters or
    exits Roman's observable zone. Optimized for single-target analysis.
    
    Args:
        ts (astropy.time.Time): Array of time samples.
        solar_angles_dict (dict): Single-entry dict mapping target name to solar angles.
        keepout_dict (dict): Single-entry dict mapping target name to boolean
            observability array.
    
    Returns:
        tuple: Two-element tuple containing:
            - fig (matplotlib.figure.Figure): Figure object
            - ax (matplotlib.axes.Axes): Axes object
    
    Raises:
        ValueError: If solar_angles_dict contains more than one target.
    
    Note:
        - Red dotted lines mark transitions between observable/unobservable states
        - X-axis ticks are positioned at transition dates for precise identification
        - Includes 5% padding on time axis for better visualization
    """
    if len(solar_angles_dict) != 1:
        raise ValueError("plot_solar_angle_with_windows is intended for single-target use only.")

    fig, ax = plt.subplots(figsize=(14, 6))
    target_name = list(solar_angles_dict.keys())[0]
    sun = solar_angles_dict[target_name]
    ko = keepout_dict[target_name]
    dates = [datetime.fromisoformat(t.iso) for t in ts]

    ax.plot(dates, sun.to(u.deg), label=target_name, linewidth=2)
    ax.axhline(54, color="k", linestyle="--", linewidth=1)
    ax.axhline(126, color="k", linestyle="--", linewidth=1)

    raw_inds = np.where(np.diff(ko.astype(int)) != 0)[0]
    transition_inds = raw_inds + 1

    for idx in transition_inds:
        ax.axvline(x=dates[idx], color="red", linestyle=":", linewidth=1, alpha=0.7)

    time_range = (dates[-1] - dates[0]).total_seconds()
    buffer = timedelta(seconds=time_range * 0.05)
    ax.set_xlim(dates[0] - buffer, dates[-1] + buffer)

    xlim = ax.get_xlim()
    ax.fill_between(xlim, [54, 54], [0, 0], hatch="/", color="none", edgecolor="k", alpha=0.3)
    ax.fill_between(xlim, [126, 126], [180, 180], hatch="\\", color="none", edgecolor="k", alpha=0.3)

    tick_inds = np.concatenate([transition_inds, [len(ts) - 1]])
    tick_dates = [dates[i] for i in tick_inds]
    ax.set_xticks(tick_dates)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    ax.set_xlabel("Date")
    ax.set_ylabel("Solar Angle (deg)")
    ax.set_title(f"Solar Angle vs Time — {target_name}\nRed lines mark true observability transitions")
    ax.legend(loc="upper left")
    fig.tight_layout()

    return fig, ax


def plot_pitch(ts, pitch_dict):
    """Plot spacecraft pitch angle requirements for target observation.
    
    Creates a line plot showing the pitch angle Roman must maintain to observe
    each target over time. Pitch angle represents the spacecraft's orientation
    relative to its reference frame.
    
    Args:
        ts (astropy.time.Time): Array of time samples.
        pitch_dict (dict): Maps target names (str) to arrays of pitch angles
            (astropy.units.Quantity in degrees).
    
    Returns:
        tuple: Two-element tuple containing:
            - fig (matplotlib.figure.Figure): Figure object
            - ax (matplotlib.axes.Axes): Axes object
    
    Note:
        - Different targets may require significantly different pitch angles
        - Useful for understanding spacecraft slew requirements between targets
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    dates = [datetime.fromisoformat(t.iso) for t in ts]

    for name, pitch in pitch_dict.items():
        ax.plot(dates, pitch.to(u.deg), label=name)

    ax.set_xlabel("Date")
    ax.set_ylabel("Pitch Angle (deg)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    ax.set_title("Pitch Angle vs Time")
    ax.legend(bbox_to_anchor=(1.15, 1), loc="upper left", borderaxespad=0)
    fig.subplots_adjust(right=0.8)

    return fig, ax


def plot_keepout(keepout_dict, ts):
    """Create a visibility map showing when targets are observable.

    Generates a color-coded heatmap displaying observability windows for one or
    more targets over time. Observable periods are shown in green, while keepout
    periods are shown in black. For single-target analysis, the x-axis ticks are
    positioned at transition points between observable and non-observable states.

    Args:
        keepout_dict (dict): Dictionary mapping target names (str) to boolean
            arrays where True indicates the target is observable and False
            indicates it is in a keepout zone.
        ts (astropy.time.Time): Array of time values corresponding to the
            keepout data points.

    Returns:
        tuple: A tuple containing:
            - fig (matplotlib.figure.Figure): The figure object
            - ax (matplotlib.axes.Axes): The axes object

    Note:
        For single targets, close transitions are slightly extended in the
        x-axis tick positioning to prevent label overlap. Font size is
        automatically adjusted based on tick density. Labels may be staggered
        vertically when spacing is less than 20 days.
    """
    names = list(keepout_dict.keys())
    num_targets = len(names)
    dates = [datetime.fromisoformat(t.iso) for t in ts]
    date_nums = mdates.date2num(dates)

    if num_targets == 1:
        komap = keepout_dict[names[0]].reshape(1, -1)
    else:
        komap = np.vstack([keepout_dict[n] for n in names])

    komap_int = komap.astype(int)
    fig_height = 4 if num_targets == 1 else max(3, 1.3 * num_targets + 1)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    cmap = matplotlib.colors.ListedColormap(["black", "green"])

    extended_date_nums = np.append(date_nums, date_nums[-1] + (date_nums[-1] - date_nums[-2]))
    ax.pcolormesh(extended_date_nums, np.arange(num_targets + 1), komap_int, cmap=cmap, shading="flat")

    ax.set_yticks(np.arange(num_targets) + 0.5)
    ax.set_yticklabels(names)
    ax.set_ylim(0, num_targets)

    def format_date(x, pos=None):
        dt = mdates.num2date(x)
        if dt.hour == 0 and dt.minute == 0:
            return dt.strftime("%Y-%m-%d")
        return dt.strftime("%Y-%m-%d\n%H:%M")

    if num_targets == 1:
        ko = keepout_dict[names[0]]
        transition_inds = np.where(np.diff(ko.astype(int)) != 0)[0] + 1
        tick_inds = np.unique(np.concatenate(([0], transition_inds, [len(ts) - 1])))
        tick_dates = [dates[i] for i in tick_inds]

        min_width_days = (ts[1].mjd - ts[0].mjd) * 3
        tick_nums = mdates.date2num(tick_dates)
        for i in range(1, len(tick_nums)):
            if tick_nums[i] - tick_nums[i-1] < min_width_days:
                tick_nums[i] = tick_nums[i-1] + min_width_days

        ax.set_xticks(mdates.num2date(tick_nums))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(format_date))

        num_ticks = len(tick_nums)
        avg_spacing_days = np.mean(np.diff(tick_nums))

        if num_ticks > 10 or avg_spacing_days < 0.3:
            fontsize = 6
        elif num_ticks > 7 or avg_spacing_days < 0.5:
            fontsize = 7
        elif num_ticks > 5 or avg_spacing_days < 1.0:
            fontsize = 8
        elif avg_spacing_days < 2.0:
            fontsize = 9
        else:
            fontsize = 10

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=fontsize)

        min_label_spacing = 20
        labels = ax.xaxis.get_majorticklabels()
        for i in range(1, len(tick_nums)):
            spacing = tick_nums[i] - tick_nums[i-1]
            if spacing < min_label_spacing and i % 2 == 1:
                labels[i].set_y(-0.05)
    else:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(format_date))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    ax.set_xlabel("Date")
    ax.set_ylabel("Target")

    cbar = plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap), ticks=[0.25, 0.75], ax=ax)
    cbar.ax.set_yticklabels(["Unavailable", "Available"])

    ax.set_title(f"Roman Keepout Map\n{ts[0].iso} → {ts[-1].iso}")
    plt.tight_layout()

    return fig, ax


def compute_visibility_fraction(keepout_dict):
    """Calculate the percentage of time each target is observable.

    Computes the fraction of the observation period during which each target
    falls within the allowed solar angle range (54-126 degrees) and is therefore
    observable by the Roman Space Telescope.

    Args:
        keepout_dict (dict): Dictionary mapping target names (str) to boolean
            arrays where True indicates the target is observable and False
            indicates it is in a keepout zone.

    Returns:
        dict: Dictionary mapping target names (str) to visibility percentages
            (float), representing the fraction of time the target is observable
            expressed as a percentage (0-100).
    """
    visibility = {}

    for name, arr in keepout_dict.items():
        frac = (np.sum(arr) / len(arr)) * 100
        visibility[name] = frac

    return visibility



def print_visibility_windows(ts, keepout_dict):
    """Print formatted list of observability windows for a single target.
    
    Displays start and end dates for each continuous observability window,
    along with the duration of each window. Only processes single-target
    keepout dictionaries; returns silently for multiple targets.
    
    Args:
        ts (astropy.time.Time): Array of time samples.
        keepout_dict (dict): Single-entry dict mapping target name to boolean
            observability array.
    
    Returns:
        None: Prints formatted output directly to console.
    
    Note:
        - If target is always observable, prints single continuous window
        - If target is never observable, prints unavailability message
        - Durations calculated in days with one decimal place precision
    """
    if len(keepout_dict) != 1:
        return

    target_name = list(keepout_dict.keys())[0]
    ko = keepout_dict[target_name]
    transitions = np.where(np.diff(ko.astype(int)) != 0)[0]

    if len(transitions) == 0:
        if ko[0]:
            print(f"\n {target_name} is observable for the entire period:")
            print(f"   {ts[0].iso.split('T')[0]} to {ts[-1].iso.split('T')[0]}")
        else:
            print(f"\n {target_name} is NOT observable at all during this period")
        return

    print(f"\n📅 Visibility Windows for {target_name}:")
    print("=" * 60)

    current_state = ko[0]
    start_idx = 0
    window_num = 0

    for trans_idx in transitions:
        if current_state:
            window_num += 1
            start_date = ts[start_idx].iso.split('T')[0]
            end_date = ts[trans_idx].iso.split('T')[0]
            duration = ts[trans_idx].mjd - ts[start_idx].mjd
            print(f"Window {window_num}: {start_date} to {end_date} ({duration:.1f} days)")

        current_state = not current_state
        start_idx = trans_idx + 1

    if current_state:
        window_num += 1
        start_date = ts[start_idx].iso.split('T')[0]
        end_date = ts[-1].iso.split('T')[0]
        duration = ts[-1].mjd - ts[start_idx].mjd
        print(f"Window {window_num}: {start_date} to {end_date} ({duration:.1f} days)")

    print("=" * 60)


def generate_csv(ts, keepout_dict, solar_angles_dict, pitch_dict, visibility_dict):
    """Generate CSV file containing complete observability analysis data.
    
    Creates a CSV-formatted string with metadata header, visibility summary,
    and time-series data for all targets. Suitable for further analysis in
    spreadsheet applications or data science tools.
    
    Args:
        ts (astropy.time.Time): Array of time samples.
        keepout_dict (dict): Maps target names to boolean observability arrays.
        solar_angles_dict (dict): Maps target names to solar angle arrays.
        pitch_dict (dict): Maps target names to pitch angle arrays.
        visibility_dict (dict): Maps target names to visibility percentages.
    
    Returns:
        str: CSV-formatted string with the following structure:
            - Header lines (starting with #) containing:
                * Start and end dates
                * Time step size
                * Visibility percentages for each target
            - Column headers
            - Data rows with one row per time step
    
    CSV Columns:
        - Date: ISO 8601 timestamp
        - Days_from_Start: Decimal days since start_date
        - For each target:
            * {Target}_Solar_Angle_deg: Solar angle in degrees (4 decimal places)
            * {Target}_Pitch_Angle_deg: Pitch angle in degrees (4 decimal places)
            * {Target}_Observable: Binary flag (1 = observable, 0 = not observable)
    
    """
    output = StringIO()
    writer = csv.writer(output)
    targets = list(keepout_dict.keys())

    writer.writerow(['# Roman Space Telescope Keepout Analysis'])
    writer.writerow([f'# Start Date: {ts[0].to_value("isot")}'])
    writer.writerow([f'# End Date: {ts[-1].to_value("isot")}'])
    timestep = (ts[1].mjd - ts[0].mjd) if len(ts) > 1 else 1
    writer.writerow([f'# Time Step (days): {timestep}'])
    writer.writerow(['# Visibility Summary (% of time observable)'])

    for target, vis_pct in visibility_dict.items():
        writer.writerow([f'# {target}: {vis_pct:.2f}%'])

    headers = ['Date', 'Days_from_Start']
    for target in targets:
        headers.extend([
            f'{target}_Solar_Angle_deg',
            f'{target}_Pitch_Angle_deg',
            f'{target}_Observable'
        ])
    writer.writerow(headers)

    for i, time in enumerate(ts):
        days_from_start = time.mjd - ts[0].mjd
        date_str = time.to_value("isot")
        row = [date_str, f'{days_from_start:.6f}']

        for target in targets:
            solar_ang = solar_angles_dict[target][i].to(u.deg).value
            pitch_ang = pitch_dict[target][i].to(u.deg).value
            observable = int(keepout_dict[target][i])

            row.extend([
                f'{solar_ang:.4f}',
                f'{pitch_ang:.4f}',
                observable
            ])

        writer.writerow(row)

    return output.getvalue()


def create_download_button(csv_string, filename='roman_keepout_data.csv'):
    """Create browser download button for CSV data.
    
    Generates an HTML download link that triggers the browser's native download
    dialog when clicked. The CSV data is base64-encoded and embedded directly
    in the link's data URI.

    Args:
        csv_string (str): CSV-formatted data string to be downloaded.
        filename (str, optional): Desired filename for the downloaded file.
            Defaults to 'roman_keepout_data.csv'. The .csv extension is
            automatically added if not present.

    Returns:
        IPython.display.HTML: HTML object containing a styled download button
            that triggers browser download when clicked.
    
    Note:
        The button uses HTML5's download attribute with a data URI. This works
        in all modern browsers (Chrome, Firefox, Edge, Safari) and allows users
        to save the file anywhere on their system.
    """
    if not filename.endswith('.csv'):
        filename += '.csv'
    
    b64_data = base64.b64encode(csv_string.encode('utf-8')).decode('utf-8')
    
    html = f"""
    <a download="{filename}" 
       href="data:text/csv;base64,{b64_data}"
       style="display:inline-block;padding:12px 24px;background:#28a745;
              color:white;text-decoration:none;border-radius:5px;
              font-weight:bold;font-size:16px;cursor:pointer;
              box-shadow:0 2px 4px rgba(0,0,0,0.2);">
         Download {filename}
    </a>
    """
    
    return HTML(html)


def launch_ui():
    """Launch interactive Jupyter widget interface for Roman keepout analysis.

    Creates and displays a comprehensive user interface for analyzing Roman Space
    Telescope target observability. The interface includes:

    - Target input field for SIMBAD-recognized astronomical objects
    - Date range and time step configuration
    - Preset target collections (bright stars, exoplanet hosts)
    - Collapsible help documentation
    - Real-time visualization generation
    - CSV download functionality

    The UI generates three plots when "Generate Maps & Plots" is clicked:
    1. Keepout map showing observability windows for all targets
    2. Solar angle evolution over time (with transition markers for single targets)
    3. Pitch angle requirements over time

    Additionally displays visibility statistics showing the percentage of time
    each target is observable during the specified observation period.

    Interface Components:
        - **Target Input**: Multi-line text area for entering target names (one per line)
        - **Preset Buttons**: Quick-load curated lists of 40 bright stars or 25 exoplanet hosts
        - **Start Date**: ISO 8601 formatted date/time (YYYY-MM-DDTHH:MM:SS)
        - **Days Slider**: Analysis duration (1-730 days, up to 2 years)
        - **Time Step Slider**: Calculation interval (1-10 days)
        - **CSV Filename**: Customizable output filename
        - **Generate Maps & Plots**: Creates visualizations and downloadable CSV
        - **Generate CSV Only**: Skips plotting for faster data export
        - **Help Toggle**: Shows/hides detailed usage instructions

    Requirements:
        - Must be run in a Jupyter notebook environment
        - Requires ipywidgets for the interface
        - Requires matplotlib for plotting
        - Requires astropy and astroquery for astronomical calculations
        - Requires roman_pointing library for telescope angle calculations

    Returns:
        None: Displays the widget interface directly in the notebook output.

    Note:
        - Single-target analysis provides detailed window transition dates
        - Multi-target analysis shows comparative observability across all targets
        - SIMBAD queries may take several seconds for multiple targets
        - Processing time scales with (number of targets) × (days / time_step)
    """
    target_input = widgets.Textarea(
        value="47 UMa\n14 Her\nGalactic Bulge",
        layout=widgets.Layout(width="500px", height="100px"),
        description="Targets:",
    )

    start_date_input = widgets.Text(
        value="2027-01-01T00:00:00",
        layout=widgets.Layout(width="400px"),
        description="Start Date:",
    )

    days_input = widgets.IntSlider(
        value=365, min=1, max=730,
        description="Days:",
        layout=widgets.Layout(width="400px"),
    )

    time_step_input = widgets.IntSlider(
        value=1, min=1, max=10,
        description="Time Step:",
        layout=widgets.Layout(width="400px"),
    )

    run_button = widgets.Button(
        description="Generate Maps & Plots",
        button_style="primary",
        layout=widgets.Layout(width="200px"),
    )

    csv_button = widgets.Button(
        description="Generate CSV Only",
        button_style="success",
        layout=widgets.Layout(width="200px"),
    )

    csv_filename_input = widgets.Text(
        value="roman_keepout_data.csv",
        layout=widgets.Layout(width="400px"),
        description="CSV Filename:",
    )

    output = widgets.Output()
    download_link_widget = widgets.VBox()

    # Help toggle and panel
    help_toggle = widgets.ToggleButton(
        value=False,
        description=" Show Instructions",
        button_style="info",
        icon="question",
    )

    help_box = widgets.HTML(
        """
    <div style="font-family: Arial, sans-serif; background: #f0f8ff; border: 2px solid #4a90e2;
    border-radius: 8px; padding: 15px; margin-top: 5px; color: #000000;">
    
    <h3 style="margin-top:0; color: #000000;">📖 How to Use This Tool</h3>
    
    <div style="margin-bottom: 15px;">
        <h4 style="color: #000000; margin-bottom: 5px;">🎯 Target Input</h4>
        <p style="margin: 5px 0;">Enter astronomical object names that are recognized by the SIMBAD database. 
        Each target should be on a separate line. The tool will look up coordinates automatically.</p>
        <p style="margin: 5px 0; font-style: italic;">Examples of valid target names:</p>
        <pre style="background: rgba(0,0,0,0.05); padding: 8px; border-radius: 4px; margin: 5px 0;">Proxima Cen
Sirius
Betelgeuse
HD 209458
* 51 Peg</pre>
        <p style="margin: 5px 0;"><strong>Special case:</strong> To include the Galactic Bulge, type "Galactic Bulge" 
        or just "bulge" on a line. This uses pre-defined coordinates.</p>
        <p style="margin: 5px 0;"><strong>Tip:</strong> Use the preset buttons below for quick access to 
        curated target lists!</p>
    </div>

    <div style="margin-bottom: 15px;">
        <h4 style="color: #000000; margin-bottom: 5px;">📅 Date and Time Settings</h4>
        <p style="margin: 5px 0;"><strong>Start Date:</strong> Must be in ISO 8601 format: YYYY-MM-DDTHH:MM:SS</p>
        <p style="margin: 5px 0; padding-left: 20px;">Example: <code>2027-01-01T00:00:00</code> (January 1, 2027 at midnight UTC)</p>
        
        <p style="margin: 5px 0;"><strong>Days:</strong> How many days into the future to analyze (1-730 days, up to 2 years)</p>
        <p style="margin: 5px 0; padding-left: 20px;">• 365 days = 1 year of visibility analysis</p>
        <p style="margin: 5px 0; padding-left: 20px;">• 730 days = 2 years for long-term planning</p>
        
        <p style="margin: 5px 0;"><strong>Time Step:</strong> Time interval between calculations (1-10 days)</p>
        <p style="margin: 5px 0; padding-left: 20px;">• Smaller values (1-2 days) = More precise, but slower</p>
        <p style="margin: 5px 0; padding-left: 20px;">• Larger values (5-10 days) = Faster, but less detailed</p>
        <p style="margin: 5px 0; padding-left: 20px;">• Recommended: 1 day for accurate results</p>
    </div>

    <div style="margin-bottom: 15px;">
        <h4 style="color: #000000; margin-bottom: 5px;">📊 Output Options</h4>
        <p style="margin: 5px 0;"><strong>Generate Maps & Plots:</strong> Creates three visualizations plus downloadable CSV:</p>
        <p style="margin: 5px 0; padding-left: 20px;">1. <strong>Keepout Map</strong> - Green = observable, Black = not observable</p>
        <p style="margin: 5px 0; padding-left: 20px;">2. <strong>Solar Angle Plot</strong> - Shows why targets enter/exit keepout zones</p>
        <p style="margin: 5px 0; padding-left: 20px;">3. <strong>Pitch Angle Plot</strong> - Spacecraft orientation needed for each target</p>
        
        <p style="margin: 5px 0;"><strong>Generate CSV Only:</strong> Skip visualizations and go straight to data export (faster)</p>
        
        <p style="margin: 5px 0;"><strong>CSV Filename:</strong> Customize your download filename (automatically adds .csv if needed)</p>
    </div>

    <div style="margin-bottom: 15px;">
        <h4 style="color: #000000; margin-bottom: 5px;">🚀 Understanding Roman's Constraints</h4>
        <p style="margin: 5px 0;"><strong>Solar Angle Constraints:</strong> Roman can only observe targets when the angle 
        between the target and the Sun is between <strong>54° and 126°</strong>.</p>
        <p style="margin: 5px 0; padding-left: 20px;">• <strong>Less than 54°:</strong> Too close to Sun (thermal issues, stray light)</p>
        <p style="margin: 5px 0; padding-left: 20px;">• <strong>Greater than 126°:</strong> Solar panels can't get enough power</p>
        <p style="margin: 5px 0; padding-left: 20px;">• <strong>Green zones in plots:</strong> Target is observable</p>
        <p style="margin: 5px 0; padding-left: 20px;">• <strong>Hatched zones in plots:</strong> Keepout regions</p>
    </div>

    <div style="margin-bottom: 15px;">
        <h4 style="color: #000000; margin-bottom: 5px;">💾 CSV Data Format</h4>
        <p style="margin: 5px 0;">The exported CSV file contains:</p>
        <p style="margin: 5px 0; padding-left: 20px;">• <strong>Metadata header</strong> (lines starting with #) with analysis summary</p>
        <p style="margin: 5px 0; padding-left: 20px;">• <strong>Visibility percentages</strong> showing how often each target is observable</p>
        <p style="margin: 5px 0; padding-left: 20px;">• <strong>Time-series data</strong> with columns for each target:</p>
        <p style="margin: 5px 0; padding-left: 40px;">- Date (ISO format)</p>
        <p style="margin: 5px 0; padding-left: 40px;">- Days_from_Start</p>
        <p style="margin: 5px 0; padding-left: 40px;">- Solar_Angle_deg (angle from Sun)</p>
        <p style="margin: 5px 0; padding-left: 40px;">- Pitch_Angle_deg (spacecraft orientation)</p>
        <p style="margin: 5px 0; padding-left: 40px;">- Observable (1 = yes, 0 = no)</p>
    </div>

    <div style="margin-bottom: 10px;">
        <h4 style="color: #000000; margin-bottom: 5px;">⚠️ Important Notes</h4>
        <p style="margin: 5px 0;">• Target names must <strong>exactly match SIMBAD</strong> (case-sensitive for some objects)</p>
        <p style="margin: 5px 0;">• If a target isn't found, check the spelling or try alternative names</p>
        <p style="margin: 5px 0;">• For <strong>single targets</strong>, you'll see detailed visibility windows with exact transition dates</p>
        <p style="margin: 5px 0;">• For <strong>multiple targets</strong>, you'll see a comparison map showing all targets together</p>
        <p style="margin: 5px 0;">• Analysis assumes observations from Sun-Earth L2 Lagrange point (Roman's orbit)</p>
        <p style="margin: 5px 0;">• Processing time increases with more targets and smaller time steps</p>
    </div>

    """
    )

    help_panel = widgets.VBox([help_box])
    help_panel.layout.display = "none"

    def toggle_help(change):
        """Toggle visibility of help documentation panel."""
        help_panel.layout.display = "block" if change["new"] else "none"
        help_toggle.description = " Hide Instructions" if change["new"] else " Show Instructions"

    help_toggle.observe(toggle_help, "value")

    # Preset buttons
    preset_stars = widgets.Button(
        description="⭐ Reference Stars", button_style="primary"
    )
    preset_exoplanets = widgets.Button(
        description="🪐 Exoplanet Hosts", button_style="warning"
    )

    def load_stars(_):
        target_input.value = "kap Ori\nbet CMa\nbet Leo\nbet Car\neps Ori\ndel Cas\nalf Ara\neta Cen\nrho Pup\neta UMa\ngam Ori\nalf Cyg\nbet Lup\nalf Lep\ndel Leo\nbet UMa\neta CMa\nalf Cep\ngam TrA\neps CMa\nalf Col\nbet TrA\nalf Gru\nbet CMi\nzet Pup\nbet Cas\nzet Oph\ndel Cru\nalf Peg\nalf Hyi\neta Tau\niot Car\nbet Tau\ndel Crv\neps UMa\nbet Eri\nalf02 CVn\nbet Lib\nzet Aql\ngam Peg"

    def load_exoplanets(_):
        target_input.value = "* 14 Her\n* 23 Lib\n* 47 UMa\n* alf Cen A\n* bet Gem\n* bet Pic\n* e Eri\n* eps Eri\n* gam Cep\n* mu. Ara\n* pi. Men\n* psi01 Dra B\n* rho01 Cnc\n* tau Cet\n* ups And\nHD 100546\nHD 114613\nHD 142\nHD 154345\nHD 190360\nHD 192310\nHD 217107\nHD 219077\nHD 219134\nHD 30562"

    preset_stars.on_click(load_stars)
    preset_exoplanets.on_click(load_exoplanets)

    preset_box = widgets.HBox([preset_stars, preset_exoplanets])

    def on_csv_only_clicked(_):
        with output:
            clear_output(wait=True)
            try:
                target_names = [t.strip() for t in target_input.value.split("\n") if t.strip()]
                print(f"Processing {len(target_names)} targets...")

                coords = get_target_coords(target_names)
                print(f" {len(coords)} found in SIMBAD")

                if not coords:
                    print(" No valid targets found.")
                    download_link_widget.children = []
                    return

                ts, keepout, solar_angles = compute_keepout(
                    coords, start_date_input.value, days_input.value, time_step_input.value
                )

                visibility = compute_visibility_fraction(keepout)
                print("\n📊 Visibility (% of time observable):")
                for name, frac in visibility.items():
                    print(f"   {name}: {frac:.1f}%")

                if len(coords) == 1:
                    print_visibility_windows(ts, keepout)

                pitch_dict = {}
                for name, coord in coords.items():
                    _, _, _, pitch = compute_roman_angles(
                        coord, start_date_input.value, days_input.value, time_step_input.value
                    )
                    pitch_dict[name] = pitch

                csv_string = generate_csv(ts, keepout, solar_angles, pitch_dict, visibility)
                filename = csv_filename_input.value.strip() or "roman_keepout_data.csv"

                download_button = create_download_button(csv_string, filename)
                
                button_output = widgets.Output()
                with button_output:
                    display(download_button)
                
                download_link_widget.children = [button_output]

            except Exception as e:
                print(f" Error: {e}")
                import traceback
                traceback.print_exc()
                download_link_widget.children = []

    def on_run_clicked(_):
        with output:
            clear_output(wait=True)
            try:
                target_names = [t.strip() for t in target_input.value.split("\n") if t.strip()]
                print(f"Processing {len(target_names)} targets...")

                coords = get_target_coords(target_names)
                print(f" {len(coords)} found in SIMBAD")

                if not coords:
                    print(" No valid targets found.")
                    download_link_widget.children = []
                    return

                ts, keepout, solar_angles = compute_keepout(
                    coords, start_date_input.value, days_input.value, time_step_input.value
                )

                visibility = compute_visibility_fraction(keepout)
                print("\n📊 Visibility (% of time observable):")
                for name, frac in visibility.items():
                    print(f"   {name}: {frac:.1f}%")

                if len(coords) == 1:
                    print_visibility_windows(ts, keepout)

                pitch_dict = {}
                for name, coord in coords.items():
                    _, _, _, pitch = compute_roman_angles(
                        coord, start_date_input.value, days_input.value, time_step_input.value
                    )
                    pitch_dict[name] = pitch

                plot_keepout(keepout, ts)
                plt.show()

                if len(coords) == 1:
                    plot_solar_angle_with_windows(ts, solar_angles, keepout)
                else:
                    plot_solar_angle(ts, solar_angles)
                plt.show()

                plot_pitch(ts, pitch_dict)
                plt.show()

                csv_string = generate_csv(ts, keepout, solar_angles, pitch_dict, visibility)
                filename = csv_filename_input.value.strip() or "roman_keepout_data.csv"

                download_button = create_download_button(csv_string, filename)
                
                button_output = widgets.Output()
                with button_output:
                    display(download_button)
                
                download_link_widget.children = [button_output]

            except Exception as e:
                print(f" Error: {e}")
                import traceback
                traceback.print_exc()
                download_link_widget.children = []

    run_button.on_click(on_run_clicked)
    csv_button.on_click(on_csv_only_clicked)

    display(
        widgets.VBox([
            widgets.HTML("<h2>🔭 Roman Space Telescope Keepout Map Generator 🔭</h2>"),
            help_toggle,
            help_panel,
            widgets.HTML("<b>Target Input</b>"),
            preset_box,
            target_input,
            start_date_input,
            days_input,
            time_step_input,
            widgets.HTML("<hr><b>Output Options</b>"),
            csv_filename_input,
            widgets.HBox([run_button, csv_button]),
            output,
            widgets.HTML("<hr><b>Download CSV:</b>"),
            download_link_widget,
        ])
    )