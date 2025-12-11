from roman_pointing.roman_pointing import (
    calcRomanAngles,
    getL2Positions,
)
import astropy.units as u
from astropy.time import Time
from astroquery.simbad import Simbad
from astropy.coordinates import (
    SkyCoord,
    Distance,
    BarycentricMeanEcliptic,
)
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from IPython.display import clear_output


def get_target_coords(target_names):
    """Query SIMBAD for astronomical target coordinates and proper motions.

    Retrieves celestial coordinates, parallax, proper motion, and radial velocity
    data from SIMBAD database for specified astronomical objects. Transforms
    coordinates to Barycentric Mean Ecliptic frame for Roman Space Telescope
    pointing calculations.

    Args:
        target_names (list of str):
            List of astronomical object names recognizable by SIMBAD (e.g.,
            'Proxima Cen', 'Sirius', 'Betelgeuse').

    Returns:
        dict:
            Dictionary mapping target names to astropy SkyCoord objects in
            BarycentricMeanEcliptic frame. Targets not found in SIMBAD are
            excluded from the returned dictionary.
    """
    simbad = Simbad()
    simbad.add_votable_fields("pmra", "pmdec", "plx_value", "rvz_radvel")

    coords = {}

    for name in target_names:
        if "bulge" in name.lower():
            coords[name] = SkyCoord(
                266.41681662,
                -29.00782497,
                unit=(u.deg, u.deg),
                frame="icrs",
                distance=8277 * u.pc,
                pm_ra_cosdec=0 * u.mas / u.year,
                pm_dec=0 * u.mas / u.year,
                radial_velocity=0 * u.km / u.s,
                equinox="J2000",
                obstime="J2000",
            ).transform_to(BarycentricMeanEcliptic)
            continue

        res = simbad.query_object(name)

        if len(res) == 0:
            print(f"SIMBAD could not find {name}. Skipping.")
            continue

        c_icrs = SkyCoord(
            res["ra"].value.data[0],
            res["dec"].value.data[0],
            unit=(res["ra"].unit, res["dec"].unit),
            frame="icrs",
            distance=Distance(
                parallax=res["plx_value"].value.data[0] * res["plx_value"].unit
            ),
            pm_ra_cosdec=res["pmra"].value.data[0] * res["pmra"].unit,
            pm_dec=res["pmdec"].value.data[0] * res["pmdec"].unit,
            radial_velocity=res["rvz_radvel"].value.data[0] * res["rvz_radvel"].unit,
            equinox="J2000",
            obstime="J2000",
        ).transform_to(BarycentricMeanEcliptic)

        coords[name] = c_icrs

    return coords


def compute_roman_angles(coord, start_date, days, time_step):
    """Calculate Roman Space Telescope pointing angles for a target over time.

    Computes the solar angle, yaw, and pitch angles required for Roman to observe
    a given celestial coordinate over a specified time period, accounting for the
    telescope's position at the Sun-Earth L2 Lagrange point.

    Args:
        coord (astropy.coordinates.SkyCoord):
            Target celestial coordinates in BarycentricMeanEcliptic frame
        start_date (str):
            Start date in ISO format (e.g., '2027-01-01T00:00:00')
        days (int or float):
            Duration of observation window in days
        time_step (int or float):
            Time interval between calculations in days

    Returns:
        tuple:
            ts (astropy.time.Time): Array of time values
            sun_ang (astropy.units.Quantity): Solar angles in degrees
            yaw (astropy.units.Quantity): Yaw angles in degrees
            pitch (astropy.units.Quantity): Pitch angles in degrees
    """
    t0 = Time(start_date, format="isot", scale="utc")
    ts = t0 + np.arange(0, days, time_step) * u.d

    sun_ang, yaw, pitch, BCI = calcRomanAngles(coord, ts, getL2Positions(ts))

    return ts, sun_ang, yaw, pitch


def compute_keepout(coords_dict, start_date, days, time_step, min_sun=54, max_sun=126):
    """Determine observability windows for multiple targets based on solar angle
    constraints.

    Calculates when targets are observable by Roman Space Telescope based on solar
    exclusion angle limits. The allowed solar angle range to avoid thermal and stray
    light issues while keeping the solar panels properly oriented is 54-126 degrees.

    Args:
        coords_dict (dict):
            Dictionary mapping target names (str) to SkyCoord objects
        start_date (str):
            Start date in ISO format (e.g., '2027-01-01T00:00:00')
        days (int or float):
            Duration of observation window in days
        time_step (int or float):
            Time interval between calculations in days
        min_sun (int or float, optional):
            Minimum allowed solar angle in degrees. Default: 54
        max_sun (int or float, optional):
            Maximum allowed solar angle in degrees. Default: 126

    Returns:
        tuple:
            ts_global (astropy.time.Time): Array of time values
            keepout (dict): Dictionary mapping target names to boolean arrays
                indicating observability (True = observable, False = in keepout zone)
            solar_angles (dict): Dictionary mapping target names to solar angle
                arrays in degrees
    """
    solar_angles = {}
    keepout = {}
    ts_global = None

    for name, coord in coords_dict.items():
        ts, sun_ang, yaw, pitch = compute_roman_angles(
            coord, start_date, days, time_step
        )

        solar_angles[name] = sun_ang
        keepout[name] = (sun_ang > min_sun * u.deg) & (sun_ang < max_sun * u.deg)

        if ts_global is None:
            ts_global = ts

    return ts_global, keepout, solar_angles


def plot_solar_angle(ts, solar_angles_dict):
    """Generate a plot showing solar angles vs time for multiple targets.

    Creates a line plot displaying how the solar angle changes over time for each
    target, with shaded regions indicating the keepout zones (< 54° and > 126°)
    where observations are not permitted.

    Args:
        ts (astropy.time.Time):
            Array of time values
        solar_angles_dict (dict):
            Dictionary mapping target names (str) to solar angle arrays
            (astropy.units.Quantity)

    Returns:
        tuple:
            fig (matplotlib.figure.Figure): The figure object
            ax (matplotlib.axes.Axes): The axes object
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for name, sun in solar_angles_dict.items():
        ax.plot(range(len(ts)), sun.to(u.deg), label=name)

    ax.set_xlabel(f"Time after {ts[0].value} (days)")
    ax.set_ylabel("Solar Angle (deg)")

    # Solar keepout boundaries
    ax.plot([0, len(ts)], [54, 54], "k--")
    ax.plot([0, len(ts)], [126, 126], "k--")

    ax.fill_between(
        [0, len(ts)], [54, 54], [0, 0], hatch="/", color="none", edgecolor="k"
    )
    ax.fill_between(
        [0, len(ts)], [126, 126], [180, 180], hatch="\\", color="none", edgecolor="k"
    )

    ax.set_title("Solar Angle vs Time")
    # Move legend outside
    ax.legend(bbox_to_anchor=(1.15, 1), loc="upper left")

    # Make room for legend
    fig.subplots_adjust(right=0.8)

    return fig, ax


def plot_pitch(ts, pitch_dict):
    """Generate a plot showing pitch angles vs time for multiple targets.

    Creates a line plot displaying the spacecraft pitch angle required to observe
    each target over time. Pitch angle represents the rotation about the telescope's
    horizontal axis.

    Args:
        ts (astropy.time.Time):
            Array of time values
        pitch_dict (dict):
            Dictionary mapping target names (str) to pitch angle arrays
            (astropy.units.Quantity)

    Returns:
        tuple:
            fig (matplotlib.figure.Figure): The figure object
            ax (matplotlib.axes.Axes): The axes object
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for name, pitch in pitch_dict.items():
        ax.plot(range(len(ts)), pitch.to(u.deg), label=name)

    ax.set_xlabel(f"Time after {ts[0].value} (days)")
    ax.set_ylabel("Pitch Angle (deg)")
    ax.set_title("Pitch Angle vs Time")
    ax.legend(bbox_to_anchor=(1.15, 1), loc="upper left", borderaxespad=0)
    fig.subplots_adjust(right=0.8)
    return fig, ax


def plot_keepout(keepout_dict, ts):
    """Create a visibility map showing when targets are observable.

    Generates a color-coded heatmap displaying observability windows for multiple
    targets over time. Green indicates the target is within the allowed solar angle
    range (observable), while black indicates the target is in a keepout zone.

    Args:
        keepout_dict (dict):
            Dictionary mapping target names (str) to boolean arrays indicating
            observability (True = observable, False = in keepout)
        ts (astropy.time.Time):
            Array of time values corresponding to the keepout data

    Returns:
        tuple:
            fig (matplotlib.figure.Figure): The figure object
            ax (matplotlib.axes.Axes): The axes object
    """
    names = list(keepout_dict.keys())
    M = len(names)

    # Handle single target case properly
    if M == 1:
        komap = keepout_dict[names[0]].reshape(1, -1)
    else:
        komap = np.vstack([keepout_dict[n] for n in names])

    # Convert boolean to int for plotting
    komap_int = komap.astype(int)

    # Create figure - make it taller for single targets
    fig_height = 4 if M == 1 else max(3, 1.3 * M + 1)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    cmap = matplotlib.colors.ListedColormap(["black", "green"])

    # Use pcolormesh - works for both single and multiple targets
    im = ax.pcolormesh(
        np.arange(komap.shape[1] + 1),
        np.arange(M + 1),
        komap_int,
        cmap=cmap,
        shading="flat",
    )

    # Set y-ticks at row centers
    ax.set_yticks(np.arange(M) + 0.5)
    ax.set_yticklabels(names)
    ax.set_ylim(0, M)
    ax.set_xlim(0, komap.shape[1])
    ax.set_xlabel("Time Index")
    ax.set_ylabel("Target")

    cbar = plt.colorbar(im, ticks=[0.25, 0.75], ax=ax)
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
        keepout_dict (dict):
            Dictionary mapping target names (str) to boolean arrays where
            True indicates the target is observable and False indicates it
            is in a keepout zone.

    Returns:
        dict:
            Dictionary mapping target names (str) to visibility percentages
            (float), representing the fraction of time the target is observable
            expressed as a percentage (0-100).
    """
    visibility = {}
    for name, arr in keepout_dict.items():
        frac = (np.sum(arr) / len(arr)) * 100
        visibility[name] = frac
    return visibility


def launch_ui():
    """Launch interactive Jupyter widget interface for Roman keepout analysis.

    Creates and displays a comprehensive user interface for analyzing Roman Space
    Telescope target observability. The interface includes:

    - Target input field for SIMBAD-recognized astronomical objects
    - Date range and time step configuration
    - Preset target collections (bright stars, exoplanet hosts, galaxies)
    - Collapsible help documentation
    - Real-time visualization generation

    The UI generates three plots when run:
    1. Keepout map showing observability windows for all targets
    2. Solar angle evolution over time
    3. Pitch angle requirements over time

    Additionally displays visibility statistics showing the percentage of time
    each target is observable during the specified observation period.

    Requirements:
        - Must be run in a Jupyter notebook environment
        - Requires ipywidgets for the interface
        - Requires matplotlib for plotting
        - Requires astropy and astroquery for astronomical calculations

    Returns:
        None. Displays the widget interface directly in the notebook output.

    Note:
        The function uses IPython.display.display() to render the widget interface.
        All event handling and plotting occurs within the displayed interface.
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
        value=365,
        min=1,
        max=730,
        description="Days:",
        layout=widgets.Layout(width="400px"),
    )

    time_step_input = widgets.IntSlider(
        value=1,
        min=1,
        max=10,
        description="Time Step:",
        layout=widgets.Layout(width="400px"),
    )

    run_button = widgets.Button(
        description="Generate Maps & Plots",
        button_style="primary",
        layout=widgets.Layout(width="200px"),
    )

    output = widgets.Output()

    help_toggle = widgets.ToggleButton(
        value=False,
        description=" Input Instructions",
        button_style="info",
        icon="question",
    )

    help_box = widgets.HTML(
        """
    <div style="font-family: Arial; background: #eef6ff; border: 1px solid #bcd4ff;
    border-radius: 6px; padding: 10px; margin-top: 2px;">
    <h4 style="margin-top:2px;">Input Guidelines</h4>
    <b>Targets:</b> One SIMBAD-recognized name per line
    <pre style="margin-top:-6px;">
    Proxima Cen
    Sirius
    Betelgeuse</pre>

    <b>Start Date:</b> ISO format (YYYY-MM-DDTHH:MM:SS)
    <pre style="margin-top:-6px;">2027-01-01T00:00:00</pre>

    <b>Days:</b> 1–730
    <b>Time Step:</b> 1–10 days

    <hr>
    <b>Notes:</b>
    <ul style="margin-top:-6px; margin-bottom:4px;">
    <li>Angles less than 54 or greater than 126 degrees are keep-out zones</li>
    <li>Names must match SIMBAD exactly</li>
    <li>To include the location of the galactic bulge, add 'bulge' or 'galatcic bulge' to your list of targets.</li>
    </ul>
    </div>
    """
    )

    help_panel = widgets.VBox([help_box])
    help_panel.layout.display = "none"

    def toggle_help(change):
        """Toggle visibility of help documentation panel.

        Args:
            change (dict): Widget value change event containing 'new' key
                with boolean value indicating toggle state.
        """
        help_panel.layout.display = "block" if change["new"] else "none"

    help_toggle.observe(toggle_help, "value")

    # Preset buttons
    preset_stars = widgets.Button(
        description="⭐ Reference Stars", button_style="primary"
    )
    preset_exoplanets = widgets.Button(
        description="🪐 Exoplanet Hosts", button_style="warning"
    )
    # preset_galaxies = widgets.Button(description="🌌 Galaxies")

    def load_stars(_):
        """Load preset list of reference stars into target input field."""
        target_input.value = "kap Ori\nbet CMa\nbet Leo\nbet Car\neps Ori\ndel Cas\nalf Ara\neta Cen\nrho Pup\neta UMa\ngam Ori\nalf Cyg\nbet Lup\nalf Lep\ndel Leo\nbet UMa\neta CMa\nalf Cep\ngam TrA\neps CMa\nalf Col\nbet TrA\nalf Gru\nbet CMi\nzet Pup\nbet Cas\nzet Oph\ndel Cru\nalf Peg\nalf Hyi\neta Tau\niot Car\nbet Tau\ndel Crv\neps UMa\nbet Eri\nalf02 CVn\nbet Lib\nzet Aql\ngam Peg"

    def load_exoplanets(_):
        """Load preset list of exoplanet host stars into target input field."""
        target_input.value = "* 14 Her\n* 23 Lib\n* 47 UMa\n* alf Cen A\n* bet Gem\n* bet Pic\n* e Eri\n* eps Eri\n* gam Cep\n* mu. Ara\n* pi. Men\n* psi01 Dra B\n* rho01 Cnc\n* tau Cet\n* ups And\nHD 100546\nHD 114613\nHD 142\nHD 154345\nHD 190360\nHD 192310\nHD 217107\nHD 219077\nHD 219134\nHD 30562"

    def load_galaxies(_):
        """Load preset list of nearby galaxies into target input field."""
        target_input.value = "M31\nM81\nM87\nNGC 1300\nSombrero"

    preset_stars.on_click(load_stars)
    preset_exoplanets.on_click(load_exoplanets)
    # preset_galaxies.on_click(load_galaxies)

    preset_box = widgets.HBox([preset_stars, preset_exoplanets])  # , preset_galaxies])

    def on_run_clicked(_):
        """Process targets and generate keepout maps and plots.

        Main event handler that:
        1. Parses target names from input
        2. Queries SIMBAD for coordinates
        3. Computes keepout periods and solar/pitch angles
        4. Calculates visibility statistics
        5. Generates and displays three plots

        Args:
            _: Button click event
        """
        with output:
            clear_output(wait=True)

            try:
                target_names = [
                    t.strip() for t in target_input.value.split("\n") if t.strip()
                ]
                print(f"Processing {len(target_names)} targets...")

                coords = get_target_coords(target_names)
                print(f"✓ {len(coords)} found in SIMBAD")

                if not coords:
                    print("⚠ No valid targets found.")
                    return

                ts, keepout, solar_angles = compute_keepout(
                    coords,
                    start_date_input.value,
                    days_input.value,
                    time_step_input.value,
                )

                # Visibility fractions
                visibility = compute_visibility_fraction(keepout)
                print("\n📊 Annual Visibility Access (% of time observable):")
                for name, frac in visibility.items():
                    print(f"   {name}: {frac:.1f}%")

                pitch_dict = {}
                for name, coord in coords.items():
                    _, _, _, pitch = compute_roman_angles(
                        coord,
                        start_date_input.value,
                        days_input.value,
                        time_step_input.value,
                    )
                    pitch_dict[name] = pitch

                fig1, _ = plot_keepout(keepout, ts)
                plt.show()

                fig2, _ = plot_solar_angle(ts, solar_angles)
                plt.show()

                fig3, _ = plot_pitch(ts, pitch_dict)
                plt.show()

            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback

                traceback.print_exc()

    # Attach event handler
    run_button.on_click(on_run_clicked)

    # Display UI
    display(  # noqa
        widgets.VBox(
            [
                widgets.HTML(
                    "<h2>🔭 Roman Space Telescope Keepout Map Generator 🔭</h2>"
                ),
                help_toggle,
                help_panel,
                widgets.HTML("<b>Target Input</b>"),
                preset_box,
                target_input,
                start_date_input,
                days_input,
                time_step_input,
                run_button,
                output,
            ]
        )
    )
