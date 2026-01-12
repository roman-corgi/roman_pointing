import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, Distance, BarycentricMeanEcliptic
from astroquery.simbad import Simbad
import ipywidgets as widgets
from IPython.display import display
import traceback

from roman_pointing.roman_pointing import (
    calcRomanAngles,
    getL2Positions,
    getRomanPositionAngle,
    getEXCAMPositionAngle,
    applyRollAngle
)


def get_exoplanet_skycoord(target_name: str) -> SkyCoord:
    """Query SIMBAD for astronomical target coordinates and proper motions.
    
    Retrieves celestial coordinates, parallax, proper motion, and radial velocity
    data from SIMBAD database for specified astronomical object. Transforms
    coordinates to Barycentric Mean Ecliptic frame for Roman Space Telescope
    pointing calculations.
    
    Args:
        target_name (str): Name of the astronomical target as recognized by SIMBAD
            (e.g., '47 UMa', 'HD 189733', 'Proxima Centauri').
    
    Returns:
        astropy.coordinates.SkyCoord: Sky coordinate object in the Barycentric
            Mean Ecliptic frame, including proper motion, parallax, and radial
            velocity information.
    
    Raises:
        ValueError: If the target name is not found in the SIMBAD database.
    
    Note:
        The coordinate system is transformed to Barycentric Mean Ecliptic, which is
        useful for solar system objects and space telescope pointing calculations.
        The query includes right ascension, declination, proper motion in RA and Dec,
        parallax for distance determination, and radial velocity.

    """
    simbad = Simbad()
    simbad.add_votable_fields("pmra", "pmdec", "plx_value", "rvz_radvel")
    
    res = simbad.query_object(target_name)
    if res is None:
        raise ValueError(f"No results found in SIMBAD for '{target_name}'")
    
    sc = SkyCoord(
        ra=res["ra"].value.data[0],
        dec=res["dec"].value.data[0],
        unit=(res["ra"].unit, res["dec"].unit),
        frame="icrs",
        distance=Distance(parallax=res["plx_value"].value.data[0] * res["plx_value"].unit),
        pm_ra_cosdec=res["pmra"].value.data[0] * res["pmra"].unit,
        pm_dec=res["pmdec"].value.data[0] * res["pmdec"].unit,
        radial_velocity=res["rvz_radvel"].value.data[0] * res["rvz_radvel"].unit,
        equinox="J2000",
        obstime="J2000",
        
    )
    
    return sc.transform_to(BarycentricMeanEcliptic)


def calculate_roman_EXCAMangles(target, ts_targ):
    """Compute Roman Observatory pointing angles and position angles.
    
    Calculates the observatory orientation angles (sun angle, yaw, pitch) and
    the position angles for the Roman telescope's Z-axis and EXCAM Y-axis
    given a target position and observation time(s).
    
    Args:
        target (astropy.coordinates.SkyCoord): Target sky coordinates in
            Barycentric Mean Ecliptic frame.
        ts_targ (astropy.time.Time or array-like): Observation time(s) in UTC.
            Can be a single Time object or an array of Time objects for
            multiple epochs.
    
    Returns:
        dict: Dictionary containing:
            - sun_ang (float or ndarray): Solar angle in degrees - angle between
              target and Sun as seen from the observatory.
            - yaw (float or ndarray): Yaw angle in degrees - rotation about the
              Z-axis.
            - pitch (float or ndarray): Pitch angle in degrees - rotation about
              the Y-axis.
            - B_C_I (ndarray): Body-to-inertial rotation matrix (3x3 for single
              time, 3x3xN for multiple times).
            - PA_Z (float or ndarray): Position angle of Roman's +Z axis in
              degrees, measured East of North.
            - PA_EXCAM_Y (float or ndarray): Position angle of EXCAM's +Y axis
              in degrees, measured East of North.
    """
    sun_ang, yaw, pitch, B_C_I = calcRomanAngles(
        target, ts_targ, getL2Positions(ts_targ)
    )
    
    PA_Z = getRomanPositionAngle(B_C_I)
    PA_EXCAM_Y = getEXCAMPositionAngle(B_C_I)

    return {
        "sun_ang": sun_ang.to_value(u.deg),
        "yaw": yaw.to_value(u.deg),
        "pitch": pitch.to_value(u.deg),
        "B_C_I": B_C_I,
        "PA_Z": np.rad2deg(PA_Z),
        "PA_EXCAM_Y": np.rad2deg(PA_EXCAM_Y),
    }


def compute_observation_times(start_time, obs_window=30, num_epochs=1):
    """Compute observation epochs within a specified time window.
    
    Generates a series of observation times evenly distributed across an
    observation window starting from a given time.
    
    Args:
        start_time (str): Start time in ISO format (e.g., '2027-06-01T00:00:00.0').
        obs_window (int or float, optional): Duration of the observation window in
            days. Default is 30 days.
        num_epochs (int, optional): Number of observation epochs to generate within
            the window. Default is 1 (single epoch at start time).
    
    Returns:
        tuple: A tuple containing:
            - days_from_start (ndarray): Array of days from the start time
              (0 to obs_window).
            - ts_targ (astropy.time.Time): Array of Time objects representing
              each observation epoch.
    """
    days_from_start = np.linspace(0, obs_window, num=num_epochs)
    t0 = Time(start_time, format="isot", scale="utc")
    ts_targ = t0 + np.array(days_from_start) * u.day
    
    return days_from_start, ts_targ


def compute_angles(target, ts_targ, roll_angle_deg):
    """Compute EXCAM position angles with optional spacecraft roll.
    
    Calculates the EXCAM Y-axis position angles both with and without an
    applied spacecraft roll angle. This is useful for planning observations
    that require specific instrument orientations.
    
    Args:
        target (astropy.coordinates.SkyCoord): Target sky coordinates in
            Barycentric Mean Ecliptic frame.
        ts_targ (astropy.time.Time or array-like): Observation time(s) in UTC.
        roll_angle_deg (float): Spacecraft roll angle in degrees to apply.
            Positive roll rotates the field of view counter-clockwise as viewed
            from the spacecraft.
    
    Returns:
        dict: Dictionary containing:
            - PA_EXCAM_Y (float or ndarray): Position angle(s) of EXCAM +Y axis
              without roll, in degrees.
            - B_C_I (ndarray): Body-to-inertial rotation matrix without roll.
            - roll_angles (list of astropy.units.Quantity): List of roll angles
              applied (same value for each epoch).
            - PA_EXCAM_Y_plusroll (ndarray): Position angle(s) of EXCAM +Y axis
              with roll applied, in radians.
    
    Note:
        The roll angle allows rotating the instrument field of view to optimize
        positioning of targets or avoid bright stars. The spacecraft can roll
        within certain limits while maintaining solar panel illumination.
        PA_EXCAM_Y is returned in degrees while PA_EXCAM_Y_plusroll is in radians
        to match the output conventions of the underlying functions.

    """
    angles = calculate_roman_EXCAMangles(target, ts_targ)
    PA_EXCAM_Y = angles["PA_EXCAM_Y"]
    B_C_I = angles["B_C_I"]
    
    roll_angles = [roll_angle_deg * u.deg] * len(ts_targ)
    B_C_I_plusroll = applyRollAngle(B_C_I, roll_angles)
    PA_EXCAM_Y_plusroll = getEXCAMPositionAngle(B_C_I_plusroll)
    
    return {
        'PA_EXCAM_Y': PA_EXCAM_Y,
        'B_C_I': B_C_I,
        'roll_angles': roll_angles,
        'PA_EXCAM_Y_plusroll': PA_EXCAM_Y_plusroll
    }


def bowtie_xys(bandnum=3):
    """Calculate bowtie-shaped spectroscopic field of view boundary points.
    
    Computes the x and y coordinates that define the four bowtie-shaped
    apertures used in EXCAM's spectroscopic modes. Each bowtie consists of
    two arcs at different radii.
    
    Args:
        bandnum (int, optional): Spectroscopic band number (2 or 3). Default is 3.
            Band 2: 1.0-1.8 µm
            Band 3: 1.7-3.0 µm
    
    Returns:
        tuple: A tuple containing:
            - xs (ndarray): X-coordinates for the four bowties, shape (4, 2*num_pts+1).
              Units are arcseconds.
            - ys (ndarray): Y-coordinates for the four bowties, shape (4, 2*num_pts+1).
              Units are arcseconds.
    
    Raises:
        ValueError: If bandnum is not 2 or 3.
    
    Note:
        The bowtie apertures are positioned at bowtie 0 centered at 90° (top),
        bowtie 1 at 270° (bottom), bowtie 2 at 30° (upper right), and bowtie 3
        at 210° (lower left). Each bowtie spans 65° in angular extent and has
        inner/outer radii that differ between bands (Band 2: 172.8-524.2 mas inner,
        191.2-579.8 mas outer; Band 3: same radii but optimized for longer wavelengths).
        The coordinates form closed polygons suitable for matplotlib plotting.

    """
    if bandnum not in [2, 3]:
        raise ValueError('Spec band must be 2 or 3')
    
    open_ang = 65 * u.deg
    ctr_ang = np.array([90, 270, 30, 210]) * u.deg
    r_in_mas = [172.8, 191.2]
    r_out_mas = [524.2, 579.8]
    num_pts = 50

    xs = np.zeros([4, num_pts * 2 + 1])
    ys = np.zeros([4, num_pts * 2 + 1])

    bowtie_idx = 0 if bandnum == 2 else 1
        
    for ct in range(len(ctr_ang)):
        arc = np.linspace(
            start=ctr_ang[ct] - open_ang / 2,
            stop=ctr_ang[ct] + open_ang / 2,
            num=num_pts
        )
        thetas = np.hstack([arc, np.flip(arc), arc[0]])
        rads = np.array(
            num_pts * [r_in_mas[bowtie_idx]] +
            num_pts * [r_out_mas[bowtie_idx]] +
            [r_in_mas[bowtie_idx]]
        )

        xs[ct, :] = rads * np.cos(thetas) / 1000
        ys[ct, :] = rads * np.sin(thetas) / 1000

    return xs, ys


def gen_sky_plot(excam_pa_y, my_pa=None, my_rho=None, mode=None):
    """Create a field of view plot showing coordinate system orientations.
    
    Generates a visualization showing the relationship between sky coordinates
    (North/East), Roman observatory axes (+Z/+Y), and EXCAM instrument axes
    (+X/+Y). Optionally includes spectroscopic bowtie apertures and companion
    star positions.
    
    Args:
        excam_pa_y (float or astropy.units.Quantity): Position angle of EXCAM's
            +Y axis in degrees, measured East of North.
        my_pa (astropy.units.Quantity, optional): Position angle of a companion
            object in degrees. If provided along with my_rho, a companion marker
            will be plotted.
        my_rho (float, optional): Separation of companion object in arcseconds.
            Must be provided along with my_pa to plot companion.
        mode (str, optional): Observation mode. If 'SPEC2' or 'SPEC3', the
            corresponding bowtie apertures will be plotted. Default is None
            (no apertures).
    
    Returns:
        matplotlib.axes.Axes: The axes object containing the plot, which can be
            further modified or shown.
    
    Note:
        The plot uses a unit circle representation where red vectors show sky
        coordinates (North up, East left following convention), blue vectors show
        observatory axes (Z-axis and Y-axis in Roman SIAF frame), and black vectors
        show EXCAM instrument axes. The EXCAM +X axis always points right (0°) and
        +Y axis points up (90°) in the plot coordinate system. For spectroscopic
        modes, four bowtie apertures are shown: nominal bowties (darker) as primary
        observing positions and rotated bowties (lighter) as alternative positions
        after 180° rotation.
    

    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Define angles
    obs_z_angle = -60 * u.deg
    obs_y_angle = 30 * u.deg
    north_angle = 90 * u.deg - excam_pa_y
    east_angle = 180 * u.deg - excam_pa_y

    # Vector components
    north_x, north_y = np.cos(north_angle), np.sin(north_angle)
    east_x, east_y = np.cos(east_angle), np.sin(east_angle)
    obsz_x, obsz_y = np.cos(obs_z_angle), np.sin(obs_z_angle)
    obsy_x, obsy_y = np.cos(obs_y_angle), np.sin(obs_y_angle)

    # Plot vectors
    ax.quiver(0, 0, north_x, north_y, angles='xy', scale_units='xy', scale=1,
              color='red', width=0.01, label='North')
    ax.quiver(0, 0, east_x, east_y, angles='xy', scale_units='xy', scale=1,
              color='red', width=0.005, label='East')
    ax.quiver(0, 0, obsz_x, obsz_y, angles='xy', scale_units='xy', scale=1,
              color='blue', width=0.005, label='Observatory +Z')
    ax.quiver(0, 0, obsy_x, obsy_y, angles='xy', scale_units='xy', scale=1,
              color='blue', width=0.005, label='Observatory +Y')
    ax.quiver(0, 0, 1.1, 0, angles='xy', scale_units='xy', scale=1,
              color='black', width=0.005, label='EXCAM +X')
    ax.quiver(0, 0, 0, 1.1, angles='xy', scale_units='xy', scale=1,
              color='black', width=0.005, label='EXCAM +Y')

    # Plot FOVs if mode specified
    if mode is not None and mode.upper()[:4] == 'SPEC':
        spec_xs, spec_ys = bowtie_xys(bandnum=int(mode[-1]))
        
        for ct in range(4):
            color = 'gray' if ct < 2 else 'lightgray'
            label = f"{'nominal' if ct < 2 else 'rotated'} bowtie B{mode[-1]}"
            plt.plot(spec_xs[ct], spec_ys[ct], color=color, label=label)
        
        # Labels
        ax.text(
            spec_xs[0, int(spec_xs.shape[1] / 2)] * 1.1,
            spec_ys[0, int(spec_xs.shape[1] / 2)] * 1.1,
            f'nominal\nbowtie B{mode[-1]}',
            color='gray', ha='right', va='center',
            fontsize=12, fontweight='bold'
        )
        ax.text(
            spec_xs[3, int(spec_xs.shape[1] / 1.5)] * 1.1,
            spec_ys[3, int(spec_ys.shape[1] / 1.5)] * 1.1,
            f'rotated\nbowtie B{mode[-1]}',
            color='darkgray', ha='right', va='center',
            fontsize=12, fontweight='bold'
        )

    # Plot companion if specified
    if my_rho is not None:
        comp_x = my_rho * np.cos(my_pa + north_angle)
        comp_y = my_rho * np.sin(my_pa + north_angle)
        ax.scatter(comp_x, comp_y, marker='d', s=100, c='orange',
                  edgecolors='k', label='companion')

    # Configure plot
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    
    # Add text labels
    ax.text(north_x * 1.1, north_y * 1.1, 'North', color='red',
            ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(east_x * 1.1, east_y * 1.1, 'East', color='red',
            ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(obsz_x * 1.1 - 0.1, obsz_y * 1.1,
            'Observatory +Z\nRoman SIAF +v3', color='blue',
            ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(obsy_x * 1.1 - 0.1, obsy_y * 1.1,
            'Observatory +Y\nRoman SIAF +v2', color='blue',
            ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(1.0, -0.1, 'EXCAM +X', color='black',
            ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(0.2, 1.0, 'EXCAM +Y', color='black',
            ha='center', va='center', fontsize=10, fontweight='bold')

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    return ax


def generate_summary_plot(target_str, start_time, obs_window, num_epochs,
                         roll_angle_deg, days_from_start, ts_targ):
    """Generate a text-based summary plot of observation parameters.
    
    Creates a matplotlib figure displaying observation planning parameters
    in a formatted text layout. This provides a quick reference for the
    observation setup.
    
    Args:
        target_str (str): Name of the target object.
        start_time (str): Start time of observation window in ISO format.
        obs_window (int or float): Duration of observation window in days.
        num_epochs (int): Number of observation epochs.
        roll_angle_deg (float): Spacecraft roll angle in degrees.
        days_from_start (ndarray): Array of days from start for each epoch.
        ts_targ (astropy.time.Time): Array of Time objects for each epoch.
    
    Returns:
        None: Displays the plot directly using plt.show().
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    summary_text = f"""
    Target: {target_str}
    Start Time: {start_time}
    Observation Window: {obs_window} days
    Number of Epochs: {num_epochs}
    Roll Angle: {roll_angle_deg}°
    
    Observation Epochs:
    """
    for i, (day, epoch_time) in enumerate(zip(days_from_start, ts_targ)):
        summary_text += f"\n  Epoch {i+1}: Day {day:.1f} ({epoch_time.iso})"
    
    ax.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
           verticalalignment='center', transform=ax.transAxes)
    plt.tight_layout()
    plt.show()


def generate_nominal_plots(target_str, PA_EXCAM_Y, ts_targ):
    """Generate plots showing nominal EXCAM field of view orientations.
    
    Creates a series of plots (one per epoch) showing the basic orientation
    of the EXCAM instrument without any bowtie apertures or roll angles
    applied. Useful for understanding the fundamental pointing geometry.
    
    Args:
        target_str (str): Name of the target object for plot titles.
        PA_EXCAM_Y (float or ndarray): Position angle(s) of EXCAM +Y axis in
            degrees for each epoch.
        ts_targ (astropy.time.Time): Array of observation times for each epoch.
    
    Returns:
        None: Displays plots directly using plt.show().
    """
    for epoch_idx in range(len(ts_targ)):
        date_str = ts_targ[epoch_idx].iso.split(' ')[0]
        ax = gen_sky_plot(PA_EXCAM_Y[epoch_idx])
        ax.set_title(
            f"{target_str} - Nominal EXCAM +Y (Epoch {epoch_idx+1}, {date_str})"
        )
        plt.show()


def generate_bowtie_plots(target_str, PA_EXCAM_Y, ts_targ):
    """Generate plots showing spectroscopic bowtie apertures without roll.
    
    Creates a series of plots (one per epoch) showing the EXCAM spectroscopic
    bowtie apertures in their nominal orientation (no spacecraft roll applied).
    Includes an example companion star position for reference.
    
    Args:
        target_str (str): Name of the target object for plot titles.
        PA_EXCAM_Y (float or ndarray): Position angle(s) of EXCAM +Y axis in
            degrees for each epoch.
        ts_targ (astropy.time.Time): Array of observation times for each epoch.
    
    Returns:
        None: Displays plots directly using plt.show().

    """
    for epoch_idx in range(len(ts_targ)):
        date_str = ts_targ[epoch_idx].iso.split(' ')[0]
        ax = gen_sky_plot(
            PA_EXCAM_Y[epoch_idx],
            my_rho=0.4,
            my_pa=220 * u.deg,
            mode="SPEC3"
        )
        ax.set_title(
            f"{target_str} - SPEC3 Bowtie FOV (no roll, Epoch {epoch_idx+1}, {date_str})"
        )
        plt.show()


def generate_rolled_plots(target_str, PA_EXCAM_Y_plusroll, roll_angles, ts_targ):
    """Generate plots showing bowtie apertures with spacecraft roll applied.
    
    Creates a series of plots (one per epoch) showing how the spectroscopic
    bowtie apertures are oriented after applying a spacecraft roll angle.
    This is the most useful view for planning actual observations.
    
    Args:
        target_str (str): Name of the target object for plot titles.
        PA_EXCAM_Y_plusroll (ndarray): Position angles of EXCAM +Y axis with
            roll applied, in radians.
        roll_angles (list of astropy.units.Quantity): List of roll angles that
            were applied (for display in titles).
        ts_targ (astropy.time.Time): Array of observation times for each epoch.
    
    Returns:
        None: Displays plots directly using plt.show().
    
    Note:
        These plots show all coordinate system vectors, SPEC3 bowtie apertures
        with roll applied, and an example companion at 0.4" separation, PA=220°.
        The roll angle allows optimizing the placement of targets within the
        bowtie apertures or avoiding bright contaminating sources. Roll angles
        are typically limited to ±10-15° from nominal to maintain thermal and
        power constraints.
    """
    for epoch_idx in range(len(PA_EXCAM_Y_plusroll)):
        date_str = ts_targ[epoch_idx].iso.split(' ')[0]
        ax = gen_sky_plot(
            PA_EXCAM_Y_plusroll[epoch_idx],
            my_rho=0.4,
            my_pa=220 * u.deg,
            mode="SPEC3"
        )
        ax.set_title(
            f"{target_str} - SPEC3 Bowtie FOV with roll = {roll_angles[epoch_idx]:.1f} "
            f"(Epoch {epoch_idx+1}, {date_str})"
        )
        plt.show()

def _create_widgets(defaults):
    """Create all UI widgets for the observation planner interface.
    
    Constructs the complete set of ipywidgets controls including input fields,
    buttons, status displays, and output containers organized in tabs.
    
    Args:
        defaults (dict): Dictionary containing default values with keys:
            - 'target' (str): Default target name
            - 'start_time' (str): Default start time
            - 'roll_angle' (float): Default roll angle
    
    Returns:
        dict: Dictionary containing all widget objects with keys:
            - header (widgets.HTML): Title header widget
            - target (widgets.Text): Target name input field
            - start_time (widgets.Text): Start time input field
            - roll_slider (widgets.FloatSlider): Roll angle slider control
            - run_button (widgets.Button): Button to generate plots
            - reset_button (widgets.Button): Button to reset to defaults
            - status (widgets.HTML): Status message display
            - outputs (dict): Dictionary of Output widgets for each tab
            - plot_tabs (widgets.Tab): Tab container for organizing plots


    """
    header = widgets.HTML(
        value="<h2 style='color: #2c3e50; margin-bottom: 5px;'>"
              "🛰️ Roman EXCAM Observation Planner</h2>"
    )
    
    target_widget = widgets.Text(
        value=defaults['target'],
        description='Target:',
        placeholder='Enter target name',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px')
    )
    
    start_time_widget = widgets.Text(
        value=defaults['start_time'],
        description='Start Time (UTC):',
        placeholder='YYYY-MM-DDTHH:MM:SS',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='400px')
    )
    
    roll_slider = widgets.FloatSlider(
        value=defaults['roll_angle'],
        min=0.0,
        max=90.0,
        step=0.1,
        description='Roll Angle (°):',
        continuous_update=False,
        readout=True,
        readout_format='.1f',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='500px')
    )
    
    run_button = widgets.Button(
        description='Generate Plots',
        button_style='primary',
        icon='chart-line',
        layout=widgets.Layout(width='150px', height='40px')
    )
    
    reset_button = widgets.Button(
        description='Reset',
        button_style='',
        icon='undo',
        layout=widgets.Layout(width='100px', height='40px')
    )
    
    status = widgets.HTML(value="")
    
    outputs = {
        'summary': widgets.Output(),
        'nominal': widgets.Output(),
        'bowtie': widgets.Output(),
        'rolled': widgets.Output()
    }
    
    plot_tabs = widgets.Tab(
        children=[
            outputs['summary'],
            outputs['nominal'],
            outputs['bowtie'],
            outputs['rolled']
        ],
        layout=widgets.Layout(width='100%', min_height='500px')
    )
    plot_tabs.set_title(0, '📊 Summary')
    plot_tabs.set_title(1, '🎯 Nominal FOV')
    plot_tabs.set_title(2, '🎀 Bowtie FOV')
    plot_tabs.set_title(3, '🔄 Rolled FOV')
    plot_tabs.layout.display = 'none'
    
    return {
        'header': header,
        'target': target_widget,
        'start_time': start_time_widget,
        'roll_slider': roll_slider,
        'run_button': run_button,
        'reset_button': reset_button,
        'status': status,
        'outputs': outputs,
        'plot_tabs': plot_tabs
    }


def _create_run_callback(widgets_dict, defaults):
    """Create the callback function for the run button.
    
    Generates a function that handles the "Generate Plots" button click by
    validating inputs, computing observation parameters, generating all plot
    types, displaying results in tabs, and providing error handling and status
    updates.
    
    Args:
        widgets_dict (dict): Dictionary of widget objects created by _create_widgets.
        defaults (dict): Dictionary of default parameter values (not currently used
            in callback).
    
    Returns:
        function: Callback function that accepts a button widget parameter (standard
            ipywidgets callback signature).

    """
    
    def run_plotting(b):
        # Clear all outputs
        for output in widgets_dict['outputs'].values():
            output.clear_output()
        
        widgets_dict['status'].value = "<p style='color: #f39c12;'>⏳ Generating plots...</p>"
        
        try:
            target_str = widgets_dict['target'].value.strip()
            if not target_str:
                raise ValueError("Please enter a target name")
            
            start_time = widgets_dict['start_time'].value.strip()
            roll_angle_deg = widgets_dict['roll_slider'].value
            
            # Default parameters
            obs_window = 30
            num_epochs = 1
            
            # Compute observation times
            days_from_start, ts_targ = compute_observation_times(
                start_time, obs_window, num_epochs
            )
            
            # Get target
            target = get_exoplanet_skycoord(target_str)
            
            # Compute angles
            angle_data = compute_angles(target, ts_targ, roll_angle_deg)
            
            # Generate plots in each tab
            with widgets_dict['outputs']['summary']:
                widgets_dict['outputs']['summary'].clear_output(wait=True)
                generate_summary_plot(
                    target_str, start_time, obs_window, num_epochs,
                    roll_angle_deg, days_from_start, ts_targ
                )
            
            with widgets_dict['outputs']['nominal']:
                widgets_dict['outputs']['nominal'].clear_output(wait=True)
                generate_nominal_plots(
                    target_str, angle_data['PA_EXCAM_Y'], ts_targ
                )
            
            with widgets_dict['outputs']['bowtie']:
                widgets_dict['outputs']['bowtie'].clear_output(wait=True)
                generate_bowtie_plots(
                    target_str, angle_data['PA_EXCAM_Y'], ts_targ
                )
            
            with widgets_dict['outputs']['rolled']:
                widgets_dict['outputs']['rolled'].clear_output(wait=True)
                generate_rolled_plots(
                    target_str, angle_data['PA_EXCAM_Y_plusroll'],
                    angle_data['roll_angles'], ts_targ
                )
            
            # Show the tabs
            widgets_dict['plot_tabs'].layout.display = 'block'
            
            widgets_dict['status'].value = (
                "<p style='color: #27ae60;'>✓ Plots generated! "
                "Use tabs to compare different FOV configurations.</p>"
            )
            
        except Exception as e:
            widgets_dict['status'].value = f"<p style='color: #e74c3c;'>❌ Error: {str(e)}</p>"
            print(f"Full error traceback:")
            traceback.print_exc()
            widgets_dict['plot_tabs'].layout.display = 'none'
    
    return run_plotting


def _create_reset_callback(widgets_dict, defaults):
    """Create the callback function for the reset button.
    
    Generates a function that restores all input widgets to their default
    values and clears all outputs when the "Reset" button is clicked.
    
    Args:
        widgets_dict (dict): Dictionary of widget objects created by _create_widgets.
        defaults (dict): Dictionary of default parameter values to restore, with keys:
            - 'target' (str)
            - 'start_time' (str)
            - 'roll_angle' (float)
    
    Returns:
        function: Callback function that accepts a button widget parameter (standard
            ipywidgets callback signature).

    """
    
    def reset_defaults(b):
        widgets_dict['target'].value = defaults['target']
        widgets_dict['start_time'].value = defaults['start_time']
        widgets_dict['roll_slider'].value = defaults['roll_angle']
        widgets_dict['status'].value = ""
        
        for output in widgets_dict['outputs'].values():
            output.clear_output()
        
        widgets_dict['plot_tabs'].layout.display = 'none'
    
    return reset_defaults


def launch_ui(target='47 UMa', start_time='2027-06-01T00:00:00.0', roll_angle=15.0):
    """Launch the interactive Roman EXCAM observation planner interface.
    
    Creates and displays a complete interactive widget-based interface for
    planning Roman Space Telescope EXCAM observations. The interface allows
    users to specify targets, observation times, and spacecraft roll angles,
    then generates comprehensive field of view visualizations.
    
    Args:
        target (str, optional): Default target name as recognized by SIMBAD. The
            target should be a resolvable astronomical object name. Default is '47 UMa'.
        start_time (str, optional): Default observation start time in ISO 8601
            format (UTC). Format: 'YYYY-MM-DDTHH:MM:SS.S'. Default is
            '2027-06-01T00:00:00.0'.
        roll_angle (float, optional): Default spacecraft roll angle in degrees.
            Must be between 0 and 90. Positive roll rotates the field of view
            counter-clockwise as viewed from the spacecraft. Default is 15.0°.
    
    Returns:
        None: Displays the interactive interface directly in the notebook.
    """
    # Set matplotlib backend
    try:
        import matplotlib
        matplotlib.use('module://matplotlib_inline.backend_inline')
    except:
        pass
    
    defaults = {
        'target': target,
        'start_time': start_time,
        'roll_angle': roll_angle
    }
    
    # Create all widgets
    widgets_dict = _create_widgets(defaults)
    
    # Create layout
    input_section = widgets.VBox([
        widgets.HTML(
            "<h4 style='color: #34495e; margin: 10px 0 5px 0;'>"
            "Input Parameters</h4>"
        ),
        widgets_dict['target'],
        widgets_dict['start_time'],
        widgets_dict['roll_slider'],
    ], layout=widgets.Layout(margin='0 0 20px 0'))
    
    buttons = widgets.HBox(
        [widgets_dict['run_button'], widgets_dict['reset_button']],
        layout=widgets.Layout(justify_content='flex-start', margin='10px 0')
    )
    
    main_container = widgets.VBox([
        widgets_dict['header'],
        widgets.HTML(
            "<hr style='border: 1px solid #ecf0f1; margin: 10px 0;'>"
        ),
        input_section,
        buttons,
        widgets_dict['status'],
        widgets_dict['plot_tabs']
    ], layout=widgets.Layout(
        padding='20px',
        border='2px solid #bdc3c7',
        border_radius='8px',
        width='900px'
    ))
    
    # Connect callbacks
    run_callback = _create_run_callback(widgets_dict, defaults)
    reset_callback = _create_reset_callback(widgets_dict, defaults)
    
    widgets_dict['run_button'].on_click(run_callback)
    widgets_dict['reset_button'].on_click(reset_callback)
    
    # Display the interface
    display(main_container)