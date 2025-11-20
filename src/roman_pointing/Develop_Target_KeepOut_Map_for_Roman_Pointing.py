from roman_pointing import calcRomanAngles, getL2Positions, getSunPositions
import astropy.units as u
from astropy.time import Time
from astroquery.simbad import Simbad
from astroquery.jplhorizons import Horizons
from astropy.coordinates import (
    SkyCoord,
    Distance,
    get_body_barycentric,
    BarycentricMeanEcliptic,
)
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def get_target_coords(target_names):
    """Query SIMBAD for astronomical target coordinates and proper motions.

        Retrieves celestial coordinates, parallax, proper motion, and radial velocity
        data from SIMBAD database for specified astronomical objects. Transforms 
        coordinates to Barycentric Mean Ecliptic frame for Roman Space Telescope 
        pointing calculations.


    Args:
         target_names (list of str):
              List of astronomical object names recognizable by SIMBAD (e.

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
        res = simbad.query_object(name)
        if res is None:
            print(f" SIMBAD could not find {name}")
            continue

        c_icrs = SkyCoord(
            res["ra"].value.data[0],
            res["dec"].value.data[0],
            unit=(res["ra"].unit, res["dec"].unit),
            frame="icrs",
            distance=Distance(parallax=res["plx_value"].value.data[0] * res["plx_value"].unit),
            pm_ra_cosdec=res["pmra"].value.data[0] * res["pmra"].unit,
            pm_dec=res["pmdec"].value.data[0] * res["pmdec"].unit,
            radial_velocity=res["rvz_radvel"].value.data[0] * res["rvz_radvel"].unit,
            equinox="J2000",
            obstime="J2000"
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
            Start date in ISO format 
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

    sun_ang, yaw, pitch, BCI = calcRomanAngles(
        coord,
        ts,
        getL2Positions(ts)
    )
    
    return ts, sun_ang, yaw, pitch


def compute_keepout(coords_dict, start_date, days, time_step, min_sun=54, max_sun=126):
    """ Determine observability windows for multiple targets based on solar angle constraints.

        Calculates when targets are observable by Roman Space Telescope based on solar
        exclusion angle limits. The allowed pitch angle range to avoid thermal and stray light 
        issues while keeping the solar panels properly oriented is 54-126 degrees. 
        
    Args:
        coords_dict (dict):
            Dictionary mapping target names (str) to SkyCoord objects
        start_date (str):
            Start date in ISO format (e.g., '2027-01-01T00:00:00')
        days (int or float):
            Duration of observation window in days
        time_step (int or float):
            Time interval between calculations in days
        min_sun (int or float):
            Minimum allowed solar angle in degrees. 
        max_sun (int or float):
            Maximum allowed solar angle in degrees.

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
        keepout[name] = (sun_ang > min_sun*u.deg) & (sun_ang < max_sun*u.deg)

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

    # solar keepout boundaries
    ax.plot([0, len(ts)], [54, 54], "k--")
    ax.plot([0, len(ts)], [126, 126], "k--")

    ax.fill_between([0, len(ts)], [54, 54], [0, 0], hatch="/", color="none", edgecolor="k")
    ax.fill_between([0, len(ts)], [126, 126], [180, 180], hatch="\\", color="none", edgecolor="k")

    ax.set_title("Solar Angle vs Time")
    ax.legend()

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
    ax.legend()

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

    komap = np.vstack([keepout_dict[n] for n in names])

    fig, ax = plt.subplots(figsize=(12, 1.3*M + 3))
    cmap = matplotlib.colors.ListedColormap(["black", "green"])

    p = ax.pcolor(np.arange(komap.shape[1]), np.arange(M), komap, cmap=cmap)

    ax.set_yticks(np.arange(M))
    ax.set_yticklabels(names)
    ax.set_xlabel("Time Index")

    cbar = plt.colorbar(p, ticks=[0.25, 0.75], drawedges=True)
    cbar.ax.set_yticklabels(["Unavailable", "Available"])

    ax.set_title(f"Roman Keepout Map\n{ts[0].iso} → {ts[-1].iso}")

    return fig, ax

