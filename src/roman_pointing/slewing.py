from pathlib import Path
from roman_pointing.utils import get_cache_dir
from roman_pointing.Reference_Star_Selection_Tool import load_catalog
from astropy.io import ascii
from urllib.request import urlretrieve
from scipy.interpolate import make_interp_spline
import astropy
import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord, Distance, BarycentricMeanEcliptic, Angle, ICRS
from astropy.time import Time
import requests
from roman_pointing.roman_pointing import calcRomanAngles, getL2Positions
from angutils.angutils import genGreatCircle
import pandas

try:
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
except ModuleNotFoundError:
    print("ortools could not be imported. Optimization will not work.")
    print("To fix, run: pip install ortools")
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    print("matplotlib could not be imported. Plotting will not work.")
    print("To fix, run: pip install matplotlib")


class Slewing(object):
    def __init__(self) -> None:
        self.SLEWSETTLE_FILE_PATH = Path(get_cache_dir()) / "SlewSettle.ecsv"

        if not self.SLEWSETTLE_FILE_PATH.exists():
            print("Slew/Settle data not found on disk.  Attempting to download.")

            url = "https://raw.githubusercontent.com/RomanSpaceTelescope/roman-technical-information/refs/heads/main/roman_technical_information/data/Observatory/SlewSettle/SlewSettle.ecsv"
            urlretrieve(url, self.SLEWSETTLE_FILE_PATH)

            assert self.SLEWSETTLE_FILE_PATH.exists(), "Data download failed."

        # generate interpolator
        self.slewSettleInterpolator()

    def slewSettleInterpolator(self) -> None:
        """Generate a linear interpolant for slew/settle data."""

        # check for datafile on disk
        data = ascii.read(self.SLEWSETTLE_FILE_PATH)
        angles = data["Angle"].value
        times = data["Time"].value

        self.slewSettle = make_interp_spline(angles, times, k=1)

    def slewTime(
        self, angles: astropy.units.quantity.Quantity
    ) -> astropy.units.quantity.Quantity:
        """

        Args:
            angles (astropy.units.quantity.Quantity):
                Slew angle

        Returns:
            astropy.units.quantity.Quantity:
                Slew times

        """

        return self.slewSettle(angles.to_value(u.deg)) * u.s

    def gen_skycoords_from_df(self, stars: pandas.DataFrame) -> SkyCoord:
        """Generate SkyCoord object from dataframe of star values

        Args:
            stars (pandas.DataFrame):
                Input table

        Returns:
            astropy.coordinates.SkyCoord:
                Coordinates object.

        """
        res = SkyCoord(
            stars["ra"].values,
            stars["dec"].values,
            unit=(u.deg, u.deg),
            frame="icrs",
            distance=Distance(parallax=stars["sy_plx"].values * u.mas),
            pm_ra_cosdec=stars["sy_pmra"].values * u.mas / u.yr,
            pm_dec=stars["sy_pmdec"].values * u.mas / u.yr,
            radial_velocity=stars["st_radv"].values * u.km / u.s,
            equinox="J2000",
            obstime="J2000",
        ).transform_to(BarycentricMeanEcliptic)

        return res

    def load_refstars(self, force_refresh: bool = False) -> None:
        """Load reference star catalog if not already loaded

        Args:
            force_refresh (bool):
                Force downlaod of new catalog)

        Returns:
            None

        """

        # check whether anything needs to be done at all
        if hasattr(self, "refstars") and not (force_refresh):
            return

        # grab reference stars
        self.refstar_cat = load_catalog(force_refresh=force_refresh)
        self.refstars = SkyCoord(
            self.refstar_cat["ra"].values,
            self.refstar_cat["dec"].values,
            unit=(u.deg, u.deg),
            frame="icrs",
            distance=Distance(parallax=self.refstar_cat["sy_plx"].values * u.mas),
            pm_ra_cosdec=self.refstar_cat["sy_pmra"].values * u.mas / u.yr,
            pm_dec=self.refstar_cat["sy_pmdec"].values * u.mas / u.yr,
            radial_velocity=self.refstar_cat["st_radv"].values * u.km / u.s,
            equinox="J2000",
            obstime="J2000",
        ).transform_to(BarycentricMeanEcliptic)

    def resolve_starList_names(self, starList: list) -> list:
        """Find main_id values for list of stars

        Args:
            starList (list):
                List of star names

        Returns:
            list:
                List of resolved names.  Any entries that could not be resolved are set
                to None

        """

        out = []
        for name in starList:

            resp = requests.get(
                "https://corgidb.sioslab.com/resolve_star_name.php",
                headers={"User-Agent": "RomanRefStarPicker/1.0"},
                params={"st_name": name},
                timeout=15,
            )

            resp.raise_for_status()
            raw = resp.json()
            if raw:
                out.append(raw[0][0])
            else:
                print(f"Coult not match {name}")
                out.append(None)

        return out

    def optimize_refstar_chain(self, starList: list, daystr: str) -> tuple:
        """Minimize the slew time of a series of star observations

        Args:
            starList (list):
                List of star names to visit. Must be resolved to main_id.
            daystr (str):
                YYYY-MM-DD formatted date string representing start of chain

        Returns:
            tuple:
                route (list):
                    Indices of optimal chain, always starting with 0.
                slewtimes (list):
                    Each slew time in seconds (rounded to nearest second). Will be of
                    size n-1 for n-element starList

        """

        self.load_refstars()

        assert (
            len(set(starList) - set(self.refstar_cat["main_id"])) == 0
        ), "Unknown stars in input. Try resolving the input list first."

        # find indices of star list
        inds = np.hstack(
            [np.where(self.refstar_cat["main_id"].values == n)[0] for n in starList]
        )

        t_str = [daystr + "T00:00:00.0"]
        t0 = Time(t_str, format="isot", scale="utc")
        roman_pos = getL2Positions(t0)

        # compute all angles at initial time
        sun_angs = np.zeros(len(starList)) * u.rad
        yaws = np.zeros(len(starList)) * u.rad
        pitchs = np.zeros(len(starList)) * u.rad
        for jj, targ in enumerate(self.refstars[inds]):
            sun_ang, yaw, pitch, B_C_I = calcRomanAngles(targ, t0, roman_pos)
            sun_angs[jj], yaws[jj], pitchs[jj] = sun_ang[0], yaw[0], pitch[0]

        # compute slew between every pair of targets
        # add final dummy target with zero distances from everything to ensure open path
        dist_mat = np.zeros((len(starList) + 1, len(starList) + 1))
        for ii in range(len(starList)):
            for jj in range(len(starList)):
                if ii == jj:
                    continue
                dyaw = np.abs(yaws[jj] - yaws[ii])
                dpitch = np.abs(pitchs[jj] - pitchs[ii])
                slew_ang = dyaw if dyaw > dpitch else dpitch
                dist_mat[ii, jj] = self.slewTime(slew_ang).to_value(u.s)

        # adapted from https://developers.google.com/optimization/routing/tsp

        # create data model
        data = {}
        data["distance_matrix"] = np.round(dist_mat).astype(int)
        data["num_vehicles"] = 1
        data["start"] = [0]
        data["end"] = [len(starList)]

        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(
            len(data["distance_matrix"]),
            data["num_vehicles"],
            data["start"],
            data["end"],
        )

        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data["distance_matrix"][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)

        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )

        # Solve the problem.
        solution = routing.SolveWithParameters(search_parameters)
        assert solution, "No solution found."

        print(f"Objective: {solution.ObjectiveValue()} seconds")
        index = routing.Start(0)
        route = [manager.IndexToNode(index)]
        slewtimes = []
        while not routing.IsEnd(index):
            next_index = solution.Value(routing.NextVar(index))
            slewtimes.append(routing.GetArcCostForVehicle(index, next_index, 0))
            index = next_index
            route.append(manager.IndexToNode(index))

        return route[:-1], slewtimes[:-1]

    def plot_refstar_route(self, starList: list, route: list) -> None:
        """

        Args:
            starList (list):
                List of star names to visit. Must be resolved to main_id.
            route (list):
                Indices of observations in order. Output from optimize_refstar_chain


        """

        # Identify stars and extract coordinates
        starinds = np.hstack(
            [np.where(self.refstar_cat["main_id"].values == n)[0] for n in starList]
        )

        ra = (
            Angle(self.refstar_cat.iloc[starinds]["ra"].values * u.deg)
            .wrap_at(180 * u.degree)
            .rad
        )
        dec = Angle(self.refstar_cat.iloc[starinds]["dec"].values * u.deg).rad

        # Generate figure and scatter plot stars
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(111, projection="mollweide")
        _ = ax.scatter(ra, dec, zorder=10)
        ax.grid()

        # plot great circle arcs between all targets in order
        diffs = np.abs(np.diff(ra[route]))
        cmap = mpl.colormaps["winter"]
        for j in range(1, len(route)):
            c = cmap(round(j / (len(route) - 1) * 255))

            lam = ra[route[j - 1 : j + 1]]
            phi = dec[route[j - 1 : j + 1]]

            l1, p1 = genGreatCircle(lam, phi)

            lamsort = np.sort(lam)

            inds = (l1 > lamsort[0]) & (l1 < lamsort[1])

            if diffs[j - 1] < np.pi:
                l2 = l1[inds]
                p2 = p1[inds]
                inds2 = np.argsort(l2)
                plt.plot(l2[inds2], p2[inds2], color=c)
            else:
                l2 = l1[~inds]
                p2 = p1[~inds]
                inds2 = l2 > 0
                plt.plot(l2[inds2], p2[inds2], color=c)
                plt.plot(l2[~inds2], p2[~inds2], color=c)

        # overplot initial and final stars with visit order colors
        _ = ax.scatter(ra[0], dec[0], c=cmap(0), zorder=11)
        _ = ax.scatter(ra[route[-1]], dec[route[-1]], c=cmap(255), zorder=11)

        # add colorbar
        norm = mpl.colors.Normalize(vmin=1, vmax=len(route) - 1)
        fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=ax,
            location="right",
            label="Observation Number",
            shrink=0.5,
        )

        ax.set_xlabel("RA")
        ax.set_ylabel("DEC")
        plt.tight_layout()

        print(
            "Plotting complete.  You may need to run plt.show() for the plot to render"
        )

    def optimize_star_chain(
        self, coords: SkyCoord, daystr: str, free_start: bool = False
    ) -> tuple:
        """Minimize the slew time of a series of star observations

        Args:
            coords(astropy.coordiates.SkyCoord):
                Coordinates of stars
            daystr (str):
                YYYY-MM-DD formatted date string representing start of chain
            free_start (bool):
                If True, can start at any target.  Defaults False, in which case always
                start at 0 index

        Returns:
            tuple:
                route (list):
                    Indices of optimal chain
                slewtimes (list):
                    Each slew time in seconds (rounded to nearest second). Will be of
                    size n-1 for n-element coords

        """

        t_str = [daystr + "T00:00:00.0"]
        t0 = Time(t_str, format="isot", scale="utc")
        roman_pos = getL2Positions(t0)

        # compute all angles at initial time
        sun_angs = np.zeros(len(coords)) * u.rad
        yaws = np.zeros(len(coords)) * u.rad
        pitchs = np.zeros(len(coords)) * u.rad
        for jj, targ in enumerate(coords):
            sun_ang, yaw, pitch, B_C_I = calcRomanAngles(targ, t0, roman_pos)
            sun_angs[jj], yaws[jj], pitchs[jj] = sun_ang[0], yaw[0], pitch[0]

        # compute slew between every pair of targets
        # add final dummy target with zero distances from everything to ensure open path
        dist_mat = np.zeros((len(coords) + 1, len(coords) + 1))
        for ii in range(len(coords)):
            for jj in range(len(coords)):
                if ii == jj:
                    continue
                dyaw = np.abs(yaws[jj] - yaws[ii])
                dpitch = np.abs(pitchs[jj] - pitchs[ii])
                slew_ang = dyaw if dyaw > dpitch else dpitch
                dist_mat[ii, jj] = self.slewTime(slew_ang).to_value(u.s)

        # if free start, add dummy initial target as well
        if free_start:
            dist_mat = np.vstack((np.zeros(len(coords) + 1), dist_mat))
            dist_mat = np.hstack((np.zeros((len(coords) + 2, 1)), dist_mat))

        # adapted from https://developers.google.com/optimization/routing/tsp

        # create data model
        data = {}
        data["distance_matrix"] = np.round(dist_mat).astype(int)
        data["num_vehicles"] = 1
        data["start"] = [0]
        if free_start:
            data["end"] = [len(coords) + 1]
        else:
            data["end"] = [len(coords)]

        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(
            len(data["distance_matrix"]),
            data["num_vehicles"],
            data["start"],
            data["end"],
        )

        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data["distance_matrix"][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)

        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )

        # Solve the problem.
        solution = routing.SolveWithParameters(search_parameters)
        assert solution, "No solution found."

        print(f"Objective: {solution.ObjectiveValue()} seconds")
        index = routing.Start(0)
        route = [manager.IndexToNode(index)]
        slewtimes = []
        while not routing.IsEnd(index):
            next_index = solution.Value(routing.NextVar(index))
            slewtimes.append(routing.GetArcCostForVehicle(index, next_index, 0))
            index = next_index
            route.append(manager.IndexToNode(index))

        if free_start:
            route, slewtimes = route[1:-1], slewtimes[1:-1]
            route = [r - 1 for r in route]
        else:
            route, slewtimes = route[:-1], slewtimes[:-1]

        return route, slewtimes

    def plot_star_route(self, coords: SkyCoord, route: list) -> None:
        """

        Args:
            coords(astropy.coordiates.SkyCoord):
                Coordinates of stars
            route (list):
                Indices of observations in order. Output from optimize_refstar_chain


        """

        # Identify stars and extract coordinates
        tmp = coords.transform_to(ICRS)
        ra = tmp.ra.wrap_at(180 * u.degree).rad
        dec = tmp.dec.rad

        # Generate figure and scatter plot stars
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(111, projection="mollweide")
        _ = ax.scatter(ra, dec, zorder=10)
        ax.grid()

        # plot great circle arcs between all targets in order
        diffs = np.abs(np.diff(ra[route]))
        cmap = mpl.colormaps["winter"]
        for j in range(1, len(route)):
            c = cmap(round(j / (len(route) - 1) * 255))

            lam = ra[route[j - 1 : j + 1]]
            phi = dec[route[j - 1 : j + 1]]

            l1, p1 = genGreatCircle(lam, phi)

            lamsort = np.sort(lam)

            inds = (l1 > lamsort[0]) & (l1 < lamsort[1])

            if diffs[j - 1] < np.pi:
                l2 = l1[inds]
                p2 = p1[inds]
                inds2 = np.argsort(l2)
                plt.plot(l2[inds2], p2[inds2], color=c)
            else:
                l2 = l1[~inds]
                p2 = p1[~inds]
                inds2 = l2 > 0
                plt.plot(l2[inds2], p2[inds2], color=c)
                plt.plot(l2[~inds2], p2[~inds2], color=c)

        # overplot initial and final stars with visit order colors
        _ = ax.scatter(ra[route[0]], dec[route[0]], c=cmap(0), zorder=11)
        _ = ax.scatter(ra[route[-1]], dec[route[-1]], c=cmap(255), zorder=11)

        # add colorbar
        norm = mpl.colors.Normalize(vmin=1, vmax=len(route) - 1)
        fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=ax,
            location="right",
            label="Observation Number",
            shrink=0.5,
        )

        ax.set_xlabel("RA")
        ax.set_ylabel("DEC")
        plt.tight_layout()

        print(
            "Plotting complete.  You may need to run plt.show() for the plot to render"
        )
