from roman_pointing.roman_observability import get_target_coords
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from astropy.time import Time
import re
from roman_pointing.roman_pointing import (
    calcRomanAngles,
    getL2Positions,
)
import numpy as np
import astropy.units as u
from astroquery.jplhorizons import Horizons
from scipy.interpolate import CubicSpline
import warnings
from erfa import ErfaWarning

warnings.filterwarnings("ignore", category=ErfaWarning, message=".*dubious year.*")

cpgs_url = "https://cpgs-roman.ipac.caltech.edu/"


def gen_cpgs_driver():
    """Create a selenium driver object"""

    driver = webdriver.Firefox()
    driver.get(cpgs_url)
    _ = input("Hit enter once you've authenticated and the page is fully loaded.")

    return driver


def get_for_intervals(targ="eps Eri", driver=None):
    "Retrieve field of regard intervals (start & end times) from CPGS" ""

    if driver is None:
        driver = gen_cpgs_driver()
    else:
        # reload page
        driver.get(cpgs_url)

    def fill(field_id, value):
        el = driver.find_element(By.ID, field_id)
        el.clear()
        el.send_keys(value)

    # 1. Observation Label
    fill("cpgs_observation_label", "CGI_OBS_000")

    # 2. Observation Number
    fill("cpgs_observation_number", "0")

    # 3. Observer First Name
    fill("cpgs_observer_firstname", "No")

    # 4. Observer Last Name
    fill("cpgs_observer_lastname", "One")

    # 5. Requested Start Date/Time (UTC)
    fill("cpgs_requested_start", "2026-09-01T00:00:00")

    # 6. Target Name
    fill("cpgs_target_name", targ)

    # 7. SIMBAD Lookup, then wait for RA to update.
    driver.find_element(By.ID, "cpgs_target_name_simbad").click()
    WebDriverWait(driver, 30).until(
        lambda d: d.find_element(By.ID, "cpgs_target_RA").get_attribute("value") != ""
    )

    # 7a. Default V Magnitude if SIMBAD didn't populate it
    v_mag = driver.find_element(By.ID, "cpgs_target_v_mag")
    if v_mag.get_attribute("value") == "":
        fill("cpgs_target_v_mag", "0")

    # 8. Observe Target Only (check)
    observe_target_only = driver.find_element(By.ID, "cpgs_observe_target_only")
    if not observe_target_only.is_selected():
        observe_target_only.click()

    # 9. Obtain satellite spot image every visit (uncheck)
    satspot_every_visit = driver.find_element(
        By.ID, "cpgs_obtain_satspot_image_every_visit"
    )
    if satspot_every_visit.is_selected():
        satspot_every_visit.click()

    # 10. Automatic EXCAM parameters (Target Integration Parameters) -> Yes
    driver.find_element(By.ID, "cpgs_target_autogain_1").click()

    # 11. Number of Frames (acquisition)
    fill("cpgs_howfsc_acq_nframes", "1")

    # 12. Create Obs Specs
    driver.find_element(By.ID, "create_obs_specs").click()

    # 13. Wait for results
    pre = WebDriverWait(driver, 30).until(
        EC.presence_of_element_located(
            (By.XPATH, "//pre[contains(., 'Field of regard intervals')]")
        )
    )
    report_text = pre.text

    n_intervals = int(re.search(r"#intervals\s*=\s*(\d+)", report_text).group(1))
    row_re = re.compile(
        r"^\*?\s*(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\s*-\s*"
        r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\s*$"
    )

    starts, ends = [], []
    for line in report_text.splitlines():
        m = row_re.match(line)
        if m:
            starts.append(m.group(1))
            ends.append(m.group(2))
            if len(starts) == n_intervals:
                break

    for_start = Time(starts, format="isot", scale="utc")
    for_end = Time(ends, format="isot", scale="utc")

    return for_start, for_end


def find_crossing_time_diff(t0, coord):
    """Compare CPGS FOR intervals with Roman Pointing predictions"""

    ts = t0 + (np.arange(0, 86400) - 86400 / 2) * u.s

    """
    # get L2 location from Horizons
    obj = Horizons(
        id="32",
        location="@0",
        epochs={"start": ts[0].value, "stop": (ts[-1] + 1 * u.h).value, "step": "1h"},
    )
    L2_vectors_table = obj.vectors()
    L2pos = (
        np.vstack([L2_vectors_table[l].data.filled() for l in ["x", "y", "z"]]) * u.AU
    )
    L2ts = L2_vectors_table["datetime_jd"].data.filled()
    L2spline = CubicSpline(L2ts, L2pos.T, extrapolate=False)

    L2 = L2spline(ts.to_value("jd")).T * u.AU
    """
    L2 = getL2Positions(ts)

    sun_ang, _, _, _ = calcRomanAngles(
        coord,
        ts,
        L2,
    )

    ko = (sun_ang > 54 * u.deg) & (sun_ang < 126 * u.deg)
    t1 = ts[np.where(np.diff(ko))]
    dt = t0 - t1

    return dt


def exercise_cpgs(targ):
    driver = gen_cpgs_driver()

    fs, fe = get_for_intervals(targ, driver)

    coords = get_target_coords([targ])

    tdiffs_start = np.zeros(fs.size)
    for jj in range(len(fs)):
        print(jj)
        if fs[jj].jd == 2461284.5:
            continue

        dt = find_crossing_time_diff(fs[jj], coords[targ])
        tdiffs_start[jj] = dt.value[0]

    tdiffs_end = np.zeros(fe.size)
    for jj in range(len(fe)):
        print(jj)
        if fe[jj].jd == 2463107.5:
            continue

        dt = find_crossing_time_diff(fe[jj], coords[targ])
        tdiffs_end[jj] = dt.value[0]
