"""
Compare CGI header pointing (and, optionally, WFI telemetry) with roman_pointing nominal-attitude models:

  1. V1 on the target, observatory at L2 (the main-branch model)
  2. aperture (SIAF CGI_CEN) on the target, observatory at L2
  3. V1 on the target, observatory on Roman's JPL Horizons trajectory
  4. aperture on the target, Horizons trajectory
  5. as 4, with the velocity aberration of the target and Sun directions (physical attitude)

Conventions established with CAR-55.1, 55.4 and 193.1 data (see docs/attitude_model/README.md, section 6):
  * The observation plan fixes PA_V3 for the planned attitude epoch (the visit-slew start in the OPUP, or the Jira
    "Planned Execution Time"), and the attitude is held inertially fixed when the visit executes at another time.
    Give that epoch per visit with --epoch; otherwise the frame time is used.
  * CGI L1 pointing keywords (RA_BORE, DEC_BORE, RA_APER, DEC_APER, PA_V3, PA_APER) are the raw ACS attitude
    quaternion, i.e. in the apparent frame. They are compared with model 5 in the apparent frame; models 1-4 have
    no aberration.
  * WFI meta.pointing (CALCULATED) is aberration-corrected (catalog frame, as the plans). With --wfi, the WFI
    exposure covering each CGI frame gives the measured attitude, compared with model 5 in the catalog frame.
  * With --plan (the OPUP "CGI Fixed v3pa@T" of a visit) and --wfi, the held attitude is predicted at the WFI
    mid-exposure time: the model-5 nominal attitude rolled about the CGI aperture until its PA_V3 there equals the
    planned value. Its PA_V3 at V1 is compared with the WFI-reported meta.pointing.pa_v3, which refers to V1.
  * With --wfi, a second table compares the raw ACS attitude quaternion in the WFI meta.pointing (SCF_AC_SDR_QBJ,
    apparent frame) with the CGI header pointing, and, with the aberration removed, with the WFI meta.pointing
    values. It also lists the rotation of the WFI Gaia alignment (meta.wcs_fit_results).
  * Headers from CDMS releases before 520 (no SIAF keyword) used the preflight SIAF; their aperture and PA keywords
    are not compared.

Usage:
  python tools/compare_header_pa.py ~/CGI_Commissioning_Data/CAR193.1_L1_MAST \\
      --epoch 0116001001001001001=2026-11-01T04:57:00 --plan 0116001001001001001=193.115 \\
      --wfi ~/CGI_Commissioning_Data/CAR193.1_WFI_MAST/*_wcs.asdf

"""

import argparse
import glob
import os
import warnings

import astropy.units as u
import numpy as np
from astropy.coordinates import BarycentricMeanEcliptic, CartesianRepresentation, SkyCoord
from astropy.io import fits
from astropy.time import Time

from roman_pointing.roman_pointing import (
    applyAberration,
    applyRollAngle,
    calcRomanAngles,
    getL2Positions,
    getRomanPositionAngle,
    getRomanPositions,
    getRomanVelocities,
    getSIAFAperture,
    getSunPositions,
    removeAberration,
)

MODELS = {1: (False, "L2"), 2: (True, "L2"), 3: (False, "horizons"), 4: (True, "horizons")}
LABELS = {1: "V1, L2 (main)", 2: "aperture, L2", 3: "V1, Horizons", 4: "aperture, Horizons",
          5: "aperture, Horizons, aberration"}


def wrap(angle):
    """Wrap angles to [-180, 180) deg.

    Args:
        angle (numpy.ndarray or float):
            Angle (deg)

    Returns:
        numpy.ndarray or float:
            Wrapped angle (deg)
    """
    return (np.asarray(angle) + 180) % 360 - 180


def ecliptic_vector(ra, dec):
    """BarycentricMeanEcliptic unit vector of an ICRS direction.

    Args:
        ra (float):
            Right ascension (deg)
        dec (float):
            Declination (deg)

    Returns:
        numpy.ndarray:
            Unit vector (3)
    """
    return SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs").transform_to(BarycentricMeanEcliptic).cartesian.xyz.value


def read_wfi_pointing(files):
    """Read the attitude, exposure times and Gaia alignment from WFI _wcs.asdf (or L1/L2 ASDF) files.

    Args:
        files (list):
            ASDF file names

    Returns:
        list:
            dicts with start and end (astropy.time.Time), B_C_I (3x3, catalog frame, BarycentricMeanEcliptic),
            pa_v3 (deg, the reported PA_V3 at V1), B_q (3x3, the raw quaternion attitude, apparent frame,
            BarycentricMeanEcliptic), and the Gaia alignment rotation fit_rot (deg) and nmatches (None if absent)
    """
    import asdf  # optional dependency, only needed with --wfi

    out = []
    for f in files:
        with warnings.catch_warnings(), asdf.open(f, lazy_load=True) as af:
            warnings.simplefilter("ignore")
            meta = af.tree["roman"]["meta"]
            p, exposure = dict(meta["pointing"]), dict(meta["exposure"])
            fit = dict(meta["wcs_fit_results"]) if "wcs_fit_results" in meta else {}
        b_1 = ecliptic_vector(p["ra_v1"], p["dec_v1"])
        north = ecliptic_vector(0, 90)
        north = north - (north @ b_1) * b_1
        north /= np.linalg.norm(north)
        east = np.cross(north, b_1)
        for sign in (1, -1):  # resolve the handedness of "east" against getRomanPositionAngle
            b_3 = np.cos(np.radians(p["pa_v3"])) * north + sign * np.sin(np.radians(p["pa_v3"])) * east
            B_C_I = np.array([b_1, np.cross(b_3, b_1), b_3])
            if abs(wrap(getRomanPositionAngle(B_C_I[:, :, None]).to_value(u.deg)[0] - p["pa_v3"])) < 1e-6:
                break
        # raw ACS quaternion (scalar last) as an ICRS-to-body DCM; its rows are b_1, b_2, b_3 in ICRS
        q = np.asarray(p["quaternion"], dtype=float)
        q /= np.linalg.norm(q)
        qv, qs = q[:3], q[3]
        q_cross = np.array([[0, -qv[2], qv[1]], [qv[2], 0, -qv[0]], [-qv[1], qv[0], 0]])
        A = (qs**2 - qv @ qv) * np.eye(3) + 2 * np.outer(qv, qv) - 2 * qs * q_cross
        B_q = SkyCoord(CartesianRepresentation(A.T), frame="icrs").transform_to(BarycentricMeanEcliptic).cartesian.xyz.value.T
        out.append({"file": os.path.basename(f), "start": Time(exposure["start_time"]), "end": Time(exposure["end_time"]),
                    "B_C_I": B_C_I, "pa_v3": p["pa_v3"], "source": p["pointing_engineering_source"], "B_q": B_q,
                    "fit_rot": fit.get("proper_rot"), "nmatches": fit.get("nmatches")})
    return out


def model5_pa(target, ts, aperture):
    """Apparent- and catalog-frame PA_V3 at the aperture for the physical nominal attitude (model 5).

    Args:
        target (astropy.coordinates.SkyCoord):
            Target (catalog coordinates)
        ts (astropy.time.Time):
            Times (array)
        aperture (dict):
            Aperture from getSIAFAperture()

    Returns:
        tuple:
            apparent PA_V3, catalog PA_V3 (numpy.ndarray, deg)
    """
    v_obs_G = getRomanVelocities(ts)
    B_C_I = calcRomanAngles(target, ts, getRomanPositions(ts), getSunPositions(ts), aperture, v_obs_G)[3]
    pa_app = getRomanPositionAngle(B_C_I, aperture).to_value(u.deg)
    pa_cat = getRomanPositionAngle(removeAberration(B_C_I, v_obs_G, aperture), aperture).to_value(u.deg)
    return pa_app, pa_cat


def held_pa_v3_at_v1(target, ts, aperture, pa_plan):
    """Predict PA_V3 at V1 of the held attitude: CGI aperture on the target, planned PA_V3 at the aperture.

    The model-5 nominal attitude (catalog frame) is rolled about the aperture by its roll from nominal,
    R = PA_nom - pa_plan, so that the PA_V3 at the aperture equals pa_plan.

    Args:
        target (astropy.coordinates.SkyCoord):
            Target (catalog coordinates)
        ts (astropy.time.Time):
            Times (array of length 1)
        aperture (dict):
            Aperture from getSIAFAperture()
        pa_plan (float):
            Planned PA_V3 at the aperture (deg, catalog frame)

    Returns:
        tuple:
            predicted PA_V3 at V1 (deg), roll from nominal R (deg)
    """
    v_obs_G = getRomanVelocities(ts)
    B_C_I = calcRomanAngles(target, ts, getRomanPositions(ts), getSunPositions(ts), aperture, v_obs_G)[3]
    B_C_I = removeAberration(B_C_I, v_obs_G, aperture)
    roll = wrap(getRomanPositionAngle(B_C_I, aperture).to_value(u.deg)[0] - pa_plan)
    B_C_I_held = applyRollAngle(B_C_I, [roll] * u.deg, aperture)
    return getRomanPositionAngle(B_C_I_held).to_value(u.deg)[0], roll


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="compare_header_pa.py",
                                 description="Compare CGI header pointing and WFI telemetry with roman_pointing models")
    ap.add_argument("datadir", help="Directory of CGI L1/L2b FITS files")
    ap.add_argument("--epoch", action="append", default=[], metavar="VISITID=TIME",
                    help="Planned attitude epoch (UTC, ISO) for a visit; repeat for several visits")
    ap.add_argument("--wfi", nargs="+", default=[], help="WFI ASDF files (e.g. *_wcs.asdf) of simultaneous exposures")
    ap.add_argument("--plan", action="append", default=[], metavar="VISITID=PA",
                    help="Planned PA_V3 at the target (OPUP 'CGI Fixed v3pa@T', deg) for a visit; repeat for several")
    args = ap.parse_args()

    aperture = getSIAFAperture()
    epochs = dict(e.split("=", 1) for e in args.epoch)
    plans = {k: float(v) for k, v in (e.split("=", 1) for e in args.plan)}
    wfi = read_wfi_pointing(args.wfi) if args.wfi else []
    files = sorted(glob.glob(os.path.join(os.path.expanduser(args.datadir), "*.fits")))

    resid = {m: [] for m in LABELS}
    quat_rows = []
    print(f"{'file':<46} {'CDMS':>4} {'epoch (UTC)':<19} | {'header PA_V3 (apparent) - model (deg)':^47} | "
          f"{'5: epoch':>8} | {'WFI telemetry (catalog) - model 5':^44}")
    print(f"{'':<46} {'':>4} {'':<19} | " + " ".join(f"{m:>8}" for m in LABELS) + f" | {'fit (min)':>8} | "
          f"{'at epoch':>10} {'held, at V1':>12} {'R(t_exp)':>9} {'t_exp':>9}")
    for f in files:
        h0, h1 = fits.getheader(f, 0), fits.getheader(f, 1)
        t_frame = Time(h1["DATETIME"], scale="utc")
        t_epoch = Time(epochs.get(str(h0["VISITID"]), h1["DATETIME"]), scale="utc")
        ts = Time([t_epoch.isot], scale="utc")
        has_siaf = "SIAF" in h0

        # catalog target: planned RA/DEC if given, else the header aperture position with the aberration removed
        if isinstance(h0.get("RA"), float) and h0["RA"] > -900:
            ra, dec = h0["RA"], h0["DEC"]
        else:
            v_frame = getRomanVelocities(Time([t_frame.isot], scale="utc"))
            r_cat = applyAberration(ecliptic_vector(h0["RA_APER"], h0["DEC_APER"])[:, None], v_frame, inverse=True)[:, 0]
            c = SkyCoord(*r_cat, representation_type="cartesian", frame=BarycentricMeanEcliptic).icrs
            ra, dec = c.ra.deg, c.dec.deg
        target = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, distance=1e6 * u.pc, pm_ra_cosdec=0 * u.mas / u.yr,
                          pm_dec=0 * u.mas / u.yr, radial_velocity=0 * u.km / u.s, frame="icrs",
                          obstime=Time("J2000")).transform_to(BarycentricMeanEcliptic)

        cols, fit = [], "-"
        if has_siaf:
            for m, (use_aperture, ephemeris) in MODELS.items():
                aper = aperture if use_aperture else None
                r_obs = getL2Positions(ts) if ephemeris == "L2" else getRomanPositions(ts)
                B_C_I = calcRomanAngles(target, ts, r_obs, getSunPositions(ts), aper)[3]
                cols.append(wrap(h0["PA_V3"] - getRomanPositionAngle(B_C_I, aper).to_value(u.deg)[0]))
            pa_app, _ = model5_pa(target, ts, aperture)
            cols.append(wrap(h0["PA_V3"] - pa_app[0]))
            # time offset from the epoch at which model 5 (apparent) equals the header PA_V3
            dt = np.linspace(-12, 12, 289) * u.hour
            d = wrap(h0["PA_V3"] - model5_pa(target, t_epoch + dt, aperture)[0])
            k = np.flatnonzero(np.diff(np.sign(d)) != 0)
            if k.size:
                i = k[np.argmin(np.abs(dt[k]))]
                fit = f"{(dt[i] - d[i] * (dt[i + 1] - dt[i]) / (d[i + 1] - d[i])).to_value(u.min):+8.1f}"
            for m, c in zip(LABELS, cols):
                resid[m].append(c)
        pa_cols = " ".join(f"{c:+8.4f}" for c in cols) if cols else f"{'preflight-SIAF header: not compared':^47}"

        telemetry = "-"
        match = [w for w in wfi if w["start"] - 60 * u.s <= t_frame <= w["end"] + 60 * u.s]
        if match:
            w = match[0]
            pa_tel = getRomanPositionAngle(w["B_C_I"][:, :, None], aperture).to_value(u.deg)[0]
            t_mid = Time([(w["start"] + (w["end"] - w["start"]) / 2).isot], scale="utc")
            telemetry = f"{wrap(pa_tel - model5_pa(target, ts, aperture)[1][0]):+10.4f}"
            if str(h0["VISITID"]) in plans:
                pa_v1, roll = held_pa_v3_at_v1(target, t_mid, aperture, plans[str(h0["VISITID"])])
                telemetry += f" {wrap(w['pa_v3'] - pa_v1):+12.4f} {roll:+9.4f} {t_mid.isot[0][11:19]:>9}"

            # quaternion check: header vs raw quaternion (apparent); WFI meta.pointing vs quaternion, aberration removed
            B_q = w["B_q"]
            B_q_cat = removeAberration(B_q[:, :, None], getRomanVelocities(t_mid), aperture)[:, :, 0]
            pa_q = getRomanPositionAngle(B_q[:, :, None], aperture).to_value(u.deg)[0]
            pa_q_cat = getRomanPositionAngle(B_q_cat[:, :, None], aperture).to_value(u.deg)[0]
            bore = np.degrees(np.linalg.norm(ecliptic_vector(h0["RA_BORE"], h0["DEC_BORE"]) - B_q[0])) * 3600
            if has_siaf:
                aper = np.degrees(np.linalg.norm(ecliptic_vector(h0["RA_APER"], h0["DEC_APER"]) - B_q.T @ aperture["u_B"])) * 3600
                hdr = f"{bore:7.3f} {aper:7.3f} {wrap(h0['PA_V3'] - pa_q):+10.6f}"
            else:
                hdr = f"{bore:7.3f} {'-':>7} {'-':>10}"
            v1_wfi = np.degrees(np.linalg.norm(w["B_C_I"][0] - B_q_cat[0])) * 3600
            pa_v1_wfi = wrap(w["pa_v3"] - getRomanPositionAngle(B_q_cat[:, :, None]).to_value(u.deg)[0])
            v1_frames = np.degrees(np.linalg.norm(B_q[0] - B_q_cat[0])) * 3600
            gaia = f"{w['fit_rot']:.4f} ({w['nmatches']})" if w["fit_rot"] is not None else "-"
            quat_rows.append(f"{os.path.basename(f):<46} {w['file'][:26]:<26} {t_mid.isot[0][11:19]:>8} | {hdr} | "
                             f"{v1_wfi:7.3f} {pa_v1_wfi:+10.6f} | {v1_frames:6.2f} {wrap(pa_q - pa_q_cat):+8.4f} | {gaia}")
        print(f"{os.path.basename(f):<46} {h0['CDMSVERS']:>4} {t_epoch.isot[:19]:<19} | {pa_cols} | {fit:>8} | {telemetry}")

    if quat_rows:
        print(f"\n{'Raw attitude quaternion check':<46} {'WFI exposure':<26} {'t_mid':>8} | "
              f"{'CGI header - quaternion':^26} | {'WFI - quat. (VA rem.)':^18} | {'quat. - VA rem.':^15} | Gaia fit")
        print(f"{'':<46} {'':<26} {'':>8} | {'BORE':>7} {'APER':>7} {'PA_V3':>10} | {'V1':>7} {'pa_v3':>10} | "
              f"{'V1':>6} {'PA_V3':>8} | rot (stars)")
        for row in quat_rows:
            print(row)
        print("Quaternion check: positions in arcsec, angles in deg. 'CGI header - quaternion' compares RA/DEC_BORE with "
              "V1, RA/DEC_APER with CGI_CEN, and PA_V3 at CGI_CEN (apparent frame). 'WFI - quat. (VA rem.)' compares the "
              "WFI ra_v1/dec_v1 and pa_v3 (at V1) with the quaternion after removeAberration. 'quat. - VA rem.' is the "
              "size of the aberration at V1 and in PA_V3 at CGI_CEN. Gaia fit: meta.wcs_fit_results proper_rot "
              "(nmatches).")

    print("\nHeader PA_V3 minus model (deg), files with in-flight SIAF headers:")
    for m, d in resid.items():
        if d:
            print(f"  {m} {LABELS[m]:<32} mean {np.mean(d):+.4f}  rms {np.sqrt(np.mean(np.square(d))):.4f}  "
                  f"max |.| {np.max(np.abs(d)):.4f}")
    print("Telemetry columns (catalog frame): 'at epoch' = PA_V3 at CGI_CEN from the WFI attitude minus model 5 at the "
          "epoch; 'held, at V1' = WFI-reported pa_v3 (at V1) minus the predicted held attitude's PA_V3 at V1 (needs "
          "--plan); R(t_exp) = roll from nominal of that prediction at the WFI mid-exposure time t_exp.")
