import datetime as dt
from datetime import timedelta
import io
import tempfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from astropy.io import ascii
from dateutil import parser

import streamlit as st
import plotly.graph_objects as go

from matplotlib.colors import Normalize, TwoSlopeNorm, to_rgba
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch, Circle  # for legend entries and hodogram circles

from netCDF4 import Dataset as NetCDFDataset, num2date

from modules import solgaleo, remapping, Finding_v_leading_and_trailing as vfit

AU = 149_597_870.7  # km

# =============================================================================
# Helper: timezone normalization (robust)
# =============================================================================


def _ensure_utc(dt_obj):
    """
    Return a timezone-aware datetime in UTC.

    Accepts:
      - python datetime
      - pandas.Timestamp
      - numpy.datetime64
      - string (will be parsed)
    If naive, treat as UTC (this is the policy for in-situ spacecraft times).
    """
    import dateutil.parser as _parser

    if dt_obj is None:
        return None

    # pandas Timestamp
    if isinstance(dt_obj, pd.Timestamp):
        dt_obj = dt_obj.to_pydatetime()

    # numpy.datetime64
    try:
        if type(dt_obj).__name__ == "datetime64":
            dt_obj = pd.to_datetime(dt_obj).to_pydatetime()
    except Exception:
        pass

    # If it's already a datetime-like object with tzinfo
    if hasattr(dt_obj, "tzinfo"):
        if dt_obj.tzinfo is None:
            return dt_obj.replace(tzinfo=dt.timezone.utc)
        else:
            return dt_obj.astimezone(dt.timezone.utc)

    # Strings etc. — parse
    try:
        parsed = _parser.parse(str(dt_obj))
    except Exception:
        raise TypeError(f"Can't interpret {dt_obj!r} as datetime")
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


# =============================================================================
# Spacecraft configuration
# =============================================================================

SPACECRAFT_PROFILES = {
    "Solar Orbiter": {
        "key": "solo",
        "has_sample_data": True,
        "cols": {
            "time": "EPOCH_yyyy-mm-ddThh:mm:ss.sssZ",
            "B": ["B_R_nT", "B_T_nT", "B_N_nT"],
            "V": ["VR_RTN_km/s", "VT_RTN_km/s", "VN_RTN_km/s"],
            "R": ["RAD_AU_AU", "HGI_LAT_deg", "HGI_LON_deg"],
        },
        "cdaweb": {
            "mag": "SOLO_L2_MAG-RTN-NORMAL-1-MINUTE",
            "plasma": "SOLO_L2_SWA-PAS-GRND-MOM",
            "pos": "SOLO_HELIO1HR_POSITION",
        },
    },
    "WIND": {
        "key": "wind",
        "has_sample_data": True,
        "cols": {
            "time": "EPOCH_yyyy-mm-ddThh:mm:ss.sssZ",
            "B": [
                "BR_(RTN)_nT_(1min)",
                "BT_(RTN)_nT_(1min)",
                "BN_(RTN)_nT_(1min)",
            ],
            "V": [
                "P+_VR_MOMENT_km/s",
                "P+_VT_MOMENT_km/s",
                "P+_VN_MOMENT_km/s",
            ],
            "R": ["RAD_AU_AU", "HGI_LAT_deg", "HGI_LON_deg"],
        },
        "cdaweb": {
            "mag": "WI_H3-RTN_MFI",
            "plasma": "WI_H1_SWE_RTN",
            "pos": "WIND_HELIO1HR_POSITION",
        },
    },
    "Aditya L1": {
        "key": "al1",
        "has_sample_data": False,
        # Aditya L1: MAG in GSM only (typical .nc variable names guessed)
        "cols": {
            "time": "time",
            "B": ["Bx_gsm", "By_gsm", "Bz_gsm"],
        },
    },
}

# =============================================================================
# Streamlit page config
# =============================================================================

st.set_page_config(
    page_title="CME Viz",
    layout="wide",
)


# =============================================================================
# Utility helpers
# =============================================================================

import base64


def show_gif_inline(gif_bytes: bytes, caption: str | None = None):
    """Display an animated GIF in Streamlit using raw HTML so it actually animates."""
    b64 = base64.b64encode(gif_bytes).decode("utf-8")
    html = f'<img src="data:image/gif;base64,{b64}" style="max-width:100%; height:auto;" />'
    st.markdown(html, unsafe_allow_html=True)
    if caption:
        st.caption(caption)


def check_time_coverage(name, t_series, t_start, t_end):
    """
    Verify that [t_start, t_end] lies within the t_series available interval.

    t_series: list-like of datetimes (may be naive) -> normalized to UTC.
    t_start/t_end: datetimes (likely timezone-aware) -> normalized to UTC.
    """
    if t_series is None or len(t_series) == 0:
        raise ValueError(f"{name} has no timestamps; cannot check coverage.")

    t0 = _ensure_utc(t_series[0])
    t1 = _ensure_utc(t_series[-1])
    t_start_u = _ensure_utc(t_start)
    t_end_u = _ensure_utc(t_end)

    if t0 > t_start_u or t1 < t_end_u:
        raise ValueError(
            f"{name} time range [{t0} – {t1}] does not fully cover "
            f"requested interval [{t_start_u} – {t_end_u}].\n\n"
            "Adjust CME start/end times to lie within the data range or upload data that covers the requested interval."
        )


def format_tdelta(td: timedelta) -> str:
    total_sec = int(td.total_seconds())
    sign = "-" if total_sec < 0 else ""
    total_sec = abs(total_sec)
    h, rem = divmod(total_sec, 3600)
    m, s = divmod(rem, 60)
    return f"{sign}{h:02d}:{m:02d}:{s:02d}"


def colormap_preview_bytes(cmap_name: str) -> bytes:
    """Return a small PNG gradient for a Matplotlib colormap.""" 
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    fig, ax = plt.subplots(figsize=(2.4, 0.4))
    ax.imshow(gradient, aspect="auto", cmap=plt.colormaps.get_cmap(cmap_name))
    ax.set_axis_off()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# =============================================================================
# CSV loaders (bundled + uploaded) — ensure times normalized to UTC
# =============================================================================


def parse_csv_tables(spacecraft_name, tbl_B, tbl_V, tbl_R):
    profile = SPACECRAFT_PROFILES[spacecraft_name]
    cols = profile["cols"]
    time_col = cols["time"]

    # MAG
    t_B = []
    for t in tbl_B[time_col]:
        try:
            dt_parsed = parser.isoparse(str(t))
        except Exception:
            dt_parsed = pd.to_datetime(str(t))
        t_B.append(_ensure_utc(dt_parsed))

    B_r = np.asarray(tbl_B[cols["B"][0]], dtype=float)
    B_t = np.asarray(tbl_B[cols["B"][1]], dtype=float)
    B_n = np.asarray(tbl_B[cols["B"][2]], dtype=float)
    B_rtn = np.column_stack((B_r, B_t, B_n))

    # V
    t_V = []
    for t in tbl_V[time_col]:
        try:
            dt_parsed = parser.isoparse(str(t))
        except Exception:
            dt_parsed = pd.to_datetime(str(t))
        t_V.append(_ensure_utc(dt_parsed))
    V_r = np.asarray(tbl_V[cols["V"][0]], dtype=float)
    V_t = np.asarray(tbl_V[cols["V"][1]], dtype=float)
    V_n = np.asarray(tbl_V[cols["V"][2]], dtype=float)
    V_rtn = np.column_stack((V_r, V_t, V_n))

    # Position (HGI)
    t_R = []
    for t in tbl_R[time_col]:
        try:
            dt_parsed = parser.isoparse(str(t))
        except Exception:
            dt_parsed = pd.to_datetime(str(t))
        t_R.append(_ensure_utc(dt_parsed))

    R_sc_r = np.asarray(tbl_R[cols["R"][0]], dtype=float) * AU
    R_sc_lat = np.asarray(tbl_R[cols["R"][1]], dtype=float)
    R_sc_lon = np.asarray(tbl_R[cols["R"][2]], dtype=float)
    theta = 90 - np.array(R_sc_lat)
    R_sc_hgi = solgaleo.batch_spherical_to_cartesian(
        R_sc_r, theta, R_sc_lon, degrees=True
    )

    return (t_B, B_rtn), (t_V, V_rtn), (t_R, R_sc_hgi)


@st.cache_data(show_spinner=True)
def load_sample_solo():
    mag_file = "sample_data/SOLO_L2_MAG-RTN-NORMAL-1-MINUTE_1727420.csv"
    swa_file = "sample_data/SOLO_L2_SWA-PAS-GRND-MOM_1727420.csv"
    pos_file = "sample_data/SOLO_HELIO1HR_POSITION_1727420.csv"

    tbl_B = ascii.read(mag_file)
    tbl_V = ascii.read(swa_file)
    tbl_R = ascii.read(pos_file)

    return parse_csv_tables("Solar Orbiter", tbl_B, tbl_V, tbl_R)


@st.cache_data(show_spinner=True)
def load_sample_wind():
    mag_file = "sample_data/WI_H3-RTN_MFI_1739706.csv"
    swa_file = "sample_data/WI_H1_SWE_RTN_1739706.csv"
    pos_file = "sample_data/WIND_HELIO1HR_POSITION_1739706.csv"

    tbl_B = ascii.read(mag_file)
    tbl_V = ascii.read(swa_file)
    tbl_R = ascii.read(pos_file)

    return parse_csv_tables("WIND", tbl_B, tbl_V, tbl_R)


# =============================================================================
# NetCDF loader for MAG uploads (.nc) - Generic (Aditya/GSM-friendly)
# =============================================================================


def _nc_try_num2date(nc, time_var_name):
    tvar = nc.variables[time_var_name]
    if hasattr(tvar, "units"):
        try:
            dates = num2date(tvar[:], units=tvar.units)
            dates = [pd.to_datetime(str(d)).to_pydatetime() for d in dates]
            return [ _ensure_utc(d) for d in dates ]
        except Exception:
            pass
    # fallback: epoch seconds
    try:
        return [ _ensure_utc(pd.to_datetime(float(x), unit="s").to_pydatetime()) for x in tvar[:] ]
    except Exception:
        raise


def load_from_uploaded_netcdf_list(nc_uploaded_files):
    """
    Accepts a list of streamlit UploadedFile objects (or a single object).
    Returns: (t_B, B_arr), ([], []), ([], [])
    B_arr shape: (N,3) arranged as (Bx, By, Bz) in that order (guessed from var names)
    """
    if nc_uploaded_files is None:
        raise ValueError("No netCDF file(s) provided.")

    if not isinstance(nc_uploaded_files, (list, tuple)):
        uploaded_list = [nc_uploaded_files]
    else:
        uploaded_list = nc_uploaded_files

    times_all = []
    Bx_all = []
    By_all = []
    Bz_all = []

    # variable name candidates (common variations)
    var_names = {
        "Bx": ["Bx_gsm", "Bx_gsm_1", "Bx", "B_x", "B1", "BX"],
        "By": ["By_gsm", "By_gsm_1", "By", "B_y", "B2", "BY"],
        "Bz": ["Bz_gsm", "Bz_gsm_1", "Bz", "B_z", "B3", "BZ"],
        "time": ["time", "Time", "epoch", "Epoch"],
    }

    for up in uploaded_list:
        # write uploaded bytes to temp file for netCDF4 to open
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp.write(up.read())
            tmp.flush()
            tmp_name = tmp.name

        nc = NetCDFDataset(tmp_name, mode="r")

        # find time variable name
        tname = None
        for cand in var_names["time"]:
            if cand in nc.variables:
                tname = cand
                break
        if tname is None:
            # fallback: pick first 1D variable
            for v in nc.variables:
                arr = nc.variables[v][:]
                if getattr(arr, "ndim", 0) == 1:
                    tname = v
                    break
        if tname is None:
            nc.close()
            raise KeyError("No time variable found in netCDF")

        # robust conversion to datetimes
        try:
            t_list = _nc_try_num2date(nc, tname)
        except Exception:
            tdata = nc.variables[tname][:]
            try:
                t_list = [ _ensure_utc(pd.to_datetime(str(x)).to_pydatetime()) for x in tdata ]
            except Exception:
                nc.close()
                raise

        # B components
        def _find_var_or_none(nc, candidates):
            for n in candidates:
                if n in nc.variables:
                    try:
                        return np.array(nc.variables[n][:], dtype=float)
                    except Exception:
                        return np.array(nc.variables[n][:].astype(float))
            return None

        Bx = _find_var_or_none(nc, var_names["Bx"])
        By = _find_var_or_none(nc, var_names["By"])
        Bz = _find_var_or_none(nc, var_names["Bz"])

        nc.close()

        if Bx is None or By is None or Bz is None:
            raise KeyError("Could not find Bx/By/Bz variables in .nc (tried common names)")

        times_all.extend(t_list)
        Bx_all.extend(Bx.tolist())
        By_all.extend(By.tolist())
        Bz_all.extend(Bz.tolist())

    B_arr = np.column_stack((np.array(Bx_all), np.array(By_all), np.array(Bz_all)))
    t_B = times_all
    return (t_B, B_arr), ([], []), ([], [])


# =============================================================================
# CDAWeb helpers (unchanged except time normalization)
# =============================================================================


def _import_sunpy():
    try:
        from sunpy.net import Fido, attrs as a
        from sunpy.timeseries import TimeSeries
    except ImportError as exc:
        raise RuntimeError(
            "CDAWeb download requires SunPy.\n"
            "Install with:  pip install 'sunpy[all]' cdflib"
        ) from exc
    return Fido, a, TimeSeries


def _find_column(df, substrings, what="column"):
    subs = [s.lower() for s in substrings]
    for col in df.columns:
        low = col.lower()
        if all(s in low for s in subs):
            return col
    raise KeyError(
        f"Could not find {what} with substrings {substrings} in columns {list(df.columns)}"
    )


def load_from_cdaweb_solo(t_start, t_end):
    Fido, a, TimeSeries = _import_sunpy()
    cfg = SPACECRAFT_PROFILES["Solar Orbiter"]["cdaweb"]

    pad = timedelta(hours=24)
    t0 = (t_start - pad).strftime("%Y-%m-%dT%H:%M:%S")
    t1 = (t_end + pad).strftime("%Y-%m-%dT%H:%M:%S")
    time_attr = a.Time(t0, t1)

    # MAG
    ds_mag = a.cdaweb.Dataset(cfg["mag"])
    res_mag = Fido.search(time_attr, ds_mag)
    if len(res_mag) == 0 or len(res_mag[0]) == 0:
        raise RuntimeError(f"No SolO MAG ({cfg['mag']}) data found in this interval.")
    files_mag = Fido.fetch(res_mag)
    ts_mag = TimeSeries(files_mag, concatenate=True)
    df_mag = ts_mag.to_dataframe().sort_index()
    t_B = [ _ensure_utc(d.to_pydatetime()) for d in pd.DatetimeIndex(df_mag.index).tz_localize("UTC") ]

    mag_cols = list(df_mag.columns)
    if all(c in mag_cols for c in ["B_RTN_0", "B_RTN_1", "B_RTN_2"]):
        br_col, bt_col, bn_col = "B_RTN_0", "B_RTN_1", "B_RTN_2"
    else:
        br_col = _find_column(df_mag, ["br", "rtn"], "SolO BR (RTN)")
        bt_col = _find_column(df_mag, ["bt", "rtn"], "SolO BT (RTN)")
        bn_col = _find_column(df_mag, ["bn", "rtn"], "SolO BN (RTN)")
    B_rtn = df_mag[[br_col, bt_col, bn_col]].to_numpy()

    # Plasma
    ds_pl = a.cdaweb.Dataset(cfg["plasma"])
    res_pl = Fido.search(time_attr, ds_pl)
    if len(res_pl) == 0 or len(res_pl[0]) == 0:
        raise RuntimeError(
            f"No SolO plasma ({cfg['plasma']}) data found in this interval."
        )
    files_pl = Fido.fetch(res_pl)
    ts_pl = TimeSeries(files_pl, concatenate=True)
    df_pl = ts_pl.to_dataframe().sort_index()
    t_V = [ _ensure_utc(d.to_pydatetime()) for d in pd.DatetimeIndex(df_pl.index).tz_localize("UTC") ]

    pl_cols = list(df_pl.columns)
    if all(c in pl_cols for c in ["VR_RTN_km/s", "VT_RTN_km/s", "VN_RTN_km/s"]):
        vr_col, vt_col, vn_col = "VR_RTN_km/s", "VT_RTN_km/s", "VN_RTN_km/s"
    elif all(c in pl_cols for c in ["V_RTN_0", "V_RTN_1", "V_RTN_2"]):
        vr_col, vt_col, vn_col = "V_RTN_0", "V_RTN_1", "V_RTN_2"
    else:
        vr_col = _find_column(df_pl, ["vr", "rtn"], "SolO VR (RTN)")
        vt_col = _find_column(df_pl, ["vt", "rtn"], "SolO VT (RTN)")
        vn_col = _find_column(df_pl, ["vn", "rtn"], "SolO VN (RTN)")
    V_rtn = df_pl[[vr_col, vt_col, vn_col]].to_numpy()

    # Position
    ds_pos = a.cdaweb.Dataset(cfg["pos"])
    res_pos = Fido.search(time_attr, ds_pos)
    if len(res_pos) == 0 or len(res_pos[0]) == 0:
        raise RuntimeError(
            f"No SolO position ({cfg['pos']}) data found in this interval"
        )
    files_pos = Fido.fetch(res_pos)
    ts_pos = TimeSeries(files_pos, concatenate=True)
    df_pos = ts_pos.to_dataframe().sort_index()
    t_R = [ _ensure_utc(d.to_pydatetime()) for d in pd.DatetimeIndex(df_pos.index).tz_localize("UTC") ]

    rad_col = _find_column(df_pos, ["rad", "au"], "SolO radial distance [AU]")
    lat_col = _find_column(df_pos, ["hgi", "lat"], "SolO HGI latitude")
    lon_col = _find_column(df_pos, ["hgi", "lon"], "SolO HGI longitude")

    R_sc_r_au = df_pos[rad_col].to_numpy()
    R_sc_lat = df_pos[lat_col].to_numpy()
    R_sc_lon = df_pos[lon_col].to_numpy()

    R_sc_r = R_sc_r_au * AU
    theta = 90 - R_sc_lat
    R_sc_hgi = solgaleo.batch_spherical_to_cartesian(
        R_sc_r, theta, R_sc_lon, degrees=True
    )

    return (t_B, B_rtn), (t_V, V_rtn), (t_R, R_sc_hgi)


def load_from_cdaweb_wind(t_start, t_end):
    Fido, a, TimeSeries = _import_sunpy()
    cfg = SPACECRAFT_PROFILES["WIND"]["cdaweb"]

    pad = timedelta(hours=24)
    t0 = (t_start - pad).strftime("%Y-%m-%dT%H:%M:%S")
    t1 = (t_end + pad).strftime("%Y-%m-%dT%H:%M:%S")
    time_attr = a.Time(t0, t1)

    # MAG
    ds_mag = a.cdaweb.Dataset(cfg["mag"])
    res_mag = Fido.search(time_attr, ds_mag)
    if len(res_mag) == 0 or len(res_mag[0]) == 0:
        raise RuntimeError(f"No WIND MAG ({cfg['mag']}) data found in this interval.")
    files_mag = Fido.fetch(res_mag)
    ts_mag = TimeSeries(files_mag, concatenate=True)
    df_mag = ts_mag.to_dataframe().sort_index()
    t_B = [ _ensure_utc(d.to_pydatetime()) for d in pd.DatetimeIndex(df_mag.index).tz_localize("UTC") ]

    mag_cols = list(df_mag.columns)
    if all(c in mag_cols for c in ["BRTN_0", "BRTN_1", "BRTN_2"]):
        br_col, bt_col, bn_col = "BRTN_0", "BRTN_1", "BRTN_2"
    elif all(c in mag_cols for c in ["B_RTN_0", "B_RTN_1", "B_RTN_2"]):
        br_col, bt_col, bn_col = "B_RTN_0", "B_RTN_1", "B_RTN_2"
    else:
        br_col = _find_column(df_mag, ["brtn_0"], "WIND BR (RTN)")
        bt_col = _find_column(df_mag, ["brtn_1"], "WIND BT (RTN)")
        bn_col = _find_column(df_mag, ["brtn_2"], "WIND BN (RTN)")
    B_rtn = df_mag[[br_col, bt_col, bn_col]].to_numpy()

    # SWE (plasma)
    ds_swe = a.cdaweb.Dataset(cfg["plasma"])
    res_swe = Fido.search(time_attr, ds_swe)
    if len(res_swe) == 0 or len(res_swe[0]) == 0:
        raise RuntimeError(
            f"No WIND SWE ({cfg['plasma']}) data found in this interval"
        )
    files_swe = Fido.fetch(res_swe)
    ts_swe = TimeSeries(files_swe, concatenate=True)
    df_swe = ts_swe.to_dataframe().sort_index()
    t_V = [ _ensure_utc(d.to_pydatetime()) for d in pd.DatetimeIndex(df_swe.index).tz_localize("UTC") ]

    swe_cols = list(df_swe.columns)
    if all(c in swe_cols for c in ["VR_RTN", "VT_RTN", "VN_RTN"]):
        vr_col, vt_col, vn_col = "VR_RTN", "VT_RTN", "VN_RTN"
    elif all(c in swe_cols for c in ["V_RTN_0", "V_RTN_1", "V_RTN_2"]):
        vr_col, vt_col, vn_col = "V_RTN_0", "V_RTN_1", "V_RTN_2"
    else:
        vr_col = _find_column(df_swe, ["vr", "rtn"], "WIND VR (RTN)")
        vt_col = _find_column(df_swe, ["vt", "rtn"], "WIND VT (RTN)")
        vn_col = _find_column(df_swe, ["vn", "rtn"], "WIND VN (RTN)")
    V_rtn = df_swe[[vr_col, vt_col, vn_col]].to_numpy()

    # Position
    ds_pos = a.cdaweb.Dataset(cfg['pos'])
    res_pos = Fido.search(time_attr, ds_pos)
    if len(res_pos) == 0 or len(res_pos[0]) == 0:
        raise RuntimeError(
            f"No WIND position ({cfg['pos']}) data found in this interval."
        )
    files_pos = Fido.fetch(res_pos)
    ts_pos = TimeSeries(files_pos, concatenate=True)
    df_pos = ts_pos.to_dataframe().sort_index()
    t_R = [ _ensure_utc(d.to_pydatetime()) for d in pd.DatetimeIndex(df_pos.index).tz_localize("UTC") ]

    rad_col = _find_column(df_pos, ["rad", "au"], "WIND radial distance [AU]")
    lat_col = _find_column(df_pos, ["hgi", "lat"], "WIND HGI latitude")
    lon_col = _find_column(df_pos, ["hgi", "lon"], "WIND HGI longitude")

    R_sc_r_au = df_pos[rad_col].to_numpy()
    R_sc_lat = df_pos[lat_col].to_numpy()
    R_sc_lon = df_pos[lon_col].to_numpy()

    R_sc_r = R_sc_r_au * AU
    theta = 90 - R_sc_lat
    R_sc_hgi = solgaleo.batch_spherical_to_cartesian(
        R_sc_r, theta, R_sc_lon, degrees=True
    )

    return (t_B, B_rtn), (t_V, V_rtn), (t_R, R_sc_hgi)


def load_data_for_spacecraft(spacecraft_name, data_mode, uploads, t_cme_start, t_cme_end):
    profile = SPACECRAFT_PROFILES[spacecraft_name]
    key = profile["key"]

    if data_mode == "Bundled sample":
        if not profile["has_sample_data"]:
            raise ValueError(f"No bundled sample data for {spacecraft_name}.")
        if key == "solo":
            return load_sample_solo()
        elif key == "wind":
            return load_sample_wind()

    if data_mode == "Upload CSVs":
        mag_file, swa_file, pos_file = uploads
        if mag_file is None:
            raise ValueError("MAG CSV is required for Upload CSVs mode.")
        # Parse what is present; caller logic decides if V/R needed
        tbl_B = ascii.read(mag_file)
        tbl_V = ascii.read(swa_file) if (swa_file is not None) else tbl_B
        tbl_R = ascii.read(pos_file) if (pos_file is not None) else tbl_B
        return parse_csv_tables(spacecraft_name, tbl_B, tbl_V, tbl_R)

    if data_mode == "CDAWeb (SolO)":
        span = (t_cme_end - t_cme_start).total_seconds()
        if span > 2 * 24 * 3600:
            raise RuntimeError(
                "CDAWeb mode is disabled for time ranges longer than 2 days.\n"
                "Please shorten the CME interval or use CSV upload mode."
            )

        if spacecraft_name == "Solar Orbiter":
            return load_from_cdaweb_solo(t_cme_start, t_cme_end)
        elif spacecraft_name == "WIND":
            return load_from_cdaweb_wind(t_cme_start, t_cme_end)
        else:
            raise RuntimeError(
                f"CDAWeb is not available for {spacecraft_name}. "
                "Select 'Upload CSVs' instead."
            )

    raise ValueError(f"Unknown data mode '{data_mode}'.")


# =============================================================================
# Remapping core (kept unchanged)
# =============================================================================

def compute_remap_for_spacecraft(
    t_B,
    B_rtn,
    t_V,
    V_rtn,
    t_R_sc,
    R_sc_hgi,
    t_cme_start,
    t_mo_start,
    t_cme_end,
):
    idx_cme_start_orig = np.argmin([abs(t - t_cme_start) for t in t_B])
    idx_cme_end_orig = np.argmin([abs(t - t_cme_end) for t in t_B])

    t_B_clip = t_B[idx_cme_start_orig: idx_cme_end_orig + 1]
    B_rtn_clip = B_rtn[idx_cme_start_orig: idx_cme_end_orig + 1]

    idx_v_start = np.argmin([abs(t - t_cme_start) for t in t_V])
    idx_v_end = np.argmin([abs(t - t_cme_end) for t in t_V])
    t_V_clip = t_V[idx_v_start: idx_v_end + 1]
    V_rtn_clip = V_rtn[idx_v_start: idx_v_end + 1]

    idx_r_start = np.argmin([abs(t - t_cme_start) for t in t_R_sc])
    idx_r_end = np.argmin([abs(t - t_cme_end) for t in t_R_sc])
    t_R_sc_clip = t_R_sc[idx_r_start: idx_r_end + 1]
    R_sc_hgi_clip = R_sc_hgi[idx_r_start: idx_r_end + 1]

    V_bounds = [(-2e4, 2e4), (-2e4, 2e4), (-2e4, 2e4)]
    B_bounds = [(-1e4, 1e4)] * 3
    R_bounds = [(-1e10, 1e10)] * 3

    B_mask = solgaleo.mask_vectors_by_components(B_rtn_clip, B_bounds)
    V_mask = solgaleo.mask_vectors_by_components(V_rtn_clip, V_bounds)
    R_mask = solgaleo.mask_vectors_by_components(R_sc_hgi_clip, R_bounds)

    t_B_clean = [t for t, ok in zip(t_B_clip, B_mask) if ok]
    B_rtn_clean = B_rtn_clip[B_mask]

    t_V_clean = [t for t, ok in zip(t_V_clip, V_mask) if ok]
    V_rtn_clean = V_rtn_clip[V_mask]

    t_R_sc_clean = [t for t, ok in zip(t_R_sc_clip, R_mask) if ok]
    R_sc_hgi_clean = R_sc_hgi_clip[R_mask]

    V_rtn_interp = solgaleo.interpolate_vector_series(
        t_V_clean, V_rtn_clean, t_B_clean
    )
    R_sc_hgi_interp = solgaleo.pchip_vector_interp(
        t_R_sc_clean, R_sc_hgi_clean, t_B_clean
    )

    V_hgi = solgaleo.batch_rtn_to_hgi(V_rtn_interp, R_sc_hgi_interp)
    B_hgi = solgaleo.batch_rtn_to_hgi(B_rtn_clean, R_sc_hgi_interp)
    B_mag = np.sqrt(np.sum(B_rtn_clean ** 2, axis=1))

    idx_cme_start = np.argmin([abs(t - t_cme_start) for t in t_B_clean])
    idx_mo_start = np.argmin([abs(t - t_mo_start) for t in t_B_clean])
    idx_cme_end = np.argmin([abs(t - t_cme_end) for t in t_B_clean])

    v_le_sheath, v_te_sheath, v_cm_sheath, v_exp_sheath = vfit.fit_velocity_edges(
        t_B_clean,
        V_hgi,
        t_cme_start,
        t_mo_start,
        plot=False,
        title=None,
    )
    v_le_mo, v_te_mo, v_cm_mo, v_exp_mo = vfit.fit_velocity_edges(
        t_B_clean,
        V_hgi,
        t_mo_start,
        t_cme_end,
        plot=False,
        title=None,
    )

    R_remapped, is_sheath = remapping.calculate_cme_positions(
        t_B_clean,
        R_sc_hgi_interp,
        t_cme_start,
        t_mo_start,
        t_cme_end,
        v_cm_sheath,
        v_exp_sheath,
        v_cm_mo,
        v_exp_mo,
    )

    R_remapped_rotated = solgaleo.custom_rotate_frame(
        R_remapped, R_remapped[idx_cme_start]
    )
    B_hgi_rotated = solgaleo.custom_rotate_frame(B_hgi, R_remapped[idx_cme_start])

    X = R_remapped_rotated[:, 0] / AU
    Y = R_remapped_rotated[:, 1] / AU
    Z = R_remapped_rotated[:, 2] / AU

    Bx = B_hgi_rotated[:, 0]
    By = B_hgi_rotated[:, 1]
    Bz = B_hgi_rotated[:, 2]

    mask_posX = X >= 0
    if not np.any(mask_posX):
        raise ValueError("No remapped points with X ≥ 0; cannot plot.")

    X_plot = X[mask_posX]
    Y_plot = Y[mask_posX]
    Z_plot = Z[mask_posX]

    Bx_plot = Bx[mask_posX]
    By_plot = By[mask_posX]
    Bz_plot = Bz[mask_posX]
    B_mag_plot = B_mag[mask_posX]
    t_B_plot = [t for i, t in enumerate(t_B_clean) if mask_posX[i]]
    is_sheath_plot = np.array(is_sheath)[mask_posX]

    # Map original indices to masked indices (X >= 0)
    original_to_masked = {}
    k = 0
    for i, ok in enumerate(mask_posX):
        if ok:
            original_to_masked[i] = k
            k += 1

    idx_cme_start_masked = original_to_masked.get(idx_cme_start, None)
    idx_mo_start_masked = original_to_masked.get(idx_mo_start, None)
    idx_cme_end_masked = original_to_masked.get(idx_cme_end, None)

    return {
        "t_B_clean": t_B_clean,
        "X": X_plot,
        "Y": Y_plot,
        "Z": Z_plot,
        "Bx": Bx_plot,
        "By": By_plot,
        "Bz": Bz_plot,
        "Bmag": B_mag_plot,
        "is_sheath": is_sheath_plot,
        "mask_posX": mask_posX,
        "idx_cme_start": idx_cme_start,
        "idx_mo_start": idx_mo_start,
        "idx_cme_end": idx_cme_end,
        "idx_cme_start_masked": idx_cme_start_masked,
        "idx_mo_start_masked": idx_mo_start_masked,
        "idx_cme_end_masked": idx_cme_end_masked,
        "t_B_plot": t_B_plot,
    }


# =============================================================================
# GIF helpers (generate_cme_frames unchanged except denser arrows)
# =============================================================================


def generate_cme_frames(
    X,
    Y,
    Z,
    B_mag,
    Bx,
    By,
    Bz,
    idx_cme_start_masked=None,
    idx_mo_start_masked=None,
    idx_cme_end_masked=None,
    plane_extent=0.45,
    colormap_mag="viridis",
    bg_choice="white",
    n_frames=40,
    dpi=140,
    show_colorbar=True,
    arrow_density=150,
):
    import imageio.v2 as imageio

    X_arr = np.asarray(X, dtype=float)
    Y_arr = np.asarray(Y, dtype=float)
    Z_arr = np.asarray(Z, dtype=float)
    B_arr = np.asarray(B_mag, dtype=float)
    Bx_arr = np.asarray(Bx, dtype=float)
    By_arr = np.asarray(By, dtype=float)
    Bz_arr = np.asarray(Bz, dtype=float)

    n_points = len(X_arr)
    if n_points < 2:
        raise ValueError("Not enough points to build animation.")

    n_frames = max(5, min(n_frames, n_points))
    frame_indices = np.linspace(1, n_points, n_frames, dtype=int)

    # Fixed spatial scaling
    xrange = 1.1 * (X_arr.max() - X_arr.min())
    if xrange <= 0:
        xrange = 0.1
    xmid = 0.5 * (X_arr.max() + X_arr.min())
    xlim = (xmid - xrange / 2, xmid + xrange / 2)

    xlen, ylen, zlen = 4, 1, 1
    yrange = xrange * ylen / xlen
    zrange = xrange * zlen / xlen
    ylim = (-yrange / 2, yrange / 2)
    zlim = (-zrange / 2, zrange / 2)

    cmap = plt.colormaps.get_cmap(colormap_mag)

    finite = np.isfinite(B_arr)
    if not np.any(finite):
        vmin, vmax = -1.0, 1.0
    else:
        vmin = float(np.nanmin(B_arr[finite]))
        vmax = float(np.nanmax(B_arr[finite]))
        if vmin == vmax:
            vmin -= 1.0
            vmax += 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)

    # B length scaling for quivers
    Bvec = np.sqrt(Bx_arr ** 2 + By_arr ** 2 + Bz_arr ** 2)
    Bmax = np.nanmax(Bvec) if np.any(np.isfinite(Bvec)) else 1.0
    B_scale_factor = 0.707 * yrange / Bmax if Bmax > 0 else 1.0

    # Plane extents
    y_min_all, y_max_all = float(Y_arr.min()), float(Y_arr.max())
    z_min_all, z_max_all = float(Z_arr.min()), float(Z_arr.max())
    y_span = y_max_all - y_min_all
    z_span = z_max_all - z_min_all

    y_mid = 0.5 * (y_max_all + y_min_all)
    z_mid = 0.5 * (z_max_all + z_min_all)

    y_min = y_mid - 0.5 * plane_extent * y_span
    y_max = y_mid + 0.5 * plane_extent * y_span
    z_min = z_mid - 0.5 * plane_extent * z_span
    z_max = z_mid + 0.5 * plane_extent * z_span

    frames = []

    for idx in frame_indices:
        fig = plt.figure(figsize=(8.5, 8.5))

        if show_colorbar:
            ax = fig.add_axes([0.05, 0.32, 0.84, 0.63], projection="3d")
            cax = fig.add_axes([0.18, 0.16, 0.64, 0.055])
        else:
            ax = fig.add_axes([0.05, 0.12, 0.84, 0.8], projection="3d")
            cax = None

        if bg_choice == "transparent":
            fig.patch.set_alpha(0.0)
            ax.set_facecolor("none")
        else:
            fig.patch.set_facecolor(bg_choice)
            ax.set_facecolor(bg_choice)

        # Scatter path (up to current frame)
        ax.scatter(
            X_arr[:idx],
            Y_arr[:idx],
            Z_arr[:idx],
            c=B_arr[:idx],
            cmap=cmap,
            norm=norm,
            s=8,
            alpha=0.95,
        )

        # Quivers (subsampled) — densified by factor ~2
        step = max(1, idx // max(1, int(arrow_density * 2)))
        for j in range(0, idx, step):
            if not np.isfinite(B_arr[j]):
                continue
            val_norm = (B_arr[j] - vmin) / (vmax - vmin)
            val_norm = float(np.clip(val_norm, 0.0, 1.0))
            rgba = to_rgba(cmap(val_norm), alpha=1.0)
            u = Bx_arr[j] * B_scale_factor
            v = By_arr[j] * B_scale_factor
            w = Bz_arr[j] * B_scale_factor
            ax.quiver(
                X_arr[j],
                Y_arr[j],
                Z_arr[j],
                u,
                v,
                w,
                color=rgba,
                arrow_length_ratio=0.1,
                linewidth=0.6,
            )

        # CME boundary planes (if indices available)
        def add_plane(masked_idx, color):
            if masked_idx is None or masked_idx >= len(X_arr):
                return
            x0 = float(X_arr[masked_idx])
            Xp = np.array([[x0, x0], [x0, x0]])
            Yp = np.array([[y_min, y_max], [y_min, y_max]])
            Zp = np.array([[z_min, z_min], [z_max, z_max]])
            ax.plot_surface(
                Xp,
                Yp,
                Zp,
                color=color,
                alpha=0.2,
                linewidth=0.0,
            )

        add_plane(idx_cme_start_masked, "blue")
        add_plane(idx_mo_start_masked, "blueviolet")
        add_plane(idx_cme_end_masked, "red")

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.set_box_aspect([xlen, ylen, zlen])

        ax.set_xlabel("$R_0$ (AU)", labelpad=12)
        ax.set_ylabel("$T_0$ (AU)", labelpad=8)
        ax.set_zlabel("$N_0$ (AU)", labelpad=8)

        if show_colorbar and cax is not None:
            sm = ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
            cb.set_label("|B| (nT)")
            cb.ax.tick_params(labelsize=8)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        plt.close(fig)
        buf.seek(0)
        frames.append(imageio.imread(buf))

    return frames


def make_cme_gif(
    X,
    Y,
    Z,
    B_mag,
    Bx,
    By,
    Bz,
    idx_cme_start_masked=None,
    idx_mo_start_masked=None,
    idx_cme_end_masked=None,
    plane_extent=0.45,
    colormap_mag="viridis",
    bg_choice="white",
    n_frames=40,
    dpi=140,
    show_colorbar=True,
    frame_duration=0.15,
    arrow_density=150,
):
    import imageio.v2 as imageio

    # generate_cme_frames does not accept frame_duration — duration supplied to mimsave
    frames = generate_cme_frames(
        X,
        Y,
        Z,
        B_mag,
        Bx,
        By,
        Bz,
        idx_cme_start_masked=idx_cme_start_masked,
        idx_mo_start_masked=idx_mo_start_masked,
        idx_cme_end_masked=idx_cme_end_masked,
        plane_extent=plane_extent,
        colormap_mag=colormap_mag,
        bg_choice=bg_choice,
        n_frames=n_frames,
        dpi=dpi,
        show_colorbar=show_colorbar,
        arrow_density=arrow_density,
    )

    gif_bytes = io.BytesIO()
    imageio.mimsave(gif_bytes, frames, format="GIF", duration=frame_duration)
    gif_bytes.seek(0)
    return gif_bytes


def make_rotation_gif_from_axes(
    fig,
    ax,
    dpi=140,
    n_frames=60,
    elev=20,
    direction=1,
    frame_duration=0.12,
):
    import imageio.v2 as imageio

    frames = []

    base_az = np.linspace(0, 360, n_frames, endpoint=False)
    azims = direction * base_az

    w, h = fig.get_size_inches()
    fig.set_size_inches(w, h)

    for az in azims:
        ax.view_init(elev=elev, azim=float(az))

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        frames.append(imageio.imread(buf))

    gif_bytes = io.BytesIO()
    imageio.mimsave(gif_bytes, frames, format="GIF", duration=frame_duration)
    gif_bytes.seek(0)
    return gif_bytes


# =============================================================================
# UI: Title
# =============================================================================

st.title("☀️ CME Viz")

st.caption(
    "Interactively remap in-situ CME measurements into a spatial view, "
    "using Solar Orbiter, WIND, or Aditya L1 (temporal-only) data."
)

# =============================================================================
# Sidebar – compact control panel
# =============================================================================

with st.sidebar:
    st.markdown("# ☀️ CME Details")
    event_name = st.text_input("# Event Label*", value="")
    st.markdown("---")
    st.markdown("### CME timing (UTC)")

    default_cme_start = dt.datetime(2024, 3, 23, 13, 32, tzinfo=dt.timezone.utc)
    default_mo_start = dt.datetime(2024, 3, 23, 15, 5, tzinfo=dt.timezone.utc)
    default_cme_end = dt.datetime(2024, 3, 24, 2, 10, tzinfo=dt.timezone.utc)

    cme_start_date = st.date_input(
        "CME start date",
        value=default_cme_start.date(),
        key="cme_start_date",
    )
    cme_start_time = st.time_input(
        "CME start time",
        value=default_cme_start.time(),
        key="cme_start_time",
    )

    mo_start_date = st.date_input(
        "MO start date",
        value=default_mo_start.date(),
        key="mo_start_date",
    )
    mo_start_time = st.time_input(
        "MO start time",
        value=default_mo_start.time(),
        key="mo_start_time",
    )

    cme_end_date = st.date_input(
        "CME end date",
        value=default_cme_end.date(),
        key="cme_end_date",
    )
    cme_end_time = st.time_input(
        "CME end time",
        value=default_cme_end.time(),
        key="cme_end_time",
    )

    st.markdown("---")
    st.markdown("### Spacecraft & data")

    spacecraft_name = st.selectbox(
        "Spacecraft",
        list(SPACECRAFT_PROFILES.keys()),
        index=0,
    )
    profile = SPACECRAFT_PROFILES[spacecraft_name]

    data_mode = st.radio(
        "Data source",
        [
            "Bundled sample",
            "Upload CSVs",
            "CDAWeb (SolO)",
        ],
        index=0 if profile["has_sample_data"] else 1,
    )

    st.markdown("---")
    st.markdown("### Uploads / Optional NETCDF")

    # Show NETCDF uploader only when Aditya L1 selected
    uploaded_netcdf = None
    if spacecraft_name == "Aditya L1":
        uploaded_netcdf = st.file_uploader(
            "Upload Aditya L1 MAG netCDF (.nc) files (GSM)",
            type=["nc"],
            accept_multiple_files=False,
        )

    # Temporal-only checkbox BEFORE CSV upload widgets so we can hide V/R when temporal-only
    temporal_only = st.checkbox(
        "Temporal-only plots (skip spatial remap)",
        value=False,
        help="If set, create only temporal panels (angle vs time, hodogram). Useful when V or R not available or for Aditya L1."
    )

    # Upload CSVs — show MAG always (unless Aditya L1: encourage .nc upload only); show V and R only if NOT temporal-only
    uploaded_mag = uploaded_swa = uploaded_pos = None
    if data_mode == "Upload CSVs":
        if spacecraft_name == "Aditya L1":
            st.warning("For Aditya L1 prefer uploading .nc files (use the NETCDF uploader shown above). CSV RTN uploads are not provided for Aditya L1.")
        else:
            st.caption("Use the same column naming convention as your existing CSVs.")
            uploaded_mag = st.file_uploader("MAG RTN CSV", type=["csv"])
            if not temporal_only:
                uploaded_swa = st.file_uploader("Plasma V CSV", type=["csv"])
                uploaded_pos = st.file_uploader("Position (HGI) CSV", type=["csv"])
            else:
                st.caption("Temporal-only selected: Plasma V & Position upload disabled.")

    st.markdown("---")
    st.markdown("### Run")
    do_plot = st.button("Compute & plot")

# Convert sidebar times to aware datetimes (guarded combine)
if (
    cme_start_date is None
    or cme_start_time is None
    or mo_start_date is None
    or mo_start_time is None
    or cme_end_date is None
    or cme_end_time is None
):
    st.error("Please select CME/MO start/end date *and* time in the sidebar before running. One or more date/time inputs are empty.")
    do_plot = False
else:
    try:
        t_cme_start = _ensure_utc(dt.datetime.combine(cme_start_date, cme_start_time))
        t_mo_start = _ensure_utc(dt.datetime.combine(mo_start_date, mo_start_time))
        t_cme_end = _ensure_utc(dt.datetime.combine(cme_end_date, cme_end_time))
    except TypeError as e:
        st.error(f"Date/time inputs invalid: {e}")
        do_plot = False

dt1 = t_mo_start - t_cme_start
dt2 = t_cme_end - t_mo_start
dt_tot = t_cme_end - t_cme_start

st.info(
    f"**Selected times (UTC)**  \n"
    f"- CME start: `{t_cme_start}`  \n"
    f"- MO start: `{t_mo_start}`  \n"
    f"- CME end: `{t_cme_end}`  \n\n"
    f"**Durations**  \n"
    f"- CME → MO: `{format_tdelta(dt1)}`  \n"
    f"- MO → CME end: `{format_tdelta(dt2)}`  \n"
    f"- Total CME: `{format_tdelta(dt_tot)}`"
)
st.warning(
    "⚠️ CDAWeb mode is limited to 48 hours\n\n"
    "⚠️ CDAWeb mode for WIND is not yet supported."
)


if t_cme_start >= t_mo_start or t_mo_start >= t_cme_end:
    st.error("Times must satisfy: **CME start < MO start < CME end**.")
    do_plot = False

# =============================================================================
# Main tabs
# =============================================================================

tab_remap, tab_donki = st.tabs(["Visualizer", "DONKI helper"])

# -----------------------------------------------------------------------------#
# VISUALIZER TAB
# -----------------------------------------------------------------------------#

with tab_remap:
    # ----------------------- Plot style & animation --------------------------#
    with st.expander("Advanced plot & GIF settings", expanded=False):
        st.markdown("#### 3D engine & colormaps")
        col1, col2 = st.columns(2)

        with col1:
            plot_engine = st.selectbox(
                "3D engine",
                ["Plotly (interactive)", "Matplotlib (static)"],
                index=0,
            )

            mag_cmap_options = [
                "viridis",
                "plasma",
                "inferno",
                "magma",
                "cividis",
                "turbo",
            ]
            colormap_mag = st.selectbox(
                r"$|B|$ colormap", mag_cmap_options, index=0
            )

            bg_choice = st.selectbox(
                "Background",
                ["white", "black", "transparent"],
                index=0,
            )

        with col2:
            br_cmap_options = [
                "seismic",
                "coolwarm",
                "bwr",
                "PiYG",
                "PRGn",
                "BrBG",
                "PuOr",
                "RdBu",
                "Spectral",
            ]
            colormap_br = st.selectbox(
                r"$B_{R_0}$ colormap (Matplotlib only)",
                br_cmap_options,
                index=0,
            )

            scatter_size = st.slider("Scatter size", 1, 8, 3)
            show_quivers = st.checkbox(
                "Show B-vectors (arrows/quivers)", value=True
            )
            show_planes = st.checkbox("Show CME boundary planes", value=True)

        # ---------------- PLOTLY-ONLY SETTINGS ---------------- #
        plotly_aspect_x = 2.0
        arrow_length_scale = 0.010
        arrow_density = 150
        arrow_head_size = 0.0007
        arrow_opacity = 0.5
        plane_extent = 0.45

        if plot_engine == "Plotly (interactive)":
            st.markdown("#### Plotly 3D settings")

            pcol1, pcol2 = st.columns(2)
            with pcol1:
                plotly_aspect_x = st.slider(
                    "Plotly R₀ stretch (x/y)",
                    min_value=1.0,
                    max_value=4.0,
                    value=2.0,
                    step=0.1,
                    help="Scales the R₀ axis relative to T₀ and N₀ in Plotly.",
                )

                plane_extent = st.slider(
                    "Boundary plane extent (Y/Z fraction)",
                    min_value=0.2,
                    max_value=1.0,
                    value=0.45,
                    step=0.05,
                    help="Fraction of Y/Z range occupied by CME/MO planes in Plotly.",
                )

            with pcol2:
                st.markdown("**Arrow settings (Plotly 3D)**")
                arrow_length_scale = st.slider(
                    "Arrow length scale",
                    min_value=0.001,
                    max_value=0.05,
                    value=0.018,
                    step=0.001,
                    format="%.3f",
                    help="Controls the relative arrow length in 3D.",
                )
                arrow_density = st.slider(
                    "Arrow density (Plotly)",
                    min_value=20,
                    max_value=300,
                    value=150,
                    step=10,
                    help="Approximate number of arrows in Plotly 3D plot.",
                )
                arrow_head_size = st.slider(
                    "Arrowhead size (Plotly)",
                    min_value=0.0001,
                    max_value=0.001,
                    value=0.0005,
                    step=0.0001,
                    format="%.4f",
                    help="Size of the cone head for Plotly arrows.",
                )
                arrow_opacity = st.slider(
                    "Arrow opacity",
                    min_value=0.05,
                    max_value=1.0,
                    value=0.5,
                    step=0.05,
                    help="Visual weight of arrows.",
                )

        # ------------- Colormap preview thumbnails ------------- #
        with st.expander("Colormap previews", expanded=False):
            st.markdown("**|B| colormaps**")
            cols_prev = st.columns(3)
            for i, cmap_name in enumerate(mag_cmap_options):
                with cols_prev[i % 3]:
                    st.image(
                        colormap_preview_bytes(cmap_name),
                        caption=cmap_name,
                    )

            st.markdown("---")
            st.markdown(r"**$B_{R_0}$ colormaps**")
            cols_prev2 = st.columns(3)
            for i, cmap_name in enumerate(br_cmap_options):
                with cols_prev2[i % 3]:
                    st.image(
                        colormap_preview_bytes(cmap_name),
                        caption=cmap_name,
                    )
        make_gif_flag = False
        make_rot_gif_flag = False
        # ---- GIF controls: separate, only detailed settings when enabled ----#
        st.markdown("#### GIF controls")
        gif_col1, gif_col2 = st.columns(2)
        with gif_col1:
            make_gif_flag = st.checkbox(
                "Generate CME evolution GIF",
                value=False,
                help="Creates a GIF of the growing CME path (with arrows & planes).",
            )
        with gif_col2:
            make_rot_gif_flag = st.checkbox(
                "Generate rotating 3D GIF",
                value=False,
                help="Rotates the full Matplotlib 3D plot.",
            )

        if make_gif_flag or make_rot_gif_flag:
            with st.expander("GIF settings", expanded=True):
                gif_frames = st.slider("GIF frames", 10, 80, 40)
                gif_dpi = st.slider(
                    "GIF resolution (DPI)",
                    min_value=80,
                    max_value=240,
                    value=140,
                    step=10,
                )
                gif_show_colorbar = st.checkbox(
                    "Show colorbar in evolution GIF", value=True
                )
                rot_elev = st.slider(
                    "Rotation elevation (deg)",
                    min_value=0,
                    max_value=80,
                    value=20,
                    step=5,
                )
                rot_direction_choice = st.radio(
                    "Rotation direction",
                    ["Counter-clockwise", "Clockwise"],
                    index=0,
                    horizontal=True,
                )
                rot_direction = (
                    1 if rot_direction_choice == "Counter-clockwise" else -1
                )
                rot_speed = st.slider(
                    "Rotation speed (sec/frame)",
                    min_value=0.05,
                    max_value=0.4,
                    value=0.12,
                    step=0.01,
                    help="Lower = faster rotation.",
                )
        else:
            gif_frames = 40
            gif_dpi = 140
            gif_show_colorbar = True
            rot_elev = 20
            rot_direction = 1
            rot_speed = 0.12

    # ----------------------- Compute & plot ---------------------------------#
    if do_plot:
        if event_name.strip() == "":
            st.error("Please enter an event label before running the remapping.")
            st.stop()
        if spacecraft_name == "WIND" and data_mode == "CDAWeb (SolO)":
            st.error("WIND CDAWeb download is not supported yet. Please use CSV upload mode.")
            st.stop()

        # If Aditya selected, enforce temporal-only and recommend .nc (GSM)
        if spacecraft_name == "Aditya L1":
            st.warning("Aditya L1 selected — spatial remap is disabled. Aditya data is GSM MAG-only; upload .nc files or CSV MAG (GSM).")
            temporal_only = True

        try:
            with st.spinner("Loading data…"):
                # prefer uploaded .nc if present for Aditya
                if spacecraft_name == "Aditya L1" and uploaded_netcdf:
                    (t_B, B_rtn), (t_V, V_rtn), (t_R_sc, R_sc_hgi) = load_from_uploaded_netcdf_list(uploaded_netcdf)
                else:
                    (t_B, B_rtn), (t_V, V_rtn), (t_R_sc, R_sc_hgi) = load_data_for_spacecraft(
                        spacecraft_name,
                        data_mode,
                        (uploaded_mag, uploaded_swa, uploaded_pos),
                        t_cme_start,
                        t_cme_end,
                    )

            st.success(
                f"Loaded {spacecraft_name} data. "
                f"Time range (MAG): [{t_B[0]} – {t_B[-1]}]"
            )

            # If temporal_only requested or V/R missing, do temporal-only branch
            v_missing = (t_V is None) or (len(t_V) == 0) or (isinstance(V_rtn, np.ndarray) and V_rtn.size == 0)
            r_missing = (t_R_sc is None) or (len(t_R_sc) == 0) or (isinstance(R_sc_hgi, np.ndarray) and R_sc_hgi.size == 0)

            # -------------------- Temporal-only branch -------------------- #
            if temporal_only or v_missing or r_missing:
                if v_missing or r_missing:
                    st.info(
                        "Velocity and/or position data are missing — running temporal-only plotting. "
                        "Spatial remap requires both velocity (V) and position (R) data."
                    )

                # STRICT: ensure CME times lie inside MAG time coverage; otherwise abort
                try:
                    tB0 = _ensure_utc(t_B[0])
                    tB1 = _ensure_utc(t_B[-1])
                    if t_cme_start < tB0 or t_cme_end > tB1:
                        st.error(
                            "CME start/end lie outside the available MAG data range for this spacecraft.\n\n"
                            f"MAG coverage (UTC): [{tB0} – {tB1}].\n\n"
                            "Fixes:  \n"
                            "- Adjust CME start/end to lie within the MAG coverage shown above, OR  \n"
                            "- Upload MAG data that covers the requested interval (include timezone info), OR  \n"
                            "- If your MAG times are naive but actually UTC, ensure the file timestamps include 'Z' or convert them to UTC when reading."
                        )
                        st.stop()
                except Exception as e:
                    st.error(f"Could not validate MAG time coverage: {e}")
                    st.stop()

                # Create temporal panels styled to match spatial figures (colormap, horizontal cbar, plane markers)
                try:
                    B_rtn = np.asarray(B_rtn)
                    if B_rtn.ndim != 2 or B_rtn.shape[1] < 3:
                        st.error("MAG data shape unexpected (need Bx, By, Bz).")
                        st.stop()

                    Bx = B_rtn[:, 0]
                    By = B_rtn[:, 1]
                    Bz = B_rtn[:, 2]
                    Bmag = np.sqrt(Bx**2 + By**2 + Bz**2)

                    t_times = [ _ensure_utc(t) for t in t_B ]
                    t_times_pd = pd.to_datetime(t_times)

                    # Angle (unwrap) — consistent with earlier code (atan2(Bz, By))
                    angles = np.degrees(np.unwrap(np.arctan2(Bz, By)))

                    # Styling variables reused from spatial UI controls (fallback defaults)
                    cmap_name = colormap_mag if 'colormap_mag' in locals() else "viridis"
                    bg = bg_choice if 'bg_choice' in locals() else "white"
                    s_size = scatter_size if 'scatter_size' in locals() else 3

                    cmap_obj = plt.colormaps.get_cmap(cmap_name)
                    finite_mask = np.isfinite(Bmag)
                    if np.any(finite_mask):
                        vmin = float(np.nanmin(Bmag[finite_mask]))
                        vmax = float(np.nanmax(Bmag[finite_mask]))
                        if vmin == vmax:
                            vmin -= 1.0
                            vmax += 1.0
                    else:
                        vmin, vmax = -1.0, 1.0

                    # determine label color based on bg
                    label_color = "white" if bg in ("black", "transparent") else "black"

                    # ---- Angle vs Time (styled like spatial main panel) ----
                    fig1 = plt.figure(figsize=(12, 4))
                    ax1 = fig1.add_axes([0.05, 0.28, 0.86, 0.60])
                    cax1 = fig1.add_axes([0.18, 0.12, 0.64, 0.05])  # horizontal colorbar below

                    if bg == "transparent":
                        fig1.patch.set_alpha(0.0)
                        ax1.set_facecolor("none")
                    else:
                        fig1.patch.set_facecolor(bg)
                        ax1.set_facecolor(bg)

                    sc = ax1.scatter(t_times_pd, angles, c=Bmag, s=max(4, s_size*3), cmap=cmap_obj, vmin=vmin, vmax=vmax, alpha=0.95)

                    # Add radial-field (B_R0 / Bx) on a secondary axis for direct comparison
                    try:
                        ax1_right = ax1.twinx()
                        br_label = r"$B_{R_0}$ (nT)" if spacecraft_name != "Aditya L1" else r"$B_x$ (nT) — GSM"
                        br_color = "C1"
                        try:
                            radial_for_plot = Bx
                        except NameError:
                            try:
                                radial_for_plot = B_rtn[:,0]
                            except Exception:
                                radial_for_plot = None
                        if radial_for_plot is not None:
                            ax1_right.plot(t_times_pd, radial_for_plot, color=br_color, linewidth=0.9, alpha=0.9, label=br_label)
                            ax1_right.set_ylabel(br_label, color=br_color)
                            ax1_right.tick_params(axis="y", colors=br_color)
                            ax1_right.legend(loc="upper right", fontsize=8)
                    except Exception:
                        pass

                    # Vertical plane-like markers (drawn as translucent vertical rectangles)
                    def add_vplane(ax, tval, color):
                        ax.axvline(tval, color=color, linestyle="--", linewidth=1.0)
                        # translucent rectangle to mimic plane breadth
                        span = (t_cme_end - t_cme_start).total_seconds() * 0.001
                        ax.axvspan(tval - timedelta(seconds=0), tval + timedelta(seconds=0), color=color, alpha=0.05)

                    add_vplane(ax1, t_cme_start, "blue")
                    add_vplane(ax1, t_mo_start, "blueviolet")
                    add_vplane(ax1, t_cme_end, "red")

                    ax1.set_ylabel("Angle (deg) — GSM if Aditya", color=label_color)
                    ax1.set_xlabel("Time (UTC)", color=label_color)
                    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
                    ax1.grid(True, which="both", linestyle="--", alpha=0.25)
                    # horizontal colorbar below
                    sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap_obj)
                    sm.set_array([])
                    cbar = fig1.colorbar(sm, cax=cax1, orientation="horizontal")
                    cbar.set_label("|B| (nT)", color=label_color)
                    cbar.ax.tick_params(colors=label_color)
                    ax1.tick_params(colors=label_color)
                    fig1.autofmt_xdate()

                    st.pyplot(fig1)

                    # ---- Hodogram: By vs Bz with concentric rotation grid lines ----
                    fig2 = plt.figure(figsize=(6, 6))
                    ax2 = fig2.add_subplot(111)
                    if bg == "transparent":
                        fig2.patch.set_alpha(0.0)
                        ax2.set_facecolor("none")
                    else:
                        fig2.patch.set_facecolor(bg)
                        ax2.set_facecolor(bg)

                    sc2 = ax2.scatter(By, Bz, c=Bmag, s=max(4, s_size*2), cmap=cmap_obj, vmin=vmin, vmax=vmax, alpha=0.9)

                    # concentric grid circles (rotation-style)
                    max_rad = np.nanmax(np.sqrt(By**2 + Bz**2))
                    if not np.isnan(max_rad) and max_rad > 0:
                        levels = np.linspace(max_rad/4, max_rad, 4)
                        for r in levels:
                            circ = Circle((0.0, 0.0), radius=r, edgecolor="gray", facecolor="none", linestyle="--", alpha=0.35)
                            ax2.add_artist(circ)
                    ax2.axhline(0, color="k", linewidth=0.4, alpha=0.6)
                    ax2.axvline(0, color="k", linewidth=0.4, alpha=0.6)

                    ax2.set_xlabel(r"$B_Y$ (nT) — GSM" if spacecraft_name == "Aditya L1" else r"$B_Y$ (nT)", color=label_color)
                    ax2.set_ylabel(r"$B_Z$ (nT) — GSM" if spacecraft_name == "Aditya L1" else r"$B_Z$ (nT)", color=label_color)
                    ax2.grid(True, linestyle="--", alpha=0.2)
                    cbar2 = fig2.colorbar(sc2, ax=ax2, orientation="vertical", pad=0.02)
                    cbar2.set_label("|B| (nT)", color=label_color)
                    cbar2.ax.tick_params(colors=label_color)
                    ax2.tick_params(colors=label_color)
                    st.pyplot(fig2)

                    # ---- 3D temporal quiver to match spatial look (time numeric vs By vs Bz) ----
                    try:
                        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
                        fig3 = plt.figure(figsize=(13, 8))
                        ax3 = fig3.add_axes([0.05, 0.25, 0.86, 0.70], projection='3d')
    
                        if bg == "transparent":
                            fig3.patch.set_alpha(0.0)
                            ax3.set_facecolor("none")
                        else:
                            fig3.patch.set_facecolor(bg)
                            ax3.set_facecolor(bg)
    
                        Xtime = mdates.date2num(t_times_pd)
    
                        # keep only finite indices
                        finite_idx = np.isfinite(Bmag)
                        Xtime_f = Xtime[finite_idx]
                        Bx_f = Bx[finite_idx]
                        By_f = By[finite_idx]
                        Bz_f = Bz[finite_idx]
                        Bmag_f = Bmag[finite_idx]
    
                        if len(Xtime_f) < 2:
                            st.warning("Not enough points for 3D temporal quivers.")
                            st.session_state["last_matplotlib_fig"] = None
                            st.session_state["last_matplotlib_ax"] = None
                        else:
                            # emulate spatial axes scaling used in the static Matplotlib figure
                            xlen_local = 4.0
                            ylen_local = 1.0
                            zlen_local = 1.0
    
                            span_x = float(np.nanmax(Xtime_f) - np.nanmin(Xtime_f))
                            if span_x <= 0:
                                span_x = 0.1
                            yrange_local = span_x * ylen_local / xlen_local
                            zrange_local = span_x * zlen_local / xlen_local
    
                            xmid_num = 0.5 * (np.nanmax(Xtime_f) + np.nanmin(Xtime_f))
                            xlim_num = (xmid_num - span_x*0.55, xmid_num + span_x*0.55)
                            ylim = (-yrange_local/2, yrange_local/2)
                            zlim = (-zrange_local/2, zrange_local/2)
                            ax3.set_xlim(xlim_num)
                            ax3.set_ylim(ylim)
                            ax3.set_zlim(zlim)
                            ax3.set_box_aspect([xlen_local, ylen_local, zlen_local])
    
                            # color normalization same as elsewhere
                            cmap_obj = plt.colormaps.get_cmap(cmap_name)
                            finite_mask2 = np.isfinite(Bmag_f)
                            if np.any(finite_mask2):
                                vmin_f = float(np.nanmin(Bmag_f[finite_mask2]))
                                vmax_f = float(np.nanmax(Bmag_f[finite_mask2]))
                                if vmin_f == vmax_f:
                                    vmin_f -= 1.0
                                    vmax_f += 1.0
                            else:
                                vmin_f, vmax_f = -1.0, 1.0
    
                            # scale arrows similar to spatial quiver scaling
                            Bvec = np.sqrt(Bx_f**2 + By_f**2 + Bz_f**2)
                            Bmax = np.nanmax(Bvec) if np.any(np.isfinite(Bvec)) else 1.0
                            B_scale_factor = 0.707 * yrange_local / Bmax if Bmax > 0 else 1.0
    
                            # subsample arrows to avoid clutter — densified by factor ~2
                            arrow_step = max(1, len(Xtime_f) // max(1, int(arrow_density * 2)))
                            idx_ar = np.arange(0, len(Xtime_f), arrow_step)
    
                            # Draw quivers: start at (time_numeric, 0, 0) with arrow components (u,v,w)
                            for j in idx_ar:
                                if not np.isfinite(Bmag_f[j]):
                                    continue
                                val = Bmag_f[j]
                                val_norm = (val - vmin_f) / (vmax_f - vmin_f) if (vmax_f - vmin_f) != 0 else 0.5
                                val_norm = float(np.clip(val_norm, 0.0, 1.0))
                                rgba = to_rgba(cmap_obj(val_norm), alpha=1.0)
    
                                x0 = float(Xtime_f[j])
                                y0 = 0.0  # baseline like spatial quivers (they originate on the mid-plane)
                                z0 = 0.0
    
                                u = float(Bx_f[j]) * B_scale_factor  # extent along X (time axis)
                                v = float(By_f[j]) * B_scale_factor
                                w = float(Bz_f[j]) * B_scale_factor
    
                                ax3.quiver(
                                    x0, y0, z0,
                                    u, v, w,
                                    length=1.0,
                                    normalize=False,
                                    pivot='tail',
                                    arrow_length_ratio=0.08,
                                    linewidth=0.9,
                                    color=rgba,
                                )
    
                            # cosmetic labels and horizontal colormap bar (like spatial)
                            label_color = "white" if bg in ("black", "transparent") else "black"
                            ax3.set_xlabel("Time (UTC numeric)", color=label_color)
                            ax3.set_ylabel(r"$T_0$-like (arb)", color=label_color)
                            ax3.set_zlabel(r"$N_0$-like (arb)", color=label_color)
                            ax3.tick_params(colors=label_color)
    
                            # add horizontal colorbar under the axes (use a ScalarMappable)
                            sm3 = ScalarMappable(norm=Normalize(vmin=vmin_f, vmax=vmax_f), cmap=cmap_obj)
                            sm3.set_array([])
                            cax_pos = fig3.add_axes([0.18, 0.16, 0.64, 0.055])
                            cbar3 = fig3.colorbar(sm3, cax=cax_pos, orientation="horizontal")
                            cbar3.set_label("|B| (nT)", color=label_color)
                            cbar3.ax.tick_params(colors=label_color)
                            cbar3.outline.set_edgecolor(label_color)
    
                            # annotate CME planes vertically (using numeric time) as translucent rectangular surfaces
                            def add_vplane_num(ax, tnum, color, x_span_frac=0.01, y_frac=0.5, z_frac=0.5):
                                """
                                Draw a small translucent Y-Z rectangle at numeric x=tnum.
                                x_span_frac: fraction of total X numeric span used to set the plane width (small).
                                y_frac, z_frac: fraction of Y/Z ranges to set plane extents (0..1)
                                """
                                try:
                                    # numeric span and midpoints
                                    x_min_num = np.nanmin(Xtime_f)
                                    x_max_num = np.nanmax(Xtime_f)
                                    x_span_num = x_max_num - x_min_num if (x_max_num - x_min_num) != 0 else 1.0
    
                                    # plane width in numeric X units (very small so it looks like a slice)
                                    half_dx = 0.5 * x_span_frac * x_span_num
                                    x_plane = np.array([[tnum - half_dx, tnum - half_dx], [tnum + half_dx, tnum + half_dx]])
    
                                    # y/z spans centered around zero
                                    y_center = 0.0
                                    z_center = 0.0
                                    y_half = 0.5 * (y_frac * (ylim[1] - ylim[0]))
                                    z_half = 0.5 * (z_frac * (zlim[1] - zlim[0]))
    
                                    Yp = np.array([[y_center - y_half, y_center + y_half], [y_center - y_half, y_center + y_half]])
                                    Zp = np.array([[z_center - z_half, z_center - z_half], [z_center + z_half, z_center + z_half]])
    
                                    ax.plot_surface(
                                        x_plane,
                                        Yp,
                                        Zp,
                                        color=color,
                                        alpha=0.20,
                                        linewidth=0.0,
                                        shade=False,
                                    )
                                except Exception:
                                    # fallback to a vertical dashed line if surface plotting fails
                                    try:
                                        ax.plot([tnum, tnum], [ylim[0], ylim[1]], [zlim[0], zlim[0]], color=color, linestyle="--", linewidth=1.0, alpha=0.8)
                                    except Exception:
                                        pass
    
                            add_vplane_num(ax3, mdates.date2num(t_cme_start), "blue")
                            add_vplane_num(ax3, mdates.date2num(t_mo_start), "blueviolet")
                            add_vplane_num(ax3, mdates.date2num(t_cme_end), "red")
    
                            # attempt to format x-axis ticks as date strings
                            try:
                                from matplotlib.ticker import FuncFormatter
                                def num_to_date_short(x, pos=None):
                                    return mdates.num2date(x).strftime("%m-%d\n%H:%M")
                                ax3.xaxis.set_major_formatter(FuncFormatter(num_to_date_short))
                            except Exception:
                                pass
    
                            # store figure & axis for rotating GIF generation
                            st.session_state["last_matplotlib_fig"] = fig3
                            st.session_state["last_matplotlib_ax"] = ax3
    
                            # show the static figure inside a tabbed view (Temporal plot | GIFs)
                            tab_temp_plot, tab_temp_gif = st.tabs(["Temporal plot", "GIFs"])
                            with tab_temp_plot:
                                st.pyplot(fig3)
    
                            # --- GIF tab: generate rotating temporal GIF or evolution GIF on demand ---
                            with tab_temp_gif:
                                if make_rot_gif_flag:
                                    fig_static = st.session_state.get("last_matplotlib_fig", None)
                                    ax_static = st.session_state.get("last_matplotlib_ax", None)
                                    if fig_static is None or ax_static is None:
                                        st.warning("No Matplotlib temporal figure available. Generate the static plot first (Matplotlib engine).")
                                    else:
                                        with st.spinner("Generating rotating temporal 3D GIF…"):
                                            try:
                                                rot_gif_bytes = make_rotation_gif_from_axes(
                                                    fig_static,
                                                    ax_static,
                                                    dpi=gif_dpi,
                                                    n_frames=gif_frames,
                                                    elev=rot_elev,
                                                    direction=rot_direction,
                                                    frame_duration=rot_speed,
                                                )
                                                show_gif_inline(rot_gif_bytes.getvalue(), caption="Rotating temporal 3D (Matplotlib)")
                                                st.download_button(
                                                    "⬇️ Download rotating temporal 3D GIF",
                                                    data=rot_gif_bytes.getvalue(),
                                                    file_name=f"{spacecraft_name.replace(' ', '_')}_{event_name}_temporal_rotating.gif",
                                                    mime="image/gif",
                                                )
                                            except Exception as _e:
                                                st.warning(f"Could not create rotating temporal GIF: {_e}")
                                else:
                                    st.info("Enable 'Generate rotating 3D GIF' in Advanced plot & GIF settings to create a rotating GIF for the temporal Matplotlib view.")
    
                    except Exception as e:
                        st.session_state["last_matplotlib_fig"] = None
                        st.session_state["last_matplotlib_ax"] = None
                        st.error(f"Temporal 3D (matplotlib) failed: {e}")
                    # CSV download for temporal data
                    df_temp = pd.DataFrame({
                        "time": [t.isoformat() for t in t_times],
                        "B_x_nT": Bx,
                        "B_y_nT": By,
                        "B_z_nT": Bz,
                        "B_mag_nT": Bmag,
                        "angle_deg": angles
                    })
                    csv_data = df_temp.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download temporal CSV",
                        data=csv_data,
                        file_name=f"{spacecraft_name}_temporal.csv",
                        mime="text/csv",
                    )
    
                except Exception as e:
                    st.error(f"Failed to create temporal plots: {e}")
                st.stop()

            # If we reach here, V and R exist -> permit spatial remap
            try:
                check_time_coverage(
                    f"{spacecraft_name} MAG", t_B, t_cme_start, t_cme_end
                )
                check_time_coverage(
                    f"{spacecraft_name} V", t_V, t_cme_start, t_cme_end
                )
                check_time_coverage(
                    f"{spacecraft_name} Position",
                    t_R_sc,
                    t_cme_start,
                    t_cme_end,
                )
            except ValueError as e:
                st.error(str(e))
                st.stop()

            with st.spinner("Computing spatial remap…"):
                res = compute_remap_for_spacecraft(
                    t_B,
                    B_rtn,
                    t_V,
                    V_rtn,
                    t_R_sc,
                    R_sc_hgi,
                    t_cme_start,
                    t_mo_start,
                    t_cme_end,
                )

            st.success("Remap complete.")

            X_plot = res["X"]
            Y_plot = res["Y"]
            Z_plot = res["Z"]
            Bx_plot = res["Bx"]
            By_plot = res["By"]
            Bz_plot = res["Bz"]
            B_mag_plot = np.asarray(res["Bmag"], dtype=float)
            t_B_plot = res["t_B_plot"]
            mask_posX = res["mask_posX"]
            idx_cme_start = res["idx_cme_start"]
            idx_mo_start = res["idx_mo_start"]
            idx_cme_end = res["idx_cme_end"]
            idx_cme_start_masked = res["idx_cme_start_masked"]
            idx_mo_start_masked = res["idx_mo_start_masked"]
            idx_cme_end_masked = res["idx_cme_end_masked"]

            df_out = pd.DataFrame(
                {
                    "time": t_B_plot,
                    "X_AU": X_plot,
                    "Y_AU": Y_plot,
                    "Z_AU": Z_plot,
                    "B_R0_nT": Bx_plot,
                    "B_T0_nT": By_plot,
                    "B_N0_nT": Bz_plot,
                    "|B|_nT": B_mag_plot,
                    "is_sheath": res["is_sheath"],
                }
            )
            csv_bytes = df_out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download remapped coordinates (CSV)",
                data=csv_bytes,
                file_name=f"{spacecraft_name.replace(' ', '_')}_{event_name}_remapped.csv",
                mime="text/csv",
            )

            sub_tab_plot, sub_tab_gif = st.tabs(["3D plot", "GIFs"])

            # ================= 3D plot tab ===================#
            with sub_tab_plot:
                if plot_engine == "Plotly (interactive)":
                    # ---- Plotly 3D ----#
                    color_vals = np.asarray(B_mag_plot, dtype=float)
                    plotly_cmap_map = {
                        "viridis": "Viridis",
                        "plasma": "Plasma",
                        "inferno": "Inferno",
                        "magma": "Magma",
                        "cividis": "Cividis",
                        "turbo": "Turbo",
                    }
                    plotly_cmap = plotly_cmap_map.get(colormap_mag, "Viridis")

                    fig3d = go.Figure()

                    # label / legend colors based on background
                    label_color = "black" if bg_choice == "white" else "white"

                    fig3d.add_trace(
                        go.Scatter3d(
                            x=X_plot,
                            y=Y_plot,
                            z=Z_plot,
                            mode="markers",
                            marker=dict(
                                size=scatter_size,
                                color=color_vals,
                                colorscale=plotly_cmap,
                                opacity=0.9
                                if bg_choice != "transparent"
                                else 0.8,
                                showscale=True,
                                colorbar=dict(
                                title=dict(text="|B| (nT)", font=dict(color=label_color)),
                                tickfont=dict(color=label_color),
                                len=0.70,
                                y=0.48,
                                thickness=12,
                                outlinewidth=0,
                                ),
                            ),
                            name=f"{spacecraft_name}",
                        )
                    )
                    fig3d.update_layout(
                        margin=dict(t=0, l=0, r=10, b=0),
                        title=dict(text=f"{spacecraft_name} — spatial remap ({event_name})", y=0.94, x=0.2),
                    )

                    # Arrows
                    if show_quivers:
                        finite = np.isfinite(B_mag_plot)
                        if np.any(finite):
                            vmin = float(np.nanmin(B_mag_plot[finite]))
                            vmax = float(np.nanmax(B_mag_plot[finite]))
                            if vmin == vmax:
                                vmin -= 1.0
                                vmax += 1.0
                        else:
                            vmin, vmax = -1.0, 1.0
                        cmap_obj = plt.colormaps.get_cmap(colormap_mag)

                        arrow_step = max(1, len(X_plot) // arrow_density)
                        idx_ar = np.arange(0, len(X_plot), arrow_step)

                        span_x = float(X_plot.max() - X_plot.min())
                        if span_x <= 0:
                            span_x = 0.1

                        Bvec = np.sqrt(Bx_plot ** 2 + By_plot ** 2 + Bz_plot ** 2)
                        Bmax = np.nanmax(Bvec) if np.any(np.isfinite(Bvec)) else 1.0
                        scale = (
                            arrow_length_scale * span_x / Bmax
                            if Bmax > 0
                            else 1.0
                        )

                        for i in idx_ar:
                            if not np.isfinite(B_mag_plot[i]):
                                continue

                            val_norm = (B_mag_plot[i] - vmin) / (vmax - vmin)
                            val_norm = float(np.clip(val_norm, 0.0, 1.0))
                            rgba = cmap_obj(val_norm)
                            color_str = "#{:02x}{:02x}{:02x}".format(
                                int(rgba[0] * 255),
                                int(rgba[1] * 255),
                                int(rgba[2] * 255),
                            )

                            x0, y0, z0 = (
                                float(X_plot[i]),
                                float(Y_plot[i]),
                                float(Z_plot[i]),
                            )
                            x1 = x0 + float(Bx_plot[i]) * scale
                            y1 = y0 + float(By_plot[i]) * scale
                            z1 = z0 + float(Bz_plot[i]) * scale

                            fig3d.add_trace(
                                go.Scatter3d(
                                    x=[x0, x1],
                                    y=[y0, y1],
                                    z=[z0, z1],
                                    mode="lines",
                                    line=dict(width=3, color=color_str),
                                    showlegend=False,
                                )
                            )

                            fig3d.add_trace(
                                go.Cone(
                                    x=[x1],
                                    y=[y1],
                                    z=[z1],
                                    u=[Bx_plot[i] * scale * 0.2],
                                    v=[By_plot[i] * scale * 0.2],
                                    w=[Bz_plot[i] * scale * 0.2],
                                    colorscale=[
                                        [0.0, color_str],
                                        [1.0, color_str],
                                    ],
                                    showscale=False,
                                    sizemode="absolute",
                                    sizeref=arrow_head_size,
                                    anchor="tip",
                                    opacity=arrow_opacity,
                                    name="B arrows",
                                )
                            )

                    # CME boundary planes (using masked indices)
                    if show_planes:
                        def add_plane(masked_idx, plane_name):
                            if masked_idx is None or masked_idx >= len(X_plot):
                                return
                            i = masked_idx
                            x0 = float(X_plot[i])

                            y_min_all, y_max_all = float(Y_plot.min()), float(
                                Y_plot.max()
                            )
                            z_min_all, z_max_all = float(Z_plot.min()), float(
                                Z_plot.max()
                            )

                            y_span = y_max_all - y_min_all
                            z_span = z_max_all - z_min_all

                            y_mid = 0.5 * (y_max_all + y_min_all)
                            z_mid = 0.5 * (z_max_all + z_min_all)

                            y_min = y_mid - 0.5 * plane_extent * y_span
                            y_max = y_mid + 0.5 * plane_extent * y_span
                            z_min = z_mid - 0.5 * plane_extent * z_span
                            z_max = z_mid + 0.5 * plane_extent * z_span

                            fig3d.add_trace(
                                go.Surface(
                                    x=[[x0, x0], [x0, x0]],
                                    y=[[y_min, y_max], [y_min, y_max]],
                                    z=[[z_min, z_min], [z_max, z_max]],
                                    showscale=False,
                                    opacity=0.16,
                                    surfacecolor=np.zeros((2, 2)),
                                    name=plane_name,
                                )
                            )

                        add_plane(idx_cme_start_masked, "CME start")
                        add_plane(idx_mo_start_masked, "MO start")
                        add_plane(idx_cme_end_masked, "CME end")

                    bg_rgba = (
                        "rgba(0,0,0,0)"
                        if bg_choice == "transparent"
                        else ("black" if bg_choice == "black" else "white")
                    )

                    fig3d.update_layout(
                        title=dict(
                            text=f"{spacecraft_name} — spatial remap ({event_name})",
                            font=dict(color=label_color),
                        ),
                        scene=dict(
                            xaxis=dict(
                                title=dict(
                                    text="R₀ (AU)", font=dict(color=label_color)
                                ),
                                backgroundcolor=bg_rgba,
                                tickfont=dict(color=label_color),
                            ),
                            yaxis=dict(
                                title=dict(
                                    text="T₀ (AU)", font=dict(color=label_color)
                                ),
                                backgroundcolor=bg_rgba,
                                tickfont=dict(color=label_color),
                            ),
                            zaxis=dict(
                                title=dict(
                                    text="N₀ (AU)", font=dict(color=label_color)
                                ),
                                backgroundcolor=bg_rgba,
                                tickfont=dict(color=label_color),
                            ),
                            aspectmode="manual",
                            aspectratio=dict(x=plotly_aspect_x, y=1, z=1),
                        ),
                        paper_bgcolor=bg_rgba,
                        plot_bgcolor=bg_rgba,
                        showlegend=True,
                        legend=dict(font=dict(color=label_color)),
                        font=dict(color=label_color),
                    )

                    st.plotly_chart(fig3d, use_container_width=True)

                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        try:
                            png_bytes = fig3d.to_image(format="png")
                            st.download_button(
                                "⬇️ Download Plot (PNG)",
                                data=png_bytes,
                                file_name=f"{spacecraft_name.replace(' ', '_')}_{event_name}_3d_plot.png",
                                mime="image/png",
                            )
                        except Exception:
                            st.caption(
                                "PNG export needs `kaleido` in requirements.txt."
                            )
                    with col_dl2:
                        html_bytes = fig3d.to_html(
                            full_html=True, include_plotlyjs="cdn"
                        ).encode("utf-8")
                        st.download_button(
                            "⬇️ Download Plot (HTML)",
                            data=html_bytes,
                            file_name=f"{spacecraft_name.replace(' ', '_')}_{event_name}_3d_plot.html",
                            mime="text/html",
                        )

                    # No Matplotlib fig to rotate in this mode
                    st.session_state["last_matplotlib_fig"] = None
                    st.session_state["last_matplotlib_ax"] = None

                else:
                    # ---- Matplotlib static 3D ----#
                    fig = plt.figure(figsize=(13, 8))

                    # Larger main plot, shorter & separated colorbars
                    ax = fig.add_axes([0.05, 0.25, 0.86, 0.70], projection="3d")
                    cax_mag = fig.add_axes([0.15, 0.20, 0.70, 0.035])
                    cax_br = fig.add_axes([0.15, 0.10, 0.70, 0.035])


                    xlen, ylen, zlen = 4, 1, 1
                    xrange = 1.1 * (X_plot.max() - X_plot.min())
                    yrange = xrange * ylen / xlen
                    zrange = xrange * zlen / xlen

                    xmid = 0.5 * (np.max(X_plot) + np.min(X_plot))
                    xlim = (xmid - xrange / 2, xmid + xrange / 2)
                    ylim = (-yrange / 2, yrange / 2)
                    zlim = (-zrange / 2, zrange / 2)

                    label_color = (
                        "white" if bg_choice in ("black", "transparent") else "black"
                    )

                    if bg_choice == "transparent":
                        fig.patch.set_alpha(0.0)
                        ax.set_facecolor("none")
                    else:
                        fig.patch.set_facecolor(bg_choice)
                        ax.set_facecolor(bg_choice)

                    cmap_mag_obj = plt.colormaps.get_cmap(colormap_mag)
                    vmin_mag = float(np.nanmin(B_mag_plot))
                    vmax_mag = float(np.nanmax(B_mag_plot))
                    if vmin_mag == vmax_mag:
                        vmin_mag -= 1.0
                        vmax_mag += 1.0
                    norm_mag = Normalize(vmin=vmin_mag, vmax=vmax_mag)
                    normed_mag = np.clip(norm_mag(B_mag_plot), 0, 1)

                    # B-vector quivers
                    if show_quivers:
                        B_scale_factor = 0.707 * yrange / np.max(
                            np.abs(B_mag_plot)
                        )
                        for i in range(len(X_plot)):
                            if np.isnan(normed_mag[i]):
                                continue
                            u = Bx_plot[i] * B_scale_factor
                            v = By_plot[i] * B_scale_factor
                            w = Bz_plot[i] * B_scale_factor
                            rgba = to_rgba(cmap_mag_obj(normed_mag[i]), alpha=1)
                            ax.quiver(
                                X_plot[i],
                                Y_plot[i],
                                Z_plot[i],
                                u,
                                v,
                                w,
                                color=rgba,
                                arrow_length_ratio=0.1,
                                linewidth=0.7,
                            )

                    # Ground projection colored by B_R0
                    br_cmap = plt.colormaps.get_cmap(colormap_br)
                    br_min = float(np.nanmin(Bx_plot))
                    br_max = float(np.nanmax(Bx_plot))
                    max_abs = max(abs(br_min), abs(br_max))
                    if max_abs == 0:
                        max_abs = 1.0
                    br_norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
                    y_offset = -yrange * 0.5 * 0.9
                    z_offset = -zrange * 0.5 * 0.9

                    sc = ax.scatter(
                        X_plot,
                        Y_plot + y_offset,
                        Z_plot + z_offset,
                        c=Bx_plot,
                        cmap=br_cmap,
                        norm=br_norm,
                        s=scatter_size * 2,
                        alpha=1.0,
                    )

                    ax.set_box_aspect([xlen, ylen, zlen])
                    ax.set_title(
                        f"{spacecraft_name} — spatial remap ({event_name})",
                        pad=18,
                        color=label_color,
                    )
                    ax.set_xlabel("$R_0$ (AU)", labelpad=12, color=label_color)
                    ax.set_ylabel("$T_0$ (AU)", labelpad=8, color=label_color)
                    ax.set_zlabel("$N_0$ (AU)", labelpad=8, color=label_color)

                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
                    ax.set_zlim(zlim)

                    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
                    ax.zaxis.set_major_locator(MaxNLocator(nbins=3))
                    ax.tick_params(colors=label_color)

                    sm_mag = ScalarMappable(cmap=cmap_mag_obj, norm=norm_mag)
                    sm_mag.set_array([])
                    cbar_mag = fig.colorbar(
                        sm_mag, cax=cax_mag, orientation="horizontal"
                    )
                    cbar_mag.set_label("|B| (nT)", color=label_color)
                    cbar_mag.ax.tick_params(colors=label_color)
                    cbar_mag.outline.set_edgecolor(label_color)

                    cbar_br = fig.colorbar(
                        sc, cax=cax_br, orientation="horizontal"
                    )
                    cbar_br.set_label(r"$B_{R_0}$ (nT)", color=label_color)
                    cbar_br.ax.tick_params(colors=label_color)
                    cbar_br.outline.set_edgecolor(label_color)

                    # Time labels along R0
                    time_seconds = np.array(
                        [(t - t_B_plot[0]).total_seconds() for t in t_B_plot]
                    )
                    X_arr = np.array(X_plot, dtype=float)
                    sort_idx = np.argsort(X_arr)
                    X_unique, idxu = np.unique(X_arr[sort_idx], return_index=True)
                    time_unique = time_seconds[sort_idx][idxu]

                    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
                    r0_ticks = ax.get_xticks()
                    r0_ticks = r0_ticks[
                        (r0_ticks >= xlim[0]) & (r0_ticks <= xlim[1])
                    ]

                    n_labels = 8
                    if len(r0_ticks) > n_labels:
                        tick_positions = r0_ticks[
                            np.linspace(0, len(r0_ticks) - 1, n_labels, dtype=int)
                        ]
                    else:
                        tick_positions = r0_ticks

                    tick_seconds = np.interp(tick_positions, X_unique, time_unique)
                    tick_times = [
                        t_B_plot[0] + timedelta(seconds=s) for s in tick_seconds
                    ]

                    axis_y = 0
                    axis_z = zrange / 2 * 1.7
                    time_label_color = (
                        "white"
                        if bg_choice in ("black", "transparent")
                        else "gray"
                    )
                    for x_pos, tlab in zip(tick_positions, tick_times):
                        ax.text(
                            x_pos,
                            axis_y,
                            axis_z,
                            tlab.strftime("%m-%d\n%H:%M"),
                            ha="center",
                            va="bottom",
                            fontsize=7,
                            color=time_label_color,
                        )

                    # Boundary planes + legend entries
                    plane_legend_handles = []

                    def add_plane(masked_idx, color, label):
                        if masked_idx is None or masked_idx >= len(X_plot):
                            return
                        i = masked_idx
                        x_pos = X_plot[i]
                        solgaleo.draw_time_plane(
                            ax, x_pos, color=color, alpha=0.2
                        )
                        ax.plot(
                            [X_plot[i], X_plot[i]],
                            [Y_plot[i], Y_plot[i]],
                            [ax.get_zlim()[0], Z_plot[i]],
                            color="k",
                            linestyle="--",
                            linewidth=1,
                            alpha=1.0,
                        )
                        plane_legend_handles.append(
                            Patch(facecolor=color, alpha=0.2, label=label)
                        )

                    if show_planes:
                        add_plane(idx_cme_start_masked, "blue", "CME start")
                        add_plane(idx_mo_start_masked, "blueviolet", "MO start")
                        add_plane(idx_cme_end_masked, "red", "CME end")

                        if plane_legend_handles:
                            leg = ax.legend(
                                handles=plane_legend_handles,
                                loc="upper left",
                                fontsize=8,
                            )
                            for text in leg.get_texts():
                                text.set_color(label_color)
                            leg.get_frame().set_facecolor("none")
                            leg.get_frame().set_edgecolor(label_color)

                    # Store fig/ax for rotating GIF
                    st.session_state["last_matplotlib_fig"] = fig
                    st.session_state["last_matplotlib_ax"] = ax

                    buf_png = io.BytesIO()
                    fig.savefig(buf_png, format="png", dpi=220, bbox_inches="tight")
                    buf_png.seek(0)

                    st.pyplot(fig, clear_figure=False)

                    st.download_button(
                        "⬇️ Download Plot (PNG)",
                        data=buf_png.getvalue(),
                        file_name=f"{spacecraft_name.replace(' ', '_')}_{event_name}_3d_plot.png",
                        mime="image/png",
                    )

            # ================= GIF tab ===================#
            with sub_tab_gif:
                if make_gif_flag or make_rot_gif_flag:
                    if make_gif_flag:
                        with st.spinner("Generating CME evolution GIF…"):
                            gif_bytes = make_cme_gif(
                                X_plot,
                                Y_plot,
                                Z_plot,
                                B_mag_plot,
                                Bx_plot,
                                By_plot,
                                Bz_plot,
                                idx_cme_start_masked=idx_cme_start_masked,
                                idx_mo_start_masked=idx_mo_start_masked,
                                idx_cme_end_masked=idx_cme_end_masked,
                                plane_extent=plane_extent,
                                colormap_mag=colormap_mag,
                                bg_choice=bg_choice,
                                n_frames=gif_frames,
                                dpi=gif_dpi,
                                show_colorbar=gif_show_colorbar,
                                frame_duration=0.15,
                                arrow_density=arrow_density,
                            )
                        show_gif_inline(
                            gif_bytes.getvalue(),
                            caption="CME evolution (growing path, with arrows & planes)",
                        )
                        st.download_button(
                            "⬇️ Download CME evolution GIF",
                            data=gif_bytes.getvalue(),
                            file_name=f"{spacecraft_name.replace(' ', '_')}_{event_name}_evolution.gif",
                            mime="image/gif",
                        )

                    if make_rot_gif_flag:
                        fig_static = st.session_state.get("last_matplotlib_fig", None)
                        ax_static = st.session_state.get("last_matplotlib_ax", None)

                        if fig_static is None or ax_static is None:
                            st.warning(
                                "No Matplotlib figure available. Generate the static "
                                "plot first (Matplotlib engine or temporal 3D)."
                            )
                        else:
                            with st.spinner("Generating rotating 3D GIF…"):
                                rot_gif_bytes = make_rotation_gif_from_axes(
                                    fig_static,
                                    ax_static,
                                    dpi=gif_dpi,
                                    n_frames=gif_frames,
                                    elev=rot_elev,
                                    direction=rot_direction,
                                    frame_duration=rot_speed,
                                )
                            show_gif_inline(
                                rot_gif_bytes.getvalue(),
                                caption="Rotating full Matplotlib CME plot",
                            )
                            st.download_button(
                                "⬇️ Download rotating 3D GIF",
                                data=rot_gif_bytes.getvalue(),
                                file_name=f"{spacecraft_name.replace(' ', '_')}_{event_name}_rotating.gif",
                                mime="image/gif",
                            )
                else:
                    st.info(
                        "Enable one of the GIF options in 'Advanced plot & GIF settings' "
                        "to generate a GIF."
                    )

        except Exception as e:
            st.error(f"Something went wrong:\n\n{e}")
            st.stop()

# -----------------------------------------------------------------------------#
# DONKI helper tab (unchanged)
# -----------------------------------------------------------------------------#

with tab_donki:
    st.subheader("NASA DONKI CME catalog helper")
    st.markdown(
        """
Use this tab to query NASA's DONKI CME catalog and inspect events.

1. Set a **date range** and **API key** (use `DEMO_KEY` for quick tests).  
2. Fetch CMEs and select an event.  
3. Copy the suggested timings into the sidebar if they match your event.

If the request times out, try a smaller date range.
"""
    )

    import requests

    col1, col2 = st.columns(2)
    with col1:
        donki_start = st.date_input(
            "DONKI start date",
            value=dt.date(2023, 1, 1),
            key="donki_start",
        )
    with col2:
        donki_end = st.date_input(
            "DONKI end date",
            value=dt.date(2023, 1, 5),
            key="donki_end",
        )

    api_key = st.text_input(
        "NASA API key",
        value="DEMO_KEY",
        help="Get a personal key from api.nasa.gov (DEMO_KEY is rate-limited).",
    )

    if st.button("Fetch CMEs from DONKI"):
        try:
            url = "https://api.nasa.gov/DONKI/CME"
            params = dict(
                startDate=donki_start.strftime("%Y-%m-%d"),
                endDate=donki_end.strftime("%Y-%m-%d"),
                api_key=api_key,
            )
            resp = requests.get(url, params=params, timeout=40)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                st.warning("No CME events found for this interval.")
            else:
                st.session_state["donki_cmes"] = data
                st.success(f"Fetched {len(data)} CME events.")
        except requests.exceptions.Timeout:
            st.error(
                "DONKI request timed out. Try again, or use a smaller date range."
            )
        except Exception as e:
            st.error(f"Failed to fetch DONKI data: {e}")

    if "donki_cmes" in st.session_state and st.session_state["donki_cmes"]:
        cmes = st.session_state["donki_cmes"]

        options = []
        for i, ev in enumerate(cmes):
            stime = ev.get("startTime") or "?"
            note = (ev.get("note") or "").replace("\n", " ")
            if len(note) > 60:
                note = note[:57] + "..."
            options.append(f"{i:02d} | {stime} | {note}")

        idx = st.selectbox(
            "Select DONKI CME event",
            options=list(range(len(cmes))),
            format_func=lambda i: options[i],
        )

        ev = cmes[idx]

        st.markdown("### Event summary")
        st.write(f"**activityID:** `{ev.get('activityID')}`")
        st.write(f"**catalog:** `{ev.get('catalog')}`")
        st.write(f"**startTime:** `{ev.get('startTime')}`")
        if ev.get("sourceLocation"):
            st.write(f"**sourceLocation:** `{ev.get('sourceLocation')}`")
        if ev.get("activeRegionNum"):
            st.write(f"**activeRegion:** `{ev.get('activeRegionNum')}`")

        if ev.get("note"):
            short_note = ev["note"].strip().replace("\n", " ")
            if len(short_note) > 200:
                short_note = short_note[:197] + "..."
            st.markdown(f"**note:** {short_note}")

        analyses = ev.get("cmeAnalyses") or []
        best = None
        for a in analyses:
            if a.get("isMostAccurate"):
                best = a
                break
        if best is None and analyses:
            best = analyses[0]

        if best:
            st.markdown("#### CME analysis (best available)")
            st.write(f"- **time21_5:** `{best.get('time21_5')}`")
            st.write(f"- **speed:** `{best.get('speed')}` km/s")
            st.write(f"- **halfAngle:** `{best.get('halfAngle')}`°")
            st.write(
                f"- **latitude / longitude:** "
                f"`{best.get('latitude')}`, `{best.get('longitude')}`"
            )

            enlil_list = best.get("enlilList") or []
            if enlil_list:
                impacts = enlil_list[0].get("impactList") or []
                if impacts:
                    lines = []
                    for imp in impacts:
                        loc = imp.get("location")
                        arr = imp.get("arrivalTime")
                        if loc and arr:
                            lines.append(f"  - **{loc}:** `{arr}`")
                    if lines:
                        st.markdown(
                            "**Predicted arrivals (ENLIL):**\n" + "\n".join(lines)
                        )

        with st.expander("Raw DONKI JSON"):
            st.json(ev)

        start_str = ev.get("startTime")
        start_dt = None
        if start_str:
            try:
                start_dt = parser.parse(start_str).replace(tzinfo=dt.timezone.utc)
            except Exception:
                start_dt = None

        if start_dt:
            st.markdown(f"**DONKI startTime (UTC):** `{start_dt}`")
        else:
            st.warning("Selected event has no valid `startTime` field.")
