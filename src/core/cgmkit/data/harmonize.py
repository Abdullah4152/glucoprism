"""Readers that turn each raw corpus into one common long table.

Common schema (one row per CGM reading):

    dataset  subject  device  sampling_min  timestamp  glucose_mgdl

Unit handling is per source: Shanghai's summary sheet is already mg/dL for
glucose but mmol/L for lipids, Colas ships mg/dL, everything else is mg/dL.
Subject ids are prefixed with the dataset so that a pooled corpus can never
silently merge two cohorts' "S01".
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[3] / "data" / "raw"

SCHEMA = ["dataset", "subject", "device", "sampling_min", "timestamp", "glucose_mgdl"]

MMOL_TO_MGDL_GLUCOSE = 18.0182
MMOL_TO_MGDL_CHOL = 38.67       # GlucoFM App. A.3
MMOL_TO_MGDL_TG = 88.57         # GlucoFM App. A.3
PMOL_TO_MICROU_INSULIN = 6.945  # GlucoFM App. A.3


def _finish(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.dropna(subset=["timestamp", "glucose_mgdl"])
    rows = rows[np.isfinite(rows["glucose_mgdl"])]
    rows = rows.sort_values(["subject", "timestamp"]).reset_index(drop=True)
    return rows[SCHEMA]


# ------------------------------------------------------------------ Stanford

def read_stanford(root: Path = RAW) -> pd.DataFrame:
    """Metwally et al. 2025 -- Dexcom G6, 5 min, 56 subjects."""
    p = (root / "stanford/extracted/Metabolic_Subphenotype_Predictor-main/data"
         / "filtered_cgm_03222026.csv")
    d = pd.read_csv(p, parse_dates=["timestamp"])
    return _finish(pd.DataFrame({
        "dataset": "stanford",
        "subject": "stanford:" + d["subject"].astype(str),
        "device": "dexcom",
        "sampling_min": 5,
        "timestamp": d["timestamp"],
        "glucose_mgdl": pd.to_numeric(d["glucose_value"], errors="coerce"),
    }))


# ---------------------------------------------------------------------- Hall

def read_hall(root: Path = RAW) -> pd.DataFrame:
    """Hall et al. 2018 glucotypes -- Dexcom G4, 5 min, 57 subjects (S1 Data)."""
    import gzip
    p = root / "hall/pbio.2005143.s010.gz"
    with gzip.open(p, "rt") as f:
        d = pd.read_csv(f, sep="\t")
    return _finish(pd.DataFrame({
        "dataset": "hall",
        "subject": "hall:" + d["subjectId"].astype(str),
        "device": "dexcom",
        "sampling_min": 5,
        "timestamp": pd.to_datetime(d["DisplayTime"], errors="coerce"),
        "glucose_mgdl": pd.to_numeric(d["GlucoseValue"], errors="coerce"),
    }))


# ------------------------------------------------------------------ Shanghai

_SHANGHAI_CGM_COLS = ["CGM (mg / dl)", "CGM (mg/dl)", "CGM"]


def read_shanghai(root: Path = RAW, cohort: str = "T2DM") -> pd.DataFrame:
    """Zhao et al. 2023 -- FreeStyle Libre H, 15 min.

    Each file is one recording session; the paper (and GlucoFM App. A.2) treats a
    session as a distinct subject-entry because clinical labels can differ between
    sessions separated by long intervals. The file stem *is* the session id, e.g.
    `2000_0_20201230` = patient 2000, session 0, start date 2020-12-30.
    """
    d = root / f"shanghai/extracted/Shanghai_{cohort}"
    frames = []
    for f in sorted(d.glob("*.xls*")):
        if f.name.startswith("~$") or f.name.startswith("."):
            continue
        try:
            t = pd.read_excel(f)
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {f.name}: {e}")
            continue
        col = next((c for c in _SHANGHAI_CGM_COLS if c in t.columns), None)
        if col is None:
            col = next((c for c in t.columns if str(c).strip().upper().startswith("CGM")), None)
        if col is None:
            print(f"  [skip] {f.name}: no CGM column in {list(t.columns)[:4]}")
            continue
        frames.append(pd.DataFrame({
            "dataset": f"shanghai{cohort.lower()}",
            "subject": f"shanghai{cohort.lower()}:" + f.stem,
            "device": "libre",
            "sampling_min": 15,
            "timestamp": pd.to_datetime(t["Date"], errors="coerce"),
            "glucose_mgdl": pd.to_numeric(t[col], errors="coerce"),
        }))
    return _finish(pd.concat(frames, ignore_index=True))


# --------------------------------------------------------------------- Colas

def read_colas(root: Path = RAW) -> pd.DataFrame:
    """Colas et al. 2019 -- Medtronic iPro, 5 min, 208 case files.

    The released files carry only a time-of-day column (`hora`), no date. We
    reconstruct a monotone timeline by detecting midnight wraps and anchoring the
    first reading to an arbitrary fixed date -- absolute *time of day* is what
    GlucoFM's circadian encoding needs, and that is preserved exactly. The
    arbitrary calendar date never enters any feature.
    """
    d = root / "colas/extracted/S1"
    anchor = pd.Timestamp("2015-01-01")
    frames = []
    for f in sorted(d.glob("case*.csv"), key=lambda p: int(re.search(r"(\d+)", p.stem).group(1))):
        case = int(re.search(r"(\d+)", f.stem).group(1))
        t = pd.read_csv(f)
        hora = pd.to_timedelta(t["hora"].astype(str))
        day = (hora.diff().dt.total_seconds().fillna(0) < 0).cumsum()
        ts = anchor + pd.to_timedelta(day, unit="D") + hora
        frames.append(pd.DataFrame({
            "dataset": "colas",
            "subject": f"colas:case{case:03d}",
            "device": "ipro",
            "sampling_min": 5,
            "timestamp": ts,
            "glucose_mgdl": pd.to_numeric(t["glucemia"], errors="coerce"),
        }))
    return _finish(pd.concat(frames, ignore_index=True))


# ----------------------------------------------------------------- BIG IDEAs

def read_bigideas(root: Path = RAW) -> pd.DataFrame:
    """Cho et al. 2023 -- Dexcom G6, 5 min, 16 subjects.

    Files are raw Dexcom Clarity exports: the first ~10 rows are patient metadata
    with `Event Type` in {FirstName, LastName, DateOfBirth, ...}; only rows with
    `Event Type == "EGV"` are sensor glucose readings.
    """
    frames = []
    for sub in sorted(p for p in root.glob("bigideas/[0-9][0-9][0-9]") if p.is_dir()):
        f = sub / f"Dexcom_{sub.name}.csv"
        if not f.exists():
            continue
        t = pd.read_csv(f, low_memory=False)
        t = t[t["Event Type"].astype(str).str.upper() == "EGV"]
        g = t["Glucose Value (mg/dL)"].astype(str)
        # Dexcom writes "Low" / "High" at the reporting limits.
        g = g.replace({"Low": "40", "High": "400"})
        frames.append(pd.DataFrame({
            "dataset": "bigideas",
            "subject": f"bigideas:{sub.name}",
            "device": "dexcom",
            "sampling_min": 5,
            "timestamp": pd.to_datetime(t["Timestamp (YYYY-MM-DDThh:mm:ss)"], errors="coerce"),
            "glucose_mgdl": pd.to_numeric(g, errors="coerce"),
        }))
    return _finish(pd.concat(frames, ignore_index=True))


# ------------------------------------------------------------------- D1namo

def read_d1namo(root: Path = RAW) -> pd.DataFrame:
    """Dubosson et al. 2018 -- Medtronic iPro2, 5 min, 29 subjects.

    Two arms on disk, `diabetes_subset_*` (9 subjects) and `healthy_subset_*`
    (20), each holding one `glucose.csv` per subject with columns
    `date, time, glucose, type, comments`.

    Two things this reader must get right:

    * **Units are mmol/L**, not mg/dL. Everything downstream assumes mg/dL, so
      convert at read time (x 18.018) exactly as the ShanghaiT2DM reader does.
    * **`type` is `cgm` or `manual`.** Manual rows are fingerstick reference
      values, not sensor readings; keeping them would inject a second, differently
      calibrated instrument into a trace the model is told is one sensor.

    MEASURED: only the **diabetes arm carries CGM**. The healthy arm's `type`
    column holds meal-relative fingersticks only -- `BB`/`AB`/`BL`/`AL`/`BD`/`AD`
    (before/after breakfast, lunch, dinner), about six readings a day. So this
    reader yields 9 subjects, not 29, and that is correct rather than a parse
    failure: the other 20 have no sensor trace to window.

    Subject ids are prefixed with the arm because both arms number from 001.
    """
    frames = []
    for f in sorted(root.glob("d1namo/**/glucose.csv")):
        arm = "diabetes" if "diabetes_subset" in str(f) else "healthy"
        sid = f.parent.name
        t = pd.read_csv(f)
        t = t[t.get("type", pd.Series(index=t.index, dtype=object))
              .astype(str).str.lower() == "cgm"]
        if t.empty:
            continue
        ts = pd.to_datetime(t["date"].astype(str) + " " + t["time"].astype(str),
                            errors="coerce")
        frames.append(pd.DataFrame({
            "dataset": "d1namo",
            "subject": f"d1namo:{arm}-{sid}",
            "device": "ipro",
            "sampling_min": 5,
            "timestamp": ts,
            "glucose_mgdl": pd.to_numeric(t["glucose"], errors="coerce") * 18.018,
        }))
    if not frames:
        raise FileNotFoundError(f"no d1namo glucose.csv under {root / 'd1namo'}")
    return _finish(pd.concat(frames, ignore_index=True))


# ----------------------------------------------------------------- CGMacros

def _interpolation_knots(x: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """Boolean mask of the true samples in a piecewise-linearly interpolated series.

    Between two real readings the released CGMacros CSV writes a straight line at
    1-minute resolution, so the first difference is constant *within* a segment and
    changes only at a real sample. Those breakpoints are the sensor's actual
    readings; everything else is interpolation and must not be treated as observed.
    """
    n = len(x)
    keep = np.zeros(n, dtype=bool)
    finite = np.isfinite(x)
    if finite.sum() < 3:
        keep[finite] = True
        return keep

    idx = np.flatnonzero(finite)
    v = x[idx]
    d = np.diff(v)
    # A breakpoint is where the slope changes; also keep both endpoints.
    brk = np.ones(len(v), dtype=bool)
    brk[1:-1] = np.abs(np.diff(d)) > tol
    keep[idx[brk]] = True
    return keep


def read_cgmacros(root: Path = RAW) -> pd.DataFrame:
    """Das et al. 2025 -- Dexcom G6 *and* FreeStyle Libre Pro on the same 45
    subjects over the same days. This is the paired-sensor natural experiment
    (proposal V1), so both device streams are emitted as separate rows sharing the
    subject id -- the pairing is (subject, date, {dexcom, libre}).

    The released per-subject CSV is on a **1-minute** grid with both glucose
    columns linearly interpolated between real readings. Keeping the interpolated
    points would make the observation mask uniformly 1 and erase the very
    device-cadence difference (5 min vs 15 min) this cohort exists to expose, so we
    recover the interpolation knots and emit only those.
    """
    base = next((p for p in [root / "cgmacros/extracted", root / "cgmacros"]
                 if p.exists()), None)
    files = sorted(base.rglob("CGMacros-*.csv")) if base else []
    frames = []
    for f in files:
        m = re.search(r"CGMacros-(\d+)", f.stem)
        if not m:
            continue
        sid = f"cgmacros:{int(m.group(1)):03d}"
        t = pd.read_csv(f, low_memory=False)
        tcol = next((c for c in t.columns if "timestamp" in c.lower() or c.lower() == "time"), None)
        if tcol is None:
            continue
        ts = pd.to_datetime(t[tcol], errors="coerce")
        for col, dev, rate in [("Dexcom GL", "dexcom", 5), ("Libre GL", "libre", 15)]:
            if col not in t.columns:
                continue
            g = pd.to_numeric(t[col], errors="coerce").to_numpy(dtype=float)
            keep = _interpolation_knots(g)
            frames.append(pd.DataFrame({
                "dataset": "cgmacros",
                "subject": sid,
                "device": dev,
                "sampling_min": rate,
                "timestamp": ts[keep].reset_index(drop=True),
                "glucose_mgdl": g[keep],
            }))
    if not frames:
        raise FileNotFoundError(f"no CGMacros-*.csv found under {base}")
    return _finish(pd.concat(frames, ignore_index=True))


def read_replacebg(root: Path = RAW, chunksize: int = 2_000_000) -> pd.DataFrame:
    """REPLACE-BG (Aleppo et al. 2017, JAEB) -- Dexcom G4 Platinum, 5 min, 226 T1D adults.

    The largest corpus in the plan by an order of magnitude: 14.8 M CGM readings,
    ~1.23 M monitoring hours, a median of 254 days per subject.

    De-identification replaces calendar dates with `DeviceDtTmDaysFromEnroll`
    (integer days relative to enrollment) plus `DeviceTm` (time of day). As with
    Colas, we anchor day 0 to a fixed arbitrary date: **absolute time of day is
    preserved exactly**, which is what the circadian encoding consumes, and the
    calendar date never enters a feature.

    `RecordType` distinguishes `CGM` from `Calibration` fingersticks; only CGM rows
    are sensor glucose.
    """
    f = next((root / "replacebg/extracted").rglob("HDeviceCGM.txt"), None)
    if f is None:
        raise FileNotFoundError(f"HDeviceCGM.txt not found under {root / 'replacebg'}")
    anchor = pd.Timestamp("2015-01-01")

    frames = []
    cols = ["PtID", "DeviceDtTmDaysFromEnroll", "DeviceTm", "RecordType", "GlucoseValue"]
    for chunk in pd.read_csv(f, sep="|", usecols=cols, chunksize=chunksize, low_memory=False):
        chunk = chunk[chunk["RecordType"].astype(str).str.upper() == "CGM"]
        if chunk.empty:
            continue
        ts = (anchor
              + pd.to_timedelta(chunk["DeviceDtTmDaysFromEnroll"].astype("int64"), unit="D")
              + pd.to_timedelta(chunk["DeviceTm"].astype(str), errors="coerce"))
        frames.append(pd.DataFrame({
            "dataset": "replacebg",
            "subject": "replacebg:" + chunk["PtID"].astype(str),
            "device": "dexcom",
            "sampling_min": 5,
            "timestamp": ts,
            "glucose_mgdl": pd.to_numeric(chunk["GlucoseValue"], errors="coerce"),
        }))
    return _finish(pd.concat(frames, ignore_index=True))


READERS = {
    "stanford": read_stanford,
    "replacebg": read_replacebg,
    "hall": read_hall,
    "shanghait2dm": lambda root=RAW: read_shanghai(root, "T2DM"),
    "shanghait1dm": lambda root=RAW: read_shanghai(root, "T1DM"),
    "colas": read_colas,
    "bigideas": read_bigideas,
    "cgmacros": read_cgmacros,
    "d1namo": read_d1namo,
}


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Per-(dataset, device) subject count and monitoring hours, for the dataset docs."""
    g = df.groupby(["dataset", "device", "sampling_min"])
    return pd.DataFrame({
        "subjects": g["subject"].nunique(),
        "readings": g.size(),
        "hours": (g.size() * g["sampling_min"].first() / 60.0).round(0),
        "first": g["timestamp"].min(),
        "last": g["timestamp"].max(),
        "glucose_min": g["glucose_mgdl"].min(),
        "glucose_median": g["glucose_mgdl"].median(),
        "glucose_max": g["glucose_mgdl"].max(),
    }).reset_index()
