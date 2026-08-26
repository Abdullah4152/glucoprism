"""Downstream clinical labels, built exactly as GlucoFM Appendix A.3 defines them.

Every threshold below is quoted from the paper. They are evaluation conventions
for a benchmark, not diagnostic criteria -- the paper says so explicitly and we
repeat it here so nobody lifts these numbers into a clinical context.

Task coverage (GlucoFM Table 6):

    CGMacros (45)      diabetes(3-class), IR, obesity, hyperlipidemia
    Hall (56)          diabetes, glucotype, IR, hyperlipidemia
    Stanford (37)      diabetes, beta-cell dysfunction, IR
    ShanghaiT2DM (65)  hypoglycemia, IR, hyperlipidemia

Shared definitions:
    HOMA-IR   = insulin_uU/mL * fasting_glucose_mg/dL / 405.0 ;  positive if > 2.9
    obesity   = BMI >= 30
    hyperlip. = total cholesterol >= 240  OR  LDL >= 160  OR  triglycerides >= 200 (mg/dL)
"""

from __future__ import annotations

import gzip
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from .harmonize import (RAW, MMOL_TO_MGDL_CHOL, MMOL_TO_MGDL_TG,
                        PMOL_TO_MICROU_INSULIN)

HOMA_IR_THRESHOLD = 2.9
BMI_OBESE = 30.0
TC_HIGH, LDL_HIGH, TG_HIGH = 240.0, 160.0, 200.0
HBA1C_ABNORMAL = 5.7


def homa_ir(insulin_uU_mL, glucose_mg_dL):
    return np.asarray(insulin_uU_mL, float) * np.asarray(glucose_mg_dL, float) / 405.0


def hyperlipidemia(tc_mgdl, ldl_mgdl, tg_mgdl):
    tc = pd.to_numeric(tc_mgdl, errors="coerce")
    ldl = pd.to_numeric(ldl_mgdl, errors="coerce")
    tg = pd.to_numeric(tg_mgdl, errors="coerce")
    hit = (tc >= TC_HIGH) | (ldl >= LDL_HIGH) | (tg >= TG_HIGH)
    # A subject is negative only if at least one lipid was measured and none is high.
    known = tc.notna() | ldl.notna() | tg.notna()
    return hit.where(known)


# ------------------------------------------------------------------ Stanford

def stanford_labels(root: Path = RAW) -> pd.DataFrame:
    """IR (SSPG classes), beta-cell dysfunction (median DI split), diabetes (HbA1c >= 5.7)."""
    d = (root / "stanford/extracted/Metabolic_Subphenotype_Predictor-main/data")
    tests = pd.read_csv(d / "filtered_metabolic_tests.csv")
    chars = pd.read_csv(d / "filtered_study_participants_characteristics.csv")
    m = tests.merge(chars, on="SubjectID", how="outer", suffixes=("", "_c"))
    # A subject can appear under both exp_type flags (CGM-JEPA calls these the
    # "dual-cohort overlaps"), and the merge multiplies those rows. Collapse to one
    # row per subject, keeping the first non-null value of every field -- otherwise
    # a subject whose second row has a NaN disposition index looks unlabelled.
    m = (m.sort_values(["SubjectID", "exp_type"])
           .groupby("SubjectID", as_index=False)
           .first())

    out = pd.DataFrame({"subject": "stanford:" + m["SubjectID"].astype(str)})
    out["exp_type"] = m["exp_type"].fillna(m.get("ExperimentType"))
    out["ir"] = m["sspg_2_classes"].map({"IR": 1, "IS": 0})
    out["beta_cell"] = m["di_2_classes_median"].map({"Dysfunction": 1, "Normal": 0})
    out["diabetes"] = (pd.to_numeric(m["HbA1c"], errors="coerce") >= HBA1C_ABNORMAL).astype("Int64")
    out.loc[pd.to_numeric(m["HbA1c"], errors="coerce").isna(), "diabetes"] = pd.NA
    out["sspg"] = pd.to_numeric(m["sspg"], errors="coerce")
    out["di"] = pd.to_numeric(m["di"], errors="coerce")
    out["hba1c"] = pd.to_numeric(m["HbA1c"], errors="coerce")
    out["age_band"] = m.get("Age")
    out["bmi_band"] = m.get("BMI")
    out["sex"] = m.get("Sex")
    out["ethnicity"] = m.get("Ethnicity")
    return out


# ---------------------------------------------------------------------- Hall

def hall_labels(root: Path = RAW) -> pd.DataFrame:
    """Diabetes risk, glucotype (severe vs not), IR (SSPG > 120 else HOMA-IR), hyperlipidemia."""
    con = sqlite3.connect(root / "hall/pbio.2005143.s014.db")
    c = pd.read_sql_query("select * from clinical", con)
    con.close()

    out = pd.DataFrame({"subject": "hall:" + c["userID"].astype(str)})

    # (1) diabetes risk: prediabetic or diabetic -> abnormal glucose regulation
    diag = c["diagnosis"].astype(str).str.lower()
    out["diabetes"] = np.where(diag.isin(["pre-diabetic", "diabetic"]), 1,
                               np.where(diag == "non-diabetic", 0, np.nan))

    # (2) glucotype: severe vs {low, moderate}
    gt = c["glucotype"].astype(str).str.lower()
    out["glucotype"] = np.where(gt == "severe", 1,
                                np.where(gt.isin(["low", "moderate"]), 0, np.nan))

    # (3) IR: SSPG > 120 when available, otherwise HOMA-IR > 2.9
    sspg = pd.to_numeric(c["SSPG"], errors="coerce")
    hi = homa_ir(pd.to_numeric(c["insulin"], errors="coerce"),
                 pd.to_numeric(c["FBG"], errors="coerce"))
    out["ir"] = np.where(sspg.notna(), (sspg > 120).astype(float),
                         np.where(np.isfinite(hi), (hi > HOMA_IR_THRESHOLD).astype(float), np.nan))

    # (4) hyperlipidemia (Tchol / LDL / Trg already mg/dL in this table)
    out["hyperlipidemia"] = hyperlipidemia(c["Tchol"], c["LDL"], c["Trg"]).astype(float)

    out["sspg"] = sspg
    out["hba1c"] = pd.to_numeric(c["A1C"], errors="coerce")
    out["bmi"] = pd.to_numeric(c["BMI"], errors="coerce")
    out["age"] = pd.to_numeric(c["Age"], errors="coerce")
    return out


# ------------------------------------------------------------------ Shanghai

def shanghai_labels(root: Path = RAW, cohort: str = "T2DM") -> pd.DataFrame:
    """Hypoglycemia (clinical record), IR (HOMA-IR after pmol/L -> uU/mL), hyperlipidemia
    (lipids converted mmol/L -> mg/dL)."""
    s = pd.read_excel(root / f"shanghai/extracted/Shanghai_{cohort}_Summary.xlsx")
    tag = f"shanghai{cohort.lower()}"
    out = pd.DataFrame({"subject": tag + ":" + s["Patient Number"].astype(str)})

    hyp = s["Hypoglycemia (yes/no)"].astype(str).str.strip().str.lower()
    out["hypoglycemia"] = np.where(hyp == "yes", 1.0, np.where(hyp == "no", 0.0, np.nan))

    insulin_uU = pd.to_numeric(s["Fasting Insulin (pmol/L)"], errors="coerce") / PMOL_TO_MICROU_INSULIN
    fpg = pd.to_numeric(s["Fasting Plasma Glucose (mg/dl)"], errors="coerce")
    hi = homa_ir(insulin_uU, fpg)
    out["ir"] = np.where(np.isfinite(hi), (hi > HOMA_IR_THRESHOLD).astype(float), np.nan)
    out["homa_ir"] = hi

    tc = pd.to_numeric(s["Total Cholesterol (mmol/L)"], errors="coerce") * MMOL_TO_MGDL_CHOL
    ldl = pd.to_numeric(s["Low-Density Lipoprotein Cholesterol (mmol/L)"], errors="coerce") * MMOL_TO_MGDL_CHOL
    tg = pd.to_numeric(s["Triglyceride (mmol/L)"], errors="coerce") * MMOL_TO_MGDL_TG
    out["hyperlipidemia"] = hyperlipidemia(tc, ldl, tg).astype(float)

    out["bmi"] = pd.to_numeric(s["BMI (kg/m2)"], errors="coerce")
    out["age"] = pd.to_numeric(s["Age (years)"], errors="coerce")
    return out


# ----------------------------------------------------------------- CGMacros

A1C_PREDIABETES, A1C_DIABETES = 5.7, 6.5


def cgmacros_labels(root: Path = RAW) -> pd.DataFrame:
    """Diabetes risk (3-class), IR (HOMA-IR), obesity (BMI), hyperlipidemia,
    from `bio.csv` in the PhysioNet release.

    `bio.csv` carries no recruitment-group column, so the three diabetes classes
    are reconstructed from the lab HbA1c using the standard ADA cut-points that the
    cohort was recruited against (< 5.7 normoglycemic, 5.7-6.4 prediabetes,
    >= 6.5 type 2 diabetes). CGMacros' own description -- 15 healthy /
    16 prediabetes / 14 T2D -- is the check on this.
    """
    base = next((p for p in [root / "cgmacros/extracted", root / "cgmacros"] if p.exists()), None)
    bio_path = next(iter(base.rglob("bio.csv")), None) if base else None
    if bio_path is None:
        raise FileNotFoundError(f"bio.csv not found under {base}")
    b = pd.read_csv(bio_path)

    def col(*names):
        """bio.csv column names carry stray trailing spaces and parenthetical units."""
        for n in names:                      # exact match on the stripped name
            for c in b.columns:
                if c.strip().lower() == n.lower():
                    return b[c]
        for n in names:                      # then substring
            for c in b.columns:
                if n.lower() in c.strip().lower():
                    return b[c]
        return pd.Series([np.nan] * len(b))

    sid = pd.to_numeric(col("subject"), errors="coerce")
    out = pd.DataFrame({"subject": ["cgmacros:%03d" % v if np.isfinite(v) else "cgmacros:NA"
                                    for v in sid]})

    a1c = pd.to_numeric(col("A1c PDL (Lab)", "a1c"), errors="coerce")
    out["diabetes_3class"] = np.select(
        [a1c < A1C_PREDIABETES,
         (a1c >= A1C_PREDIABETES) & (a1c < A1C_DIABETES),
         a1c >= A1C_DIABETES],
        [0.0, 1.0, 2.0], default=np.nan)
    out["hba1c"] = a1c

    insulin = pd.to_numeric(col("Insulin"), errors="coerce")            # uU/mL
    fpg = pd.to_numeric(col("Fasting GLU - PDL (Lab)", "fasting glu"), errors="coerce")
    hi = homa_ir(insulin, fpg)
    out["ir"] = np.where(np.isfinite(hi), (hi > HOMA_IR_THRESHOLD).astype(float), np.nan)
    out["homa_ir"] = hi

    bmi = pd.to_numeric(col("BMI"), errors="coerce")
    out["obesity"] = np.where(bmi.notna(), (bmi >= BMI_OBESE).astype(float), np.nan)
    out["bmi"] = bmi

    out["hyperlipidemia"] = hyperlipidemia(
        col("Cholesterol"), col("LDL (Cal)", "ldl"), col("Triglycerides")).astype(float)
    out["age"] = pd.to_numeric(col("Age"), errors="coerce")
    out["sex"] = col("Gender")
    out["ethnicity"] = col("Self-identify")
    return out


# --------------------------------------------------------------------- Colas

def colas_labels(root: Path = RAW) -> pd.DataFrame:
    """Colas is pretraining-only in both prior papers; we still parse the released
    clinical table so the incident-T2DM label is available for future work."""
    p = root / "colas/extracted/S1/clinical_data.txt"
    c = pd.read_csv(p, sep=r"\s+", quotechar='"')
    c = c.reset_index().rename(columns={"index": "case"})
    case = pd.to_numeric(c["case"], errors="coerce")
    if case.isna().all():
        case = pd.Series(np.arange(1, len(c) + 1))
    return pd.DataFrame({
        "subject": ["colas:case%03d" % i for i in case.astype(int)],
        "incident_t2dm": c["T2DM"].astype(str).str.upper().map({"TRUE": 1.0, "FALSE": 0.0}),
        "age": pd.to_numeric(c["age"], errors="coerce"),
        "bmi": pd.to_numeric(c["BMI"], errors="coerce"),
        "hba1c": pd.to_numeric(c["HbA1c"], errors="coerce"),
        "fasting_glucose": pd.to_numeric(c["glycaemia"], errors="coerce"),
        "followup_days": pd.to_numeric(c["follow.up"], errors="coerce"),
    })


LABEL_READERS = {
    "stanford": stanford_labels,
    "hall": hall_labels,
    "shanghait2dm": lambda root=RAW: shanghai_labels(root, "T2DM"),
    "cgmacros": cgmacros_labels,
    "colas": colas_labels,
}

# Task -> (cohort, column). Mirrors GlucoFM Table 3's 14 task-dataset cells.
TASK_MATRIX = {
    "cgmacros": ["diabetes_3class", "ir", "hyperlipidemia", "obesity"],
    "shanghait2dm": ["ir", "hyperlipidemia", "hypoglycemia"],
    "stanford": ["diabetes", "beta_cell", "ir"],
    "hall": ["diabetes", "ir", "hyperlipidemia", "glucotype"],
}
