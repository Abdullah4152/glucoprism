"""Download the public CGM corpora used by GlucoPRISM from their authoritative sources.

Every URL here points at the primary distributor named by the originating publication
(PhysioNet, figshare, PLOS, GitHub, Hugging Face) -- no mirrors, no re-uploads.

Usage:
    python scripts/download_datasets.py --all
    python scripts/download_datasets.py cgmacros shanghai
    python scripts/download_datasets.py --list
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) GlucoPRISM-research-downloader/1.0"


@dataclass
class Source:
    key: str
    name: str
    landing: str            # human-facing landing page / DOI
    files: list[tuple[str, str]] = field(default_factory=list)  # (url, local filename)
    unzip: bool = True
    note: str = ""
    manual: bool = False    # requires a DUA / account -- cannot be auto-fetched


SOURCES: dict[str, Source] = {}


def add(src: Source) -> None:
    SOURCES[src.key] = src


add(Source(
    key="cgmacros",
    name="CGMacros (Das et al. 2025) -- paired Dexcom G6 + FreeStyle Libre Pro, 45 subjects",
    landing="https://physionet.org/content/cgmacros/1.0.0/",
    files=[("https://physionet.org/static/published-projects/cgmacros/"
            "cgmacros-a-scientific-dataset-for-personalized-nutrition-and-diet-monitoring-1.0.0.zip",
            "cgmacros-1.0.0.zip")],
    note="CC BY-NC-SA 4.0. Open access, no credentialing.",
))

_BIGIDEAS_BASE = "https://physionet.org/files/big-ideas-glycemic-wearable/1.1.2"
_BIGIDEAS_SUBJECTS = [f"{i:03d}" for i in range(1, 17)]

add(Source(
    key="bigideas",
    name="BIG IDEAs Lab Glycemic Variability and Wearable Device Data (Cho et al. 2023)",
    landing="https://physionet.org/content/big-ideas-glycemic-wearable/1.1.2/",
    files=(
        [(f"{_BIGIDEAS_BASE}/Demographics.csv", "Demographics.csv"),
         (f"{_BIGIDEAS_BASE}/LICENSE.txt", "LICENSE.txt"),
         (f"{_BIGIDEAS_BASE}/SHA256SUMS.txt", "SHA256SUMS.txt")]
        + [(f"{_BIGIDEAS_BASE}/{s}/Dexcom_{s}.csv", f"{s}/Dexcom_{s}.csv") for s in _BIGIDEAS_SUBJECTS]
        + [(f"{_BIGIDEAS_BASE}/{s}/Food_Log_{s}.csv", f"{s}/Food_Log_{s}.csv") for s in _BIGIDEAS_SUBJECTS]
    ),
    unzip=False,
    note="ODC-By 1.0, open access. We pull only the Dexcom CGM + food logs + demographics "
         "(the full archive is 4.7 GB because of the 64 Hz Empatica E4 streams, which "
         "GlucoFM/GlucoPRISM do not use).",
))

add(Source(
    key="shanghai",
    name="ShanghaiT1DM + ShanghaiT2DM (Zhao et al. 2023, Sci Data)",
    landing="https://doi.org/10.6084/m9.figshare.c.6310860",
    files=[("https://ndownloader.figshare.com/files/42966622", "diabetes_datasets.zip")],
    note="figshare article 21600933, CC BY 4.0.",
))

add(Source(
    key="stanford",
    name="Stanford metabolic-subphenotype cohort (Metwally et al. 2025, Nat Biomed Eng)",
    landing="https://github.com/aametwally/Metabolic_Subphenotype_Predictor",
    files=[("https://github.com/aametwally/Metabolic_Subphenotype_Predictor/archive/refs/heads/main.zip",
            "Metabolic_Subphenotype_Predictor-main.zip")],
    note="Public GitHub release of the filtered CGM / OGTT / metabolic-test tables.",
))

add(Source(
    key="cgmjepa",
    name="CGM-JEPA released corpora (Muhammad et al. 2026) -- Stanford+Colas pretrain CSV, labelled splits",
    landing="https://huggingface.co/CRUISEResearchGroup/CGM-JEPA",
    files=[],  # handled by _fetch_hf
    note="Hugging Face datasets; also carries the Colas subset used by both prior papers.",
))

add(Source(
    key="colas",
    name="Colas et al. 2019 (PLOS ONE) -- iPro CGM, 208 at-risk subjects",
    landing="https://doi.org/10.1371/journal.pone.0225817",
    files=[],  # filled in by probe_plos_supplements()
    unzip=False,
    note="Primary CGM stream reaches us via the CGM-JEPA pretraining corpus (206 subjects). "
         "The PLOS supplement carries the DFA feature tables.",
))

add(Source(
    key="hall",
    name="Hall et al. 2018 glucotypes (PLOS Biology) -- Dexcom G4, 57 subjects",
    landing="https://doi.org/10.1371/journal.pbio.2005143",
    files=[],  # filled in by probe_plos_supplements()
    unzip=False,
    note="S1 Data = raw CGM; S5 Data = SQLite master table with glucotype + clinical labels.",
))

add(Source(
    key="diatrend",
    name="DiaTrend (Prioleau et al. 2023) -- 54 subjects, 27,561 CGM-days, multi-year",
    landing="https://doi.org/10.7303/syn38187184",
    manual=True,
    note="Synapse project syn38187184. Verified gate: ManagedACTAccessRequirement 9606040 and "
         "SelfSignAccessRequirement 9606041, both accessType=DOWNLOAD, requiring "
         "isCertifiedUserRequired=true, isValidatedProfileRequired=true, isIDURequired=true. "
         "Anonymous file access returns HTTP 403. Validated Profile is reviewed manually by "
         "Synapse ACT (several days). Once you hold a personal access token with Download+View "
         "scope, set SYNAPSE_AUTH_TOKEN (or put it in .synapse/token.txt) and run: "
         "`pip install synapseclient && synapse get -r syn38187184 "
         "--downloadLocation data/raw/diatrend`. See docs/datasets/diatrend.md.",
))

_JAEB_S3 = "https://live-jchrpublicdatasets.s3.amazonaws.com/Diabetes/Public%20Datasets"

add(Source(
    key="replacebg",
    name="REPLACE-BG (Aleppo et al. 2017, JAEB) -- Dexcom G4 Platinum, 226 T1D adults, 6 months",
    landing="https://public.jaeb.org/datasets/diabetes",
    files=[(f"{_JAEB_S3}/Replace-BG%20Dataset.zip", "Replace-BG_Dataset.zip")],
    note="JAEB Center public datasets portal (NCT02258373). The page renders download links "
         "as ASP.NET __doPostBack calls, but each row also carries the real object URL in the "
         "download icon's alt attribute, pointing at a public S3 bucket -- no login, no "
         "click-through DUA. Redistribution terms are in the archive's own documentation.",
))

add(Source(
    key="ohiot1dm",
    name="OhioT1DM (Marling & Bunescu 2018/2020) -- Medtronic Enlite, 12 T1D subjects, 8 weeks",
    landing="https://webpages.charlotte.edu/rbunescu/data/ohiot1dm/OhioT1DM-dataset.html",
    manual=True,
    note="NOTE: the dataset has moved from Ohio University to UNC Charlotte. Email "
         "razvan.bunescu@charlotte.edu, subject 'OhioT1DM Request', from an INSTITUTIONAL "
         "address (personal addresses are rejected), including researcher name/title/email and "
         "the institution's full postal address. Distribution is governed by a Data Use "
         "Agreement signed by the institution. ~1 week turnaround; you receive an encrypted "
         "archive plus a password. Cannot be automated. See docs/datasets/ohiot1dm.md.",
))


# --------------------------------------------------------------------------- helpers

def _open(url: str, offset: int = 0):
    headers = {"User-Agent": UA, "Accept": "*/*"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    req = urllib.request.Request(url, headers=headers)
    return urllib.request.urlopen(req, timeout=120)


def download(url: str, dest: Path, retries: int = 8) -> Path:
    """Resumable fetch. PhysioNet in particular drops long connections, so we keep
    the .part file and continue with a Range request instead of starting over."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"    [skip] {dest.name} already present ({dest.stat().st_size:,} B)")
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")

    for attempt in range(1, retries + 1):
        offset = tmp.stat().st_size if tmp.exists() else 0
        try:
            r = _open(url, offset)
            resumed = r.status == 206
            if offset and not resumed:
                offset = 0                       # server ignored Range; restart
            total = int(r.headers.get("Content-Length") or 0) + offset

            with r, open(tmp, "ab" if offset else "wb") as f:
                got, t0, last = offset, time.time(), 0.0
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    got += len(chunk)
                    if time.time() - last > 5:
                        pct = f"{100*got/total:5.1f}%" if total else f"{got/1e6:8.1f} MB"
                        rate = (got - offset) / 1e6 / max(time.time() - t0, 0.1)
                        print(f"      {pct}  ({got/1e6:.1f} MB, {rate:.1f} MB/s)")
                        last = time.time()

            if total and tmp.stat().st_size < total:
                raise IOError(f"short read: {tmp.stat().st_size:,} of {total:,}")
            tmp.replace(dest)
            print(f"    [ok]   {dest.name} ({dest.stat().st_size:,} B)")
            return dest
        except Exception as e:  # noqa: BLE001
            have = tmp.stat().st_size if tmp.exists() else 0
            print(f"    [retry {attempt}/{retries}] {e}  (have {have:,} B, will resume)")
            time.sleep(min(3 * attempt, 20))
    raise RuntimeError(f"failed to download {url}")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def extract(zpath: Path, outdir: Path) -> None:
    marker = outdir / ".extracted"
    if marker.exists():
        print(f"    [skip] {zpath.name} already extracted")
        return
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"    [unzip] {zpath.name} -> {outdir}")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(outdir)
    marker.write_text(f"{zpath.name}\n{sha256(zpath)}\n")


def _fetch_hf(outdir: Path) -> None:
    """Pull the three CGM-JEPA Hugging Face repos (weights + labelled splits + pretrain CSV)."""
    from huggingface_hub import snapshot_download

    for repo, kind, sub in [
        ("CRUISEResearchGroup/CGM-JEPA", "model", "weights"),
        ("CRUISEResearchGroup/CGM-JEPA-Downstream", "dataset", "downstream"),
        ("CRUISEResearchGroup/CGM-JEPA-Pretraining", "dataset", "pretraining"),
    ]:
        target = outdir / sub
        print(f"    [hf] {repo} -> {target}")
        snapshot_download(repo_id=repo, repo_type=kind, local_dir=str(target))


def probe_plos_supplements(doi: str, journal: str, n: int = 30) -> list[tuple[str, str]]:
    """PLOS numbers supplements s001..sNNN without exposing the S-label; resolve real filenames."""
    out = []
    for i in range(1, n + 1):
        sid = f"s{i:03d}"
        url = f"https://journals.plos.org/{journal}/article/file?type=supplementary&id={doi}.{sid}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA}, method="GET")
            with urllib.request.urlopen(req, timeout=60) as r:
                final = r.geturl()
                name = final.split("?")[0].rsplit("/", 1)[-1]
                size = r.headers.get("Content-Length")
                print(f"      {sid}: {name}  ({size} B)")
                out.append((url, name))
        except Exception as e:  # noqa: BLE001
            print(f"      {sid}: -- ({e})")
            break
    return out


# --------------------------------------------------------------------------- main

def run(keys: list[str]) -> None:
    manifest_path = RAW / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    for key in keys:
        src = SOURCES[key]
        outdir = RAW / key
        print(f"\n=== {key}: {src.name}")
        print(f"    landing: {src.landing}")

        if src.manual:
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / "HOW_TO_OBTAIN.txt").write_text(
                f"{src.name}\n\nLanding page: {src.landing}\n\n{src.note}\n", encoding="utf-8")
            print(f"    [manual] {src.note}")
            manifest[key] = {"status": "manual", "landing": src.landing, "note": src.note}
            continue

        if key == "cgmjepa":
            _fetch_hf(outdir)
            manifest[key] = {"status": "ok", "landing": src.landing, "via": "huggingface_hub"}
            continue

        if key == "hall":
            src.files = probe_plos_supplements("10.1371/journal.pbio.2005143", "plosbiology")
        if key == "colas":
            src.files = probe_plos_supplements("10.1371/journal.pone.0225817", "plosone")

        entries = []
        for url, fname in src.files:
            p = download(url, outdir / fname)
            entries.append({"file": fname, "url": url, "bytes": p.stat().st_size,
                            "sha256": sha256(p)})
            if src.unzip and fname.endswith(".zip"):
                extract(p, outdir / "extracted")
        manifest[key] = {"status": "ok", "landing": src.landing, "note": src.note,
                         "files": entries}

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nManifest written to {manifest_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("keys", nargs="*", help="dataset keys to fetch")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    if a.list:
        for k, s in SOURCES.items():
            flag = "MANUAL" if s.manual else "auto  "
            print(f"  {flag}  {k:<12} {s.name}")
        sys.exit(0)

    keys = list(SOURCES) if a.all else a.keys
    if not keys:
        ap.error("give dataset keys, or --all")
    bad = [k for k in keys if k not in SOURCES]
    if bad:
        ap.error(f"unknown keys: {bad}")
    run(keys)
