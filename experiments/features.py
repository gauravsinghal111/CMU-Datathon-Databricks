"""Shared data loading + feature building for local bed-count experiments (no dataset values are modified)."""
import json, re, os
import numpy as np
import pandas as pd

DATA = r"c:\Users\mihir\Documents\Databricks\data"

def parse_list(s):
    if isinstance(s, str) and s.startswith("["):
        try:
            return json.loads(s)
        except Exception:
            pass
    if isinstance(s, str) and s:
        return [x.strip() for x in s.split("; ") if x.strip()]
    return []

BED_RE = re.compile(r"(\d{2,4})\s*\+?\s*[- ]?(?:bed(?:s|ded)?)\b")

def from_text(t, rx, lo, hi):
    vals = [int(m.group(1)) for m in rx.finditer(t) if lo <= int(m.group(1)) <= hi]
    return float(max(vals)) if vals else np.nan

def load():
    fac = pd.read_csv(os.path.join(DATA, "facilities_clean.csv"), low_memory=False)
    fac = fac[fac["include_in_search"].astype(str).str.lower() == "true"].drop_duplicates("unique_id").reset_index(drop=True)
    for c in ["specialties_list", "procedure_list", "equipment_list", "capability_list"]:
        fac[c] = fac[c].map(parse_list)
    for c in ["beds", "doctors", "loc_lat", "loc_lon", "pin_radius_km", "n_sources"]:
        fac[c] = pd.to_numeric(fac[c], errors="coerce")
    fac["description"] = fac["description"].fillna("")
    fac["text"] = fac.apply(lambda r: " | ".join(r["specialties_list"] + r["procedure_list"] + r["equipment_list"]
                                                 + r["capability_list"] + [r["description"]]).lower(), axis=1)
    fac["beds_text"] = fac["text"].map(lambda t: from_text(t, BED_RE, 5, 5000))
    # website-crawled beds (fills only)
    wpath = os.path.join(DATA, "website_size.csv")
    if os.path.exists(wpath):
        w = pd.read_csv(wpath).drop_duplicates("unique_id")[["unique_id", "beds_fill"]]
        fac = fac.merge(w, on="unique_id", how="left")
    else:
        fac["beds_fill"] = np.nan
    fac["beds_fill"] = pd.to_numeric(fac["beds_fill"], errors="coerce")
    fac["beds_known"] = fac["beds"].fillna(fac["beds_text"]).fillna(fac["beds_fill"])
    return fac

def clean_text(s):
    s = re.sub(r"\d[\d,]*\s*\+?\s*-?\s*(?:bedded|beds?|doctors?|consultants?|specialists?|physicians?)", " ", s)
    s = re.sub(r"\b(?:bedded|beds?)\b", " ", s)
    return re.sub(r"\d+", " ", s)

FLAGS = {
    "icu": r"\bicu\b|intensive care", "nicu": r"\bnicu\b|neonatal intensive", "cathlab": r"cath ?lab|angioplasty",
    "mri": r"\bmri\b", "ct": r"\bct\b|computed tomography", "bloodbank": r"blood bank", "dialysis": r"dialysis",
    "emergency": r"emergency|24\s*/\s*7|24x7|casualty", "ot": r"operation theat|operating theat|modular ot",
    "trauma": r"trauma", "cancer": r"oncolog|chemotherap|radiotherap", "ambulance": r"ambulance",
    "transplant": r"transplant", "lab": r"laborator|pathology", "ventilator": r"ventilator",
    "teaching": r"medical college|teaching hospital|\bmbbs\b|postgraduate|residency|\bdnb\b",
    "accredited": r"\bnabh\b|\bjci\b|\bnabl\b|accredit",
}

def base_features(fac):
    X = pd.DataFrame(index=fac.index)
    for k, rx in FLAGS.items():
        X["svc_" + k] = fac["text"].str.contains(rx, regex=True).astype(int)
    X["n_services"] = X.filter(like="svc_").sum(axis=1)
    X["n_specialties"] = fac["specialties_list"].map(len)
    X["n_procedures"] = fac["procedure_list"].map(len)
    X["n_equipment"] = fac["equipment_list"].map(len)
    X["n_capability"] = fac["capability_list"].map(len)
    X["public"] = (fac["operator_type"] == "public").astype(int)
    for t in ["hospital", "clinic", "dentist", "doctor", "pharmacy"]:
        X["type_" + t] = (fac["facility_type"] == t).astype(int)
    X["doctors_log"] = np.log1p(fac["doctors"])
    return X

def band(b):
    return np.where(b < 20, 0, np.where(b <= 100, 1, 2))      # 0 small, 1 medium, 2 large
