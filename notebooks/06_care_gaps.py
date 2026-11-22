# Databricks notebook source
# =====================================================================
#  NOTEBOOK 6 - CARE GAPS & POLICY (for planners)
#
#  Question: where do people live too far from VERIFIED care for a service, and where is the need highest?
#
#  Method
#   1. Demand points: every PIN-code area (~19,000), weighted by its district's population (Census 2011)
#   2. For each service: distance from every PIN area to the nearest facility with STRONG evidence
#      (HIGH confidence in Notebook 2)
#   3. Travel standard per service (documented assumption, see REACH_KM)
#   4. Need: NFHS-5 district indicators (diabetes / blood pressure / births / tobacco) per service
#   5. Priority = people beyond reach x (1 + need), per district and service
#   6. Policy recommendations generated from the actual numbers
#
#  Honesty: a "gap" = no VERIFIED facility in THIS dataset (~10,000 facilities, not all of India).
#           Some gaps are data gaps, not care gaps - that is itself a policy finding.
#
#  Input : facility_service_evidence (NB2), pin_lookup (NB1), facilities_clean (NB1), cleaning_report (NB1),
#          NFHS-5 table (organisers), Wikipedia district populations (needs lxml, optional)
#  Output: care_gaps_pin       per PIN area x service: nearest verified facility, within reach?
#          care_gaps_district  per district x service: people beyond reach, need, priority, nearest facility
#          care_gaps_summary   per service: national coverage numbers
#          care_gap_policies   ranked policy recommendations with the evidence behind each
# =====================================================================
import io, json, re, math
import numpy as np
import pandas as pd
import requests
from sklearn.neighbors import BallTree

OUT = "workspace.default"
SRCDB = "databricks_virtue_foundation_dataset_dais_2026.virtue_foundation_dataset"

# Travel standard per service (km, straight line). Emergencies ~1 hour; repeat care close; specialist further.
REACH_KM = {
    "Emergency C-section": 50, "NICU (newborn intensive care)": 50, "Trauma / Accident care": 50,
    "ICU (intensive care)": 50, "Dialysis": 50, "Blood bank": 50,
    "CT scan": 100, "MRI scan": 100, "Angioplasty / Cath lab": 100, "Burn care": 100,
    "Cardiac surgery": 150, "Cancer care (chemo / radiation)": 150,
}
# Which NFHS-5 indicator signals NEED for each service (keywords to find the column)
NEED_KEYWORDS = {
    "Dialysis": [("high or very high", "blood sugar"), ("with high bp",)],
    "Cardiac surgery": [("with high bp",), ("high or very high", "blood sugar")],
    "Angioplasty / Cath lab": [("with high bp",), ("high or very high", "blood sugar")],
    "Emergency C-section": [("institutional birth 5y",), ("delivered by csection",)],
    "NICU (newborn intensive care)": [("institutional birth 5y",)],
    "Cancer care (chemo / radiation)": [("tobacco",)],
}

def norm(s):
    return re.sub(r"[^a-z]", "", str(s or "").lower().replace("&", "and"))

# =====================================================================
# 1. LOAD
# =====================================================================
ev = spark.table(f"{OUT}.facility_service_evidence").select(
    "service", "unique_id", "name", "city", "district", "state", "loc_lat", "loc_lon", "confidence").toPandas()
for c in ["loc_lat", "loc_lon"]:
    ev[c] = pd.to_numeric(ev[c], errors="coerce")
ev = ev.dropna(subset=["loc_lat", "loc_lon"])
pins = spark.table(f"{OUT}.pin_lookup").toPandas()
for c in ["pin_lat", "pin_lon"]:
    pins[c] = pd.to_numeric(pins[c], errors="coerce")
pins = pins.dropna(subset=["pin_lat", "pin_lon"]).reset_index(drop=True)
pins["dkey"], pins["skey"] = pins["pin_district"].map(norm), pins["pin_state"].map(norm)
print(f"{len(ev):,} facility-service rows | {len(pins):,} PIN areas")

# ---- district population (Census 2011 via Wikipedia); PINs share their district's population equally
try:
    r = requests.get("https://en.wikipedia.org/wiki/List_of_districts_in_India", timeout=30,
                     headers={"User-Agent": "CareFinderResearchBot/1.0 (student datathon project)"})
    rows = []
    for t in pd.read_html(io.StringIO(r.text)):
        cols = [" ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in t.columns]
        lc = [c.lower() for c in cols]
        d = [i for i, c in enumerate(lc) if "district" in c and "code" not in c]
        p = [i for i, c in enumerate(lc) if "population" in c]
        if d and p:
            for _, row in t.iterrows():
                m = re.search(r"\d[\d,]*", str(row.iloc[p[0]]))
                if m:
                    rows.append((norm(row.iloc[d[0]]), float(m.group(0).replace(",", ""))))
    pop = pd.DataFrame(rows, columns=["dkey", "population"])
    pop = pop[(pop["dkey"].str.len() > 2) & ~pop["dkey"].duplicated(keep=False)].set_index("dkey")["population"]
    pins["district_pop"] = pins["dkey"].map(pop)
    print(f"population matched for {pins['district_pop'].notna().mean():.0%} of PIN areas")
except Exception as e:
    print(f"population unavailable ({str(e)[:80]}); using PIN-area counts instead (run  %pip install lxml  to fix)")
    pins["district_pop"] = np.nan
pins_per_district = pins.groupby(["skey", "dkey"])["pin"].transform("count")
median_pin_pop = (pins["district_pop"] / pins_per_district).median()
pins["people"] = (pins["district_pop"] / pins_per_district).fillna(median_pin_pop if median_pin_pop == median_pin_pop
                                                                  else 1.0)

# ---- NFHS-5 need indicators (organisers' data)
nf = spark.table(f"{SRCDB}.nfhs_5_district_health_indicators").toPandas()
ncols = {c: c.lower().replace("_", " ") for c in nf.columns}   # NFHS uses snake_case
def find_col(words):
    return next((c for c, lc in ncols.items() if all(w in lc for w in words)), None)
def pct(x):
    try:
        return float(str(x).strip().replace("(", "").replace(")", ""))
    except ValueError:
        return np.nan
ndc = find_col(("district",))
nsc = find_col(("state",))
nf["dkey"], nf["skey"] = nf[ndc].map(norm), nf[nsc].map(norm)

def need_for(service):
    """District need score in [0,1] for a service (rank-normalised NFHS indicator), or None."""
    for words in NEED_KEYWORDS.get(service, []):
        col = find_col(words)
        if col:
            v = nf[col].map(pct)
            score = v.rank(pct=True)
            d = pd.DataFrame({"skey": nf["skey"], "dkey": nf["dkey"], "need": score}).dropna()
            return d.drop_duplicates(["skey", "dkey"]), col
    return None, None

# =====================================================================
# 2. DISTANCE FROM EVERY PIN AREA TO THE NEAREST VERIFIED FACILITY
# =====================================================================
P_RAD = np.radians(pins[["pin_lat", "pin_lon"]].values)
pin_rows, dist_rows, summ = [], [], []
for svc, reach in REACH_KM.items():
    strong = ev[(ev["service"] == svc) & (ev["confidence"] == "HIGH")].reset_index(drop=True)
    if len(strong) == 0:
        continue
    tree = BallTree(np.radians(strong[["loc_lat", "loc_lon"]].values), metric="haversine")
    d, i = tree.query(P_RAD, k=1)
    km = d[:, 0] * 6371
    near = strong.iloc[i[:, 0]].reset_index(drop=True)
    p = pins[["pin", "pin_district", "pin_state", "skey", "dkey", "pin_lat", "pin_lon", "people"]].copy()
    p["service"], p["reach_km"] = svc, reach
    p["nearest_km"] = km.round(1)
    p["within_reach"] = km <= reach
    p["nearest_name"], p["nearest_city"] = near["name"].values, near["city"].values
    # ---- any CLAIM of the service (any confidence): separates "verification gaps" from "possible care gaps"
    anyc = ev[ev["service"] == svc].reset_index(drop=True)
    tree_any = BallTree(np.radians(anyc[["loc_lat", "loc_lon"]].values), metric="haversine")
    km_any = tree_any.query(P_RAD, k=1)[0][:, 0] * 6371
    p["nearest_claim_km"] = km_any.round(1)
    p["gap_type"] = np.select([p["within_reach"], km_any <= reach],
                              ["covered (verified)", "verification gap"], default="possible care gap")
    pin_rows.append(p)

    # ---- district level
    p["people_beyond"] = np.where(p["within_reach"], 0, p["people"])
    p["people_verify"] = np.where(p["gap_type"] == "verification gap", p["people"], 0)
    p["people_nothing"] = np.where(p["gap_type"] == "possible care gap", p["people"], 0)
    g = p.groupby(["pin_state", "pin_district", "skey", "dkey"]).agg(
        people=("people", "sum"), people_beyond=("people_beyond", "sum"),
        people_verify=("people_verify", "sum"), people_nothing=("people_nothing", "sum"),
        pin_areas=("pin", "count"),
        pin_areas_beyond=("within_reach", lambda s: int((~s).sum())), median_km=("nearest_km", "median"),
        max_km=("nearest_km", "max"), nearest_claim_km=("nearest_claim_km", "median")).reset_index()
    # nearest verified facility to the district's centre
    cen = p.groupby(["skey", "dkey"])[["pin_lat", "pin_lon"]].median().reset_index()
    dc, ic = tree.query(np.radians(cen[["pin_lat", "pin_lon"]].values), k=1)
    cen["centre_nearest_km"] = (dc[:, 0] * 6371).round(0)
    cen["centre_nearest_facility"] = (strong.iloc[ic[:, 0]]["name"] + ", " + strong.iloc[ic[:, 0]]["city"].fillna("")).values
    cen["lat"], cen["lon"] = cen["pin_lat"], cen["pin_lon"]
    g = g.merge(cen[["skey", "dkey", "lat", "lon", "centre_nearest_km", "centre_nearest_facility"]], on=["skey", "dkey"])
    n_in = strong.assign(skey=strong["state"].map(norm), dkey=strong["district"].map(norm)) \
                 .groupby(["skey", "dkey"]).size().rename("verified_in_district")
    g = g.merge(n_in, on=["skey", "dkey"], how="left").fillna({"verified_in_district": 0})
    need, need_col = need_for(svc)
    if need is not None:
        g = g.merge(need, on=["skey", "dkey"], how="left")
    else:
        g["need"] = np.nan
    g["need_indicator"] = need_col or "none (no matching NFHS-5 indicator)"
    g["share_beyond"] = (g["people_beyond"] / g["people"]).round(3)
    g["share_verify"] = (g["people_verify"] / g["people"]).round(3)
    g["share_nothing"] = (g["people_nothing"] / g["people"]).round(3)
    # priority for NEW CAPACITY = people with nothing even claimed within reach x local need
    g["priority"] = g["people_nothing"] * (1 + g["need"].fillna(0.5))
    # priority for VERIFICATION = people whose only nearby options are unverified claims
    g["verify_priority"] = g["people_verify"] * (1 + g["need"].fillna(0.5))
    g["service"], g["reach_km"] = svc, reach
    dist_rows.append(g)

    tot = p["people"].sum()
    summ.append({"service": svc, "reach_km": reach, "verified_facilities": len(strong),
                 "claiming_facilities": len(anyc),
                 "people_share_within_reach": float(1 - p["people_beyond"].sum() / tot),
                 "people_share_verification_gap": float(p["people_verify"].sum() / tot),
                 "people_share_possible_care_gap": float(p["people_nothing"].sum() / tot),
                 "people_beyond_reach": float(p["people_beyond"].sum()),
                 "people_verification_gap": float(p["people_verify"].sum()),
                 "people_possible_care_gap": float(p["people_nothing"].sum()),
                 "pin_areas_beyond": int((~p["within_reach"]).sum()),
                 "districts_with_no_verified_within_reach": int((g["share_beyond"] >= 0.999).sum()),
                 "districts_with_nothing_claimed_within_reach": int((g["share_nothing"] >= 0.999).sum()),
                 "districts_total": int(len(g))})
    s_ = summ[-1]
    print(f"  {svc:32s} verified {len(strong):4d} / claiming {len(anyc):5d} | within {reach} km: "
          f"verified {s_['people_share_within_reach']:.0%}, only unverified claims "
          f"{s_['people_share_verification_gap']:.0%}, nothing claimed {s_['people_share_possible_care_gap']:.0%}")

pin_df = pd.concat(pin_rows, ignore_index=True)
dist_df = pd.concat(dist_rows, ignore_index=True)
dist_df["priority_rank"] = dist_df.groupby("service")["priority"].rank(ascending=False, method="first").astype(int)
summary = pd.DataFrame(summ)

# =====================================================================
# 3. POLICY RECOMMENDATIONS (numbers filled from the analysis above)
# =====================================================================
def fmt_people(x):
    return f"{x / 1e7:.1f} crore" if x >= 1e7 else (f"{x / 1e5:.1f} lakh" if x >= 1e5 else f"{x:,.0f}")

rep = spark.table(f"{OUT}.cleaning_report").toPandas()
def rep_n(pattern):
    m = rep[rep["check"].str.contains(pattern, regex=True)]
    return int(m["count"].iloc[0]) if len(m) else None

fac = spark.table(f"{OUT}.facilities_clean").select("beds", "doctors", "n_sources").toPandas()
no_beds = pd.to_numeric(fac["beds"], errors="coerce").isna().mean()
single_src = (pd.to_numeric(fac["n_sources"], errors="coerce") <= 1).mean()

policies = []
HIGH_STAKES = ["Emergency C-section", "NICU (newborn intensive care)", "Trauma / Accident care", "Dialysis"]

def top_districts(svc, n=5, by="priority"):
    t = dist_df[(dist_df["service"] == svc) & (dist_df[by] > 0)].sort_values(by, ascending=False).head(n)
    return ", ".join(f"{r.pin_district} ({r.pin_state})" for r in t.itertuples()), t

def add(theme, title, detail, evidence):
    policies.append({"rank": len(policies) + 1, "theme": theme, "title": title, "detail": detail, "evidence": evidence})

# 1. VERIFY FIRST - the cheapest, fastest win: claims exist nearby but evidence is weak
hs = summary[summary["service"].isin(HIGH_STAKES)].sort_values("people_verification_gap", ascending=False)
r0 = hs.iloc[0]
names, _ = top_districts(r0.service, by="verify_priority")
add("Verify first",
    "Verify existing claims before building anything: start with high-stakes services",
    f"For {r0.service.lower()}, {fmt_people(r0.people_verification_gap)} people "
    f"({r0.people_share_verification_gap:.0%}) have a facility within {r0.reach_km} km that CLAIMS the service but "
    f"without strong evidence ({int(r0.claiming_facilities):,} facilities claim it; only {int(r0.verified_facilities):,} "
    f"show strong evidence). A phone call or a visit by a district health officer can confirm these quickly. "
    f"Start in: {names}.",
    f"care_gaps_district.verify_priority, service = {r0.service}")

# 2. ADD CAPACITY only where NOTHING within reach even claims the service
worst = summary.sort_values("people_possible_care_gap", ascending=False)
for r in worst.head(2).itertuples():
    names, t = top_districts(r.service)
    add("Where to add capacity",
        f"Add {r.service.lower()} capacity where nothing is available within {r.reach_km} km",
        f"{fmt_people(r.people_possible_care_gap)} people ({r.people_share_possible_care_gap:.0%}) live more than "
        f"{r.reach_km} km from ANY facility that even claims {r.service.lower()}; "
        f"{r.districts_with_nothing_claimed_within_reach} districts have nothing claimed within reach. "
        f"Highest priority (people with nothing in reach x local need): {names}.",
        f"care_gaps_district.priority, service = {r.service}")

# 3. DIALYSIS programme
dia = summary[summary["service"] == "Dialysis"]
if len(dia):
    d = dia.iloc[0]
    names, _ = top_districts("Dialysis", 8)
    add("Dialysis",
        "Direct the national dialysis programme to the districts with nothing in reach",
        f"Dialysis is needed 2-3 times a week, so distance matters. {d.people_share_within_reach:.0%} of people live "
        f"within {d.reach_km} km of verified dialysis, {d.people_share_verification_gap:.0%} only of unverified claims, "
        f"and {d.people_share_possible_care_gap:.0%} ({fmt_people(d.people_possible_care_gap)}) of nothing at all. "
        f"Prioritise district-hospital dialysis units in: {names}.",
        "care_gaps_district, service = Dialysis")

# 4. EMERGENCY REFERRAL corridors where even claims are far away
em = dist_df[dist_df["service"].isin(["Emergency C-section", "Trauma / Accident care"])]
far = em[em["nearest_claim_km"] > 100]
add("Emergency referral",
    "Create referral corridors where emergency care is more than 100 km away",
    f"In {far['pin_district'].nunique()} districts, the nearest facility that even claims emergency C-section or "
    f"trauma care is more than 100 km away for a typical resident. Until capacity is added, link these districts to "
    f"named receiving hospitals with pre-arranged ambulance (108) transfer.",
    "care_gaps_district.nearest_claim_km > 100")

# 5. DATA QUALITY / registry
add("Data quality",
    "Make facilities report beds, services and verified location to the national facility registry",
    f"{no_beds:.0%} of facilities in this dataset do not state their bed count; "
    f"{rep_n('Location confidence: low') or 'hundreds of'} have contradictory addresses or coordinates; "
    f"{rep_n('Merge risk: high') or 'many'} records look like several facilities merged together. "
    f"Structured, mandatory registration (beds, services, geo-location) in the Health Facility Registry would turn "
    f"verification gaps into known coverage - or confirm real gaps.",
    "cleaning_report")

# 6. CLAIM QUALITY
weak_share = float((ev["confidence"] != "HIGH").mean())
add("Verification",
    "Require evidence behind service claims on public listings",
    f"{weak_share:.0%} of all facility-service claims in this dataset are not backed by strong evidence (they appear "
    f"in only one field, rest on a related specialty, or come from records with location or merge problems). "
    f"Listings used for referrals should show when and how a service was last confirmed.",
    "facility_service_evidence.confidence")
pol_df = pd.DataFrame(policies)

# =====================================================================
# 4. SAVE + REPORT
# =====================================================================
def save(df, name):
    spark.createDataFrame(df.reset_index(drop=True)).write.mode("overwrite") \
        .option("overwriteSchema", "true").saveAsTable(f"{OUT}.{name}")
    print(f"Saved {OUT}.{name} ({len(df):,} rows)")

save(pin_df.drop(columns=["skey", "dkey"]), "care_gaps_pin")
save(dist_df.drop(columns=["skey", "dkey"]), "care_gaps_district")
save(summary, "care_gaps_summary")
save(pol_df, "care_gap_policies")

print("\n" + "=" * 90 + "\nNATIONAL COVERAGE: share of people whose nearest option within reach is ...")
print(f"  {'service':34s} {'reach':>6s} {'verified':>9s} {'only unverified claims':>23s} {'nothing claimed':>16s}")
for r in summary.sort_values("people_share_within_reach").itertuples():
    print(f"  {r.service:34s} {r.reach_km:4d}km {r.people_share_within_reach:9.0%} "
          f"{r.people_share_verification_gap:23.0%} {r.people_share_possible_care_gap:16.0%}")
print("\n" + "=" * 90 + "\nPOLICY RECOMMENDATIONS")
for p in policies:
    print(f"\n{p['rank']}. [{p['theme']}] {p['title']}\n   {p['detail']}")
print("\nTop 10 priority districts for dialysis:")
display(spark.createDataFrame(dist_df[dist_df["service"] == "Dialysis"].sort_values("priority", ascending=False)
        .head(10)[["pin_district", "pin_state", "people_nothing", "share_nothing", "share_verify",
                   "nearest_claim_km", "centre_nearest_facility", "need"]].astype(str)))
