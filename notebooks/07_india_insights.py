# Databricks notebook source
# =====================================================================
#  NOTEBOOK 7 - INDIA INSIGHTS
#
#   1. The C-section paradox       too few (<5%) vs too many (>30%) caesareans, against verified access (NFHS-5)
#   2. North-South divide          EAG states vs southern states: facilities and verified access per person
#   3. Metro concentration         share of advanced care in 8 metros vs their share of people
#   4. Insured but nowhere to go   high health-insurance districts with nothing in reach (NFHS-5)
#   5. Diabetes double burden      high blood sugar + no dialysis in reach (NFHS-5)
#   6. Aspirational districts      do our priority districts match NITI Aayog's list?
#   7. Public care is invisible    private vs public in web-sourced data; PHCs/CHCs vs official counts
#   8. State health scorecard      verified access to 6 life-critical services + data quality, by state
#   9. The emergency distance gap  typical distance to verified emergency C-section / trauma care, by region
#
#  Input : care_gaps_pin, care_gaps_district (NB6), facility_service_evidence (NB2), facilities_clean (NB1),
#          NFHS-5 (organisers), Wikipedia (aspirational districts list; needs lxml, optional)
#  Output: india_insights, india_state_scorecard, india_region_coverage, india_csection_districts,
#          india_insight_points (the numbers behind every chart, for the VeriCare app),
#          facility_contacts (email + websites from the original dataset, for the app)
#          + PNG charts in /Volumes/workspace/default/exports/charts (for the video)
# =====================================================================
import io, json, os, re, shutil, warnings
import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

OUT = "workspace.default"
SRCDB = "databricks_virtue_foundation_dataset_dais_2026.virtue_foundation_dataset"
CHARTS = "/Volumes/workspace/default/exports/charts"
spark.sql("CREATE VOLUME IF NOT EXISTS workspace.default.exports")
os.makedirs(CHARTS, exist_ok=True)

EAG = ["Bihar", "Uttar Pradesh", "Madhya Pradesh", "Rajasthan", "Jharkhand", "Odisha", "Chhattisgarh", "Uttarakhand"]
SOUTH = ["Kerala", "Tamil Nadu", "Karnataka", "Andhra Pradesh", "Telangana"]
LIFE_CRITICAL = ["Emergency C-section", "NICU (newborn intensive care)", "Trauma / Accident care",
                 "ICU (intensive care)", "Dialysis", "Blood bank"]
ADVANCED = ["Cardiac surgery", "Angioplasty / Cath lab", "MRI scan", "Cancer care (chemo / radiation)"]
METROS = {"Delhi": (28.614, 77.209), "Mumbai": (19.076, 72.878), "Kolkata": (22.573, 88.364),
          "Chennai": (13.083, 80.271), "Bengaluru": (12.972, 77.595), "Hyderabad": (17.385, 78.487),
          "Pune": (18.520, 73.857), "Ahmedabad": (23.023, 72.571)}

def norm(s):
    return re.sub(r"[^a-z]", "", str(s or "").lower().replace("&", "and"))

def region(state):
    return "EAG states" if state in EAG else ("South" if state in SOUTH else "Rest of India")

def crore(x):
    return f"{x / 1e7:.1f} crore" if x >= 1e7 else f"{x / 1e5:.1f} lakh"

def save_chart(fig, name):
    tmp = f"/tmp/{name}.png"
    fig.savefig(tmp, dpi=160, bbox_inches="tight")
    shutil.copy(tmp, f"{CHARTS}/{name}.png")
    plt.close(fig)

INSIGHTS = []
def add(key, headline, detail, ref=None):
    INSIGHTS.append({"n": len(INSIGHTS) + 1, "key": key, "headline": headline, "detail": detail,
                     "ref": float(ref) if ref is not None else float("nan")})
    print(f"\n{len(INSIGHTS)}. {headline}\n   {detail}")

# every number behind a chart, so the app can draw it and show the details when a bar/point is clicked
POINTS = []
def num(v):
    try:
        v = float(v)
        return v if v == v else None
    except (TypeError, ValueError):
        return None
def pt(key, label, value, value2=None, series="", info=None):
    POINTS.append({"key": key, "label": str(label), "series": str(series), "value": num(value),
                   "value2": num(value2), "info": json.dumps(info or {}, default=str)})
def pctf(x):
    return "" if x is None or x != x else f"{x:.0%}"

# contact details from the original dataset (the app shows email + website in every result)
try:
    spark.sql(f"""CREATE OR REPLACE TABLE {OUT}.facility_contacts AS
                  SELECT unique_id, email, officialWebsite AS official_website, websites
                  FROM {SRCDB}.facilities""")
    print("Saved", f"{OUT}.facility_contacts", spark.table(f"{OUT}.facility_contacts").count(), "rows")
except Exception as e:
    print("facility_contacts not created:", str(e)[:120])

# =====================================================================
# LOAD
# =====================================================================
pin = spark.table(f"{OUT}.care_gaps_pin").toPandas()
for c in ["people", "nearest_km", "nearest_claim_km", "pin_lat", "pin_lon", "reach_km"]:
    pin[c] = pd.to_numeric(pin[c], errors="coerce")
pin["within_reach"] = pin["within_reach"].astype(str).str.lower() == "true"
dist = spark.table(f"{OUT}.care_gaps_district").toPandas()
for c in ["people", "people_nothing", "people_verify", "share_beyond", "share_nothing", "share_verify", "priority",
          "need", "centre_nearest_km"]:
    dist[c] = pd.to_numeric(dist[c], errors="coerce")
dist["skey"], dist["dkey"] = dist["pin_state"].map(norm), dist["pin_district"].map(norm)
ev = spark.table(f"{OUT}.facility_service_evidence").select(
    "service", "unique_id", "name", "city", "state", "loc_lat", "loc_lon", "confidence").toPandas()
for c in ["loc_lat", "loc_lon"]:
    ev[c] = pd.to_numeric(ev[c], errors="coerce")
fac = spark.table(f"{OUT}.facilities_clean").select(
    "unique_id", "name", "facility_type", "operator_type", "state", "loc_confidence", "include_in_search").toPandas()
fac = fac[fac["include_in_search"].astype(str).str.lower() == "true"]
nf = spark.table(f"{SRCDB}.nfhs_5_district_health_indicators").toPandas()

pins = pin.drop_duplicates("pin")[["pin", "pin_state", "pin_district", "people", "pin_lat", "pin_lon"]]
pins["region"] = pins["pin_state"].map(region)
TOTAL = pins["people"].sum()
state_people = pins.groupby("pin_state")["people"].sum()
print(f"{len(pins):,} PIN areas, {crore(TOTAL)} people | {len(fac):,} facilities | {len(nf):,} NFHS districts")

# ---- NFHS helpers
ncols = {c: c.lower().replace("_", " ") for c in nf.columns}   # NFHS uses snake_case
def cols_with(*words, exclude=()):
    return [c for c, lc in ncols.items() if all(w in lc for w in words) and not any(x in lc for x in exclude)]
def pct(x):
    try:
        return float(str(x).strip().replace("(", "").replace(")", ""))
    except ValueError:
        return np.nan
dcol = cols_with("district", exclude=("code",))[0]
scol = cols_with("state", exclude=("code",))[0]
nf["skey"], nf["dkey"] = nf[scol].map(norm), nf[dcol].map(norm)
def nfhs_join(df, series_name, values):
    """attach an NFHS district value to df (by state+district, then by unique district name)"""
    t = pd.DataFrame({"skey": nf["skey"], "dkey": nf["dkey"], series_name: values})
    by_sd = t.dropna().drop_duplicates(["skey", "dkey"]).set_index(["skey", "dkey"])[series_name]
    by_d = t.dropna()[~t.dropna()["dkey"].duplicated(keep=False)].set_index("dkey")[series_name]
    v = pd.Series([by_sd.get((s, d), np.nan) for s, d in zip(df["skey"], df["dkey"])], index=df.index)
    return v.fillna(df["dkey"].map(by_d))

# people-weighted verified coverage of a service in a group of PIN areas
def coverage(rows):
    return float((rows["people"] * rows["within_reach"]).sum() / rows["people"].sum()) if len(rows) else np.nan

# =====================================================================
# 1. THE C-SECTION PARADOX
# =====================================================================
cs_cols = cols_with("csection", exclude=("private", "public"))
if cs_cols:
    nf["cs_rate"] = nf[cs_cols[0]].map(pct)
    priv = cols_with("csection", "private"); pub = cols_with("csection", "public")
    cs = dist[dist["service"] == "Emergency C-section"].copy()
    cs["cs_rate"] = nfhs_join(cs, "cs_rate", nf["cs_rate"])
    cs["verified_share"] = 1 - cs["share_beyond"]
    cs["band"] = pd.cut(cs["cs_rate"], [-1, 5, 15, 30, 101],
                        labels=["under 5% (unmet need)", "5-15% (near WHO range)", "15-30% (high)", "over 30% (very high)"])
    low = cs[(cs["cs_rate"] < 5) & (cs["verified_share"] < 0.5)]
    st_rate = cs.groupby("pin_state")["cs_rate"].mean().dropna().sort_values()
    high_states = ", ".join(f"{s} {v:.0f}%" for s, v in st_rate.tail(3)[::-1].items())
    low_states = ", ".join(f"{s} {v:.0f}%" for s, v in st_rate.head(3).items())
    priv_pub = ""
    if priv and pub:
        pv, pb = nf[priv[0]].map(pct).mean(), nf[pub[0]].map(pct).mean()
        priv_pub = f" In private hospitals the average district C-section rate is {pv:.0f}% vs {pb:.0f}% in public ones."
    add("csection_paradox", "The C-section paradox: India has too few AND too many caesareans",
        f"The WHO considers roughly 10-15% of births by caesarean healthy. Highest state averages: {high_states}; "
        f"lowest: {low_states}.{priv_pub} {len(low)} districts have a C-section rate below 5% (a sign women cannot "
        f"get one when needed) AND verified C-section care within 50 km for less than half their people: "
        f"{', '.join(low.sort_values('people', ascending=False)['pin_district'].head(6))}.")
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = cs["pin_state"].map(lambda s: "#d73027" if s in EAG else ("#1a9850" if s in SOUTH else "#999999"))
    ax.scatter(cs["cs_rate"], cs["verified_share"] * 100, s=12, c=colors, alpha=0.7)
    ax.axvspan(10, 15, color="#2c7fb8", alpha=0.12, label="WHO healthy range (10-15%)")
    ax.axvline(5, color="#d73027", ls="--", lw=1)
    ax.set_xlabel("District C-section rate, % of births (NFHS-5)")
    ax.set_ylabel("People with verified C-section care within 50 km (%)")
    ax.set_title("The C-section paradox (red = EAG states, green = South)")
    ax.legend(loc="lower right")
    save_chart(fig, "1_csection_paradox")
    cs_out = cs[["pin_state", "pin_district", "cs_rate", "band", "verified_share", "share_nothing", "people"]].copy()
    cs_out["band"] = cs_out["band"].astype(str)
else:
    cs_out = pd.DataFrame()
    print("C-section rate column not found in NFHS-5 - insight 1 skipped")

# =====================================================================
# 2. NORTH-SOUTH DIVIDE  +  9. EMERGENCY DISTANCE GAP
# =====================================================================
fac["region"] = fac["state"].map(region)
reg_rows = []
for reg in ["EAG states", "South", "Rest of India"]:
    pp = pins[pins["region"] == reg]
    row = {"region": reg, "people_share": pp["people"].sum() / TOTAL,
           "facilities_per_crore": (fac["region"] == reg).sum() / (pp["people"].sum() / 1e7)}
    for svc in LIFE_CRITICAL + ADVANCED:
        row[svc] = coverage(pin[(pin["service"] == svc) & pin["pin"].isin(pp["pin"])])
    for svc in ["Emergency C-section", "Trauma / Accident care"]:
        r = pin[(pin["service"] == svc) & pin["pin"].isin(pp["pin"])]
        row[f"median_km_{svc}"] = float(np.average(r["nearest_km"], weights=r["people"])) if len(r) else np.nan
        row[f"typical_km_{svc}"] = float(r["nearest_km"].median()) if len(r) else np.nan
    reg_rows.append(row)
region_cov = pd.DataFrame(reg_rows)
E, S = region_cov.set_index("region").loc["EAG states"], region_cov.set_index("region").loc["South"]
add("north_south", "The North-South divide is real, even in web data",
    f"The EAG states hold {E.people_share:.0%} of people but have {E.facilities_per_crore:.0f} facilities per crore "
    f"people in this data, vs {S.facilities_per_crore:.0f} in the South. Verified dialysis within 50 km: "
    f"{E['Dialysis']:.0%} of people in EAG states vs {S['Dialysis']:.0%} in the South; verified ICU: "
    f"{E['ICU (intensive care)']:.0%} vs {S['ICU (intensive care)']:.0%}; verified NICU: "
    f"{E['NICU (newborn intensive care)']:.0%} vs {S['NICU (newborn intensive care)']:.0%}.")
add("emergency_distance", "The emergency distance gap",
    f"For a typical person, the nearest facility with strong evidence of emergency C-section is "
    f"{E['typical_km_Emergency C-section']:.0f} km away in the EAG states vs {S['typical_km_Emergency C-section']:.0f} km "
    f"in the South; for trauma care, {E['typical_km_Trauma / Accident care']:.0f} km vs "
    f"{S['typical_km_Trauma / Accident care']:.0f} km. In an emergency, every extra kilometre is minutes lost.")
fig, ax = plt.subplots(figsize=(9, 4.5))
x = np.arange(len(LIFE_CRITICAL)); w = 0.27
for i, reg in enumerate(["EAG states", "Rest of India", "South"]):
    vals = region_cov.set_index("region").loc[reg, LIFE_CRITICAL].astype(float).values * 100
    ax.bar(x + (i - 1) * w, vals, w, label=reg, color=["#d73027", "#999999", "#1a9850"][i])
ax.set_xticks(x); ax.set_xticklabels([s.split(" (")[0] for s in LIFE_CRITICAL], rotation=20, ha="right")
ax.set_ylabel("People with VERIFIED care within 50 km (%)")
ax.set_title("Verified access to life-critical care: EAG states vs South")
ax.legend()
save_chart(fig, "2_north_south")

# =====================================================================
# 3. METRO CONCENTRATION
# =====================================================================
def near_metro(lat, lon, km=30):
    lat, lon = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    out = np.zeros(len(lat), dtype=bool)
    for mlat, mlon in METROS.values():
        a = np.sin((np.radians(mlat) - lat) / 2) ** 2 + np.cos(lat) * np.cos(np.radians(mlat)) * \
            np.sin((np.radians(mlon) - lon) / 2) ** 2
        out |= 6371 * 2 * np.arcsin(np.sqrt(a)) <= km
    return out
metro_people = pins.loc[near_metro(pins["pin_lat"], pins["pin_lon"]), "people"].sum() / TOTAL
def which_metro(lat, lon, km=30):
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    out = np.array([""] * len(lat), dtype=object)
    for name, (mlat, mlon) in METROS.items():
        a = np.sin((np.radians(mlat) - np.radians(lat)) / 2) ** 2 + np.cos(np.radians(lat)) * np.cos(np.radians(mlat)) \
            * np.sin((np.radians(mlon) - np.radians(lon)) / 2) ** 2
        out[(out == "") & (6371 * 2 * np.arcsin(np.sqrt(a)) <= km)] = name
    return out
met = []
for svc in ADVANCED + ["Dialysis", "ICU (intensive care)"]:
    h = ev[(ev["service"] == svc) & (ev["confidence"] == "HIGH")].dropna(subset=["loc_lat"])
    share = float(near_metro(h["loc_lat"], h["loc_lon"]).mean()) if len(h) else np.nan
    met.append({"service": svc, "metro_share": share})
    m_of = pd.Series(which_metro(h["loc_lat"].values, h["loc_lon"].values))
    by = m_of[m_of != ""].value_counts()
    pt("metro_concentration", svc, share * 100, len(h), info={
        "Service": svc, "Facilities with strong evidence (India)": f"{len(h):,}",
        "Of these, in the 8 metros": f"{int((m_of != '').sum()):,} ({pctf(share)})",
        "Metros' share of people": pctf(metro_people),
        "_table": [{"Metro": k, "Verified facilities": int(v)} for k, v in by.items()]})
met = pd.DataFrame(met)
cardiac = met.set_index("service").loc["Cardiac surgery", "metro_share"]
add("metro_concentration", "Advanced care is concentrated in 8 metros",
    f"The 8 largest metros (within 30 km of Delhi, Mumbai, Kolkata, Chennai, Bengaluru, Hyderabad, Pune, Ahmedabad) "
    f"hold about {metro_people:.0%} of people but {cardiac:.0%} of facilities with strong evidence of cardiac surgery"
    + "".join(f", {r.metro_share:.0%} of {r.service.split(' (')[0].lower()}" for r in met.itertuples()
              if r.service in ("MRI scan", "Cancer care (chemo / radiation)", "Angioplasty / Cath lab"))
    + ". Patients from smaller cities and villages travel for specialist care.", ref=metro_people * 100)
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh([s.split(" (")[0] for s in met["service"]], met["metro_share"] * 100, color="#4c78a8")
ax.axvline(metro_people * 100, color="#d73027", ls="--", label=f"metros' share of people ({metro_people:.0%})")
ax.set_xlabel("% of verified facilities located in the 8 metros")
ax.set_title("Specialist care clusters in metros")
ax.legend()
save_chart(fig, "3_metro_concentration")

# =====================================================================
# 4. INSURED BUT NOWHERE TO GO   +   5. DIABETES DOUBLE BURDEN
# =====================================================================
ins_cols = cols_with("health insurance")
if ins_cols:
    nf["insurance"] = nf[ins_cols[0]].map(pct)
    icu = dist[dist["service"] == "ICU (intensive care)"].copy()
    icu["insurance"] = nfhs_join(icu, "insurance", nf["insurance"])
    q = icu["insurance"].quantile(0.75)
    hit = icu[(icu["insurance"] >= q) & (icu["share_nothing"] >= 0.5)].sort_values("people", ascending=False)
    for r in hit.itertuples():
        pt("insured_nowhere", f"{r.pin_district}, {r.pin_state}", r.people_nothing, r.insurance, r.pin_state, {
            "District": r.pin_district, "State": r.pin_state,
            "Households with health insurance (NFHS-5)": f"{r.insurance:.0f}%",
            "People with no ICU even claimed within 50 km": crore(r.people_nothing),
            "Share of district": pctf(r.share_nothing), "District population": crore(r.people)})
    add("insured_nowhere", "Insured, but nowhere to go",
        f"In the quarter of districts with the highest health-insurance coverage (at least {q:.0f}% of households), "
        f"{len(hit)} still have no facility even claiming ICU care within 50 km for most of their people - a card "
        f"without a hospital: {', '.join(hit['pin_district'].head(6))}.", ref=q)
sugar_cols = cols_with("blood sugar", "high or very high")
if sugar_cols:
    nf["sugar"] = pd.concat([nf[c].map(pct) for c in sugar_cols], axis=1).mean(axis=1)
    dia = dist[dist["service"] == "Dialysis"].copy()
    dia["sugar"] = nfhs_join(dia, "sugar", nf["sugar"])
    q = dia["sugar"].quantile(0.75)
    db = dia[(dia["sugar"] >= q) & (dia["share_nothing"] >= 0.5)].sort_values("people", ascending=False)
    for r in db.itertuples():
        pt("diabetes_double_burden", f"{r.pin_district}, {r.pin_state}", r.people_nothing, r.sugar, r.pin_state, {
            "District": r.pin_district, "State": r.pin_state,
            "Adults with high blood sugar (NFHS-5)": f"{r.sugar:.1f}%",
            "People with no dialysis even claimed within 50 km": crore(r.people_nothing),
            "Share of district": pctf(r.share_nothing), "District population": crore(r.people)})
    add("diabetes_double_burden", "The diabetes double burden: tomorrow's kidney-failure hotspots",
        f"Diabetes is the leading cause of kidney failure. In {len(db)} districts, high blood sugar is in India's top "
        f"quarter (at least {q:.0f}% of adults, NFHS-5) AND most people have no dialysis even claimed within 50 km "
        f"({crore(db['people_nothing'].sum())} people): {', '.join(db['pin_district'].head(6))}.", ref=q)

# =====================================================================
# 6. ASPIRATIONAL DISTRICTS (NITI Aayog)
# =====================================================================
asp = set()
for title in ["Aspirational_Districts_Programme", "Transformation_of_Aspirational_Districts_programme",
              "Aspirational_Districts_Program"]:
    try:
        r = requests.get(f"https://en.wikipedia.org/wiki/{title}", timeout=30,
                         headers={"User-Agent": "CareFinderResearchBot/1.0 (student datathon project)"})
        if r.status_code != 200:
            continue
        for t in pd.read_html(io.StringIO(r.text)):
            cols = [" ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in t.columns]
            for i, c in enumerate(cols):
                if "district" in c.lower():
                    for cell in map(str, t.iloc[:, i].values.ravel().tolist()):
                        # a cell may hold one district or a whole list: "Araria, Aurangabad, Banka ..."
                        for part in re.split(r"[,;\n•]|\band\b|\d+\.", cell):
                            k = norm(re.sub(r"\[.*?\]|\(.*?\)", "", part))
                            if 2 < len(k) < 30 and k not in ("nan", "district", "districts"):
                                asp.add(k)
        if len(asp) > 50:
            break
    except Exception as e:
        print(f"  aspirational list ({title}) not loaded: {str(e)[:80]}")
if len(asp) > 50:
    hs = dist[dist["service"].isin(LIFE_CRITICAL)].groupby(["pin_state", "pin_district", "dkey"])["priority"] \
        .sum().reset_index().sort_values("priority", ascending=False)
    top = hs.head(50)
    overlap = top["dkey"].isin(asp).mean()
    base = hs["dkey"].isin(asp).mean()
    lc = dist[dist["service"].isin(LIFE_CRITICAL)]
    for r in top.itertuples():
        rows_d = lc[(lc["pin_state"] == r.pin_state) & (lc["pin_district"] == r.pin_district)]
        lacking = rows_d[rows_d["share_nothing"] >= 0.5]["service"].tolist()
        on = r.dkey in asp
        pt("aspirational", f"{r.pin_district}, {r.pin_state}", r.priority, None,
           "NITI Aayog Aspirational District" if on else "Not on the list", {
               "District": r.pin_district, "State": r.pin_state, "Aspirational District": "Yes" if on else "No",
               "Life-critical services with nothing claimed within 50 km": ", ".join(lacking) or "none",
               "_table": [{"Service": x.service, "Verified within 50 km": pctf(1 - x.share_beyond),
                           "Nothing claimed": pctf(x.share_nothing)} for x in rows_d.itertuples()]})
    add("aspirational", "Our priority districts independently match the government's own list",
        f"{overlap:.0%} of our 50 highest-priority districts for life-critical care are NITI Aayog Aspirational "
        f"Districts, vs {base:.0%} of all districts ({overlap / base if base else float('nan'):.1f}x more than chance). "
        f"Our data-driven ranking agrees with the government's targeting, and adds which service each one lacks: "
        f"{', '.join(top[top['dkey'].isin(asp)]['pin_district'].head(6))}.", ref=base * 100)
else:
    print("Aspirational districts list unavailable - insight 6 skipped (run  %pip install lxml  and re-run)")

# =====================================================================
# 7. PUBLIC CARE IS INVISIBLE ONLINE
# =====================================================================
ops = fac["operator_type"].value_counts()
priv_share = ops.get("private", 0) / max(ops.get("private", 0) + ops.get("public", 0), 1)
names = fac["name"].fillna("").str.lower()
n_phc = names.str.contains(r"primary health cent(?:re|er)|\bphc\b").sum()
n_chc = names.str.contains(r"community health cent(?:re|er)|\bchc\b").sum()
for k in ["private", "public"]:
    pt("public_invisible", k.title(), ops.get(k, 0), None, "owner",
       {"Owner": k.title(), "Facilities in this dataset": f"{ops.get(k, 0):,}",
        "Share of facilities with a known owner": pctf(ops.get(k, 0) / max(ops.get("private", 0) + ops.get("public", 0), 1))})
for label, n_in, official in [("Primary Health Centres", n_phc, 25000), ("Community Health Centres", n_chc, 5500)]:
    pt("public_invisible", label, n_in, official, "rural", {
        "Type": label, "Found in this dataset": f"{n_in:,}", "Run by the government (approx.)": f"{official:,}",
        "Share visible online": f"{n_in / official:.2%}", "Source for official count": "Rural Health Statistics"})
add("public_invisible", "Public healthcare is almost invisible in web-sourced data",
    f"{priv_share:.0%} of facilities with a known owner in this dataset are private. Only {n_phc} Primary Health "
    f"Centres and {n_chc} Community Health Centres appear, while India runs roughly 25,000 PHCs and 5,500 CHCs "
    f"(Rural Health Statistics). Web data systematically misses the rural public system, so any tool built on it "
    f"under-counts rural care - another reason for a complete government facility registry.")

# =====================================================================
# 8. STATE HEALTH SCORECARD
# =====================================================================
rows = []
for stt, pp in pins.groupby("pin_state"):
    if pp["people"].sum() < 1e6:
        continue
    r = {"state": stt, "region": region(stt), "people": pp["people"].sum(),
         "facilities_per_crore": (fac["state"] == stt).sum() / (pp["people"].sum() / 1e7)}
    for svc in LIFE_CRITICAL:
        r[svc] = coverage(pin[(pin["service"] == svc) & pin["pin"].isin(pp["pin"])])
    fs = fac[fac["state"] == stt]
    r["data_quality"] = 1 - (fs["loc_confidence"] == "low").mean() if len(fs) else np.nan
    r["life_critical_access"] = np.nanmean([r[s] for s in LIFE_CRITICAL])
    rows.append(r)
score = pd.DataFrame(rows).sort_values("life_critical_access", ascending=False).reset_index(drop=True)
score["rank"] = np.arange(1, len(score) + 1)
big = score[score["people"] >= 2e7]                      # large states (2 crore+ people)
best = ", ".join(f"{r.state} ({r.life_critical_access:.0%})" for r in big.head(3).itertuples())
worst = ", ".join(f"{r.state} ({r.life_critical_access:.0%})" for r in big.tail(3)[::-1].itertuples())
delhi = score.set_index("state")["life_critical_access"].get("Delhi", float("nan"))
add("state_scorecard", "State health scorecard: verified access to six life-critical services",
    f"Average share of people with verified C-section, NICU, trauma, ICU, dialysis and blood-bank care within 50 km. "
    f"Among large states (2 crore+ people) - best: {best}; worst: {worst}. Among smaller states and union "
    f"territories, Delhi leads ({delhi:.0%}) while the north-east has the least verified care. Full ranking in the app.")
fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(score))))
sc = score.sort_values("life_critical_access")
ax.barh(sc["state"], sc["life_critical_access"] * 100,
        color=sc["region"].map({"EAG states": "#d73027", "South": "#1a9850", "Rest of India": "#999999"}))
ax.set_xlabel("Average % of people with verified life-critical care within 50 km")
ax.set_title("State health scorecard (red = EAG, green = South)")
save_chart(fig, "8_state_scorecard")

# =====================================================================
# SAVE
# =====================================================================
def safe(c):                 # Delta column names can't contain spaces / brackets / slashes
    return re.sub(r"[^a-z0-9]+", "_", str(c).lower()).strip("_")

def save(df, name):
    if len(df):
        spark.createDataFrame(df.reset_index(drop=True).rename(columns=safe)).write.mode("overwrite") \
            .option("overwriteSchema", "true").saveAsTable(f"{OUT}.{name}")
        print(f"Saved {OUT}.{name} ({len(df):,} rows)")

save(pd.DataFrame(INSIGHTS), "india_insights")
save(score, "india_state_scorecard")
save(region_cov, "india_region_coverage")
save(cs_out, "india_csection_districts")
save(pd.DataFrame(POINTS), "india_insight_points")
print(f"\nCharts for the video: Catalog -> workspace -> default -> Volumes -> exports -> charts")
display(spark.createDataFrame(score[["rank", "state", "region", "life_critical_access", "facilities_per_crore",
                                     "data_quality"]].astype(str)))
