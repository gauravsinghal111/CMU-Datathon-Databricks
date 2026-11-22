# Databricks notebook source
# =====================================================================
#  NOTEBOOK 2 - EVIDENCE + SEARCH
#
#  Input : workspace.default.facilities_clean, pin_lookup, place_lookup   (from Notebook 1)
#  Output: workspace.default.facility_service_evidence
#            one row per (facility, service) that has ANY evidence:
#            score, score breakdown, quotes, warnings, overall confidence,
#            plus the facility details the app needs (one table = fast app)
#          workspace.default.service_coverage
#            per service x state: how many facilities with HIGH / MEDIUM / LOW evidence
#  Also   : find_care(service, location) to test searches before building the app
# =====================================================================
import json, re, math
import numpy as np
import pandas as pd
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType, BooleanType

OUT = "workspace.default"
# Use the Google-verified table from Notebook 3 if it exists, otherwise the cleaned table from Notebook 1
SRC_TABLE = (f"{OUT}.facilities_verified" if spark.catalog.tableExists(f"{OUT}.facilities_verified")
             else f"{OUT}.facilities_clean")
fac = spark.table(SRC_TABLE).toPandas()
pin_lookup = spark.table(f"{OUT}.pin_lookup").toPandas()
place_lookup = spark.table(f"{OUT}.place_lookup").toPandas()
fac = fac[fac["include_in_search"].astype(str).str.lower() == "true"].reset_index(drop=True)

GOOGLE_COLS = ["google_match", "google_note", "g_name", "g_rating", "g_reviews_count", "g_maps_url", "g_status",
               "g_reviews", "checks", "aliases"]
for c in GOOGLE_COLS:
    if c not in fac.columns:
        fac[c] = None
LIST_COLS = ["specialties_list", "procedure_list", "equipment_list", "capability_list", "source_urls_list",
             "phones_all", "g_reviews", "checks", "aliases"]
for c in LIST_COLS:
    fac[c] = fac[c].map(lambda s: json.loads(s) if isinstance(s, str) and s.startswith("[") else [])
fac["description"] = fac["description"].fillna("")
for c in ["loc_lat", "loc_lon", "loc_precision_km", "doctors", "beds", "year_established", "g_rating", "g_reviews_count"]:
    fac[c] = pd.to_numeric(fac[c], errors="coerce")
print(f"{len(fac):,} facilities loaded from {SRC_TABLE}")

# Beds / doctors: listed, stated in text, or predicted (Notebook 4). Falls back to listed values only.
SIZE_COLS = ["beds_value", "beds_source", "beds_low", "beds_high", "beds_p_large", "beds_class", "beds_class_shown",
             "doctors_value", "doctors_source", "doctors_low", "doctors_high", "size_band"]
if spark.catalog.tableExists(f"{OUT}.facility_size"):
    size = spark.table(f"{OUT}.facility_size").toPandas().drop_duplicates("unique_id")
    fac = fac.merge(size, on="unique_id", how="left")
    print("  + bed/doctor estimates from facility_size")
for c in SIZE_COLS:
    if c not in fac.columns:
        fac[c] = None
fac["beds_value"] = pd.to_numeric(fac["beds_value"], errors="coerce").fillna(fac["beds"])
fac["doctors_value"] = pd.to_numeric(fac["doctors_value"], errors="coerce").fillna(fac["doctors"])
fac["beds_source"] = fac["beds_source"].fillna(pd.Series(np.where(fac["beds"].notna(), "listed", None), index=fac.index))
fac["doctors_source"] = fac["doctors_source"].fillna(pd.Series(np.where(fac["doctors"].notna(), "listed", None), index=fac.index))

# =====================================================================
# 1. SERVICES - what text counts as evidence
#    terms     : words that mean the service itself is offered
#    specialty : related specialty (weak evidence on its own)
#    support   : extra signal that matters for this service (e.g. 24/7)
# =====================================================================
EMERGENCY_247 = r"24\s*/\s*7|24 ?x ?7|24 ?hours?|24 ?hrs|round the clock|emergency"
SERVICES = {
    "Dialysis": {
        "terms": r"dialysis|ha?emodialysis|renal replacement|\bcrrt\b",
        "specialty": r"nephrology"},
    "Emergency C-section": {
        "terms": r"c[- ]?section|caesarean|cesarean|\blscs\b",
        "specialty": r"obstetric|gynecologyandobstetrics",
        "support": EMERGENCY_247},
    "NICU (newborn intensive care)": {
        "terms": r"\bnicu\b|neonatal (intensive|icu|critical)|newborn intensive|level i{1,3} nicu",
        "specialty": r"neonatology"},
    "ICU (intensive care)": {
        "terms": r"\bicu\b|intensive care|critical care unit|\bccu\b|\bmicu\b|\bsicu\b",
        "specialty": r"criticalcare"},
    "Cardiac surgery": {
        "terms": r"cardiac surgery|heart surgery|open[- ]heart|\bcabg\b|bypass surgery|cardiothoracic|valve replacement",
        "specialty": r"cardiothoracic|cardiacsurgery|cardiovascularsurgery"},
    "Angioplasty / Cath lab": {
        "terms": r"angioplasty|cath ?lab|catheteri[sz]ation lab|\bptca\b|coronary angiograph|stent",
        "specialty": r"interventionalcardiology|cardiology"},
    "MRI scan": {
        "terms": r"\bmri\b|magnetic resonance",
        "specialty": None},
    "CT scan": {
        "terms": r"\bct\b|\bct scan|computed tomography|\bcat scan",
        "specialty": None},
    "Burn care": {
        "terms": r"\bburns? (unit|ward|care|centre|center|icu|treatment|management|injur)|\bburns\b",
        "specialty": r"burn"},
    "Trauma / Accident care": {
        # "trauma" but not dental trauma ("traumatic teeth", "dental trauma")
        "terms": r"(?<!dental )trauma(?!tic (?:teeth|tooth))|polytrauma|accident (and|&) emergency|road traffic accident",
        "specialty": r"trauma|emergencymedicine",
        "support": EMERGENCY_247},
    "Blood bank": {
        "terms": r"blood bank|blood storage|blood transfusion",
        "specialty": None},
    "Cancer care (chemo / radiation)": {
        "terms": r"chemotherap|radiotherap|radiation therap|oncolog|linac|brachytherap",
        "specialty": r"oncology"},
}

# "google review" = a patient review on Google Maps mentions the service: independent of the web listing
WEIGHTS = {"procedure": 3, "equipment": 3, "capability": 2, "description": 2, "google review": 2,
           "specialty": 1, "support": 1}
NEG = re.compile(r"\b(no|not|without|unavailable|referred|refers?|lacks?)\b", re.I)
CHILD = re.compile(r"p(a)?ediatric|child|neonat|infant|newborn", re.I)
CHILD_SERVICES = {"NICU (newborn intensive care)"}
SHORTENERS = {"goo.gl", "maps.app.goo.gl", "bit.ly", "tinyurl.com", "wa.me", "g.page", "linktr.ee"}

def snippet(text, pat, width=200):
    m = pat.search(text)
    if not m or len(text) <= width:
        return text
    start = max(0, m.start() - width // 2)
    end = min(len(text), start + width)
    return ("…" if start > 0 else "") + text[start:end] + ("…" if end < len(text) else "")

def norm_key(t):
    return re.sub(r"[^a-z]", "", t.lower().replace("paediatric", "pediatric").replace("haemo", "hemo"))

def evidence_for(r, svc, pat):
    """Return quotes per field: [(field, text)], negated quotes, support quote."""
    quotes, negated, seen = [], [], set()
    sources = [("procedure", r["procedure_list"]), ("equipment", r["equipment_list"]),
               ("capability", r["capability_list"])]
    if str(r["flag_filler_description"]).lower() != "true":
        sources.append(("description", re.split(r"(?<=[.!?])\s+", r["description"])))
    reviews = [rv.get("text", "") for rv in r["g_reviews"] if isinstance(rv, dict)]
    if reviews:
        sources.append(("google review", [s for rv in reviews for s in re.split(r"(?<=[.!?])\s+", rv) if s]))
    for field, items in sources:
        for item in items:
            if pat.search(item):
                s = snippet(item, pat)
                k = norm_key(s)
                if k in seen:
                    continue
                seen.add(k)
                (negated if NEG.search(s) else quotes).append((field, s))
    spec = None
    if svc.get("specialty"):
        sp = re.compile(svc["specialty"], re.I)
        spec = next((x for x in r["specialties_list"] if sp.search(x)), None)
    return quotes, negated, spec

# =====================================================================
# 2. SCORE EVERY FACILITY x SERVICE
# =====================================================================
rows = []
records = fac.to_dict("records")          # faster than iterrows
for svc_name, svc in SERVICES.items():
    pat = re.compile(svc["terms"], re.I)
    sup = re.compile(svc["support"], re.I) if svc.get("support") else None
    for r in records:
        quotes, negated, spec = evidence_for(r, svc, pat)
        if not quotes and not spec and not negated:
            continue
        fields = sorted({f for f, _ in quotes}, key=lambda f: -WEIGHTS[f])
        breakdown = [(f, WEIGHTS[f]) for f in fields]
        if spec:
            breakdown.append(("specialty", WEIGHTS["specialty"]))
        support_quote = None
        if sup is not None and fields:
            support_quote = next((snippet(x, sup) for x in r["capability_list"] + [r["description"]] if sup.search(x)), None)
            if support_quote:
                breakdown.append(("support", WEIGHTS["support"]))
        score = sum(w for _, w in breakdown)
        if score == 0:
            continue

        # ---- evidence level: how directly the record says it offers the service
        n_direct = len(fields)
        evidence_level = "HIGH" if n_direct >= 2 else ("MEDIUM" if n_direct == 1 else "LOW")

        # ---- overall confidence = evidence level, lowered by trust problems
        level = {"HIGH": 2, "MEDIUM": 1, "LOW": 0}[evidence_level]
        warnings, lowered_by = [], []
        if n_direct == 0:
            warnings.append(f"Only the related specialty '{spec}' is listed; {svc_name.lower()} itself is never mentioned")
        if negated:
            level -= 1; lowered_by.append("possible 'not offered' wording")
            warnings.append(f'Some text may say it is NOT offered: "{negated[0][1][:140]}"')
        if r["loc_confidence"] == "low":
            level -= 1; lowered_by.append("uncertain location")
        if r["loc_confidence"] != "high":
            warnings.append(r["loc_note"])
        if r["merge_risk"] == "high":
            level -= 1; lowered_by.append("record may mix several facilities")
            warnings.append("Record may mix up several facilities: " + str(r["merge_notes"]))
        elif r["merge_risk"] == "possible":
            warnings.append("Note: " + str(r["merge_notes"]))
        children_only = bool(quotes) and svc_name not in CHILD_SERVICES and all(CHILD.search(t) for _, t in quotes)
        if children_only:
            warnings.append("Evidence only mentions services for children; confirm adults are treated")
        if r["n_sources"] <= 1:
            warnings.append("Based on a single web source")
        if r["phone_flag"]:
            warnings.append("Phone: " + r["phone_flag"])
        if r["size_note"]:
            warnings.append("Size data: " + r["size_note"])
        if r["google_note"]:
            warnings.append(r["google_note"])
        confidence = ["LOW", "MEDIUM", "HIGH"][max(level, 0)]

        rows.append({
            "service": svc_name, "unique_id": r["unique_id"],
            "evidence_score": score,
            "score_breakdown": " + ".join(f"{f} {w}" for f, w in breakdown) + f" = {score}",
            "evidence_level": evidence_level, "confidence": confidence,
            "confidence_lowered_by": "; ".join(lowered_by) or None,
            "n_direct_fields": n_direct, "children_only": children_only,
            "quotes": quotes, "negated_quotes": negated, "specialty_match": spec,
            "support_quote": support_quote, "warnings": warnings,
            # facility details for the app
            "name": r["name"], "facility_type": r["facility_type"], "operator_type": r["operator_type"],
            "address": ", ".join(x for x in [r["address_line1"], r["address_line2"], r["address_line3"]] if isinstance(x, str)),
            "city": r["city"], "district": r["district"], "state": r["state"], "pin": r["pin"],
            "loc_lat": r["loc_lat"], "loc_lon": r["loc_lon"], "loc_confidence": r["loc_confidence"],
            "loc_precision_km": r["loc_precision_km"], "loc_note": r["loc_note"],
            "phone": r["phone_best"], "phone_flag": r["phone_flag"],
            "website": None if r["website"] in SHORTENERS else r["website"],
            "doctors": r["doctors"], "beds": r["beds"], "year_established": r["year_established"],
            "n_sources": r["n_sources"], "source_urls": r["source_urls_list"][:5],
            "merge_risk": r["merge_risk"],
            # Google Maps checks (Notebook 3); empty if not run
            "google_match": r["google_match"], "g_name": r["g_name"], "g_rating": r["g_rating"],
            "g_reviews_count": r["g_reviews_count"], "g_maps_url": r["g_maps_url"], "g_status": r["g_status"],
            "checks": r["checks"], "aliases": r["aliases"],
            # size: value + where it came from (listed / stated in text / predicted) + likely range
            **{c: r[c] for c in SIZE_COLS},
        })
    print(f"  {svc_name:34s} {sum(1 for x in rows if x['service'] == svc_name):5d} facilities with some evidence")

ev = pd.DataFrame(rows)

# =====================================================================
# 3. COVERAGE TABLE (for "evidence is thin here" messages and a gap map)
# =====================================================================
cov = (ev.pivot_table(index=["service", "state"], columns="confidence", values="unique_id",
                      aggfunc="count", fill_value=0).reset_index())
for c in ["HIGH", "MEDIUM", "LOW"]:
    if c not in cov.columns:
        cov[c] = 0
cov = cov.rename(columns={"HIGH": "n_high", "MEDIUM": "n_medium", "LOW": "n_low"})[
    ["service", "state", "n_high", "n_medium", "n_low"]]
cov.columns.name = None

# =====================================================================
# 4. SAVE
# =====================================================================
def to_spark(df):
    out, fields = df.copy(), []
    for c in out.columns:
        dt = out[c].dtype
        if pd.api.types.is_bool_dtype(dt):
            out[c] = out[c].map(bool); fields.append(StructField(c, BooleanType()))
        elif pd.api.types.is_integer_dtype(dt):
            out[c] = out[c].map(int); fields.append(StructField(c, LongType()))
        elif pd.api.types.is_float_dtype(dt):
            out[c] = out[c].map(lambda v: None if pd.isna(v) else float(v)); fields.append(StructField(c, DoubleType()))
        else:
            out[c] = out[c].map(lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, tuple))
                                else (None if v is None or (isinstance(v, float) and math.isnan(v)) else str(v)))
            fields.append(StructField(c, StringType()))
    return spark.createDataFrame(out.astype(object).values.tolist(), StructType(fields))

def save(df, name):
    to_spark(df).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{OUT}.{name}")
    print(f"Saved table {OUT}.{name}  ({len(df):,} rows)")

save(ev, "facility_service_evidence")
save(cov, "service_coverage")

print("\nEVIDENCE SUMMARY (facilities per service, by overall confidence)")
summary = ev.pivot_table(index="service", columns="confidence", values="unique_id", aggfunc="count", fill_value=0)
summary = summary.reindex(columns=[c for c in ["HIGH", "MEDIUM", "LOW"] if c in summary.columns])
summary["total"] = summary.sum(axis=1)
print(summary.sort_values("total", ascending=False).to_string())

# =====================================================================
# 5. SEARCH (same logic the app will use)
# =====================================================================
CITY_ALIASES = {
    "bangalore": "bengaluru", "bengaluru urban": "bengaluru", "bangalore urban": "bengaluru", "bombay": "mumbai",
    "calcutta": "kolkata", "madras": "chennai", "gurgaon": "gurugram", "trivandrum": "thiruvananthapuram",
    "cochin": "kochi", "poona": "pune", "mysore": "mysuru", "mangalore": "mangaluru", "belgaum": "belagavi",
    "hubli": "hubballi", "baroda": "vadodara", "allahabad": "prayagraj", "benares": "varanasi",
    "banaras": "varanasi", "pondicherry": "puducherry", "new delhi": "delhi", "calicut": "kozhikode",
    "trichy": "tiruchirappalli", "tiruchirapalli": "tiruchirappalli", "vizag": "visakhapatnam",
    "shimoga": "shivamogga", "gulbarga": "kalaburagi", "bellary": "ballari", "tumkur": "tumakuru",
    # renamed cities: the India Post directory often still uses the OLD name
    "chhatrapati sambhajinagar": "aurangabad", "chh sambhajinagar": "aurangabad", "sambhajinagar": "aurangabad",
    "dharashiv": "osmanabad", "ahilyanagar": "ahmednagar", "ahilya nagar": "ahmednagar", "ahmadnagar": "ahmednagar",
    "narmadapuram": "hoshangabad", "faizabad": "ayodhya", "kanpur nagar": "kanpur", "mumbai suburban": "mumbai",
}

def ckey(s):
    s = str(s or "").lower().replace("&", " and ")
    s = re.sub(r"\b(city|district|dist|urban|rural|municipal corporation)\b", " ", s)
    s = re.sub(r"[^a-z ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return CITY_ALIASES.get(s, s) or None

def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))

KIND_PRIO = {"facility city": 0, "district": 1, "post office": 2}

def parse_location(q):
    """'Jaipur' / 'Jaipur, Rajasthan' / '302001' / '26.9, 75.8'  ->  (lat, lon, description, precision_km)"""
    q = str(q).strip()
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)", q)
    if m:
        return float(m.group(1)), float(m.group(2)), f"coordinates {q}", 0
    if re.fullmatch(r"\d{6}", q):
        row = pin_lookup[pin_lookup["pin"] == q]
        if len(row):
            r = row.iloc[0]
            return r["pin_lat"], r["pin_lon"], f"PIN {q} ({r['pin_district']}, {r['pin_state']})", r["pin_radius_km"]
        return None
    parts = [x.strip() for x in q.split(",")]
    place, state = ckey(parts[0]), (parts[1].lower() if len(parts) > 1 else None)
    hit = place_lookup[place_lookup["place"] == place].copy()
    if state:
        hit = hit[hit["state"].str.lower().str.contains(state, na=False)]
    if not len(hit):
        return None
    hit["prio"] = hit["kind"].map(KIND_PRIO)
    r = hit.sort_values(["prio", "n"], ascending=[True, False]).iloc[0]
    others = hit["state"].nunique() - 1
    desc = f"{parts[0].title()}, {r['state']} ({r['kind']} centre)"
    if others > 0:
        desc += f" - {others} other state(s) also have a place with this name; add the state to be sure"
    return r["lat"], r["lon"], desc, r["radius_km"]

CONF_MULT = {"HIGH": 1.0, "MEDIUM": 0.85, "LOW": 0.6}

# Ranking: service evidence first, then size (beds), then closeness (distance is mainly the radius filter)
W_EVIDENCE, W_SIZE, W_CLOSE = 0.50, 0.33, 0.17
PREDICTED_DISCOUNT = 0.8          # a predicted bed count counts 80% as much as a listed one

def size_score(d):
    """Real bed count -> log scale (300+ beds = 1.0). No real count -> the classifier's probability of
    being a large (100+ bed) facility, discounted because it is a prediction."""
    beds = pd.to_numeric(d["beds_value"], errors="coerce")
    s = (np.log1p(beds) / np.log1p(300)).clip(0, 1)
    p_large = pd.to_numeric(d.get("beds_p_large"), errors="coerce")
    s = np.where(beds.isna() & p_large.notna(), PREDICTED_DISCOUNT * p_large, s)
    return pd.Series(s, index=d.index).fillna(0.3)

def rank_results(service, ulat, ulon, max_km=150):
    d = ev[ev["service"] == service].copy()
    d["distance_km"] = haversine_km(ulat, ulon, d["loc_lat"].values, d["loc_lon"].values)
    d = d[d["distance_km"] <= max_km].copy()
    d["evidence_strength"] = (d["evidence_score"] / 8).clip(upper=1.0)
    d["closeness"] = 1 / (1 + d["distance_km"] / 25)
    d["size_score"] = size_score(d)
    d["rank_score"] = (W_EVIDENCE * d["evidence_strength"] + W_SIZE * d["size_score"]
                      + W_CLOSE * d["closeness"]) * d["confidence"].map(CONF_MULT)
    return d.sort_values("rank_score", ascending=False)

def find_care(service, location, top_n=5, max_km=150):
    match = [s for s in SERVICES if service.lower() in s.lower()]
    if not match:
        print("Unknown service. Choose from:", ", ".join(SERVICES)); return None
    service = match[0]
    loc = parse_location(location)
    if loc is None:
        print(f"Could not find '{location}'. Try a 6-digit PIN, 'City, State', or 'lat, lon'."); return None
    ulat, ulon, loc_desc, _ = loc
    d = rank_results(service, ulat, ulon, max_km)
    top = d.head(top_n)

    print("=" * 90)
    print(f"{service.upper()} near {loc_desc}")
    n_high = (d["confidence"] == "HIGH").sum()
    print(f"{len(d)} facilities within {max_km} km have some evidence ({n_high} with HIGH confidence).")
    if n_high == 0:
        far = rank_results(service, ulat, ulon, 2000)
        far = far[far["confidence"] == "HIGH"].sort_values("distance_km").head(1)
        if len(far):
            f = far.iloc[0]
            print(f"⚠ No HIGH-confidence option within {max_km} km. Nearest HIGH: {f['name']}, {f['city']} ({f['distance_km']:.0f} km).")
    print("HOW: rank = (50% evidence + 33% size [beds, predicted x0.8] + 17% closeness) x confidence (HIGH 1.0, MEDIUM 0.85, LOW 0.6)\n")

    for i, (_, r) in enumerate(top.iterrows(), 1):
        approx = f" (±{r['loc_precision_km']:.0f} km)" if r["loc_precision_km"] and r["loc_precision_km"] > 2 else ""
        print(f"#{i}  {r['name']}  -  {r['city']}, {r['state']}")
        print(f"    {r['distance_km']:.1f} km{approx} | Confidence: {r['confidence']} | Score: {r['score_breakdown']}")
        print("    WHY (quoted from the record):")
        for f, t in r["quotes"][:3]:
            print(f'      [{f}] "{t}"')
        if r["specialty_match"]:
            print(f"      [specialty] {r['specialty_match']}")
        if r["support_quote"]:
            print(f'      [support] "{r["support_quote"]}"')
        for w in r["warnings"]:
            print(f"    ⚠ {w}")
        print(f"    Phone: {r['phone'] or 'none'} | Website: {r['website'] or 'none'}\n")
    return top

# =====================================================================
# 6. TEST SEARCHES  (the challenge's own example scenarios)
# =====================================================================
find_care("dialysis", "Jaipur")
find_care("dialysis", "Patna")
find_care("c-section", "Lucknow")
find_care("trauma", "Pune")
find_care("nicu", "Nagpur")
