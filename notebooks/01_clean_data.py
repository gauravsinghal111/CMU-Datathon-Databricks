# Databricks notebook source
# =====================================================================
#  NOTEBOOK 1 - DATA CLEANUP  (v2)
#
#  Input : marketplace tables  facilities, india_post_pincode_directory
#  Output: workspace.default.facilities_clean   one row per facility, cleaned + flags
#          workspace.default.pin_lookup         PIN -> centre, radius, district, state
#          workspace.default.place_lookup       city / district / post-office name -> centre
#          workspace.default.cleaning_report    counts for the slides
#          /Volumes/workspace/default/exports/facilities_clean.csv   (for Excel)
#
#  What it fixes
#   A. Rows whose values slid into the wrong columns
#   B. State names (36 states + UTs) and which ones have no data
#   C. PIN lookup with a robust centre and an "area size" (radius)
#   D. Place lookup: cities, districts, post offices
#   E. LOCATION: vote between up to 6 independent sources
#        map coordinates | postcode | city field | address locality |
#        "located in ..." text | city in directory-listing URLs
#   F. Phone numbers: normalise, detect numbers shared by many facilities
#   G. Websites: drop values that are not real web addresses
#   H. Doctor / bed / year values that are placeholders or impossible
#   J. Duplicate records (same name, same city, same place)
#   I. Merge risk: weighted signals that one record mixes several facilities
#   K. Which records are safe to use in search
# =====================================================================
import json, re, math
from collections import defaultdict
from urllib.parse import urlparse
import numpy as np
import pandas as pd
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType, BooleanType

SRC = "databricks_virtue_foundation_dataset_dais_2026.virtue_foundation_dataset"
OUT = "workspace.default"
EXPORT_DIR = "/Volumes/workspace/default/exports"

fac = spark.table(f"{SRC}.facilities").toPandas()
pins_raw = spark.table(f"{SRC}.india_post_pincode_directory").toPandas()

report = []
def log(section, check, n):
    report.append((section, check, int(n)))

# =====================================================================
# HELPERS + CONSTANTS
# =====================================================================
def clean_str(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    s = str(x).strip()
    return None if s.lower() in ("", "null", "none", "nan") else s

def parse_list(x):
    """'["a","b","a"]' text -> ['a','b'] (deduplicated)."""
    s = clean_str(x)
    if s is None or s == "[]":
        return []
    try:
        v = json.loads(s)
        items = v if isinstance(v, list) else [v]
    except Exception:
        items = [s]
    out = []
    for i in items:
        i = clean_str(i)
        if i and i not in out:
            out.append(i)
    return out

def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorised distance (numpy / pandas)."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))

def hav(lat1, lon1, lat2, lon2):
    """Fast scalar distance."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(min(1.0, a)))

def in_india(lat, lon):
    return (lat >= 6) & (lat <= 37.5) & (lon >= 68) & (lon <= 97.5)

def domain(u):
    """Website domain, or None if the text isn't a real web address."""
    u = clean_str(u)
    if not u or " " in u or '"' in u or "[" in u:
        return None
    try:
        d = urlparse(u if "//" in u else "http://" + u).netloc.lower().replace("www.", "")
    except ValueError:
        return None
    return d if re.fullmatch(r"[a-z0-9.-]+\.[a-z]{2,}", d) else None

def url_path(u):
    try:
        return urlparse(u if "//" in u else "http://" + u).path
    except ValueError:
        return ""

DIRECTORY_SITES = (r"justdial|practo|facebook|linkedin|instagram|twitter|x\.com|youtube|google|"
                   r"onefivenine|indiamart|magicbricks|wheree|lybrate|sulekha|medindia|credihealth|"
                   r"restaurant-guru|gov\.in|wikipedia|indiabyroad|palliumindia|palliativecare|meddco|"
                   r"bajajfinserv|myupchar|kayawell|365doctor|drlogy|crunchbase|mappls|cashlesshospital|"
                   r"hospitalsin|hospitallist|zaubacorp|tofler|quickerala|asklaila|yellowpages|clinicspots|"
                   r"healthfrog|docprime|apollo247|1mg|pharmeasy|tripadvisor|yelp|nearbuy|magicpin")
FACILITY_WORDS = r"hosp|clinic|nursing|maternity|dialysis|medicalcent|healthcent"
GENERIC = {"hospital", "hospitals", "clinic", "clinics", "centre", "center", "multi", "multispeciality", "speciality",
           "specialty", "super", "care", "health", "healthcare", "medical", "research", "surgical", "nursing", "home"}

# =====================================================================
# A. STRUCTURE: misaligned rows, types, lists
# =====================================================================
for c in ["name", "facilityTypeId", "operatorTypeId", "address_line1", "address_line2", "address_line3",
          "address_city", "address_stateOrRegion", "officialWebsite", "officialPhone", "description"]:
    fac[c] = fac[c].map(clean_str)

VALID_TYPES = {"hospital", "clinic", "dentist", "doctor", "pharmacy"}
VALID_OPS = {"private", "public", "government"}
ftype = fac["facilityTypeId"].str.lower().replace({"farmacy": "pharmacy"})
op = fac["operatorTypeId"].str.lower()
bad_type = ftype.notna() & ~ftype.isin(VALID_TYPES)
bad_op = op.notna() & ~op.isin(VALID_OPS)
fac["facility_type"] = ftype.where(~bad_type)
fac["operator_type"] = op.where(~bad_op).replace({"government": "public"})
fac["flag_misaligned"] = (bad_type | bad_op).fillna(False).astype(bool)
fac["flag_no_name"] = fac["name"].isna()
log("A structure", "Rows with values in the wrong columns", fac.flag_misaligned.sum())
log("A structure", "Rows with no facility name", fac.flag_no_name.sum())

for col in ["specialties", "procedure", "equipment", "capability", "source_urls", "phone_numbers"]:
    fac[col + "_list"] = fac[col].apply(parse_list)

fac["description"] = fac["description"].fillna("")
FILLER = re.compile(r"listed as (?:a )?nearby|is listed (?:on|in|under|as)|nearby hospital|appears on .{0,40}(?:map|listing)", re.I)
fac["flag_filler_description"] = fac["description"].str.contains(FILLER)
log("A structure", "Descriptions that are only directory filler", fac.flag_filler_description.sum())

# =====================================================================
# B. STATES
# =====================================================================
STATES = ["Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat", "Haryana",
          "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur",
          "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana",
          "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",                                   # 28 states
          "Andaman And Nicobar Islands", "Chandigarh", "Dadra And Nagar Haveli And Daman And Diu", "Delhi",
          "Jammu And Kashmir", "Ladakh", "Lakshadweep", "Puducherry"]                                # 8 UTs

def skey(s):
    s = str(s).lower().replace("&", "and")
    s = re.sub(r"^the\s+", "", s)
    return re.sub(r"[^a-z]", "", s)

CANON = {skey(s): s for s in STATES}
CANON.update({
    "orissa": "Odisha", "telengana": "Telangana", "pondicherry": "Puducherry", "uttaranchal": "Uttarakhand",
    "chattisgarh": "Chhattisgarh", "chhatisgarh": "Chhattisgarh", "nctdelhi": "Delhi", "nctofdelhi": "Delhi",
    "newdelhi": "Delhi", "westdelhi": "Delhi", "southdelhi": "Delhi", "northdelhi": "Delhi", "eastdelhi": "Delhi",
    "delhincr": "Delhi", "jammukashmir": "Jammu And Kashmir", "jandk": "Jammu And Kashmir",
    "andamanandnicobar": "Andaman And Nicobar Islands", "andamannicobarislands": "Andaman And Nicobar Islands",
    "dadraandnagarhaveli": "Dadra And Nagar Haveli And Daman And Diu",
    "damananddiu": "Dadra And Nagar Haveli And Daman And Diu",
    "dadranagarhaveli": "Dadra And Nagar Haveli And Daman And Diu",
})

def canon_state(s):
    s = clean_str(s)
    return CANON.get(skey(s)) if s else None

fac["state_canon"] = fac["address_stateOrRegion"].map(canon_state)
state_is_place = fac["address_stateOrRegion"].notna() & fac["state_canon"].isna()
fac["city_raw"] = fac["address_city"].where(fac["address_city"].notna(),
                                            fac["address_stateOrRegion"].where(state_is_place))

# =====================================================================
# C. PIN LOOKUP (robust centre + area radius)
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
    """Normalised place name used for matching."""
    s = clean_str(s)
    if not s:
        return None
    s = s.lower().replace("&", " and ")
    s = re.sub(r"\b(city|district|dist|urban|rural|municipal corporation)\b", " ", s)
    s = re.sub(r"[^a-z ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return CITY_ALIASES.get(s, s) or None

p = pins_raw.copy()
p["pin"] = p["pincode"].astype(str).str.extract(r"(\d{6})")[0]
p["lat"] = pd.to_numeric(p["latitude"], errors="coerce")
p["lon"] = pd.to_numeric(p["longitude"], errors="coerce")
p = p[in_india(p["lat"], p["lon"]) & p["pin"].notna()].copy()
p["state"] = p["statename"].map(canon_state)
p["district_name"] = p["district"].astype(str).str.title()

med = p.groupby("pin")[["lat", "lon"]].median().rename(columns={"lat": "mlat", "lon": "mlon"})
p = p.join(med, on="pin")
p = p[haversine_km(p["lat"], p["lon"], p["mlat"], p["mlon"]) <= 50].drop(columns=["mlat", "mlon"])

pin_lookup = p.groupby("pin").agg(pin_lat=("lat", "median"), pin_lon=("lon", "median"),
                                  n_offices=("lat", "size")).reset_index()
p = p.merge(pin_lookup[["pin", "pin_lat", "pin_lon"]], on="pin")
p["d_pin"] = haversine_km(p["lat"], p["lon"], p["pin_lat"], p["pin_lon"])

def top_by(df, key, col):
    t = df.groupby([key, col]).size().reset_index(name="n").sort_values([key, "n"], ascending=[True, False])
    return t.drop_duplicates(key).set_index(key)[col]

pin_lookup = (pin_lookup.set_index("pin")
              .join(p.groupby("pin")["d_pin"].quantile(0.9).rename("pin_radius_km"))
              .join(top_by(p, "pin", "district_name").rename("pin_district"))
              .join(top_by(p, "pin", "state").rename("pin_state"))
              .reset_index())
pin_lookup["pin_radius_km"] = pin_lookup["pin_radius_km"].fillna(0).clip(lower=2)
p = p.drop(columns=["pin_lat", "pin_lon"])
log("C postcodes", "PIN codes with a usable centre", len(pin_lookup))

# =====================================================================
# D. PLACE LOOKUP (districts, post offices; facility cities added in E)
# =====================================================================
def centres(df, keys, latc, lonc, kind, min_n=1):
    d = df[keys + [latc, lonc]].dropna().rename(columns={latc: "_y", lonc: "_x"})
    g = d.groupby(keys).agg(lat=("_y", "median"), lon=("_x", "median"), n=("_y", "size")).reset_index()
    g = g[g["n"] >= min_n]
    m = d.merge(g, on=keys)
    m["_d"] = haversine_km(m["_y"], m["_x"], m["lat"], m["lon"])
    g = g.merge(m.groupby(keys)["_d"].quantile(0.9).rename("radius_km").reset_index(), on=keys)
    g["radius_km"] = g["radius_km"].clip(lower=3)
    g["kind"] = kind
    return g

p["office_clean"] = (p["officename"].astype(str).str.lower()
                     .str.replace(r"\s+(b\.?o|s\.?o|h\.?o|g\.?p\.?o)\.?$", "", regex=True).str.strip())
p["okey"] = p["office_clean"].map(ckey)
district_c = centres(p.assign(place=p["district"].map(ckey)), ["place", "state"], "lat", "lon", "district")
office_c = centres(p.rename(columns={"okey": "place"}), ["place", "state"], "lat", "lon", "post office")
office_c = office_c[office_c["radius_km"] <= 30]          # same office name far apart = ambiguous

# =====================================================================
# E. LOCATION - vote between up to 6 independent sources
# =====================================================================
fac["pin"] = fac["address_zipOrPostcode"].astype(str).str.extract(r"(\d{6})")[0]
fac = fac.merge(pin_lookup, on="pin", how="left")

def nearest_pin(lat, lon, chunk=256):
    """Nearest PIN centre to each point (gives the state/district at a coordinate)."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    idx = np.full(len(lat), -1)
    ok = np.where(in_india(lat, lon))[0]
    plat, plon = pin_lookup["pin_lat"].values[None, :], pin_lookup["pin_lon"].values[None, :]
    for s in range(0, len(ok), chunk):
        ii = ok[s:s + chunk]
        d = haversine_km(lat[ii][:, None], lon[ii][:, None], plat, plon)
        idx[ii] = d.argmin(axis=1)
    return idx

def at_idx(col, idx):
    vals = pin_lookup[col].values
    return np.where(idx >= 0, vals[np.clip(idx, 0, None)], None)

fac["coord_state"] = at_idx("pin_state", nearest_pin(fac["latitude"], fac["longitude"]))

# ---- final state: address -> postcode -> coordinates
fac["state"] = fac["state_canon"]
fac["state_source"] = np.where(fac["state_canon"].notna(), "address", None)
m = fac["state"].isna() & fac["pin_state"].notna()
fac.loc[m, "state"] = fac.loc[m, "pin_state"]; fac.loc[m, "state_source"] = "postcode"
m = fac["state"].isna() & fac["coord_state"].notna()
fac.loc[m, "state"] = fac.loc[m, "coord_state"]; fac.loc[m, "state_source"] = "coordinates"
present = set(fac["state"].dropna())
missing = [s for s in STATES if s not in present]
log("B states", "Distinct state values before cleaning", fac["address_stateOrRegion"].nunique())
log("B states", "States + UTs with facilities (of 36)", len(present))
log("B states", "Missing (no facilities in data): " + (", ".join(missing) or "none"), len(missing))
log("B states", "State filled from postcode or coordinates", fac["state_source"].isin(["postcode", "coordinates"]).sum())

fac["city_key"] = fac["city_raw"].map(ckey)
fac["city"] = fac["city_raw"].str.title()
fac["coords_ok"] = in_india(fac["latitude"], fac["longitude"]).fillna(False).astype(bool)

# ---- city centres learned from facilities whose coordinates AND postcode agree
gap_cp = haversine_km(fac["latitude"], fac["longitude"], fac["pin_lat"], fac["pin_lon"])
thr_p = (1.5 * fac["pin_radius_km"]).clip(lower=25, upper=60).fillna(25)
verified = fac[fac["coords_ok"] & (gap_cp <= thr_p)]
fac_city_c = centres(verified.rename(columns={"city_key": "place"}), ["place", "state"],
                     "latitude", "longitude", "facility city", min_n=3)

place_lookup = pd.concat([fac_city_c, district_c, office_c], ignore_index=True)
place_lookup["prio"] = place_lookup["kind"].map({"facility city": 0, "district": 1, "post office": 2})
best_place = place_lookup.sort_values("prio").drop_duplicates(["place", "state"])
fac = fac.merge(best_place.rename(columns={"place": "city_key", "lat": "city_lat", "lon": "city_lon",
                                           "radius_km": "city_radius_km", "kind": "city_source", "n": "city_n"})
                          .drop(columns="prio"),
                on=["city_key", "state"], how="left")

# ---- lookup dictionaries for the extra sources
OFFICE_PIN = {(k, pn): (la, lo) for k, pn, la, lo in p[["okey", "pin", "lat", "lon"]].dropna().itertuples(index=False)}
OFFICE_STATE = {(r.place, r.state): (r.lat, r.lon, max(3.0, r.radius_km)) for r in office_c.itertuples()}
PLACE_STATE = {(r.place, r.state): (r.lat, r.lon, r.radius_km) for r in best_place.itertuples()}
PLACE_ANY = defaultdict(list)                       # city / district name -> every state that has it
for r in best_place[best_place["kind"].isin(["facility city", "district"])].itertuples():
    PLACE_ANY[r.place].append((r.state, r.lat, r.lon, r.radius_km))

def lookup_place(k, st):
    """Place in the facility's state, else a place that exists in only one state."""
    if st and (k, st) in PLACE_STATE:
        return PLACE_STATE[(k, st)]
    opts = PLACE_ANY.get(k, [])
    return opts[0][1:] if len(opts) == 1 else None

SKIP = {"india", "hospital", "clinic", "road", "main road", "near", "nagar", "colony", "sector", "ward", "post",
        "village", "taluk", "tehsil", "street", "lane", "floor", "building", "complex", "market", "bazar", "bazaar",
        "chowk", "circle", "cross", "station", "bus stand", "railway station", "town", "phase", "block", "area",
        "hospitals", "clinics", "doctors", "places", "listing", "india places", "villages", "town hospitals"}
PREFIX = re.compile(r"^(near|nr|opp|opposite|behind|beside|next to|in front of|adjacent to|adj|at|po|p o|"
                    r"post|via|dist|district|taluk|tehsil|village|vill)\.?\s+", re.I)
LOCATED = re.compile(r"\b(?:located|based|situated)\s+(?:in|at|near)\s+([^.;]+)", re.I)

def phrases(text):
    out = []
    for part in re.split(r"[,;/\n]| - ", str(text or "")):
        part = PREFIX.sub("", part.strip())
        if re.search(r"\d", part):
            continue
        k = ckey(part)
        if k and len(k) >= 4 and k not in SKIP and k not in GENERIC:
            out.append(k)
    return out

def candidates(r):
    """Every location claim we can find for a facility: (source, lat, lon, precision_km, label)."""
    c, st = [], r["state"]
    if r["coords_ok"]:
        c.append(("map", r["latitude"], r["longitude"], 0.5, "map coordinates"))
    if pd.notna(r["pin_lat"]):
        c.append(("postcode", r["pin_lat"], r["pin_lon"], r["pin_radius_km"], f"postcode {r['pin']}"))
    if pd.notna(r["city_lat"]):
        c.append(("city", r["city_lat"], r["city_lon"], r["city_radius_km"], f"city '{r['city']}'"))
    seen = {r["city_key"]}
    for line in [r["address_line1"], r["address_line2"], r["address_line3"]]:
        for k in phrases(line):
            if k in seen:
                continue
            seen.add(k)
            hit = OFFICE_PIN.get((k, r["pin"]))
            if hit:
                # found INSIDE the listed PIN -> precise, but not independent of the postcode
                c.append(("address_pin", hit[0], hit[1], 2.0, f"address locality '{k.title()}' (in postcode {r['pin']})"))
                continue
            hit = OFFICE_STATE.get((k, st))
            if hit:
                c.append(("address", hit[0], hit[1], hit[2], f"address locality '{k.title()}'"))
    for t in r["capability_list"] + [r["description"]]:
        for mt in LOCATED.finditer(t):
            for k in phrases(mt.group(1)):
                hit = lookup_place(k, st)
                if hit:
                    c.append(("text", hit[0], hit[1], hit[2], f"text says located in '{k.title()}'"))
    for u in r["source_urls_list"]:
        if not re.search(DIRECTORY_SITES, u, re.I):
            continue
        for seg in url_path(u).split("/"):
            k = ckey(seg.replace("-", " ").replace("_", " "))
            if not k or len(k) < 4 or k in SKIP:
                continue
            hit = lookup_place(k, st)
            if hit:
                c.append(("web listing", hit[0], hit[1], hit[2], f"web listing for '{k.title()}'"))
    # de-duplicate (same source, same spot)
    out, keys = [], set()
    for x in c:
        key = (x[0], round(x[1], 2), round(x[2], 2))
        if key not in keys:
            keys.add(key); out.append(x)
    return out

def agree_thr(pa, pb):
    m = max(pa, pb)
    return min(60.0, max(10.0 if m <= 5 else 25.0, 1.5 * m))

# City field, "located in" text and directory URLs all repeat the same town NAME,
# so together they count as ONE independent family, not three votes.
FAMILY = {"map": "coordinates", "postcode": "postcode", "address_pin": "postcode",
          "address": "address locality", "city": "place name", "text": "place name", "web listing": "place name"}

ADDRESS_FAMILIES = {"postcode", "place name", "address locality"}

def resolve(r):
    all_cands = candidates(r)
    # Directory-URL towns are too noisy to vote on location; they are only used to detect merged records
    cands = [c for c in all_cands if c[0] != "web listing"]
    if not cands:
        return (np.nan, np.nan, "none", "none", np.nan, "No usable location", None, None, [], 0)
    support = []
    for a in cands:
        s = {FAMILY[a[0]]}
        for b in cands:
            if FAMILY[b[0]] not in s and hav(a[1], a[2], b[1], b[2]) <= agree_thr(a[3], b[3]):
                s.add(FAMILY[b[0]])
        support.append(s)
    # most families wins; on a tie, prefer the side backed by the written address; then precision
    bi = max(range(len(cands)),
             key=lambda i: (len(support[i]), len(support[i] & ADDRESS_FAMILIES), -cands[i][3]))
    a = cands[bi]
    cluster = [b for b in cands if b is a or hav(a[1], a[2], b[1], b[2]) <= agree_thr(a[3], b[3])]
    pick = min(cluster, key=lambda b: b[3])
    agree = sorted({b[0] for b in cluster})
    all_types = {b[0] for b in cands}
    dissent_types = sorted(all_types - set(agree))
    fam_agree = {FAMILY[t] for t in agree}
    # Only the record's own address fields can make the location "conflicting";
    # far-away web listings / text go to merge risk instead (Section I)
    fam_dissent = {FAMILY[t] for t in dissent_types
                   if t in ("map", "postcode", "address_pin", "city", "address")} - fam_agree
    k, precise = len(fam_agree), pick[3] <= 5          # precise = coordinates or post office

    # The address CITY is the most explicit claim of where a facility is. If a real town/district
    # named in the city field is far from the chosen location, the record contradicts itself:
    # never call that "high", whatever the other sources say (they may all come from the same wrong address).
    city_conflict = "city" in dissent_types and r["city_source"] in ("facility city", "district")

    if k >= 2 and len(fam_dissent) >= k:
        conf = "low"                                   # as many families disagree as agree
    elif city_conflict:
        conf = "low"                                   # stated city contradicts the location
    elif k >= 2 and precise:
        conf = "high"                                  # independently confirmed AND a precise point
    elif k >= 2:
        conf = "medium"                                # confirmed, but only to an area (± km)
    else:
        conf = "low"

    labels = list(dict.fromkeys(b[4] for b in cluster))
    note = "Agrees: " + ", ".join(labels[:4])
    far = []
    dis = []
    for b in all_cands:
        if b in cluster:
            continue
        d = hav(pick[1], pick[2], b[1], b[2])
        if b[0] in dissent_types:
            dis.append(f"{b[4]} ({d:.0f} km away)")
        if b[0] in ("web listing", "text", "address", "address_pin") and d > 100:
            far.append(f"{b[4]} ({d:.0f} km)")
    if dis:
        note += ". Disagrees: " + ", ".join(dict.fromkeys(dis[:3]))
    if city_conflict:
        note += f". The address says the city is '{r['city']}', which does not match"
    if pick[3] > 2:
        note += f". Position approx. ±{pick[3]:.0f} km"
    if conf == "low":
        note += ". Confirm by phone"
    return (pick[1], pick[2], "+".join(agree), conf, float(pick[3]), note,
            ", ".join(agree), ", ".join(dissent_types) or None, far, len({c[0] for c in all_cands}))

fac[["loc_lat", "loc_lon", "loc_method", "loc_confidence", "loc_precision_km", "loc_note",
     "loc_sources_agree", "loc_sources_disagree", "far_mentions", "loc_n_source_types"]] = \
    fac.apply(resolve, axis=1, result_type="expand")
for c in ["loc_lat", "loc_lon", "loc_precision_km"]:
    fac[c] = pd.to_numeric(fac[c], errors="coerce")

li = nearest_pin(fac["loc_lat"], fac["loc_lon"])
fac["loc_state"] = at_idx("pin_state", li)
fac["district"] = at_idx("pin_district", li)
fac["flag_state_mismatch"] = (fac["loc_state"].notna() & fac["state"].notna() &
                              (fac["loc_state"] != fac["state"])).astype(bool)

for k, v in fac["loc_confidence"].value_counts().items():
    log("E location", f"Location confidence: {k}", v)
for k, v in fac["loc_n_source_types"].value_counts().sort_index().items():
    log("E location", f"Facilities with {k} location source type(s) available", v)
src_counts = defaultdict(int)
for s in fac["loc_sources_agree"].dropna():
    for x in s.split(", "):
        src_counts[x] += 1
for k, v in sorted(src_counts.items(), key=lambda kv: -kv[1]):
    log("E location", f"Source '{k}' is part of the agreed location", v)
log("E location", "Final location is in a different state than the address", fac.flag_state_mismatch.sum())
log("E location", "Address city contradicts the location (set to low)",
    fac["loc_note"].str.contains("which does not match", na=False).sum())

# =====================================================================
# F. PHONES
# =====================================================================
def norm_phone(s):
    d = re.sub(r"\D", "", str(s))
    if d.startswith(("1800", "1860")):
        return d if 10 <= len(d) <= 11 else None
    if len(d) == 12 and d.startswith("91"):
        d = d[2:]
    elif len(d) == 11 and d.startswith("0"):
        d = d[1:]
    elif len(d) == 13 and d.startswith("091"):
        d = d[3:]
    return "+91" + d if len(d) == 10 and d[0] != "0" else None

STOP = GENERIC | {"dr", "the", "sri", "shri", "shree", "new", "and", "of", "pvt", "ltd"}

def brand(name):
    toks = [t for t in re.findall(r"[a-z]+", str(name).lower()) if t not in STOP and len(t) > 2]
    return toks[0] if toks else str(name).lower()

fac["brand"] = fac["name"].map(brand)
fac["phones_all"] = fac.apply(
    lambda r: list(dict.fromkeys(x for x in (norm_phone(v) for v in
                   ([r["officialPhone"]] if r["officialPhone"] else []) + r["phone_numbers_list"]) if x)), axis=1)

owners = defaultdict(set)
for b, phones in zip(fac["brand"], fac["phones_all"]):
    for ph in phones:
        owners[ph].add(b)

def phone_info(r):
    unique = [x for x in r["phones_all"] if len(owners[x]) == 1]
    off = norm_phone(r["officialPhone"]) if r["officialPhone"] else None
    best = off if off in unique else (unique[0] if unique else (off or (r["phones_all"][0] if r["phones_all"] else None)))
    n = len(owners[best]) if best else 0
    if best is None:
        flag = "No valid phone number"
    elif n >= 3:
        flag = f"Number is listed for {n} different facilities; probably a booking or directory line"
    elif n == 2:
        flag = "Number is also listed for another facility"
    else:
        flag = None
    return best, flag, n

fac[["phone_best", "phone_flag", "phone_shared_count"]] = fac.apply(phone_info, axis=1, result_type="expand")
fac["official_phone_norm"] = fac["officialPhone"].map(lambda x: norm_phone(x) if x else None)
log("F phones", "Facilities with a valid, unshared phone", fac["phone_flag"].isna().sum())
log("F phones", "Best phone is shared with 2 facilities", (fac["phone_shared_count"] == 2).sum())
log("F phones", "Best phone is shared with 3+ facilities (directory line)", (fac["phone_shared_count"] >= 3).sum())
log("F phones", "No valid phone at all", fac["phone_best"].isna().sum())
log("F phones", "Official phone was invalid, another number used",
    (fac["official_phone_norm"].isna() & fac["officialPhone"].notna() & fac["phone_best"].notna()).sum())
log("F phones", "Official phone was shared, a unique number used instead",
    (fac["official_phone_norm"].notna() & fac["phone_best"].notna() &
     (fac["phone_best"] != fac["official_phone_norm"])).sum())

# =====================================================================
# G. WEBSITES
# =====================================================================
fac["website"] = fac["officialWebsite"].map(domain)
fac["flag_bad_website"] = (fac["officialWebsite"].notna() & fac["website"].isna()).astype(bool)
log("G websites", "Official website field that is not a web address", fac.flag_bad_website.sum())

# =====================================================================
# H. SIZE NUMBERS (doctors, beds, year)
# =====================================================================
d_raw = pd.to_numeric(fac["numberDoctors"], errors="coerce")
b_raw = pd.to_numeric(fac["capacity"], errors="coerce")
y_raw = pd.to_numeric(fac["yearEstablished"], errors="coerce")
rules = [
    ((d_raw == 1) & (fac["facility_type"] == "hospital"), "doctors", "Doctor count of 1 for a hospital treated as unknown"),
    ((d_raw <= 0) | (d_raw > 5000), "doctors", "Doctor count impossible, removed"),
    ((b_raw <= 0) | (b_raw > 5000), "beds", "Bed count impossible, removed"),
    ((y_raw < 1800) | (y_raw > 2026), "year_established", "Year established impossible, removed"),
]
fac["doctors"], fac["beds"], fac["year_established"] = d_raw, b_raw, y_raw
notes = [[] for _ in range(len(fac))]
for mask, col, msg in rules:
    mask = mask.fillna(False).values
    fac.loc[mask, col] = np.nan
    for i in np.where(mask)[0]:
        notes[i].append(msg)
    log("H size", msg, mask.sum())
fac["size_note"] = ["; ".join(n) if n else None for n in notes]

# =====================================================================
# J. DUPLICATES
# =====================================================================
LIST_COLS = ["specialties_list", "procedure_list", "equipment_list", "capability_list", "source_urls_list"]
fac = fac.reset_index(drop=True)
fac["name_key"] = fac["name"].fillna("").str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
fac["evidence_len"] = fac[["procedure_list", "equipment_list", "capability_list"]].apply(
    lambda r: sum(len(x) for x in r), axis=1)
is_dup = np.zeros(len(fac), dtype=bool)
dup_of = [None] * len(fac)
merged = {c: {} for c in LIST_COLS}
cand = fac[(fac["name_key"].str.len() > 3) & fac["city_key"].notna()]
for _, g in cand.groupby(["name_key", "city_key"]):
    if len(g) < 2:
        continue
    g = g.sort_values("evidence_len", ascending=False)
    i0 = g.index[0]
    for i in g.index[1:]:
        dist = haversine_km(fac.at[i0, "loc_lat"], fac.at[i0, "loc_lon"], fac.at[i, "loc_lat"], fac.at[i, "loc_lon"])
        if pd.notna(dist) and dist > 5:
            continue                                     # same name, different branch
        is_dup[i] = True
        dup_of[i] = fac.at[i0, "unique_id"]
        for c in LIST_COLS:
            base = merged[c].get(i0, fac.at[i0, c])
            merged[c][i0] = list(dict.fromkeys(base + fac.at[i, c]))
for c in LIST_COLS:
    fac[c] = [merged[c].get(i, v) for i, v in zip(fac.index, fac[c])]
fac["is_duplicate"] = is_dup
fac["duplicate_of"] = dup_of
fac["n_sources"] = fac["source_urls_list"].apply(len)
log("J duplicates", "Duplicate records merged into another record", fac.is_duplicate.sum())

# =====================================================================
# I. MERGE RISK - weighted signals that one record mixes facilities
#    weight 2: listings / text place it far away (>100 km), location sources conflict
#    weight 1: another facility's website, size doesn't add up, phone used by another facility
#    high >= 3, possible 1-2
# =====================================================================
def other_facility_sites(r):
    toks = [t for t in re.findall(r"[a-z]{4,}", str(r["name"]).lower())
            if t not in GENERIC and t != (r["city_key"] or "")]
    doms = {domain(u) for u in r["source_urls_list"] if not re.search(DIRECTORY_SITES, u, re.I)}
    doms = {d for d in doms if d} - {r["website"]}
    return sorted(d for d in doms if re.search(FACILITY_WORDS, d) and not any(t in d for t in toks))

def merge_risk(r):
    score, notes = 0, []
    far = r["far_mentions"]
    if far:
        # Directory URLs are noisy -> weak signal (1) however many there are.
        # The record's OWN text or address pointing far away -> strong signal (2).
        own = [f for f in far if not f.startswith("web listing")]
        score += 2 if own else 1
        notes.append("Listings/text also point to places far away: " + "; ".join((own or far)[:3]))
    if r["loc_confidence"] == "low" and r["loc_sources_disagree"]:
        score += 2
        notes.append("Location sources contradict each other")
    sites = other_facility_sites(r)
    if sites:
        score += 1
        notes.append("Sources include another facility's website: " + ", ".join(sites[:3]))
    dr, bd = pd.to_numeric(r["numberDoctors"], errors="coerce"), pd.to_numeric(r["capacity"], errors="coerce")
    if pd.notna(dr) and pd.notna(bd) and dr <= 2 and bd >= 50:
        score += 1
        notes.append(f"Size does not add up ({int(dr)} doctors, {int(bd)} beds)")
    if any(len(owners[x]) >= 2 for x in r["phones_all"]):
        score += 1
        notes.append("A phone number in the record belongs to another facility too")
    level = "high" if score >= 3 else ("possible" if score >= 1 else "none")
    return level, score, "; ".join(notes) or None

fac[["merge_risk", "merge_score", "merge_notes"]] = fac.apply(merge_risk, axis=1, result_type="expand")
for k, v in fac["merge_risk"].value_counts().items():
    log("I merge risk", f"Merge risk: {k}", v)
log("I merge risk", "Signal: listings/text point >100 km away", (fac["far_mentions"].str.len() > 0).sum())
log("I merge risk", "Signal: location sources contradict",
    ((fac["loc_confidence"] == "low") & fac["loc_sources_disagree"].notna()).sum())

# =====================================================================
# K. SEARCH ELIGIBILITY
# =====================================================================
fac["exclude_reason"] = np.select(
    [fac["flag_no_name"], fac["flag_misaligned"], fac["loc_lat"].isna(), fac["is_duplicate"]],
    ["No facility name", "Fields are in the wrong columns", "No usable location", "Duplicate of another record"],
    default="")
fac["include_in_search"] = fac["exclude_reason"] == ""
log("K search", "Records usable in search", fac.include_in_search.sum())
for k, v in fac.loc[~fac.include_in_search, "exclude_reason"].value_counts().items():
    log("K search", f"Excluded: {k}", v)
s = fac[fac.include_in_search]
for k, v in s["loc_confidence"].value_counts().items():
    log("K search", f"Searchable with location confidence {k}", v)

# =====================================================================
# SAVE
# =====================================================================
fac = fac.rename(columns={"latitude": "latitude_raw", "longitude": "longitude_raw",
                          "address_stateOrRegion": "state_raw", "address_city": "city_raw_field"})
KEEP = ["unique_id", "name", "brand", "facility_type", "operator_type",
        "address_line1", "address_line2", "address_line3", "city", "city_raw_field", "district", "state",
        "state_raw", "state_source", "pin",
        "loc_lat", "loc_lon", "loc_method", "loc_confidence", "loc_precision_km", "loc_note",
        "loc_sources_agree", "loc_sources_disagree", "loc_n_source_types", "loc_state", "flag_state_mismatch",
        "latitude_raw", "longitude_raw", "pin_lat", "pin_lon", "pin_radius_km", "city_lat", "city_lon", "city_source",
        "phone_best", "phone_flag", "phone_shared_count", "phones_all",
        "website", "flag_bad_website", "n_sources", "source_urls_list",
        "doctors", "beds", "year_established", "size_note",
        "merge_risk", "merge_score", "merge_notes", "far_mentions",
        "description", "flag_filler_description",
        "specialties_list", "procedure_list", "equipment_list", "capability_list",
        "flag_misaligned", "flag_no_name", "is_duplicate", "duplicate_of", "include_in_search", "exclude_reason"]
clean = fac[KEEP].copy()

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
    out = out.astype(object)
    return spark.createDataFrame(out.values.tolist(), StructType(fields))

def save(df, name):
    to_spark(df).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{OUT}.{name}")
    print(f"Saved table {OUT}.{name}  ({len(df):,} rows)")

save(clean, "facilities_clean")
save(pin_lookup, "pin_lookup")
save(place_lookup.drop(columns="prio"), "place_lookup")
rep = pd.DataFrame(report, columns=["section", "check", "count"])
save(rep, "cleaning_report")

spark.sql("CREATE VOLUME IF NOT EXISTS workspace.default.exports")
csv = clean.copy()
for c in csv.columns:
    if csv[c].dtype == object:
        csv[c] = csv[c].map(lambda v: "; ".join(map(str, v)) if isinstance(v, list) else v)
csv.to_csv(f"{EXPORT_DIR}/facilities_clean.csv", index=False, encoding="utf-8-sig")
rep.to_csv(f"{EXPORT_DIR}/cleaning_report.csv", index=False, encoding="utf-8-sig")
print(f"Saved CSVs in {EXPORT_DIR}")

# =====================================================================
# REPORT
# =====================================================================
print("\n" + "=" * 78 + "\nCLEANING REPORT\n" + "=" * 78)
for sec, g in rep.groupby("section", sort=False):
    print(f"\n{sec}")
    for _, r in g.iterrows():
        print(f"   {r['count']:>6,}  {r['check']}")

print("\nExamples of NON-high locations and HIGH merge risk (for review):")
ex = pd.concat([
    clean[clean["loc_confidence"] == "medium"].head(5),
    clean[clean["loc_confidence"] == "low"].head(5),
    clean[clean["merge_risk"] == "high"].head(5),
])[["name", "city", "state", "loc_confidence", "loc_note", "merge_risk", "merge_notes"]]
display(spark.createDataFrame(ex.astype(str)))
