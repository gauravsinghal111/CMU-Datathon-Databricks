# Databricks notebook source
# =====================================================================
#  NOTEBOOK 3 - GOOGLE MAPS VERIFICATION
#
#  For each facility: search Google Maps (Places API, New), match the result carefully
#  (similar name + nearby / same postcode / same city / same phone), then:
#    - CONFIRM or CORRECT location, postcode, city, district, state
#    - record Google's name as an alternative name
#    - compare / fill phone and website
#    - detect permanently closed facilities (excluded from search)
#    - keep rating, review count and up to 5 review texts (used as evidence in Notebook 2)
#  Originals are always kept next to Google values (columns *_orig) so every change is traceable.
#
#  Input : workspace.default.facilities_clean           (Notebook 1)
#  Output: workspace.default.google_places_raw           raw API responses (resumable, no paying twice)
#          workspace.default.facilities_verified         facilities_clean + Google checks + corrections
#
#  HOW TO RUN

#   2. Leave "limit" = 50 for a first test run -> check results + Google Cloud billing page
#   3. Set "limit" = 0 to process everything that is left (already-fetched rows are skipped)
# =====================================================================
import json, re, math, time, difflib, datetime
from concurrent.futures import ThreadPoolExecutor
import requests
import numpy as np
import pandas as pd
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType, BooleanType

OUT = "workspace.default"
RAW = f"{OUT}.google_places_raw"

dbutils.widgets.text("google_api_key", "")
dbutils.widgets.text("limit", "50")
# "location_only" = cheapest tier (name, address, postcode, coordinates, open/closed): largest free allowance.
# "full"          = also phone, website, rating, review count, reviews: pricier tier, smaller free allowance.
dbutils.widgets.dropdown("detail", "location_only", ["location_only", "full"])
API_KEY = dbutils.widgets.get("google_api_key").strip()
LIMIT = int(dbutils.widgets.get("limit") or 0)
DETAIL = dbutils.widgets.get("detail")
if not API_KEY:
    raise ValueError("Paste your Google Maps API key into the 'google_api_key' box at the top of this notebook.")

URL = "https://places.googleapis.com/v1/places:searchText"
FIELDS = ["id", "displayName", "formattedAddress", "addressComponents", "location", "businessStatus", "types",
          "googleMapsUri"]
if DETAIL == "full":
    FIELDS += ["nationalPhoneNumber", "internationalPhoneNumber", "websiteUri", "rating", "userRatingCount", "reviews"]
print(f"Requesting fields ({DETAIL}): {', '.join(FIELDS)}")
HEADERS = {"Content-Type": "application/json", "X-Goog-Api-Key": API_KEY,
           "X-Goog-FieldMask": ",".join("places." + f for f in FIELDS)}

# =====================================================================
# 1. WHICH FACILITIES TO CHECK (most doubtful first)
# =====================================================================
fac = spark.table(f"{OUT}.facilities_clean").toPandas()
for c in ["phones_all", "source_urls_list"]:
    fac[c] = fac[c].map(lambda s: json.loads(s) if isinstance(s, str) and s.startswith("[") else [])
todo_all = fac[fac["include_in_search"]].copy()
# lower = checked first: doubtful location, then likely-merged records, then missing/shared phone
todo_all["prio"] = (todo_all["loc_confidence"].map({"none": 0, "low": 0, "medium": 1, "high": 2}).fillna(2)
                    - (todo_all["merge_risk"] == "high").astype(int)
                    - 0.5 * (todo_all["phone_best"].isna() | todo_all["phone_flag"].notna()).astype(int))
todo_all = todo_all.sort_values("prio")

done = set()
if spark.catalog.tableExists(RAW):
    done = set(spark.table(RAW).filter("status = 200").select("unique_id").toPandas()["unique_id"])
todo = todo_all[~todo_all["unique_id"].isin(done)]
if LIMIT > 0:
    todo = todo.head(LIMIT)
elif LIMIT < 0:                      # LIMIT = -1 -> make NO new calls, just re-process saved results
    todo = todo.head(0)

# ---- HARD MONTHLY CAP: never exceed Google's free 5,000 Text Search Pro calls per month
MONTHLY_CAP = 4800
used_this_month = 0
if spark.catalog.tableExists(RAW):
    month = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m")
    used_this_month = spark.table(RAW).filter(f"substr(fetched_at, 1, 7) = '{month}'").count()
allowed = max(0, MONTHLY_CAP - used_this_month - 1)          # -1 for the connection test call
if len(todo) > allowed:
    print(f"⚠️ Monthly safety cap: {used_this_month:,} calls already made this month; "
          f"only {allowed:,} more allowed (cap {MONTHLY_CAP:,}). Trimming the batch.")
    todo = todo.head(allowed)
print(f"Calls this month so far: {used_this_month:,} | this run will make: {len(todo):,} | "
      f"total after run: {used_this_month + len(todo) + 1:,} of {MONTHLY_CAP:,} cap")
print(f"{len(todo_all):,} searchable facilities | {len(done):,} already fetched | fetching {len(todo):,} now")

print()
print("Sample of the searches that will be sent (check these look sensible):")
for q in todo.head(5).apply(lambda r: ", ".join(str(x) for x in [r["name"], r["city"] or r["district"], r["state"]]
                                                if isinstance(x, str) and x), axis=1):
    print("   ", q)

if len(todo):
    # 0. CAN WE REACH GOOGLE? (1 call; skipped when there is nothing to fetch)
    try:
        t = requests.post(URL, json={"textQuery": "AIIMS New Delhi", "maxResultCount": 1},
                          headers={**HEADERS, "X-Goog-FieldMask": "places.id,places.displayName"}, timeout=20)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"This workspace cannot reach the internet / Google ({e}). "
                           "Free Edition may block outbound calls - run this step on a laptop instead.")
    if t.status_code != 200:
        raise RuntimeError(f"Google API error {t.status_code}: {t.text[:400]} | "
                           "Check: key is correct, 'Places API (New)' is enabled, billing is on.")
    print("✅ Google Places API reachable:", t.json().get("places", [{}])[0].get("displayName"))
else:
    print("Nothing new to fetch: 0 calls will be made. Re-using saved results.")

def build_query(r):
    parts = [r["name"], r["city"] or r["district"], r["state"]]
    return ", ".join(str(x) for x in parts if isinstance(x, str) and x)

def fetch(r):
    body = {"textQuery": build_query(r), "maxResultCount": 3, "regionCode": "IN", "languageCode": "en"}
    if r["loc_confidence"] == "high":                     # only bias toward OUR location if we trust it
        body["locationBias"] = {"circle": {"center": {"latitude": float(r["loc_lat"]), "longitude": float(r["loc_lon"])},
                                           "radius": 30000.0}}
    err = "retries exhausted"
    for attempt in range(4):
        try:
            resp = requests.post(URL, json=body, headers=HEADERS, timeout=30)
            if resp.status_code in (429, 500, 503):
                time.sleep(2 ** attempt); continue
            return (r["unique_id"], body["textQuery"], resp.status_code, resp.text, datetime.datetime.now(datetime.timezone.utc).isoformat())
        except requests.exceptions.RequestException as e:
            err = str(e); time.sleep(2 ** attempt)
    return (r["unique_id"], body["textQuery"], -1, err, datetime.datetime.now(datetime.timezone.utc).isoformat())

RAW_SCHEMA = StructType([StructField("unique_id", StringType()), StructField("query", StringType()),
                         StructField("status", LongType()), StructField("response", StringType()),
                         StructField("fetched_at", StringType())])

# =====================================================================
# 2. FETCH (batches of 200, saved after every batch -> safe to stop and resume)
# =====================================================================
recs = todo.to_dict("records")
for s in range(0, len(recs), 200):
    batch = recs[s:s + 200]
    with ThreadPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(fetch, batch))
    spark.createDataFrame(res, RAW_SCHEMA).write.mode("append").saveAsTable(RAW)
    bad = [x for x in res if x[2] != 200]
    print(f"  fetched {s + len(batch):,}/{len(recs):,}  (errors in batch: {len(bad)})")
    if any(x[2] in (400, 401, 403) for x in bad):
        raise RuntimeError(f"Stopping - key/billing problem: {bad[0][3][:400]}")
    if len(bad) > len(batch) / 2:
        raise RuntimeError(f"Stopping - more than half of this batch failed. Example: {bad[0][3][:400]}")

# =====================================================================
# 3. MATCH Google results to our records
# =====================================================================
raw = spark.table(RAW).toPandas()
raw = raw[raw["status"] == 200].sort_values("fetched_at").drop_duplicates("unique_id", keep="last")
RESP = dict(zip(raw["unique_id"], raw["response"]))

CITY_ALIASES = {
    "bangalore": "bengaluru", "bengaluru urban": "bengaluru", "bangalore urban": "bengaluru", "bombay": "mumbai",
    "calcutta": "kolkata", "madras": "chennai", "gurgaon": "gurugram", "trivandrum": "thiruvananthapuram",
    "cochin": "kochi", "poona": "pune", "mysore": "mysuru", "mangalore": "mangaluru", "belgaum": "belagavi",
    "hubli": "hubballi", "baroda": "vadodara", "allahabad": "prayagraj", "benares": "varanasi",
    "banaras": "varanasi", "pondicherry": "puducherry", "new delhi": "delhi", "calicut": "kozhikode",
    "trichy": "tiruchirappalli", "tiruchirapalli": "tiruchirappalli", "vizag": "visakhapatnam",
    "shimoga": "shivamogga", "gulbarga": "kalaburagi", "bellary": "ballari", "tumkur": "tumakuru",
    "chhatrapati sambhajinagar": "aurangabad", "chh sambhajinagar": "aurangabad", "sambhajinagar": "aurangabad",
    "dharashiv": "osmanabad", "ahilyanagar": "ahmednagar", "ahilya nagar": "ahmednagar", "ahmadnagar": "ahmednagar",
    "narmadapuram": "hoshangabad", "faizabad": "ayodhya", "kanpur nagar": "kanpur", "mumbai suburban": "mumbai",
}

def ckey(s):
    s = str(s or "").lower().replace("&", " and ")
    s = re.sub(r"\b(city|district|dist|division|urban|rural|municipal corporation)\b", " ", s)
    s = re.sub(r"[^a-z ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return CITY_ALIASES.get(s, s) or None

def hav(lat1, lon1, lat2, lon2):
    if None in (lat1, lon1, lat2, lon2) or any(isinstance(v, float) and math.isnan(v) for v in (lat1, lon1, lat2, lon2)):
        return float("inf")
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(min(1.0, a)))

STOPN = {"hospital", "hospitals", "clinic", "clinics", "centre", "center", "multi", "multispeciality",
         "multispecialty", "speciality", "specialty", "super", "care", "health", "healthcare", "medical",
         "research", "surgical", "nursing", "home", "and", "the", "of", "dr", "pvt", "ltd", "institute", "limited"}
HEALTH_TYPES = {"hospital", "doctor", "health", "dentist", "medical_lab", "physiotherapist", "dental_clinic",
                "medical_clinic", "general_hospital", "pharmacy"}

def toks(s):
    return [t for t in re.findall(r"[a-z0-9]+", str(s).lower()) if t not in STOPN and len(t) > 1]

def name_sim(a, b):
    ta, tb = set(toks(a)), set(toks(b))
    if not ta or not tb:
        return difflib.SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()
    jac = len(ta & tb) / len(ta | tb)
    seq = difflib.SequenceMatcher(None, " ".join(sorted(ta)), " ".join(sorted(tb))).ratio()
    contain = len(ta & tb) / min(len(ta), len(tb))
    return max(jac, seq, 0.9 * contain)

def d10(p):
    d = re.sub(r"\D", "", str(p or ""))
    return d[-10:] if len(d) >= 10 else None

def domain(u):
    m = re.match(r"^(?:https?://)?(?:www\.)?([^/?#]+)", str(u or "").strip().lower())
    return m.group(1) if m else None

def parse_place(p):
    comps = p.get("addressComponents", []) or []
    def comp(*types):
        for t in types:
            for c in comps:
                if t in c.get("types", []):
                    return c.get("longText")
        return None
    loc = p.get("location") or {}
    reviews = []
    for rv in p.get("reviews", []) or []:
        txt = (rv.get("text") or {}).get("text") or (rv.get("originalText") or {}).get("text")
        if txt:
            reviews.append({"rating": rv.get("rating"), "text": txt, "when": rv.get("relativePublishTimeDescription")})
    return {
        "g_place_id": p.get("id"), "g_name": (p.get("displayName") or {}).get("text"),
        "g_address": p.get("formattedAddress"), "g_lat": loc.get("latitude"), "g_lon": loc.get("longitude"),
        "g_postcode": comp("postal_code"), "g_city": comp("locality", "administrative_area_level_3", "sublocality"),
        "g_district": comp("administrative_area_level_3", "administrative_area_level_2"),
        "g_state": comp("administrative_area_level_1"), "g_status": p.get("businessStatus"),
        "g_types": p.get("types", []), "g_phone": p.get("internationalPhoneNumber") or p.get("nationalPhoneNumber"),
        "g_website": p.get("websiteUri"), "g_rating": p.get("rating"), "g_reviews_count": p.get("userRatingCount"),
        "g_maps_url": p.get("googleMapsUri"), "g_reviews": reviews,
    }

def best_match(r):
    txt = RESP.get(r["unique_id"])
    if txt is None:
        return {"google_match": "not checked"}
    try:
        places = json.loads(txt).get("places", []) or []
    except Exception:
        return {"google_match": "error"}
    if not places:
        return {"google_match": "not found"}
    ours10 = {d10(x) for x in r["phones_all"]} - {None}
    city_k = ckey(r["city"])
    refs = [(r["loc_lat"], r["loc_lon"]), (r["pin_lat"], r["pin_lon"]), (r["city_lat"], r["city_lon"]),
            (r["latitude_raw"], r["longitude_raw"])]
    best = None
    for p in places:
        g = parse_place(p)
        if g["g_lat"] is None:
            continue
        sim = name_sim(r["name"], g["g_name"])
        phone_match = d10(g["g_phone"]) in ours10 if g["g_phone"] else False
        pc_match = bool(g["g_postcode"]) and g["g_postcode"] == r["pin"]
        city_match = bool(city_k) and city_k in {ckey(g["g_city"]), ckey(g["g_district"])}
        d_ref = min(hav(a, b, g["g_lat"], g["g_lon"]) for a, b in refs)
        health = bool(HEALTH_TYPES & set(g["g_types"]))
        strong = (sim >= 0.75 and (d_ref <= 15 or pc_match or city_match)) or (phone_match and sim >= 0.4)
        weak = (not strong) and sim >= 0.55 and d_ref <= 40
        score = (sim + (0.5 if phone_match else 0) + (0.3 if pc_match else 0) + (0.2 if city_match else 0)
                 + (0.1 if health else 0) - min(d_ref, 100) / 200)
        cand = {**g, "g_name_sim": round(sim, 2), "g_phone_match": phone_match, "g_postcode_match": pc_match,
                "g_city_match": city_match, "g_dist_ref_km": round(d_ref, 1) if d_ref != float("inf") else None,
                "google_match": "strong" if strong else ("weak" if weak else "no match"), "_key": (strong, weak, score)}
        if best is None or cand["_key"] > best["_key"]:
            best = cand
    if best is None:
        return {"google_match": "not found"}
    best.pop("_key")
    if best["google_match"] == "no match":
        return {"google_match": "not found", "g_best_guess": best["g_name"]}
    return best

matches = pd.DataFrame([{"unique_id": r["unique_id"], **best_match(r)} for r in fac.to_dict("records")])
# a few unique_ids repeat in the source data -> keep one match per id so the join can't duplicate rows
matches = matches.drop_duplicates("unique_id")

# =====================================================================
# 4. APPLY: confirm / correct, keeping originals
# =====================================================================
vf = fac.merge(matches, on="unique_id", how="left")
STATE_FIX = {"Delhi": "Delhi", "Odisha": "Odisha", "Orissa": "Odisha", "Pondicherry": "Puducherry"}

def norm_phone(s):
    d = re.sub(r"\D", "", str(s or ""))
    if len(d) == 12 and d.startswith("91"):
        d = d[2:]
    elif len(d) == 11 and d.startswith("0"):
        d = d[1:]
    return "+91" + d if len(d) == 10 and d[0] != "0" else None

def apply(r):
    out = {}
    gm = r.get("google_match")
    for c in ["loc_lat", "loc_lon", "loc_confidence", "loc_note", "pin", "city", "district", "state",
              "phone_best", "website"]:
        out[c + "_orig"] = r[c]
    out.update({c: r[c] for c in ["loc_lat", "loc_lon", "loc_confidence", "loc_precision_km", "loc_note", "loc_method",
                                  "pin", "city", "district", "state", "phone_best", "phone_flag", "website",
                                  "include_in_search", "exclude_reason"]})
    out["aliases"], out["checks"], out["google_note"] = [], [], None
    if gm not in ("strong", "weak"):
        if gm == "not found":
            out["google_note"] = "Could not be matched to a Google Maps listing"
        return out

    d_move = hav(r["loc_lat"], r["loc_lon"], r["g_lat"], r["g_lon"])
    g_phone = norm_phone(r["g_phone"])
    g_web = domain(r["g_website"])
    out["checks"] = [
        ["Found on Google Maps" + (" (strong match)" if gm == "strong" else " (possible match)"), True],
        ["Open (" + str(r["g_status"]).replace("_", " ").lower() + ")", r["g_status"] == "OPERATIONAL"],
        [f"Location matches ours ({d_move:.1f} km apart)", d_move <= 5],
        ["Phone matches", bool(r["g_phone_match"])] if g_phone else ["Phone matches", None],
        ["Postcode matches", bool(r["g_postcode_match"])] if r["g_postcode"] else ["Postcode matches", None],
        ["Website matches", (g_web == r["website"]) if (g_web and r["website"]) else None],
    ]
    if r["g_name"] and r["g_name_sim"] < 0.95:
        out["aliases"] = [r["g_name"]]

    # ---- location / address classification
    use_google_loc = gm == "strong" or r["loc_confidence"] in ("low", "medium", "none")
    if use_google_loc:
        out["loc_lat"], out["loc_lon"] = r["g_lat"], r["g_lon"]
        out["loc_precision_km"] = 0.2
        out["loc_method"] = "google_maps"
        out["loc_confidence"] = "high" if gm == "strong" else "medium"
        moved = f"; our data placed it {d_move:.0f} km away" if d_move > 2 else "; matches our location"
        out["loc_note"] = f"Verified on Google Maps as '{r['g_name']}' ({r['g_address']}){moved}"
        if r["g_postcode"]:
            out["pin"] = r["g_postcode"]
        if r["g_city"]:
            out["city"] = r["g_city"]
        if r["g_district"]:
            out["district"] = r["g_district"]
        if r["g_state"]:
            out["state"] = STATE_FIX.get(r["g_state"], r["g_state"].title())

    # ---- phone / website (only from strong matches)
    if gm == "strong":
        if g_phone and (not r["phone_best"] or r["phone_flag"]):
            out["phone_best"], out["phone_flag"] = g_phone, None
        elif g_phone and not r["g_phone_match"]:
            out["phone_best"] = g_phone
            out["phone_flag"] = f"Google Maps lists {g_phone}; our record had {r['phone_best']}"
        if g_web and not r["website"]:
            out["website"] = g_web

    # ---- closed
    if gm == "strong" and r["g_status"] == "CLOSED_PERMANENTLY":
        out["include_in_search"], out["exclude_reason"] = False, "Permanently closed (Google Maps)"
    elif r["g_status"] == "CLOSED_TEMPORARILY":
        out["google_note"] = "Google Maps marks this facility as temporarily closed"
    return out

applied = pd.DataFrame([apply(r) for r in vf.to_dict("records")])
for c in applied.columns:
    vf[c] = applied[c].values
vf = vf.drop(columns=[c for c in ["prio"] if c in vf.columns])

# =====================================================================
# 5. SAVE + REPORT
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
            out[c] = out[c].map(lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, tuple, dict))
                                else (None if v is None or (isinstance(v, float) and math.isnan(v)) else str(v)))
            fields.append(StructField(c, StringType()))
    return spark.createDataFrame(out.astype(object).values.tolist(), StructType(fields))

to_spark(vf).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{OUT}.facilities_verified")
print(f"Saved {OUT}.facilities_verified ({len(vf):,} rows)")

chk = vf[vf["google_match"].isin(["strong", "weak", "not found", "error"])]
print("\nGOOGLE VERIFICATION REPORT")
print(f"  checked so far:                 {len(chk):,} of {len(todo_all):,} searchable facilities")
for k, v in chk["google_match"].value_counts().items():
    print(f"  match {k:12s}             {v:,}")
m = chk[chk["google_match"].isin(["strong", "weak"])]
if len(m):
    moved = m.apply(lambda r: hav(r["loc_lat_orig"], r["loc_lon_orig"], r["g_lat"], r["g_lon"]), axis=1)
    print(f"  location confirmed (<5 km):     {(moved <= 5).sum():,}")
    print(f"  location corrected (>5 km):     {(moved > 5).sum():,}")
    print(f"  postcode corrected:             {(m['pin'] != m['pin_orig']).sum():,}")
    print(f"  phone matched Google:           {(m['g_phone_match'] == True).sum():,}")
    print(f"  permanently closed (excluded):  {(m['exclude_reason'] == 'Permanently closed (Google Maps)').sum():,}")
    print(f"  with Google reviews saved:      {(m['g_reviews'].map(lambda x: len(x) if isinstance(x, list) else 0) > 0).sum():,}")
    print("\nBiggest corrections (check a few by hand):")
    m = m.assign(moved_km=moved.round(0))
    display(spark.createDataFrame(
        m.sort_values("moved_km", ascending=False).head(15)[
            ["name", "city_orig", "pin_orig", "g_name", "g_address", "moved_km", "google_match"]].astype(str)))
