# Databricks notebook source
# =====================================================================
#  NOTEBOOK 5 - WEBSITE CRAWLER: bed and doctor counts from hospitals' OWN websites
#
#  For every facility with an official website:
#   - fetch the homepage + up to 4 relevant pages (about / infrastructure / facilities / our doctors)
#   - respect robots.txt, ~1 request per site at a time with pauses, clear bot name, 2 MB page limit
#   - extract   "200 beds", "200-bedded", "bed capacity of 200"      -> total beds
#               "30 ICU beds", "10-bed NICU"                          -> unit beds (NOT total)
#               "50+ doctors", "40 consultants"                       -> doctors
#               number of distinct "Dr. ..." names on a doctors page  -> doctor profiles
#   - CHAIN GUARD: if several facilities share one website (Apollo, Fortis ...), a number is only used
#     when the page or sentence mentions that facility's city; otherwise it's a group total -> ignored
#   - every value keeps the SENTENCE and URL it came from (so the app can cite it)
#
#  Resumable: results are saved after every batch; already-crawled websites are skipped.
#  Free: plain web requests, no API keys.
#
#  Output: workspace.default.website_crawl_raw   one row per website (everything found)
#          workspace.default.website_size        one row per facility (beds / doctors + source sentence)
# =====================================================================
import re, json, time, html, datetime, warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin, urlparse
from urllib import robotparser
import requests
import numpy as np
import pandas as pd
from pyspark.sql.types import StructType, StructField, StringType

warnings.filterwarnings("ignore")
requests.packages.urllib3.disable_warnings()

OUT = "workspace.default"
RAW = f"{OUT}.website_crawl_raw"
dbutils.widgets.text("limit", "200")          # websites per run; 0 = all remaining
LIMIT = int(dbutils.widgets.get("limit") or 0)
UA = "CareFinderResearchBot/1.0 (student datathon project; reads public hospital capacity info)"
HDR = {"User-Agent": UA, "Accept": "text/html,application/xhtml+xml", "Accept-Language": "en"}

# =====================================================================
# 1. WHICH WEBSITES
# =====================================================================
SRC = (f"{OUT}.facilities_verified" if spark.catalog.tableExists(f"{OUT}.facilities_verified")
       else f"{OUT}.facilities_clean")
fac = spark.table(SRC).toPandas().drop_duplicates("unique_id")
fac["beds"] = pd.to_numeric(fac["beds"], errors="coerce")
fac["doctors"] = pd.to_numeric(fac["doctors"], errors="coerce")

SKIP_DOMAINS = re.compile(r"justdial|practo|facebook|instagram|linkedin|twitter|youtube|google|goo\.gl|bit\.ly|"
                          r"wa\.me|wix|blogspot|wordpress\.com|sulekha|lybrate|indiamart|linktr|business\.site", re.I)
fac["domain"] = fac["website"].fillna("").str.lower().str.replace(r"^www\.", "", regex=True)
# Facilities that already list BOTH beds and doctors don't need the crawler -> drop them entirely
already_complete = fac["beds"].notna() & fac["doctors"].notna()
print(f"Skipping {already_complete.sum():,} facilities that already list both beds and doctors")
sites = fac[(fac["domain"].str.len() > 3) & ~fac["domain"].str.contains(SKIP_DOMAINS) & ~already_complete]
site_facs = sites.groupby("domain").apply(
    lambda g: g[["unique_id", "name", "city"]].to_dict("records")).to_dict()
print(f"{len(fac):,} facilities | {len(site_facs):,} distinct official websites")

done = set()
if spark.catalog.tableExists(RAW):
    done = set(spark.table(RAW).select("domain").toPandas()["domain"])
todo = [d for d in site_facs if d not in done]
# websites whose facilities are missing BOTH numbers first, then those missing one
missing = fac.set_index("unique_id")[["beds", "doctors"]].isna().sum(axis=1).to_dict()
todo.sort(key=lambda d: -max(missing.get(f["unique_id"], 0) for f in site_facs[d]))
if LIMIT > 0:
    todo = todo[:LIMIT]
print(f"{len(done):,} already crawled | crawling {len(todo):,} now")

# =====================================================================
# 2. FETCHING (polite)
# =====================================================================
def get(url):
    """Returns (html_text, final_url) or (None, url). Max 2 MB, HTML only."""
    for verify in (True, False):                       # many small hospital sites have broken certificates
        try:
            r = requests.get(url, headers=HDR, timeout=(6, 12), allow_redirects=True, stream=True, verify=verify)
            if r.status_code != 200 or "html" not in r.headers.get("Content-Type", "").lower():
                r.close()
                return None, url
            buf = b""
            for chunk in r.iter_content(65536):
                buf += chunk
                if len(buf) > 2_000_000:
                    break
            enc = r.encoding or "utf-8"
            final = r.url
            r.close()
            return buf.decode(enc, errors="ignore"), final
        except requests.exceptions.SSLError:
            continue
        except Exception:
            return None, url
    return None, url

def robots_for(root):
    try:
        r = requests.get(root + "/robots.txt", headers=HDR, timeout=5, verify=False)
        if r.status_code == 200 and len(r.text) < 200_000:
            rp = robotparser.RobotFileParser()
            rp.parse(r.text.splitlines())
            return rp
    except Exception:
        pass
    return None                                        # no robots.txt -> allowed

def html_to_text(h):
    h = re.sub(r"(?is)<(script|style|noscript|svg)[^>]*>.*?</\1>", " ", h)
    h = re.sub(r"(?s)<[^>]+>", " ", h)
    return re.sub(r"\s+", " ", html.unescape(h)).strip()

LINK_KEYS = [("doctor", 3), ("consultant", 3), ("specialist", 2), ("our-team", 2), ("team", 1), ("about", 3),
             ("infrastructure", 3), ("facilit", 2), ("overview", 1), ("who-we-are", 2), ("profile", 1)]
DOC_PAGE = re.compile(r"doctor|consultant|specialist|team|physician|faculty", re.I)

def pick_links(h, final, host):
    scored = {}
    for href in re.findall(r"""href\s*=\s*["']([^"'#>]+)""", h, re.I):
        u = urljoin(final, href.strip())
        p = urlparse(u)
        if p.scheme not in ("http", "https") or p.netloc.lower().replace("www.", "") != host:
            continue
        if re.search(r"\.(pdf|jpe?g|png|gif|webp|zip|docx?|xlsx?|mp4)$", p.path, re.I):
            continue
        score = sum(w for k, w in LINK_KEYS if k in p.path.lower())
        if score:
            key = u.split("?")[0].rstrip("/")
            scored[key] = max(scored.get(key, 0), score)
    links = sorted(scored, key=lambda k: -scored[k])
    if not links:
        root = f"{urlparse(final).scheme}://{urlparse(final).netloc}"
        links = [root + p for p in ("/about-us", "/about", "/infrastructure", "/our-doctors")]
    return links[:4]

# =====================================================================
# 3. EXTRACTION
# =====================================================================
BED_RX = [re.compile(r"\b(\d{1,2},\d{3}|\d{1,4})\s*\+?\s*[- ]?\s*(?:bedded|beds?)\b", re.I),
          re.compile(r"\bbeds?\s*(?:capacity|strength)\s*(?:of|:|is|-)?\s*(\d{1,2},\d{3}|\d{1,4})\b", re.I)]
UNIT_RX = re.compile(r"\b(icu|nicu|picu|iccu|ccu|hdu|sicu|micu|dialysis|ward|isolation|emergency|casualty|"
                     r"day ?care|labour|labor|recovery|private|suite|deluxe|general|semi|ot|operation|"
                     r"chemo|burn|maternity|paediatric|pediatric)\b", re.I)
ICU_RX = re.compile(r"\b(icu|nicu|picu|iccu|ccu|hdu|sicu|micu|critical care|intensive)\b", re.I)
GROUP_RX = re.compile(r"\b(across|network|group|all (?:our|its) (?:hospitals|centres|centers|units)|"
                      r"\d+ (?:hospitals|centres|centers|locations|branches))\b", re.I)
DOC_RX = re.compile(r"\b(\d{1,4})\s*\+?\s*(?:experienced |expert |qualified |senior |eminent |renowned |highly |"
                    r"skilled |dedicated |specialist |super[- ]?specialist |full[- ]time |visiting )*"
                    r"(?:doctors|consultants|specialists|physicians|surgeons|medical professionals)\b", re.I)
NAME_RX = re.compile(r"\bDr\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]*\.?){0,2}")

def snippet(t, m, w=110):
    a, b = max(0, m.start() - w), min(len(t), m.end() + w)
    return ("…" if a else "") + t[a:b].strip() + ("…" if b < len(t) else "")

def bed_mentions(t, url):
    out = []
    for rx in BED_RX:
        for m in rx.finditer(t):
            v = int(m.group(1).replace(",", ""))
            if not 5 <= v <= 5000:
                continue
            # only the words right next to the number decide: "ICU with 20 beds", "10-bed NICU"
            pre, post = t[max(0, m.start() - 15): m.start()], t[m.end(): m.end() + 8]
            unit = bool(UNIT_RX.search(pre) or UNIT_RX.search(post))
            icu = bool(ICU_RX.search(pre) or ICU_RX.search(post))
            group = bool(GROUP_RX.search(t[m.end(): m.end() + 25]))      # "1,500 beds across 12 hospitals"
            out.append({"v": v, "unit": unit or group, "icu": icu, "snip": snippet(t, m), "url": url})
    return out

def doc_mentions(t, url):
    out = []
    for m in DOC_RX.finditer(t):
        v = int(m.group(1))
        win = t[max(0, m.start() - 25): m.end() + 25].lower()
        if 2 <= v <= 3000 and not re.search(r"year|lakh|crore|patient|bed", win):
            out.append({"v": v, "snip": snippet(t, m), "url": url})
    return out

def crawl_site(dom):
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    res = {"domain": dom, "status": "unreachable", "pages": [], "beds": [], "doctors": [], "profiles": [],
           "crawled_at": now}
    home, final = None, None
    for base in (f"https://{dom}/", f"https://www.{dom}/", f"http://{dom}/"):
        home, final = get(base)
        if home:
            break
    if not home:
        return res
    root = f"{urlparse(final).scheme}://{urlparse(final).netloc}"
    host = urlparse(final).netloc.lower().replace("www.", "")
    rp = robots_for(root)
    if rp and not rp.can_fetch(UA, final):
        res["status"] = "blocked by robots.txt"
        return res
    res["status"] = "ok"
    pages = [(final, home)]
    for u in pick_links(home, final, host):
        if rp and not rp.can_fetch(UA, u):
            continue
        time.sleep(0.6)
        h, f = get(u)
        if h:
            pages.append((f, h))
    for url, h in pages:
        t = html_to_text(h)
        res["pages"].append(url)
        res["beds"] += bed_mentions(t, url)
        res["doctors"] += doc_mentions(t, url)
        if DOC_PAGE.search(urlparse(url).path):
            names = {m.group(0) for m in NAME_RX.finditer(t)}
            if len(names) >= 3:
                res["profiles"].append({"v": len(names), "url": url})
    return res

# =====================================================================
# 4. CRAWL (batches of 300 websites, saved after each batch)
# =====================================================================
RAW_SCHEMA = StructType([StructField(c, StringType()) for c in
                         ["domain", "status", "pages", "beds", "doctors", "profiles", "crawled_at"]])

# ---- live log: one panel updated in place, showing only the LAST 5 websites
from collections import deque
from concurrent.futures import as_completed
from IPython.display import display as ipy_display, HTML
LOG = deque(maxlen=5)
stats = {"done": 0, "ok": 0, "beds": 0, "docs": 0, "saved": 0}
try:
    panel = ipy_display(HTML("<pre>starting crawler…</pre>"), display_id=True)
except Exception:
    panel = None

def log_line(r):
    if r["status"] != "ok":
        return f"✗ {r['domain']}  ({r['status']})"
    tb = [b for b in r["beds"] if not b["unit"]]
    beds = f"beds {Counter(b['v'] for b in tb).most_common(1)[0][0]}" if tb else "no bed count"
    if r["doctors"]:
        docs = f"doctors {Counter(d['v'] for d in r['doctors']).most_common(1)[0][0]}"
    elif r["profiles"]:
        docs = f"{max(p['v'] for p in r['profiles'])} doctor profiles"
    else:
        docs = "no doctor count"
    line = f"✓ {r['domain']}  | {len(r['pages'])} pages | {beds} | {docs}"
    if tb:
        line += "\n      “" + tb[0]["snip"][:150] + "”"
    return line

def refresh(total, t0):
    mins = (time.time() - t0) / 60
    eta = mins / max(stats["done"], 1) * (total - stats["done"])
    head = (f"Crawled {stats['done']:,}/{total:,} websites  ·  reachable {stats['ok']:,}  ·  bed count found "
            f"{stats['beds']:,}  ·  doctor count found {stats['docs']:,}  ·  saved {stats['saved']:,}  ·  "
            f"{mins:.1f} min elapsed, ~{eta:.0f} min left")
    body = "\n".join(LOG)
    if panel is not None:
        panel.update(HTML("<pre style='font-size:12px;line-height:1.35'>" + html.escape(head) + "\n"
                          + "─" * 100 + "\n" + html.escape(body) + "</pre>"))
    elif stats["done"] % 25 == 0:
        print(head)

t0 = time.time()
for s in range(0, len(todo), 300):
    batch = todo[s:s + 300]
    res = []
    with ThreadPoolExecutor(max_workers=24) as ex:
        for fut in as_completed([ex.submit(crawl_site, d) for d in batch]):
            r = fut.result()
            res.append(r)
            stats["done"] += 1
            stats["ok"] += r["status"] == "ok"
            stats["beds"] += any(not b["unit"] for b in r["beds"])
            stats["docs"] += bool(r["doctors"] or r["profiles"])
            LOG.append(log_line(r))
            refresh(len(todo), t0)
    rows = [(r["domain"], r["status"], json.dumps(r["pages"]), json.dumps(r["beds"], ensure_ascii=False),
             json.dumps(r["doctors"], ensure_ascii=False), json.dumps(r["profiles"]), r["crawled_at"]) for r in res]
    spark.createDataFrame(rows, RAW_SCHEMA).write.mode("append").saveAsTable(RAW)
    stats["saved"] += len(rows)
    refresh(len(todo), t0)
print(f"Crawl finished: {stats['done']:,} websites in {(time.time() - t0) / 60:.1f} min")

# =====================================================================
# 5. PER-FACILITY VALUES (with chain guard) from ALL crawled websites
# =====================================================================
raw = spark.table(RAW).toPandas().sort_values("crawled_at").drop_duplicates("domain", keep="last")
for c in ["pages", "beds", "doctors", "profiles"]:
    raw[c] = raw[c].map(lambda s: json.loads(s) if s else [])
RAWD = raw.set_index("domain").to_dict("index")

def ckey(s):
    return re.sub(r"[^a-z]", "", str(s or "").lower())

def pick(mentions, city, shared):
    """Most-repeated value; on shared (chain) websites only mentions tied to this facility's city count."""
    if shared:
        ck = ckey(city)
        mentions = [m for m in mentions if ck and len(ck) > 3 and (ck in ckey(m["url"]) or ck in ckey(m["snip"]))]
    if not mentions:
        return None
    cnt = Counter(m["v"] for m in mentions)
    best_v = max(cnt, key=lambda v: (cnt[v], v))
    return next(m for m in mentions if m["v"] == best_v)

rows = []
for dom, flist in site_facs.items():
    r = RAWD.get(dom)
    if r is None:
        continue
    shared = len(flist) >= 2
    for f in flist:
        out = {"unique_id": f["unique_id"], "domain": dom, "crawl_status": r["status"],
               "shared_website": shared, "pages_read": len(r["pages"])}
        b = pick([m for m in r["beds"] if not m["unit"]], f["city"], shared)
        icu = pick([m for m in r["beds"] if m["icu"]], f["city"], shared)
        d = pick(r["doctors"], f["city"], shared)
        prof = max(r["profiles"], key=lambda p: p["v"]) if (r["profiles"] and not shared) else None
        out.update({
            "beds_web": b["v"] if b else None, "beds_web_snippet": b["snip"] if b else None,
            "beds_web_url": b["url"] if b else None,
            "icu_beds_web": icu["v"] if icu else None,
            "doctors_web": d["v"] if d else (prof["v"] if prof else None),
            "doctors_web_source": "stated on website" if d else ("doctor profiles counted" if prof else None),
            "doctors_web_snippet": d["snip"] if d else None,
            "doctors_web_url": d["url"] if d else (prof["url"] if prof else None),
            "group_number_ignored": bool(shared and not b and any(not m["unit"] for m in r["beds"])),
        })
        rows.append(out)
ws = pd.DataFrame(rows)

# ---- RULE: the dataset always wins. A website value only FILLS a missing number, never replaces one.
ws = ws.merge(fac[["unique_id", "beds", "doctors"]].rename(columns={"beds": "beds_dataset", "doctors": "doctors_dataset"}),
              on="unique_id", how="left")
ws["beds_fill"] = np.where(ws["beds_dataset"].isna(), ws["beds_web"], np.nan)          # used downstream
ws["doctors_fill"] = np.where(ws["doctors_dataset"].isna(), ws["doctors_web"], np.nan)  # used downstream
# record big disagreements (kept for review only; the dataset value is still the one used)
for col in ["beds", "doctors"]:
    a, b = ws[f"{col}_dataset"], ws[f"{col}_web"]
    ws[f"{col}_web_disagrees"] = a.notna() & b.notna() & ((b - a).abs() / a > 0.5)

spark.createDataFrame(ws.astype(object).where(ws.notna(), None)).write.mode("overwrite") \
    .option("overwriteSchema", "true").saveAsTable(f"{OUT}.website_size")

# =====================================================================
# 6. REPORT + VALIDATION against facilities that already list beds/doctors
# =====================================================================
print("\n" + "=" * 90 + "\nCRAWL REPORT (all runs so far)")
print(f"  websites crawled:          {len(raw):,} of {len(site_facs):,}")
print("  status:                    " + ", ".join(f"{k}: {v:,}" for k, v in raw["status"].value_counts().items()))
print(f"  facilities with beds from website:    {ws['beds_web'].notna().sum():,}")
print(f"  facilities with doctors from website: {ws['doctors_web'].notna().sum():,} "
      f"(stated: {(ws['doctors_web_source'] == 'stated on website').sum():,}, "
      f"profiles counted: {(ws['doctors_web_source'] == 'doctor profiles counted').sum():,})")
print(f"  group-level numbers ignored (chain sites): {ws['group_number_ignored'].sum():,}")

for col in ["beds", "doctors"]:
    both = ws[ws[f"{col}_dataset"].notna() & ws[f"{col}_web"].notna()]
    if len(both):
        diff = (both[f"{col}_web"] - both[f"{col}_dataset"]).abs() / both[f"{col}_dataset"]
        print(f"\n  CHECK {col.upper()}: {len(both):,} facilities have BOTH a dataset and a website value "
              f"(dataset value kept)")
        print(f"    exactly equal: {(both[f'{col}_web'] == both[f'{col}_dataset']).mean():.0%} | "
              f"within ±20%: {(diff <= 0.2).mean():.0%} | big disagreement (>50%): {(diff > 0.5).sum():,}")
for col in ["beds", "doctors"]:
    before = fac[col].notna().sum()
    gained = ws[f"{col}_fill"].notna().sum()
    print(f"\n  {col.upper()}: {gained:,} missing values FILLED from websites "
          f"-> facilities with a real {col} count: {before:,} -> {before + gained:,}")
print("\nExamples:")
display(spark.createDataFrame(
    ws[ws["beds_web"].notna()].head(10)[["domain", "beds_web", "beds_web_snippet", "beds_web_url"]].astype(str)))
