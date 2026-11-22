# Databricks notebook source
# =====================================================================
#  NOTEBOOK 4 - PREDICTING HOSPITAL SIZE (beds, doctors) WITH MACHINE LEARNING
#
#  TARGET (real numbers, priority order; the dataset always wins):
#     dataset field  ->  number stated in the dataset's description  ->  hospital's own website (Notebook 5)
#
#  FEATURES (selected by the team)
#   District / state context
#     1. district population (Census 2011)
#     2. state income per person (NSDP per capita)
#     3. district wealth index (NFHS-5: electricity, clean fuel, sanitation, drinking water, health insurance)
#     4. district women's literacy (NFHS-5)
#     5. distance to the state capital and to the nearest major city
#   The facility itself
#     6. services offered (ICU, NICU, MRI, CT, cath lab, dialysis, transplant, teaching, accreditation ...)
#     7. breadth: number of specialties / procedures / equipment / capabilities listed
#     8. ownership (public / private) and facility type
#     9. the other count (beds model uses doctors, doctors model uses beds) when known
#    10. description text -> 40 NLP topics (numbers and bed/doctor phrases REMOVED -> no leakage)
#
#  MODELS: naive | Ridge | Random forest | Extra trees | Gradient boosting | Ensemble, 5-fold cross-validation;
#          best chosen automatically; quantile models + conformal calibration give an honest 80% range.
#  A live panel shows the training stage and % complete. Everything is logged to MLflow.
#
#  Output: facility_size, size_model_metrics, size_model_comparison, size_feature_importance
# =====================================================================
import io, json, re, math, time, warnings
from collections import deque
import numpy as np
import pandas as pd
import requests
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.inspection import permutation_importance
from scipy import sparse
try:
    import lightgbm as lgb
except ImportError:
    lgb = None                 # falls back to sklearn's HistGradientBoostingClassifier
try:
    import mlflow, mlflow.sklearn
except ImportError:
    mlflow = None
from IPython.display import display as ipy_display, HTML

# ---- BED SIZE: chosen after 17 local experiments (see /experiments). Two classes split at BED_CUT beds,
#      shown to users only for the SHOW_SHARE most-confident facilities.
BED_CUT = 50           # "under 50 beds" vs "50+ beds"  (team choice; 100 was also tested: balanced classes)
SHOW_SHARE = 0.80      # show a predicted size class for the 80% most confident facilities

OUT = "workspace.default"
SRCDB = "databricks_virtue_foundation_dataset_dais_2026.virtue_foundation_dataset"
UA = {"User-Agent": "CareFinderResearchBot/1.0 (student datathon project)"}

# =====================================================================
# LIVE PROGRESS PANEL
# =====================================================================
class Progress:
    def __init__(self, total):
        self.total, self.done, self.t0 = total, 0, time.time()
        self.stage, self.log = "starting", deque(maxlen=6)
        try:
            self.h = ipy_display(HTML("<pre>starting…</pre>"), display_id=True)
        except Exception:
            self.h = None

    def stage_(self, name):
        self.stage = name
        self.render()

    def step(self, msg, n=1):
        self.done = min(self.total, self.done + n)
        self.log.append(f"{time.strftime('%H:%M:%S')}  {msg}")
        self.render()

    def render(self):
        pct = self.done / self.total
        el = time.time() - self.t0
        eta = el / pct - el if pct > 0.02 else float("nan")
        bar = "█" * int(pct * 40) + "░" * (40 - int(pct * 40))
        txt = (f"STAGE: {self.stage}\n[{bar}] {pct:5.1%}   step {self.done}/{self.total}   "
               f"{el / 60:.1f} min elapsed" + (f", ~{eta / 60:.1f} min left" if eta == eta else "") +
               "\n" + "─" * 90 + "\n" + "\n".join(self.log))
        if self.h is not None:
            self.h.update(HTML("<pre style='font-size:12px;line-height:1.4'>" +
                               txt.replace("&", "&amp;").replace("<", "&lt;") + "</pre>"))
        else:
            print(txt.splitlines()[1])

N_FOLDS = 5
FITS_PER_FOLD = 6                                   # 4 models + 2 quantile models
P = Progress(total=8 + (N_FOLDS * 2 + 5) + 2 * (N_FOLDS * FITS_PER_FOLD + 6))   # features + beds clf + beds range + doctors

# =====================================================================
# 1. LOAD
# =====================================================================
P.stage_("1/6 loading data")
SRC = (f"{OUT}.facilities_verified" if spark.catalog.tableExists(f"{OUT}.facilities_verified")
       else f"{OUT}.facilities_clean")
fac = spark.table(SRC).toPandas().drop_duplicates("unique_id").reset_index(drop=True)
if spark.catalog.tableExists(f"{OUT}.website_size"):
    web = spark.table(f"{OUT}.website_size").toPandas().drop_duplicates("unique_id")
    fac = fac.merge(web[["unique_id", "beds_fill", "doctors_fill"]], on="unique_id", how="left")
else:
    fac["beds_fill"], fac["doctors_fill"] = np.nan, np.nan
for c in ["specialties_list", "procedure_list", "equipment_list", "capability_list"]:
    fac[c] = fac[c].map(lambda s: json.loads(s) if isinstance(s, str) and s.startswith("[") else [])
for c in ["doctors", "beds", "loc_lat", "loc_lon", "beds_fill", "doctors_fill"]:
    fac[c] = pd.to_numeric(fac[c], errors="coerce")
fac["description"] = fac["description"].fillna("")
fac["text"] = fac.apply(lambda r: " | ".join(r["specialties_list"] + r["procedure_list"] + r["equipment_list"]
                                             + r["capability_list"] + [r["description"]]).lower(), axis=1)
N = len(fac)
P.step(f"{N:,} facilities loaded from {SRC}")

# =====================================================================
# 2. TARGETS
# =====================================================================
P.stage_("2/6 building targets (real bed / doctor counts)")
BED_RE = re.compile(r"(\d{2,4})\s*\+?\s*[- ]?(?:bed(?:s|ded)?)\b")
DOC_RE = re.compile(r"(\d{1,4})\s*\+?\s*(?:doctors|physicians|consultants|specialist doctors)\b")

def from_text(t, rx, lo, hi):
    vals = [int(m.group(1)) for m in rx.finditer(t) if lo <= int(m.group(1)) <= hi]
    return float(max(vals)) if vals else np.nan

fac["beds_text"] = fac["text"].map(lambda t: from_text(t, BED_RE, 5, 5000))
fac["doctors_text"] = fac["text"].map(lambda t: from_text(t, DOC_RE, 2, 5000))
beds_known = fac["beds"].fillna(fac["beds_text"]).fillna(fac["beds_fill"])
docs_known = fac["doctors"].fillna(fac["doctors_text"]).fillna(fac["doctors_fill"])
P.step(f"real bed counts {beds_known.notna().sum():,} | real doctor counts {docs_known.notna().sum():,}")

# =====================================================================
# 3. FEATURES
# =====================================================================
P.stage_("3/6 building features")
X = pd.DataFrame(index=fac.index)

def norm(s):
    return re.sub(r"[^a-z]", "", str(s or "").lower().replace("&", "and"))

# ---- (6) services offered
FLAGS = {
    "icu": r"\bicu\b|intensive care", "nicu": r"\bnicu\b|neonatal intensive", "cathlab": r"cath ?lab|angioplasty",
    "mri": r"\bmri\b", "ct": r"\bct\b|computed tomography", "bloodbank": r"blood bank", "dialysis": r"dialysis",
    "emergency": r"emergency|24\s*/\s*7|24x7|casualty", "ot": r"operation theat|operating theat|modular ot",
    "trauma": r"trauma", "cancer": r"oncolog|chemotherap|radiotherap", "ambulance": r"ambulance",
    "transplant": r"transplant", "lab": r"laborator|pathology", "ventilator": r"ventilator",
    "teaching": r"medical college|teaching hospital|\bmbbs\b|postgraduate|residency|\bdnb\b",
    "accredited": r"\bnabh\b|\bjci\b|\bnabl\b|accredit",
}
for k, rx in FLAGS.items():
    X["svc_" + k] = fac["text"].str.contains(rx, regex=True).astype(int)
X["n_services"] = X[[c for c in X.columns if c.startswith("svc_")]].sum(axis=1)

# ---- (7) breadth
X["n_specialties"] = fac["specialties_list"].map(len)
X["n_procedures"] = fac["procedure_list"].map(len)
X["n_equipment"] = fac["equipment_list"].map(len)
X["n_capability"] = fac["capability_list"].map(len)

# ---- (8) ownership and type
X["public"] = (fac["operator_type"] == "public").astype(int)
for t in ["hospital", "clinic", "dentist", "doctor", "pharmacy"]:
    X["type_" + t] = (fac["facility_type"] == t).astype(int)
P.step("services, breadth, ownership")

# ---- (5) distance to the state capital and to the nearest major city
CAPITALS = {
    "Andhra Pradesh": (16.515, 80.515), "Arunachal Pradesh": (27.084, 93.605), "Assam": (26.143, 91.790),
    "Bihar": (25.594, 85.138), "Chhattisgarh": (21.251, 81.630), "Goa": (15.491, 73.828),
    "Gujarat": (23.216, 72.637), "Haryana": (30.733, 76.779), "Himachal Pradesh": (31.105, 77.173),
    "Jharkhand": (23.344, 85.310), "Karnataka": (12.972, 77.595), "Kerala": (8.524, 76.937),
    "Madhya Pradesh": (23.260, 77.413), "Maharashtra": (19.076, 72.878), "Manipur": (24.817, 93.937),
    "Meghalaya": (25.579, 91.893), "Mizoram": (23.727, 92.718), "Nagaland": (25.675, 94.109),
    "Odisha": (20.296, 85.825), "Punjab": (30.733, 76.779), "Rajasthan": (26.912, 75.787),
    "Sikkim": (27.339, 88.607), "Tamil Nadu": (13.083, 80.271), "Telangana": (17.385, 78.487),
    "Tripura": (23.832, 91.287), "Uttar Pradesh": (26.847, 80.946), "Uttarakhand": (30.317, 78.032),
    "West Bengal": (22.573, 88.364), "Andaman And Nicobar Islands": (11.623, 92.727),
    "Chandigarh": (30.733, 76.779), "Dadra And Nagar Haveli And Daman And Diu": (20.397, 72.833),
    "Delhi": (28.614, 77.209), "Jammu And Kashmir": (34.084, 74.797), "Ladakh": (34.153, 77.577),
    "Lakshadweep": (10.567, 72.642), "Puducherry": (11.942, 79.808),
}
MAJOR_CITIES = [  # cities with ~1 million+ people
    (19.076, 72.878), (28.614, 77.209), (12.972, 77.595), (17.385, 78.487), (23.023, 72.571), (13.083, 80.271),
    (22.573, 88.364), (21.170, 72.831), (18.520, 73.857), (26.912, 75.787), (26.847, 80.946), (26.450, 80.332),
    (21.146, 79.088), (22.720, 75.858), (19.218, 72.978), (23.260, 77.413), (17.687, 83.219), (25.594, 85.138),
    (22.307, 73.181), (28.669, 77.454), (30.901, 75.857), (27.177, 78.008), (19.998, 73.790), (28.409, 77.318),
    (28.985, 77.706), (22.304, 70.802), (25.318, 82.974), (34.084, 74.797), (19.876, 75.343), (23.796, 86.430),
    (31.634, 74.872), (25.436, 81.846), (23.344, 85.310), (22.596, 88.264), (11.017, 76.956), (23.182, 79.986),
    (26.218, 78.183), (16.506, 80.648), (26.239, 73.024), (9.925, 78.120), (21.251, 81.630), (25.214, 75.865),
    (26.145, 91.736), (30.733, 76.779), (8.524, 76.937), (9.931, 76.267), (20.296, 85.825), (12.296, 76.639),
]

def hav(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))

cap = fac["state"].map(CAPITALS)
X["dist_state_capital_km"] = [hav(la, lo, c[0], c[1]) if isinstance(c, tuple) and la == la else np.nan
                              for la, lo, c in zip(fac["loc_lat"], fac["loc_lon"], cap)]
mc = np.array(MAJOR_CITIES)
X["dist_major_city_km"] = np.where(
    fac["loc_lat"].notna(),
    hav(fac["loc_lat"].values[:, None], fac["loc_lon"].values[:, None], mc[None, :, 0], mc[None, :, 1]).min(axis=1),
    np.nan)
P.step("distance to state capital / nearest major city")

# ---- (1) district population, Census 2011 (Wikipedia)
def wiki_tables(title):
    r = requests.get(f"https://en.wikipedia.org/wiki/{title}", headers=UA, timeout=30)
    return pd.read_html(io.StringIO(r.text))

def num(x):
    m = re.search(r"\d[\d,]*(?:\.\d+)?", str(x))
    return float(m.group(0).replace(",", "")) if m else np.nan

def colnames(t):
    return [" ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in t.columns]

dkey = fac["district"].map(norm)
skey_ = fac["state"].map(norm)
try:
    rows = []
    for t in wiki_tables("List_of_districts_in_India"):
        lc = [c.lower() for c in colnames(t)]
        d = [i for i, c in enumerate(lc) if "district" in c and "code" not in c]
        p = [i for i, c in enumerate(lc) if "population" in c]
        if d and p:
            for _, row in t.iterrows():
                rows.append((norm(row.iloc[d[0]]), num(row.iloc[p[0]])))
    pop = pd.DataFrame(rows, columns=["key", "population"]).dropna()
    pop = pop[(pop["key"].str.len() > 2) & ~pop["key"].duplicated(keep=False)].set_index("key")["population"]
    X["district_population_log"] = np.log1p(dkey.map(pop))
    P.step(f"district population: {len(pop):,} districts, matched {X['district_population_log'].notna().mean():.0%} "
           f"of facilities")
except Exception as e:
    P.step(f"district population SKIPPED ({str(e)[:80]})")

# ---- (2) state income per person (NSDP per capita, Wikipedia)
try:
    best = None
    for t in wiki_tables("List_of_Indian_states_and_union_territories_by_GDP_per_capita"):
        lc = [c.lower() for c in colnames(t)]
        s = [i for i, c in enumerate(lc) if "state" in c or "union territory" in c]
        v = [i for i, c in enumerate(lc) if "nsdp" in c or "per capita" in c or "₹" in c or "inr" in c]
        if s and v and len(t) >= 20:
            best = {norm(row.iloc[s[0]]): num(row.iloc[v[0]]) for _, row in t.iterrows()}
            break
    if not best:
        raise ValueError("no per-capita table found")
    best = {k: v for k, v in best.items() if v == v and v > 1000}
    X["state_income_per_capita"] = skey_.map(best)
    P.step(f"state income per capita: {len(best)} states, matched {X['state_income_per_capita'].notna().mean():.0%}")
except Exception as e:
    P.step(f"state income SKIPPED ({str(e)[:80]})")

# ---- (3) + (4) NFHS-5 district wealth index and women's literacy (dataset provided by the organisers)
try:
    nf = spark.table(f"{SRCDB}.nfhs_5_district_health_indicators").toPandas()
    cols = {c: c.lower().replace("_", " ") for c in nf.columns}   # NFHS uses snake_case
    def find(*words, exclude=()):
        for c, lc in cols.items():
            if all(w in lc for w in words) and not any(x in lc for x in exclude):
                return c
        return None
    def pct(x):                                # "*" = suppressed -> NaN, "(29.5)" = small-sample estimate
        s = str(x).strip().replace("(", "").replace(")", "")
        try:
            return float(s)
        except ValueError:
            return np.nan
    dcol = find("district", exclude=("code",))
    scol = find("state", exclude=("code",))
    wealth_cols = [c for c in [find("electricity"), find("clean fuel"), find("sanitation"), find("improved water"),
                               find("health insurance")] if c]
    lit_col = find("women", "literate") or find("literate", "women")
    z = pd.DataFrame({c: nf[c].map(pct) for c in wealth_cols})
    nf["wealth_index"] = ((z - z.mean()) / z.std()).mean(axis=1)
    nf["women_literacy"] = nf[lit_col].map(pct) if lit_col else np.nan
    nf["k_ds"] = nf[scol].map(norm) + "|" + nf[dcol].map(norm)
    nf["k_d"] = nf[dcol].map(norm)
    by_ds = nf.drop_duplicates("k_ds").set_index("k_ds")
    by_d = nf[~nf["k_d"].duplicated(keep=False)].set_index("k_d")
    for feat in ["wealth_index", "women_literacy"]:
        X["district_" + feat] = (skey_ + "|" + dkey).map(by_ds[feat]).fillna(dkey.map(by_d[feat]))
    P.step(f"NFHS-5: wealth index from {len(wealth_cols)} indicators, women's literacy "
           f"{'found' if lit_col else 'NOT found'}; matched {X['district_wealth_index'].notna().mean():.0%} of facilities")
except Exception as e:
    P.step(f"NFHS-5 features SKIPPED ({str(e)[:80]})")

# ---- (10) description text -> 40 topics, with numbers and bed/doctor phrases removed (no leakage)
clean = (fac["text"]
         .str.replace(r"\d[\d,]*\s*\+?\s*-?\s*(?:bedded|beds?|doctors?|consultants?|specialists?|physicians?)", " ",
                      regex=True)
         .str.replace(r"\b(?:bedded|beds?)\b", " ", regex=True)
         .str.replace(r"\d+", " ", regex=True))
tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, stop_words="english", sublinear_tf=True)
M_text = tfidf.fit_transform(clean)
T = TruncatedSVD(n_components=100, random_state=42).fit_transform(M_text)
for i in range(T.shape[1]):
    X[f"text_topic_{i:02d}"] = T[:, i]
# facility-name words ("medical college", "clinic", "nursing home" ...) -> 15 topics
tf_name = TfidfVectorizer(max_features=3000, ngram_range=(1, 3), min_df=2, sublinear_tf=True)
M_name = tf_name.fit_transform(fac["name"].fillna("").str.lower())
TN = TruncatedSVD(n_components=15, random_state=42).fit_transform(M_name)
for i in range(TN.shape[1]):
    X[f"name_topic_{i:02d}"] = TN[:, i]
# full word matrix for the stacked text model (all words, not compressed)
base_cols = [c for c in X.columns if not c.startswith(("text_topic_", "name_topic_", "district_", "state_", "dist_"))]
M_sparse = sparse.hstack([M_text, M_name, sparse.csr_matrix(
    StandardScaler().fit_transform(X[base_cols].fillna(0)))]).tocsr()
P.step(f"text: 100 description topics + 15 name topics (numbers removed) | total features: {X.shape[1] + 1}")

# =====================================================================
# 4. MODELS
# =====================================================================
def candidates():
    return {
        "Ridge (linear)": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=3.0)),
        "Random forest": make_pipeline(SimpleImputer(strategy="median"),
                                       RandomForestRegressor(n_estimators=400, min_samples_leaf=3, max_features=0.4,
                                                             n_jobs=-1, random_state=42)),
        "Extra trees": make_pipeline(SimpleImputer(strategy="median"),
                                     ExtraTreesRegressor(n_estimators=400, min_samples_leaf=3, max_features=0.5,
                                                         n_jobs=-1, random_state=42)),
        "Gradient boosting": HistGradientBoostingRegressor(max_iter=500, learning_rate=0.04, max_leaf_nodes=31,
                                                           min_samples_leaf=20, l2_regularization=1.0, random_state=42),
    }

def quantile_model(q):
    return HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=400, learning_rate=0.04,
                                         min_samples_leaf=20, random_state=42)

def band_beds(b):
    return "small" if b < 20 else ("medium" if b <= 100 else "large")

def band_docs(d):
    return "small" if d < 5 else ("medium" if d <= 25 else "large")

def scores(y, p, bf):
    true, pred = np.expm1(y), np.expm1(p)
    return {"log_mae": float(np.mean(np.abs(y - p))),
            "median_abs_pct_error": float(np.median(np.abs(pred - true) / true)),
            "r2_log": float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)),
            "band_accuracy": float(np.mean([bf(a) == bf(b) for a, b in zip(true, pred)]))}

def run_target(label, known, other_count, bf):
    XT = X.copy()
    XT["other_count_log"] = np.log1p(other_count)                # beds model: doctors; doctors model: beds
    known_idx = np.where(known.notna().values)[0]
    y = np.log1p(known.values[known_idx])
    Xk = XT.iloc[known_idx]
    oof = {n: np.zeros(len(y)) for n in list(candidates()) + ["Ensemble (forest + boosting)", "Naive (median)"]}
    lo_oof, hi_oof = np.zeros(len(y)), np.zeros(len(y))
    for f, (tr, te) in enumerate(KFold(N_FOLDS, shuffle=True, random_state=42).split(Xk), 1):
        for n, m in candidates().items():
            m.fit(Xk.iloc[tr], y[tr]); oof[n][te] = m.predict(Xk.iloc[te])
            P.step(f"{label}: fold {f}/{N_FOLDS} · {n} trained")
        oof["Ensemble (forest + boosting)"][te] = (oof["Random forest"][te] + oof["Gradient boosting"][te]) / 2
        oof["Naive (median)"][te] = np.median(y[tr])
        lo_oof[te] = quantile_model(0.10).fit(Xk.iloc[tr], y[tr]).predict(Xk.iloc[te])
        P.step(f"{label}: fold {f}/{N_FOLDS} · low-range model trained")
        hi_oof[te] = quantile_model(0.90).fit(Xk.iloc[tr], y[tr]).predict(Xk.iloc[te])
        P.step(f"{label}: fold {f}/{N_FOLDS} · high-range model trained")

    comp = pd.DataFrame([{"target": label, "model": n, **scores(y, p, bf)} for n, p in oof.items()])
    comp = comp.sort_values("log_mae").reset_index(drop=True)
    best = comp[comp["model"] != "Naive (median)"].iloc[0]["model"]
    bst, naive = comp[comp["model"] == best].iloc[0], comp[comp["model"] == "Naive (median)"].iloc[0]
    lo_oof, hi_oof = np.minimum(lo_oof, oof[best]), np.maximum(hi_oof, oof[best])
    raw_cov = float(np.mean((lo_oof <= y) & (y <= hi_oof)))
    nonconf = np.maximum(lo_oof - y, y - hi_oof)                 # conformal calibration -> honest 80% range
    qhat = max(0.0, float(np.quantile(nonconf, min(1.0, np.ceil((len(y) + 1) * 0.80) / len(y)))))
    coverage = float(np.mean((lo_oof - qhat <= y) & (y <= hi_oof + qhat)))
    P.step(f"{label}: best model = {best} (typical error {bst['median_abs_pct_error']:.0%}); "
           f"range coverage {raw_cov:.0%} -> {coverage:.0%}")

    # ---- final fit on all known rows, predict everyone
    fitted = {}
    if best.startswith("Ensemble"):
        for n in ["Random forest", "Gradient boosting"]:
            fitted[n] = candidates()[n].fit(Xk, y)
        pred = (fitted["Random forest"].predict(XT) + fitted["Gradient boosting"].predict(XT)) / 2
    else:
        fitted[best] = candidates()[best].fit(Xk, y)
        pred = fitted[best].predict(XT)
    P.step(f"{label}: final model trained on all {len(y):,} known facilities")
    lo = np.minimum(quantile_model(0.10).fit(Xk, y).predict(XT), pred) - qhat
    hi = np.maximum(quantile_model(0.90).fit(Xk, y).predict(XT), pred) + qhat
    P.step(f"{label}: final range models trained")

    # ---- what drives it (permutation importance on a held-out 20%)
    cut = int(len(y) * 0.8)
    order = np.random.RandomState(0).permutation(len(y))
    gb = candidates()["Gradient boosting"].fit(Xk.iloc[order[:cut]], y[order[:cut]])
    pi = permutation_importance(gb, Xk.iloc[order[cut:]], y[order[cut:]], n_repeats=5, random_state=42,
                                scoring="neg_mean_absolute_error")
    imp = pd.DataFrame({"feature": Xk.columns, "importance": pi.importances_mean})
    imp["group"] = (imp["feature"].str.replace(r"^text_topic_\d+$", "description text (NLP topics)", regex=True)
                    .str.replace(r"^svc_.*$", "services offered", regex=True)
                    .str.replace(r"^type_.*$", "facility type", regex=True)
                    .str.replace(r"^n_(specialties|procedures|equipment|capability)$", "breadth", regex=True)
                    .str.replace(r"^dist_.*$", "distance to capital / major city", regex=True))
    imp = imp.groupby("group", as_index=False)["importance"].sum().sort_values("importance", ascending=False)
    imp["target"] = label
    P.step(f"{label}: feature importance computed")

    metrics = {"target": label, "n_train": int(len(y)), "best_model": best,
               "median_abs_pct_error": bst["median_abs_pct_error"], "baseline_abs_pct_error": naive["median_abs_pct_error"],
               "r2_log": bst["r2_log"], "band_accuracy": bst["band_accuracy"],
               "baseline_band_accuracy": naive["band_accuracy"], "within_range_pct": coverage}
    if mlflow is not None:
        try:
            user = spark.sql("SELECT current_user()").first()[0]
            mlflow.set_experiment(f"/Users/{user}/carefinder_size_model")
            for _, r in comp.iterrows():
                with mlflow.start_run(run_name=f"{label} - {r['model']}"):
                    mlflow.log_params({"target": label, "model": r["model"], "n_train": len(y),
                                       "n_features": Xk.shape[1], "cv_folds": N_FOLDS})
                    mlflow.log_metrics({k: float(r[k]) for k in
                                        ["log_mae", "median_abs_pct_error", "r2_log", "band_accuracy"]})
                    if r["model"] == best:
                        mlflow.log_metric("interval_coverage_raw", raw_cov)
                        mlflow.log_metric("interval_coverage_calibrated", coverage)
                        mlflow.log_table(imp, "feature_importance.json")
                        if not best.startswith("Ensemble"):
                            mlflow.sklearn.log_model(fitted[best], "model", input_example=Xk.head(3))
        except Exception as e:
            print(f"  (MLflow logging skipped: {str(e)[:120]})")
    P.step(f"{label}: logged to MLflow")
    return np.expm1(pred), np.expm1(lo), np.expm1(hi), metrics, comp, imp

# =====================================================================
# 4b. BED SIZE CLASSIFIER: "under BED_CUT" vs "BED_CUT+" (the winning setup from the local experiments)
#     Stacking: a text model (logistic regression on ALL words) -> its probability becomes a feature
#     for gradient boosting. Nested cross-validation, so no facility ever sees its own label.
# =====================================================================
def make_clf():
    if lgb is not None:
        return lgb.LGBMClassifier(n_estimators=700, learning_rate=0.03, num_leaves=31, min_child_samples=20,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0,
                                  verbose=-1, random_state=42, n_jobs=-1)
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=500, learning_rate=0.04, min_samples_leaf=20,
                                          l2_regularization=1.0, random_state=42)

def run_bed_classifier(known, other_count):
    XT = X.copy()
    XT["other_count_log"] = np.log1p(other_count)
    known_idx = np.where(known.notna().values)[0]
    y = (known.values[known_idx] >= BED_CUT).astype(int)
    Xk = XT.iloc[known_idx].reset_index(drop=True)
    Mk = M_sparse[known_idx]

    def text_model_probs(tr, M_target):
        """inner out-of-fold text-model probabilities for the training rows + a full fit for M_target"""
        inner = np.zeros(len(tr))
        for itr, ite in StratifiedKFold(5, shuffle=True, random_state=7).split(np.zeros(len(tr)), y[tr]):
            inner[ite] = LogisticRegression(C=1.0, max_iter=3000).fit(Mk[tr][itr], y[tr][itr]) \
                .predict_proba(Mk[tr][ite])[:, 1]
        full = LogisticRegression(C=1.0, max_iter=3000).fit(Mk[tr], y[tr]).predict_proba(M_target)[:, 1]
        return inner, full

    oof = np.zeros(len(y))
    for f, (tr, te) in enumerate(StratifiedKFold(N_FOLDS, shuffle=True, random_state=42)
                                 .split(np.zeros(len(y)), y), 1):
        inner, te_lin = text_model_probs(tr, Mk[te])
        P.step(f"beds: fold {f}/{N_FOLDS} · text model trained")
        Xtr = Xk.iloc[tr].copy(); Xtr["text_model_prob"] = inner
        Xte = Xk.iloc[te].copy(); Xte["text_model_prob"] = te_lin
        fold_model = make_clf().fit(Xtr, y[tr])
        oof[te] = fold_model.predict_proba(Xte)[:, 1]
        P.step(f"beds: fold {f}/{N_FOLDS} · size classifier trained "
               f"(fold accuracy {np.mean((oof[te] >= 0.5) == y[te]):.0%})")

    pred, conf = (oof >= 0.5).astype(int), np.maximum(oof, 1 - oof)
    thr = float(np.quantile(conf, 1 - SHOW_SHARE))
    sel = conf >= thr
    naive = max(y.mean(), 1 - y.mean())
    metrics = {
        "target": "beds", "n_train": int(len(y)), "best_model": "text model stacked into gradient boosting",
        "bed_cut": BED_CUT, "accuracy": float(np.mean(pred == y)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)), "naive_accuracy": float(naive),
        "recall_small": float(recall_score(y, pred, pos_label=0)), "recall_large": float(recall_score(y, pred, pos_label=1)),
        "shown_share": float(sel.mean()), "shown_accuracy": float(np.mean(pred[sel] == y[sel])),
        "shown_balanced_accuracy": float(balanced_accuracy_score(y[sel], pred[sel])), "confidence_threshold": thr,
    }
    P.step(f"beds: CV accuracy {metrics['accuracy']:.1%} (balanced {metrics['balanced_accuracy']:.1%}); "
           f"most confident {SHOW_SHARE:.0%}: {metrics['shown_accuracy']:.1%}")

    # ---- final model on all known facilities, predict everyone
    inner_all, lin_all = text_model_probs(np.arange(len(y)), M_sparse)
    Xk_final = Xk.copy(); Xk_final["text_model_prob"] = inner_all
    final = make_clf().fit(Xk_final, y)
    X_all = XT.copy(); X_all["text_model_prob"] = lin_all
    p_large = final.predict_proba(X_all)[:, 1]
    P.step(f"beds: final classifier trained on all {len(y):,} known facilities")

    # ---- what drives it (permutation importance on the last held-out fold)
    pi = permutation_importance(fold_model, Xte, y[te], n_repeats=5, random_state=42, scoring="accuracy")
    imp = pd.DataFrame({"feature": Xte.columns, "importance": pi.importances_mean})
    imp["group"] = (imp["feature"].str.replace(r"^text_topic_\d+$", "description text (NLP topics)", regex=True)
                    .str.replace(r"^name_topic_\d+$", "facility name words", regex=True)
                    .str.replace(r"^text_model_prob$", "description text (full-word model)", regex=True)
                    .str.replace(r"^svc_.*$", "services offered", regex=True)
                    .str.replace(r"^type_.*$", "facility type", regex=True)
                    .str.replace(r"^n_(specialties|procedures|equipment|capability)$", "breadth", regex=True)
                    .str.replace(r"^dist_.*$", "distance to capital / major city", regex=True))
    imp = imp.groupby("group", as_index=False)["importance"].sum().sort_values("importance", ascending=False)
    imp["target"] = "beds"
    P.step("beds: feature importance computed")

    if mlflow is not None:
        try:
            user = spark.sql("SELECT current_user()").first()[0]
            mlflow.set_experiment(f"/Users/{user}/carefinder_size_model")
            with mlflow.start_run(run_name=f"beds - classifier under/over {BED_CUT}"):
                mlflow.log_params({"target": "beds", "bed_cut": BED_CUT, "show_share": SHOW_SHARE,
                                   "n_train": len(y), "n_features": X_all.shape[1], "cv_folds": N_FOLDS,
                                   "model": metrics["best_model"]})
                mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})
                mlflow.log_table(imp, "feature_importance.json")
        except Exception as e:
            print(f"  (MLflow logging skipped: {str(e)[:120]})")
    P.step("beds: logged to MLflow")
    return p_large, thr, metrics, imp

# =====================================================================
# 5. TRAIN
# =====================================================================
out = fac[["unique_id"]].copy()
all_metrics, all_comp, all_imp = [], [], []

# ---- beds: 2-class classifier
P.stage_(f"4/6 training bed-size classifier (under {BED_CUT} vs {BED_CUT}+, {N_FOLDS}-fold CV)")
p_large, thr, bed_met, bed_imp = run_bed_classifier(beds_known, docs_known)
all_metrics.append(bed_met); all_imp.append(bed_imp)
src = np.select([fac["beds"].notna().values, fac["beds_text"].notna().values, fac["beds_fill"].notna().values],
                ["listed", "stated in text", "official website"], default="predicted")
out["beds_value"] = beds_known.values.astype(float)                       # real numbers only
out["beds_source"] = src
out["beds_p_large"] = p_large                                             # P(BED_CUT+ beds)
out["beds_class"] = np.where(p_large >= 0.5, f"{BED_CUT}+ beds", f"under {BED_CUT} beds")
out["beds_class_shown"] = (src == "predicted") & (np.maximum(p_large, 1 - p_large) >= thr)
# ---- beds: calibrated 80% range from the regression models, kept INSIDE the predicted group
P.stage_(f"4b/6 training bed-range models ({N_FOLDS}-fold cross-validation)")
b_pred, b_lo, b_hi, b_met, b_comp, b_imp = run_target("beds", beds_known, docs_known, band_beds)
b_met["target"] = "beds_range"
all_metrics.append(b_met)
is_large = p_large >= 0.5
confident = np.maximum(p_large, 1 - p_large) >= thr
lo = np.maximum(1, np.round(b_lo)); hi = np.round(b_hi)
# confident prediction -> clip the range to the group ("100+" never shows 60-140; "under 100" never shows 80-150)
lo = np.where(confident & is_large, np.maximum(lo, BED_CUT), lo)
hi = np.where(confident & is_large, np.maximum(hi, lo + 1), hi)
hi = np.where(confident & ~is_large, np.minimum(hi, BED_CUT - 1), hi)
lo = np.where(confident & ~is_large, np.minimum(lo, hi - 1).clip(1), lo)
pred_rows = src == "predicted"
out["beds_low"] = np.where(pred_rows, lo, np.nan)
out["beds_high"] = np.where(pred_rows, hi, np.nan)
out["size_band"] = np.where(out["beds_value"].notna(),
                            np.where(out["beds_value"] >= BED_CUT, "large", "small"),
                            np.where(p_large >= 0.5, "large", "small"))

# ---- doctors: regression (as before)
P.stage_(f"5/6 training doctors models ({N_FOLDS}-fold cross-validation)")
pred, lo, hi, met, comp, imp = run_target("doctors", docs_known, beds_known, band_docs)
all_metrics.append(met); all_comp.append(comp); all_imp.append(imp)
src_d = np.select([fac["doctors"].notna().values, fac["doctors_text"].notna().values,
                   fac["doctors_fill"].notna().values], ["listed", "stated in text", "official website"],
                  default="predicted")
out["doctors_value"] = np.where(docs_known.notna(), docs_known, np.round(pred)).astype(float)
out["doctors_source"] = src_d
out["doctors_low"] = np.where(src_d == "predicted", np.maximum(1, np.round(lo)), np.nan)
out["doctors_high"] = np.where(src_d == "predicted", np.round(hi), np.nan)

# =====================================================================
# 6. SAVE + REPORT
# =====================================================================
P.stage_("6/6 saving results")
def save(df, name):
    spark.createDataFrame(df.reset_index(drop=True)).write.mode("overwrite") \
        .option("overwriteSchema", "true").saveAsTable(f"{OUT}.{name}")
save(out, "facility_size")
save(pd.DataFrame(all_metrics), "size_model_metrics")
save(pd.concat(all_comp, ignore_index=True), "size_model_comparison")
save(pd.concat(all_imp, ignore_index=True), "size_feature_importance")
P.done = P.total
P.stage_("DONE ✅")

print("=" * 90 + "\nSOURCE OF EACH VALUE")
for label in ["beds", "doctors"]:
    vc = out[f"{label}_source"].value_counts()
    print(f"  {label:8s} " + " | ".join(f"{k}: {v:,}" for k, v in vc.items()))

m = bed_met
print("\n" + "=" * 90 + f"\nBEDS: 'under {BED_CUT}' vs '{BED_CUT}+' CLASSIFIER ({N_FOLDS}-fold CV, {m['n_train']:,} facilities)")
print(f"  accuracy (all facilities)          {m['accuracy']:.1%}   vs always guessing the bigger group: {m['naive_accuracy']:.0%}")
print(f"  balanced accuracy                  {m['balanced_accuracy']:.1%}")
print(f"  correctly spots small / large      {m['recall_small']:.0%} / {m['recall_large']:.0%}")
print(f"  SHOWN TO USERS (most confident {m['shown_share']:.0%}): accuracy {m['shown_accuracy']:.1%}, "
      f"balanced {m['shown_balanced_accuracy']:.1%}")
pred_rows = out["beds_source"] == "predicted"
print(f"  facilities without a real bed count: {pred_rows.sum():,} -> size class shown for "
      f"{out.loc[pred_rows, 'beds_class_shown'].sum():,}, 'size uncertain' for the rest")
print("\n  WHAT DRIVES THE BED-SIZE PREDICTION")
tot = bed_imp["importance"].clip(lower=0).sum() or 1
for _, r in bed_imp.head(12).iterrows():
    print(f"    {r['group']:36s} {max(r['importance'], 0) / tot:6.1%}")

for met, comp, imp in zip([x for x in all_metrics if x["target"] == "doctors"], all_comp,
                          [x for x in all_imp if x["target"].iloc[0] == "doctors"]):
    lab = met["target"]
    bands = "<5 / 5-25 / >25 doctors"
    print("\n" + "=" * 90 + f"\n{lab.upper()}: MODEL COMPARISON ({N_FOLDS}-fold cross-validation, {met['n_train']:,} facilities)")
    print(f"  {'model':32s} {'typical error':>14s} {'band correct':>13s} {'R² (log)':>9s}")
    for _, r in comp.iterrows():
        star = "  <- chosen" if r["model"] == met["best_model"] else ""
        print(f"  {r['model']:32s} {r['median_abs_pct_error']:>13.0%} {r['band_accuracy']:>13.0%} "
              f"{r['r2_log']:>9.2f}{star}")
    print(f"  size bands: {bands} | real value inside the predicted range: {met['within_range_pct']:.0%}")
    print(f"\n  WHAT DRIVES THE {lab.upper()} PREDICTION")
    tot = imp["importance"].clip(lower=0).sum() or 1
    for _, r in imp.head(12).iterrows():
        print(f"    {r['group']:36s} {max(r['importance'], 0) / tot:6.1%}")
print("\nSize bands (all facilities):", out["size_band"].value_counts().to_dict())
