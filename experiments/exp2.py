"""Experiment series for bed-size bands. Same 5-fold stratified CV + fixed seeds everywhere -> replicable."""
import io, re, time, warnings, json, os
import numpy as np, pandas as pd, requests
warnings.filterwarnings("ignore")
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from features import load, base_features, clean_text, band, DATA

t0 = time.time()
LOG = []
def log(name, acc, extra=""):
    LOG.append((name, acc, extra))
    print(f"{name:62s} band accuracy {acc:6.1%}  {extra}  [{time.time() - t0:.0f}s]", flush=True)

fac = load()
known = fac["beds_known"].notna().values
yb = band(fac.loc[known, "beds_known"].values)
print(f"real bed counts: {known.sum():,} (dataset+text+website)")
skf = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(known.sum()), yb))

def norm(s):
    return re.sub(r"[^a-z]", "", str(s or "").lower().replace("&", "and"))

# ---------------------------------------------------------------- feature blocks
B = base_features(fac)
clean = fac["text"].map(clean_text)
TFW = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, stop_words="english", sublinear_tf=True)
Mw = TFW.fit_transform(clean)
TFN = TfidfVectorizer(max_features=3000, ngram_range=(1, 3), min_df=2, sublinear_tf=True)
Mn = TFN.fit_transform(fac["name"].fillna("").str.lower())            # name words ("medical college", "clinic")
svd40 = TruncatedSVD(40, random_state=42).fit_transform(Mw)
svd100 = TruncatedSVD(100, random_state=42).fit_transform(Mw)
svdN = TruncatedSVD(15, random_state=42).fit_transform(Mn)

def block(arr, p):
    return pd.DataFrame(arr, columns=[f"{p}{i:03d}" for i in range(arr.shape[1])], index=fac.index)

# location / context
CAP = {"Andhra Pradesh": (16.515, 80.515), "Assam": (26.143, 91.790), "Bihar": (25.594, 85.138),
       "Chhattisgarh": (21.251, 81.630), "Goa": (15.491, 73.828), "Gujarat": (23.216, 72.637),
       "Haryana": (30.733, 76.779), "Himachal Pradesh": (31.105, 77.173), "Jharkhand": (23.344, 85.310),
       "Karnataka": (12.972, 77.595), "Kerala": (8.524, 76.937), "Madhya Pradesh": (23.260, 77.413),
       "Maharashtra": (19.076, 72.878), "Odisha": (20.296, 85.825), "Punjab": (30.733, 76.779),
       "Rajasthan": (26.912, 75.787), "Tamil Nadu": (13.083, 80.271), "Telangana": (17.385, 78.487),
       "Uttar Pradesh": (26.847, 80.946), "Uttarakhand": (30.317, 78.032), "West Bengal": (22.573, 88.364),
       "Delhi": (28.614, 77.209), "Jammu And Kashmir": (34.084, 74.797), "Chandigarh": (30.733, 76.779),
       "Puducherry": (11.942, 79.808), "Manipur": (24.817, 93.937), "Meghalaya": (25.579, 91.893),
       "Tripura": (23.832, 91.287), "Nagaland": (25.675, 94.109), "Arunachal Pradesh": (27.084, 93.605),
       "Mizoram": (23.727, 92.718), "Sikkim": (27.339, 88.607)}
MC = np.array([(19.076, 72.878), (28.614, 77.209), (12.972, 77.595), (17.385, 78.487), (23.023, 72.571),
               (13.083, 80.271), (22.573, 88.364), (21.170, 72.831), (18.520, 73.857), (26.912, 75.787),
               (26.847, 80.946), (26.450, 80.332), (21.146, 79.088), (22.720, 75.858), (23.260, 77.413),
               (17.687, 83.219), (25.594, 85.138), (22.307, 73.181), (30.901, 75.857), (27.177, 78.008),
               (19.998, 73.790), (22.304, 70.802), (25.318, 82.974), (19.876, 75.343), (31.634, 74.872),
               (25.436, 81.846), (23.344, 85.310), (11.017, 76.956), (23.182, 79.986), (26.218, 78.183),
               (16.506, 80.648), (26.239, 73.024), (9.925, 78.120), (21.251, 81.630), (25.214, 75.865),
               (26.145, 91.736), (30.733, 76.779), (8.524, 76.937), (9.931, 76.267), (20.296, 85.825)])
def hav(a, b, c, d):
    a, b, c, d = map(np.radians, [a, b, c, d])
    h = np.sin((c - a) / 2) ** 2 + np.cos(a) * np.cos(c) * np.sin((d - b) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(h))
L = pd.DataFrame(index=fac.index)
cap = fac["state"].map(CAP)
L["dist_capital"] = [hav(la, lo, c[0], c[1]) if isinstance(c, tuple) and la == la else np.nan
                     for la, lo, c in zip(fac["loc_lat"], fac["loc_lon"], cap)]
L["dist_major_city"] = np.where(fac["loc_lat"].notna(),
    hav(fac["loc_lat"].values[:, None], fac["loc_lon"].values[:, None], MC[None, :, 0], MC[None, :, 1]).min(1), np.nan)

# population (Census 2011, Wikipedia) + state income per capita
def wiki(title):
    r = requests.get(f"https://en.wikipedia.org/wiki/{title}", timeout=30,
                     headers={"User-Agent": "CareFinderResearch/1.0 (student datathon)"})
    return pd.read_html(io.StringIO(r.text))
def num(x):
    m = re.search(r"\d[\d,]*(?:\.\d+)?", str(x)); return float(m.group(0).replace(",", "")) if m else np.nan
def cn(t):
    return [" ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in t.columns]
dkey, skey = fac["district"].map(norm), fac["state"].map(norm)
C = pd.DataFrame(index=fac.index)
try:
    rows = []
    for t in wiki("List_of_districts_in_India"):
        lc = [c.lower() for c in cn(t)]
        d = [i for i, c in enumerate(lc) if "district" in c and "code" not in c]
        p = [i for i, c in enumerate(lc) if "population" in c]
        if d and p:
            rows += [(norm(r.iloc[d[0]]), num(r.iloc[p[0]])) for _, r in t.iterrows()]
    pop = pd.DataFrame(rows, columns=["k", "pop"]).dropna()
    pop = pop[(pop["k"].str.len() > 2) & ~pop["k"].duplicated(keep=False)].set_index("k")["pop"]
    C["district_pop_log"] = np.log1p(dkey.map(pop))
    print(f"population matched {C['district_pop_log'].notna().mean():.0%}")
except Exception as e:
    print("population skipped", e)
try:
    for t in wiki("List_of_Indian_states_and_union_territories_by_GDP_per_capita"):
        lc = [c.lower() for c in cn(t)]
        s = [i for i, c in enumerate(lc) if "state" in c or "union territory" in c]
        v = [i for i, c in enumerate(lc) if "nsdp" in c or "per capita" in c or "₹" in c or "inr" in c]
        if s and v and len(t) >= 20:
            inc = {norm(r.iloc[s[0]]): num(r.iloc[v[0]]) for _, r in t.iterrows()}
            inc = {k: v for k, v in inc.items() if v == v and v > 1000}
            C["state_income"] = skey.map(inc)
            print(f"state income matched {C['state_income'].notna().mean():.0%}")
            break
except Exception as e:
    print("income skipped", e)
# NFHS-5 (organisers' data)
nf = pd.read_csv(os.path.join(DATA, "nfhs5.csv"))
cols = {c: c.lower() for c in nf.columns}
def find(*w, ex=()):
    return next((c for c, lc in cols.items() if all(x in lc for x in w) and not any(e in lc for e in ex)), None)
def pct(x):
    try: return float(str(x).strip().replace("(", "").replace(")", ""))
    except ValueError: return np.nan
dcol, scol = find("district", ex=("code",)), find("state", ex=("code",))
wc = [c for c in [find("electricity"), find("clean fuel"), find("sanitation"), find("drinking water"),
                  find("health insurance")] if c]
lit = find("women", "literate")
z = pd.DataFrame({c: nf[c].map(pct) for c in wc})
nf["wealth"] = ((z - z.mean()) / z.std()).mean(1)
nf["lit"] = nf[lit].map(pct) if lit else np.nan
nf["kds"] = nf[scol].map(norm) + "|" + nf[dcol].map(norm); nf["kd"] = nf[dcol].map(norm)
bds = nf.drop_duplicates("kds").set_index("kds"); bd = nf[~nf["kd"].duplicated(keep=False)].set_index("kd")
for f in ["wealth", "lit"]:
    C["nfhs_" + f] = (skey + "|" + dkey).map(bds[f]).fillna(dkey.map(bd[f]))
print(f"NFHS matched {C['nfhs_wealth'].notna().mean():.0%} (wealth from {len(wc)} indicators, literacy {'yes' if lit else 'no'})")

# ---------------------------------------------------------------- models
def lgbm(n=700, lr=0.03, leaves=31, mcs=20, cs=0.6, cw=None):
    return lgb.LGBMClassifier(n_estimators=n, learning_rate=lr, num_leaves=leaves, min_child_samples=mcs,
                              subsample=0.8, subsample_freq=1, colsample_bytree=cs, reg_lambda=1.0,
                              class_weight=cw, verbose=-1, random_state=42, n_jobs=-1)

def cv_tree(Xdf, model_fn=lgbm, **kw):
    Xk = Xdf[known].reset_index(drop=True)
    P = np.zeros((len(yb), 3))
    for tr, te in skf:
        P[te] = model_fn(**kw).fit(Xk.iloc[tr], yb[tr]).predict_proba(Xk.iloc[te])
    return P

def cv_linear(M, C_=2.0):
    Mk = M[known]
    P = np.zeros((len(yb), 3))
    for tr, te in skf:
        P[te] = LogisticRegression(C=C_, max_iter=3000).fit(Mk[tr], yb[tr]).predict_proba(Mk[te])
    return P

def acc(P):
    return (P.argmax(1) == yb).mean()

def selective(P, covs=(0.8, 0.7, 0.6, 0.5)):
    conf = P.max(1); pr = P.argmax(1); out = []
    for c in covs:
        sel = conf >= np.quantile(conf, 1 - c)
        out.append(f"{int(c * 100)}%->{(pr[sel] == yb[sel]).mean():.0%}")
    return "selective: " + " ".join(out)

F1 = pd.concat([B, block(svd40, "t")], axis=1)
P1 = cv_tree(F1); log("E2  LightGBM: base + 40 text topics (previous best)", acc(P1), selective(P1))
F2 = pd.concat([B, block(svd100, "t")], axis=1)
P2 = cv_tree(F2); log("E3  + 100 text topics", acc(P2), selective(P2))
F3 = pd.concat([F2, block(svdN, "n")], axis=1)
P3 = cv_tree(F3); log("E4  + facility-name topics", acc(P3), selective(P3))
F4 = pd.concat([F3, L], axis=1)
P4 = cv_tree(F4); log("E5  + distance to capital / major city", acc(P4), selective(P4))
F5 = pd.concat([F4, C], axis=1)
P5 = cv_tree(F5); log("E6  + population, state income, NFHS wealth & literacy", acc(P5), selective(P5))

# linear model on the full sparse text (+ scaled base features)
Bs = sparse.csr_matrix(StandardScaler().fit_transform(B.fillna(0)))
Msp = sparse.hstack([Mw, Mn, Bs]).tocsr()
best_lin, best_c = None, None
for c_ in [0.5, 1.0, 2.0, 4.0]:
    Pl = cv_linear(Msp, c_)
    if best_lin is None or acc(Pl) > acc(best_lin):
        best_lin, best_c = Pl, c_
log(f"E7  logistic regression on full text TF-IDF (C={best_c})", acc(best_lin), selective(best_lin))

# blend tree + linear probabilities
Pt = max([P1, P2, P3, P4, P5], key=acc)
best_b, best_w = None, None
for w in np.linspace(0, 1, 11):
    Pb = w * Pt + (1 - w) * best_lin
    if best_b is None or acc(Pb) > acc(best_b):
        best_b, best_w = Pb, w
log(f"E8  blend: {best_w:.1f} x best tree + {1 - best_w:.1f} x text model", acc(best_b), selective(best_b))

# stacking: linear model's out-of-fold probabilities become features for LightGBM (nested, no leakage)
Mk = Msp[known]
stack_feat = np.zeros((len(yb), 3))
Pst = np.zeros((len(yb), 3))
Fk = F5[known].reset_index(drop=True)
for tr, te in skf:
    inner = StratifiedKFold(5, shuffle=True, random_state=7)
    oof_tr = np.zeros((len(tr), 3))
    for itr, ite in inner.split(np.zeros(len(tr)), yb[tr]):
        oof_tr[ite] = LogisticRegression(C=best_c, max_iter=3000).fit(Mk[tr][itr], yb[tr][itr]).predict_proba(Mk[tr][ite])
    lin_full = LogisticRegression(C=best_c, max_iter=3000).fit(Mk[tr], yb[tr])
    te_feat = lin_full.predict_proba(Mk[te])
    Xtr = Fk.iloc[tr].copy(); Xtr[["lin0", "lin1", "lin2"]] = oof_tr
    Xte = Fk.iloc[te].copy(); Xte[["lin0", "lin1", "lin2"]] = te_feat
    Pst[te] = lgbm().fit(Xtr, yb[tr]).predict_proba(Xte)
log("E9  stacking: text-model probabilities -> LightGBM (nested CV)", acc(Pst), selective(Pst))

best = max([("E2", P1), ("E3", P2), ("E4", P3), ("E5", P4), ("E6", P5), ("E7", best_lin), ("E8", best_b),
            ("E9", Pst)], key=lambda kv: acc(kv[1]))
print(f"\nBEST: {best[0]}  {acc(best[1]):.1%}")
print("confusion (rows = true small/medium/large):\n", pd.crosstab(yb, best[1].argmax(1)).values)
np.save("best_proba.npy", best[1]); np.save("yb.npy", yb)
pd.DataFrame(LOG, columns=["experiment", "band_accuracy", "notes"]).to_csv("experiment_log.csv", index=False)
