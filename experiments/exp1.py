"""Experiments 1-2: baseline regression vs direct band classifier. 5-fold CV on facilities with a real bed count."""
import warnings, time
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingRegressor
import lightgbm as lgb
from features import load, base_features, clean_text, band

t0 = time.time()
fac = load()
known = fac["beds_known"].notna().values
print(f"searchable facilities {len(fac):,} | real bed counts {known.sum():,}")
y_num = fac.loc[known, "beds_known"].values
yb = band(y_num)
print("band share: small %.0f%% | medium %.0f%% | large %.0f%%" % tuple(100 * np.bincount(yb, minlength=3) / len(yb)))
print("facility types among known:", fac.loc[known, "facility_type"].value_counts().to_dict())

X = base_features(fac)
clean = fac["text"].map(clean_text)
tf = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=3, stop_words="english", sublinear_tf=True)
Tm = tf.fit_transform(clean)
T = TruncatedSVD(n_components=40, random_state=42).fit_transform(Tm)
for i in range(T.shape[1]):
    X[f"t{i:02d}"] = T[:, i]
Xk = X[known].reset_index(drop=True)
ylog = np.log1p(y_num)

skf = StratifiedKFold(5, shuffle=True, random_state=42)
res = {}

# E1 baseline: regression -> band
oof = np.zeros(len(ylog))
for tr, te in skf.split(Xk, yb):
    m = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.04, min_samples_leaf=20, l2_regularization=1.0,
                                      random_state=42).fit(Xk.iloc[tr], ylog[tr])
    oof[te] = m.predict(Xk.iloc[te])
res["E1 regression -> band"] = (band(np.expm1(oof)) == yb).mean()

# E2 direct classifier (LightGBM multiclass)
proba = np.zeros((len(yb), 3))
for tr, te in skf.split(Xk, yb):
    m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=20,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
                           class_weight=None, verbose=-1, random_state=42).fit(Xk.iloc[tr], yb[tr])
    proba[te] = m.predict_proba(Xk.iloc[te])
pred = proba.argmax(1)
res["E2 LightGBM band classifier"] = (pred == yb).mean()

for k, v in res.items():
    print(f"{k:35s} band accuracy {v:.1%}")

# selective prediction curve for E2
conf = proba.max(1)
print("\nSelective prediction (E2): show a size only when the model is confident")
for cov in [1.0, 0.8, 0.6, 0.5, 0.4, 0.3]:
    thr = np.quantile(conf, 1 - cov)
    sel = conf >= thr
    print(f"  most confident {cov:4.0%} of facilities -> band accuracy {(pred[sel] == yb[sel]).mean():.1%} "
          f"(confidence >= {thr:.2f})")
print("\nconfusion (rows=true small/medium/large):\n", pd.crosstab(yb, pred).values)
print(f"\n{time.time() - t0:.0f}s")
