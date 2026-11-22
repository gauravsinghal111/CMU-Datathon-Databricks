"""How much do the band BOUNDARIES matter? Same E10 model, same folds, different size bands."""
src = open("exp3.py", encoding="utf-8").read()
exec(src.split("best_c = 1.0")[0])                       # features from exp2/exp3 (identical)
best_c = 1.0
yvals = fac.loc[known, "beds_known"].values

def stacked_k(Fdf, ylab, k):
    Fk = Fdf[known].reset_index(drop=True); Mk = Msp[known]; P = np.zeros((len(ylab), k))
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(ylab)), ylab))
    for tr, te in folds:
        oof_tr = np.zeros((len(tr), k))
        for itr, ite in StratifiedKFold(5, shuffle=True, random_state=7).split(np.zeros(len(tr)), ylab[tr]):
            oof_tr[ite] = LogisticRegression(C=best_c, max_iter=3000).fit(Mk[tr][itr], ylab[tr][itr]).predict_proba(Mk[tr][ite])
        te_feat = LogisticRegression(C=best_c, max_iter=3000).fit(Mk[tr], ylab[tr]).predict_proba(Mk[te])
        cols = [f"lin{i}" for i in range(k)]
        Xtr = Fk.iloc[tr].copy(); Xtr[cols] = oof_tr
        Xte = Fk.iloc[te].copy(); Xte[cols] = te_feat
        P[te] = lgbm().fit(Xtr, ylab[tr]).predict_proba(Xte)
    return P

SCHEMES = [
    ("3 bands: <20 / 20-100 / >100   (current)", [20, 100.5]),
    ("3 bands: <30 / 30-150 / >150", [30, 150.5]),
    ("3 bands: <50 / 50-200 / >200", [50, 200.5]),
    ("2 bands: under 100 / 100 or more", [100]),
    ("2 bands: under 50 / 50 or more", [50]),
]
print("\nBAND SCHEME COMPARISON (same model, 5-fold CV)")
for name, cuts in SCHEMES:
    ylab = np.digitize(yvals, cuts)
    shares = " / ".join(f"{s:.0%}" for s in np.bincount(ylab) / len(ylab))
    P = stacked_k(F_doc, ylab, len(cuts) + 1)
    pr, conf = P.argmax(1), P.max(1)
    sel = {c: (pr[conf >= np.quantile(conf, 1 - c)] == ylab[conf >= np.quantile(conf, 1 - c)]).mean() for c in (0.8, 0.6)}
    print(f"  {name:42s} share {shares:18s} all {np.mean(pr == ylab):.1%} | top80% {sel[0.8]:.0%} | "
          f"top60% {sel[0.6]:.0%}", flush=True)
