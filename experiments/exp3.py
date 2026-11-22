"""E10-E12 on top of the E9 setup (reuses exp2's feature code). Same folds and seeds."""
import os
src = open("exp2.py", encoding="utf-8").read()
exec(src.split("F1 = pd.concat")[0])                     # features + helpers from exp2 (identical)

import re as _re
DOC_RE = _re.compile(r"(\d{1,4})\s*\+?\s*(?:doctors|physicians|consultants|specialist doctors)\b")
fac["doctors_text"] = fac["text"].map(lambda t: (lambda v: max(v) if v else np.nan)(
    [int(m.group(1)) for m in DOC_RE.finditer(t) if 2 <= int(m.group(1)) <= 5000]))
w = pd.read_csv(os.path.join(DATA, "website_size.csv")).drop_duplicates("unique_id")[["unique_id", "doctors_fill"]]
fac = fac.merge(w, on="unique_id", how="left")
docs_all = fac["doctors"].fillna(fac["doctors_text"]).fillna(pd.to_numeric(fac["doctors_fill"], errors="coerce"))

Bs = sparse.csr_matrix(StandardScaler().fit_transform(B.fillna(0)))
Msp = sparse.hstack([Mw, Mn, Bs]).tocsr()
F = pd.concat([B, block(svd100, "t"), block(svdN, "n"), L, C], axis=1)
F_doc = F.copy(); F_doc["doctors_all_log"] = np.log1p(docs_all.values)

def stacked(Fdf, model_fn, ordinal=False):
    Fk = Fdf[known].reset_index(drop=True); Mk = Msp[known]; P = np.zeros((len(yb), 3))
    for tr, te in skf:
        oof_tr = np.zeros((len(tr), 3))
        for itr, ite in StratifiedKFold(5, shuffle=True, random_state=7).split(np.zeros(len(tr)), yb[tr]):
            oof_tr[ite] = LogisticRegression(C=best_c, max_iter=3000).fit(Mk[tr][itr], yb[tr][itr]).predict_proba(Mk[tr][ite])
        te_feat = LogisticRegression(C=best_c, max_iter=3000).fit(Mk[tr], yb[tr]).predict_proba(Mk[te])
        Xtr = Fk.iloc[tr].copy(); Xtr[["lin0", "lin1", "lin2"]] = oof_tr
        Xte = Fk.iloc[te].copy(); Xte[["lin0", "lin1", "lin2"]] = te_feat
        if not ordinal:
            P[te] = model_fn().fit(Xtr, yb[tr]).predict_proba(Xte)
        else:   # ordinal: P(>=medium), P(large) from two binary models
            p1 = model_fn().fit(Xtr, (yb[tr] >= 1).astype(int)).predict_proba(Xte)[:, 1]
            p2 = model_fn().fit(Xtr, (yb[tr] >= 2).astype(int)).predict_proba(Xte)[:, 1]
            p2 = np.minimum(p2, p1)
            P[te] = np.c_[1 - p1, p1 - p2, p2]
    return P

best_c = 1.0
P10 = stacked(F_doc, lgbm); log("E10 E9 + doctor counts from all sources (dataset/text/website)", acc(P10), selective(P10))
P11 = stacked(F_doc, lambda: lgb.LGBMClassifier(n_estimators=700, learning_rate=0.03, num_leaves=31,
                     min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.6, verbose=-1,
                     random_state=42), ordinal=True)
log("E11 ordinal model (small < medium < large)", acc(P11), selective(P11))
tuned = lambda: lgb.LGBMClassifier(n_estimators=1500, learning_rate=0.015, num_leaves=15, min_child_samples=30,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.4, reg_lambda=5.0,
                                   reg_alpha=0.5, verbose=-1, random_state=42, n_jobs=-1)
P12 = stacked(F_doc, tuned); log("E12 tuned LightGBM (slower, more regularised)", acc(P12), selective(P12))
Pavg = (P10 + P11 + P12) / 3; log("E13 average of E10-E12", acc(Pavg), selective(Pavg, covs=(0.8, 0.7, 0.6, 0.5)))
best = max([("E10", P10), ("E11", P11), ("E12", P12), ("E13", Pavg)], key=lambda kv: acc(kv[1]))
print(f"\nBEST of this round: {best[0]} {acc(best[1]):.1%}")
np.save("best_proba_r2.npy", best[1])
pd.DataFrame(LOG, columns=["experiment", "band_accuracy", "notes"]).to_csv("experiment_log_r2.csv", index=False)
