"""Simulation: BRR vs metaBRR across heritability and heterogeneity.

Genotypes: K cohorts, P SNPs, cohort allele frequencies drawn from a
Balding-Nichols model (Fst = 0.02) around ancestral frequencies U(0.05, 0.5);
standardised within cohort. Effects: beta_k = sqrt(rho) b0 + sqrt(1 - rho) d_k
with b0, d_k ~ N(0, h2 / P I), so h2 is the per-cohort SNP heritability and
rho the cross-cohort genetic correlation (heterogeneity = 1 - rho).
Noise variance 1 - h2.
"""

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge

from metabrr import MetaBayesianRidge


def simulate(rng, K, n_train, n_test, P, h2, rho, fst=0.02):
    p_anc = rng.uniform(0.05, 0.5, size=P)
    a = p_anc * (1 - fst) / fst
    b = (1 - p_anc) * (1 - fst) / fst
    b0 = rng.normal(0, np.sqrt(h2 / P), size=P)
    data = []
    for k in range(K):
        pk = np.clip(rng.beta(a, b), 0.01, 0.99)
        G = rng.binomial(2, pk, size=(n_train + n_test, P)).astype(float)
        G = (G - G.mean(0)) / (G.std(0) + 1e-12)
        bk = np.sqrt(rho) * b0 + np.sqrt(1 - rho) * rng.normal(0, np.sqrt(h2 / P), size=P)
        g = G @ bk
        y = g + rng.normal(0, np.sqrt(1 - h2), size=len(g)) + rng.normal()  # cohort mean shift
        data.append((G[:n_train], y[:n_train], G[n_train:], y[n_train:], g[n_train:]))
    return data


def r2(y, yhat):
    return 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)


def evaluate(pred_by_cohort, data):
    """Mean over cohorts of within-cohort test R^2 (phenotype) and squared corr with g."""
    r2y, r2g = [], []
    for (_, _, _, yte, gte), p in zip(data, pred_by_cohort):
        r2y.append(r2(yte, p))
        r2g.append(np.corrcoef(gte, p)[0, 1] ** 2)
    return np.mean(r2y), np.mean(r2g)


def centre_train(data):
    """Centre y within cohort (genotypes already standardised within cohort)."""
    return [(Xtr, ytr - ytr.mean(), Xte, yte - ytr.mean(), gte)
            for Xtr, ytr, Xte, yte, gte in data]


def run_one(seed, K, N, P, n_test, h2, rho):
    rng = np.random.default_rng(seed)
    data = centre_train(simulate(rng, K, N // K, n_test, P, h2, rho))
    X = np.vstack([d[0] for d in data])
    y = np.concatenate([d[1] for d in data])
    grp = np.repeat(np.arange(K), N // K)
    rows = []

    t = time.time()
    brr = BayesianRidge(fit_intercept=False).fit(X, y)
    rows.append(("BRR (pooled)", time.time() - t,
                 *evaluate([d[2] @ brr.coef_ for d in data], data), np.nan, np.nan))

    t = time.time()
    preds = []
    for d in data:
        m = BayesianRidge(fit_intercept=False).fit(d[0], d[1])
        preds.append(d[2] @ m.coef_)
    rows.append(("BRR (per cohort)", time.time() - t, *evaluate(preds, data), np.nan, np.nan))

    t = time.time()
    meta = MetaBayesianRidge(fit_intercept=False).fit(X, y, grp)
    el = time.time() - t
    rows.append(("metaBRR", el,
                 *evaluate([d[2] @ meta.coefs_[k] for k, d in enumerate(data)], data),
                 meta.cross_cohort_corr_, meta.heterogeneity_ratio_))
    rows.append(("metaBRR (shared β0)", el,
                 *evaluate([d[2] @ meta.coef_ for d in data], data), np.nan, np.nan))

    return [dict(seed=seed, h2=h2, rho=rho, method=m, time=tt, r2_y=a, r2_g=b,
                 rho_hat=rh, r_hat=rr) for m, tt, a, b, rh, rr in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--N", type=int, default=10_000)
    ap.add_argument("--P", type=int, default=2_000)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--h2", type=float, nargs="+", default=[0.1, 0.25, 0.5, 0.75])
    ap.add_argument("--rho", type=float, nargs="+", default=[1.0, 0.9, 0.75, 0.5, 0.25, 0.0])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="sim/results.csv")
    args = ap.parse_args()

    rows = []
    i = 0
    for rep in range(args.reps):
        for h2 in args.h2:
            for rho in args.rho:
                i += 1
                res = run_one(1000 * rep + i, args.K, args.N, args.P, args.n_test, h2, rho)
                rows += res
                summ = "  ".join(f"{r['method']}={r['r2_g']:.3f}" for r in res)
                print(f"rep={rep} h2={h2} rho={rho} rho_hat={res[2]['rho_hat']:.3f} "
                      f"t_meta={res[2]['time']:.1f}s | r2_g: {summ}", flush=True)
                pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
