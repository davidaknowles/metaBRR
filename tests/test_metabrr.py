import numpy as np
import pytest
from scipy.stats import multivariate_normal
from sklearn.linear_model import BayesianRidge

from metabrr import MetaBayesianRidge


def make_data(rng, K=3, n=(6, 9, 14), P=8, rho=0.5):
    b0 = rng.normal(size=P)
    X, y, g = [], [], []
    for k in range(K):
        Xk = rng.normal(size=(n[k], P))
        bk = np.sqrt(rho) * b0 + np.sqrt(1 - rho) * rng.normal(size=P)
        X.append(Xk)
        y.append(Xk @ bk + rng.normal(size=n[k]) + 3.0 * k)
        g.append(np.full(n[k], k))
    return np.vstack(X), np.concatenate(y), np.concatenate(g)


def centred_blocks(X, y, g):
    out = []
    for c in np.unique(g):
        i = g == c
        out.append((X[i] - X[i].mean(0), y[i] - y[i].mean()))
    return out


def dense_evidence(blocks, alpha, lam0, lamd):
    """log N(y | 0, Sigma) with Sigma built explicitly."""
    Xs = [Xk for Xk, _ in blocks]
    N = sum(len(Xk) for Xk in Xs)
    Xall = np.vstack(Xs)
    S = Xall @ Xall.T / lam0 + np.eye(N) / alpha
    o = 0
    for Xk in Xs:
        n = len(Xk)
        S[o:o + n, o:o + n] += Xk @ Xk.T / lamd
        o += n
    y = np.concatenate([yk for _, yk in blocks])
    return multivariate_normal(np.zeros(N), S).logpdf(y)


def dense_posterior(blocks, alpha, lam0, lamd):
    """Joint posterior mean of (beta_0, delta_1..delta_K) by direct solve."""
    K = len(blocks)
    P = blocks[0][0].shape[1]
    A = np.zeros(((K + 1) * P, (K + 1) * P))
    rhs = np.zeros((K + 1) * P)
    A[:P, :P] += lam0 * np.eye(P)
    for k, (Xk, yk) in enumerate(blocks):
        sl = slice((k + 1) * P, (k + 2) * P)
        G = alpha * Xk.T @ Xk
        A[:P, :P] += G
        A[:P, sl] += G
        A[sl, :P] += G
        A[sl, sl] += G + lamd * np.eye(P)
        rhs[:P] += alpha * Xk.T @ yk
        rhs[sl] += alpha * Xk.T @ yk
    mu = np.linalg.solve(A, rhs)
    return mu[:P], mu[:P] + mu[P:].reshape(K, P)


@pytest.mark.parametrize("P", [8, 20])  # n_k > P and n_k < P cases
@pytest.mark.parametrize("r", [None, 0.3])
def test_matches_dense_computation(P, r):
    rng = np.random.default_rng(0)
    X, y, g = make_data(rng, P=P)
    m = MetaBayesianRidge(heterogeneity_ratio=r).fit(X, y, g)
    assert np.isfinite(m.lambda_delta_)
    blocks = centred_blocks(X, y, g)
    ev = dense_evidence(blocks, m.alpha_, m.lambda0_, m.lambda_delta_)
    np.testing.assert_allclose(m.log_evidence_, ev, rtol=1e-8)
    b0, bk = dense_posterior(blocks, m.alpha_, m.lambda0_, m.lambda_delta_)
    np.testing.assert_allclose(m.coef_, b0, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(m.coefs_, bk, rtol=1e-6, atol=1e-8)


def test_hyperparameters_are_stationary():
    """Learned hyperparameters should be a local max of the dense evidence."""
    rng = np.random.default_rng(1)
    X, y, g = make_data(rng, K=4, n=(30, 30, 30, 30), P=10, rho=0.3)
    m = MetaBayesianRidge(r_xtol=1e-6).fit(X, y, g)
    blocks = centred_blocks(X, y, g)
    h = np.array([m.alpha_, m.lambda0_, m.lambda_delta_])
    f0 = dense_evidence(blocks, *h)
    for j in range(3):
        for fac in (0.9, 1.1):
            hh = h.copy()
            hh[j] *= fac
            assert dense_evidence(blocks, *hh) <= f0 + 1e-6


def test_r_zero_is_pooled_bayesian_ridge():
    rng = np.random.default_rng(2)
    X, y, g = make_data(rng, P=12)
    m = MetaBayesianRidge(heterogeneity_ratio=0.0, tol=1e-12).fit(X, y, g)
    blocks = centred_blocks(X, y, g)
    Xc = np.vstack([b[0] for b in blocks])
    yc = np.concatenate([b[1] for b in blocks])
    br = BayesianRidge(fit_intercept=False, tol=1e-12, max_iter=10000,
                       compute_score=True).fit(Xc, yc)
    np.testing.assert_allclose(m.alpha_, br.alpha_, rtol=1e-6)
    np.testing.assert_allclose(m.lambda0_, br.lambda_, rtol=1e-6)
    np.testing.assert_allclose(m.coef_, br.coef_, rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(m.scores_, br.scores_[-1], rtol=1e-8)
    for bk in m.coefs_:
        np.testing.assert_allclose(bk, m.coef_)


def test_predict_uses_cohort_coefficients():
    rng = np.random.default_rng(3)
    X, y, g = make_data(rng, P=6, n=(40, 40, 40))
    m = MetaBayesianRidge().fit(X, y, g)
    pred = m.predict(X, g)
    for j, c in enumerate(m.cohorts_):
        i = g == c
        np.testing.assert_allclose(pred[i], X[i] @ m.coefs_[j] + m.intercepts_[j])
        # per-cohort intercept should make residuals mean-zero within cohort
        assert abs(np.mean(y[i] - pred[i])) < 1e-8
    assert m.predict(X).shape == y.shape
