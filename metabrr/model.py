"""Meta-analytic Bayesian ridge regression (metaBRR).

Model, for cohorts k = 1..K with individual-level data (X_k, y_k):

    y_k   = X_k beta_k + eps_k,        eps_k   ~ N(0, alpha^{-1} I)
    beta_k = beta_0 + delta_k,          delta_k ~ N(0, lambda_d^{-1} I)
    beta_0 ~ N(0, lambda_0^{-1} I)

i.e. cohort-specific effects drawn from a learned random-effects prior centred
on a shared effect vector. Hyperparameters (alpha, lambda_0, lambda_d) are set
by type-II maximum likelihood (evidence maximisation), as in scikit-learn's
``BayesianRidge``.

Computational trick: with r = alpha / lambda_d fixed, integrating out the
delta_k gives y_k ~ N(X_k beta_0, alpha^{-1} (I + r X_k X_k^T)). Whitening by
(I + r X_k X_k^T)^{-1/2} turns this into an ordinary Bayesian ridge problem in
(alpha, lambda_0) whose design has Gram matrix

    B(r) = sum_k V_k diag(s_k^2 / (1 + r s_k^2)) V_k^T,

where X_k = U_k diag(s_k) V_k^T is a thin SVD computed once per cohort. For
each r we eigendecompose B(r) once and then run the cheap spectral MacKay
fixed-point updates of ``BayesianRidge``; an outer 1-D search maximises the
profile evidence over log r. r = 0 recovers pooled BayesianRidge exactly.
"""

from math import log

import numpy as np
from scipy import linalg
from scipy.optimize import minimize_scalar
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted, validate_data


class MetaBayesianRidge(RegressorMixin, BaseEstimator):
    """Bayesian ridge regression with cohort-specific random-effect coefficients.

    Parameters
    ----------
    max_iter : int, default=300
        Maximum number of inner MacKay iterations per value of r.
    tol : float, default=1e-6
        Inner convergence tolerance on the change in log(alpha), log(lambda_0).
    alpha_1, alpha_2, lambda_1, lambda_2 : float, default=1e-6
        Gamma hyperprior shape/rate parameters on alpha and lambda_0 (as in
        ``BayesianRidge``).
    heterogeneity_ratio : float or None, default=None
        If given, fix r = alpha / lambda_d to this value instead of learning it.
        r = 0 is pooled BayesianRidge.
    r_bounds : tuple of float, default=(1e-4, 1e4)
        Search range for r, in units of 1 / mean nonzero squared singular value
        (i.e. relative to the scale at which heterogeneity becomes noticeable).
    n_grid : int, default=9
        Number of log-spaced grid points used to bracket the optimum of r
        before Brent refinement.
    r_xtol : float, default=1e-3
        Tolerance on log r for the Brent refinement.
    fit_intercept : bool, default=True
        Fit a separate intercept per cohort (by centring within cohort).
    verbose : bool, default=False
    """

    def __init__(
        self,
        *,
        max_iter=300,
        tol=1e-6,
        alpha_1=1e-6,
        alpha_2=1e-6,
        lambda_1=1e-6,
        lambda_2=1e-6,
        heterogeneity_ratio=None,
        r_bounds=(1e-4, 1e4),
        n_grid=9,
        r_xtol=1e-3,
        fit_intercept=True,
        verbose=False,
    ):
        self.max_iter = max_iter
        self.tol = tol
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.heterogeneity_ratio = heterogeneity_ratio
        self.r_bounds = r_bounds
        self.n_grid = n_grid
        self.r_xtol = r_xtol
        self.fit_intercept = fit_intercept
        self.verbose = verbose

    # ------------------------------------------------------------------ fit
    def fit(self, X, y, groups):
        """Fit the model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,)
        groups : array-like of shape (n_samples,)
            Cohort label of each sample.
        """
        X, y = validate_data(self, X, y, dtype=np.float64, y_numeric=True)
        groups = np.asarray(groups)
        self.cohorts_ = np.unique(groups)
        n_samples, n_features = X.shape
        self._n_samples = n_samples
        y_var = 0.0

        # Per-cohort centring and thin SVD (done once).
        self._svd = []
        X_offsets, y_offsets = [], []
        for c in self.cohorts_:
            idx = groups == c
            Xk, yk = X[idx], y[idx]
            if self.fit_intercept:
                xm, ym = Xk.mean(axis=0), yk.mean()
            else:
                xm, ym = np.zeros(n_features), 0.0
            Xk = Xk - xm
            yk = yk - ym
            y_var += np.sum(yk**2)
            U, s, Vt = linalg.svd(Xk, full_matrices=False, check_finite=False)
            keep = s > s[0] * max(Xk.shape) * np.finfo(np.float64).eps
            U, s, Vt = U[:, keep], s[keep], Vt[keep]
            yt = U.T @ yk
            # squared norm of the part of y_k orthogonal to col(X_k)
            e = max(float(yk @ yk - yt @ yt), 0.0)
            self._svd.append((s, Vt, yt, e))
            X_offsets.append(xm)
            y_offsets.append(ym)
        self.X_offsets_ = np.array(X_offsets)
        self.y_offsets_ = np.array(y_offsets)
        self._y_var = y_var / n_samples

        s2_all = np.concatenate([sv[0] ** 2 for sv in self._svd])
        self._r_scale = 1.0 / s2_all.mean()

        if self.heterogeneity_ratio is not None:
            r = float(self.heterogeneity_ratio)
            best = self._profile(r)
            self.r_path_ = [(r, best["score"])]
        else:
            best = self._optimize_r()

        self._finalize(best)
        return self

    def _profile(self, r, init=None):
        """Maximise the evidence over (alpha, lambda_0) for fixed r.

        This is scikit-learn's BayesianRidge loop applied to the whitened
        problem, run entirely in the eigenbasis of B(r).
        """
        P = self.n_features_in_
        N = self._n_samples
        B = np.zeros((P, P))
        b = np.zeros(P)
        yy = 0.0
        jac = 0.0  # -1/2 log det(I + r X_k X_k^T), summed over cohorts
        for s, Vt, yt, e in self._svd:
            d = 1.0 / (1.0 + r * s**2)
            B += (Vt.T * (s**2 * d)) @ Vt
            b += Vt.T @ (s * d * yt)
            yy += np.sum(d * yt**2) + e
            jac += 0.5 * np.sum(np.log(d))
        lam, Q = linalg.eigh(B, overwrite_a=True, check_finite=False)
        lam = np.clip(lam, 0.0, None)
        qb = Q.T @ b

        eps = np.finfo(np.float64).eps
        if init is None:
            alpha_, lambda_ = 1.0 / (self._y_var + eps), 1.0
        else:
            alpha_, lambda_ = init
        for it in range(self.max_iter):
            m = qb / (lam + lambda_ / alpha_)
            sse = yy - 2 * m @ qb + np.sum(lam * m**2)
            gamma = np.sum(alpha_ * lam / (lambda_ + alpha_ * lam))
            lambda_new = (gamma + 2 * self.lambda_1) / (m @ m + 2 * self.lambda_2)
            alpha_new = (N - gamma + 2 * self.alpha_1) / (sse + 2 * self.alpha_2)
            delta = abs(log(lambda_new / lambda_)) + abs(log(alpha_new / alpha_))
            alpha_, lambda_ = alpha_new, lambda_new
            if delta < self.tol:
                break
        m = qb / (lam + lambda_ / alpha_)
        sse = yy - 2 * m @ qb + np.sum(lam * m**2)
        score = self._log_evidence(alpha_, lambda_, lam, m, sse, jac)
        log_ev = self._log_evidence(alpha_, lambda_, lam, m, sse, jac, hyperprior=False)
        if self.verbose:
            print(f"r={r:.4g} alpha={alpha_:.4g} lambda0={lambda_:.4g} "
                  f"score={score:.6f} iters={it + 1}")
        return dict(r=r, alpha=alpha_, lambda0=lambda_, Q=Q, lam=lam, m=m,
                    score=score, log_evidence=log_ev, n_iter=it + 1)

    def _log_evidence(self, alpha_, lambda_, lam, m, sse, jac, hyperprior=True):
        """log p(y | alpha, lambda_0, r) (+ Gamma hyperprior terms as sklearn)."""
        P = self.n_features_in_
        N = self._n_samples
        logdet_sigma = -np.sum(np.log(lambda_ + alpha_ * lam))
        score = 0.0
        if hyperprior:
            score += self.lambda_1 * log(lambda_) - self.lambda_2 * lambda_
            score += self.alpha_1 * log(alpha_) - self.alpha_2 * alpha_
        score += 0.5 * (P * log(lambda_) + N * log(alpha_) - alpha_ * sse
                        - lambda_ * (m @ m) + logdet_sigma - N * log(2 * np.pi))
        return score + jac

    def _optimize_r(self):
        lo, hi = self.r_bounds
        grid = np.linspace(log(lo * self._r_scale), log(hi * self._r_scale),
                           self.n_grid)
        cache = {}

        def neg(logr):
            res = self._profile(np.exp(logr))
            cache[logr] = res
            return -res["score"]

        vals = np.array([neg(g) for g in grid])
        i = int(np.argmin(vals))
        if 0 < i < len(grid) - 1:
            opt = minimize_scalar(neg, bounds=(grid[i - 1], grid[i + 1]),
                                  method="bounded", options=dict(xatol=self.r_xtol))
            cand = cache[opt.x] if opt.x in cache else self._profile(np.exp(opt.x))
        else:
            cand = cache[grid[i]]
        # also compare against the homogeneous (pooled BRR) limit r = 0
        r0 = self._profile(0.0)
        self.r_path_ = sorted((float(np.exp(k)), v["score"]) for k, v in cache.items())
        self.r_path_.insert(0, (0.0, r0["score"]))
        return cand if cand["score"] >= r0["score"] else r0

    def _finalize(self, best):
        r = best["r"]
        self.heterogeneity_ratio_ = r
        self.alpha_ = best["alpha"]
        self.lambda0_ = best["lambda0"]
        self.lambda_delta_ = self.alpha_ / r if r > 0 else np.inf
        tau0 = 1.0 / self.lambda0_
        taud = 1.0 / self.lambda_delta_
        # implied correlation of a coefficient across two cohorts
        self.cross_cohort_corr_ = tau0 / (tau0 + taud)
        self.scores_ = best["score"]  # includes Gamma hyperprior terms, as sklearn
        self.log_evidence_ = best["log_evidence"]  # log p(y | alpha, lambda_0, lambda_d)
        self.n_iter_ = best["n_iter"]

        Q, lam = best["Q"], best["lam"]
        self.coef_ = Q @ best["m"]  # posterior mean of shared beta_0
        self.sigma0_ = (Q / (self.lambda0_ + self.alpha_ * lam)) @ Q.T
        coefs = []
        for s, Vt, yt, e in self._svd:
            resid = yt - s * (Vt @ self.coef_)
            coefs.append(self.coef_ + Vt.T @ (r * s / (1.0 + r * s**2) * resid))
        self.coefs_ = np.array(coefs)  # posterior means of beta_k
        self.intercepts_ = self.y_offsets_ - np.sum(self.X_offsets_ * self.coefs_, axis=1)
        self.intercept_ = float(np.mean(self.y_offsets_ - self.X_offsets_ @ self.coef_))
        del self._svd

    # -------------------------------------------------------------- predict
    def predict(self, X, groups=None):
        """Predict with cohort-specific coefficients.

        Samples whose group is None / unseen use the shared coefficients
        ``coef_`` (the prior mean for a new cohort).
        """
        check_is_fitted(self)
        X = validate_data(self, X, dtype=np.float64, reset=False)
        out = X @ self.coef_ + self.intercept_
        if groups is None:
            return out
        groups = np.asarray(groups)
        for j, c in enumerate(self.cohorts_):
            idx = groups == c
            if idx.any():
                out[idx] = X[idx] @ self.coefs_[j] + self.intercepts_[j]
        return out
