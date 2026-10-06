# metaBRR

Random-effects meta-analytic Bayesian ridge regression for individual-level data from multiple cohorts.

Each cohort gets its own coefficient vector, `beta_k = beta_0 + delta_k`, with `delta_k ~ N(0, 1/lambda_d I)` and `beta_0 ~ N(0, 1/lambda_0 I)`. All variance hyperparameters (noise `alpha`, `lambda_0`, `lambda_d`) are learned by evidence maximisation, as in scikit-learn's `BayesianRidge`.

For a fixed heterogeneity ratio `r = alpha / lambda_d`, integrating out the `delta_k` reduces the model exactly to an ordinary BRR on whitened data. The fit therefore reuses scikit-learn's spectral MacKay updates inside a 1-D search over `r`. `r = 0` is pooled BRR. See [`docs/metaBRR.pdf`](docs/metaBRR.pdf) for the derivation and simulation results.

```python
from metabrr import MetaBayesianRidge

m = MetaBayesianRidge().fit(X, y, groups)   # groups: cohort label per row
m.coefs_               # (K, P) cohort-specific posterior means
m.coef_                # shared beta_0 (prediction for a new cohort)
m.cross_cohort_corr_   # estimated rho = (1/lambda_0) / (1/lambda_0 + 1/lambda_d)
m.predict(X_new, groups_new)
```

## Reproduce

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests
PYTHONPATH=. .venv/bin/python sim/simulate.py      # ~40 min on 12 cores
.venv/bin/python sim/plot.py
cd docs && tectonic metaBRR.tex
```
