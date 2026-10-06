# metaBRR

Random-effects meta-analytic Bayesian ridge regression for individual-level data from multiple cohorts.

Each cohort gets its own coefficient vector, `beta_k = beta_0 + delta_k`, with `delta_k ~ N(0, 1/lambda_d I)` and `beta_0 ~ N(0, 1/lambda_0 I)`. All variance hyperparameters (noise `alpha`, `lambda_0`, `lambda_d`) are learned by evidence maximisation, as in scikit-learn's `BayesianRidge`.

For a fixed heterogeneity ratio `r = alpha / lambda_d`, integrating out the `delta_k` reduces the model exactly to an ordinary BRR on whitened data. The fit therefore reuses scikit-learn's spectral MacKay updates inside a 1-D search over `r`. `r = 0` is pooled BRR. See [`docs/metaBRR.pdf`](docs/metaBRR.pdf) for the derivation and simulation results.

```python
import numpy as np
from metabrr import MetaBayesianRidge

rng = np.random.default_rng(0)
K, n, P = 5, 200, 100                       # cohorts, samples per cohort, features
beta0 = rng.normal(0, 0.2, P)               # shared effects
betas = beta0 + rng.normal(0, 0.1, (K, P))  # cohort effects = shared + deviation
X = rng.normal(size=(K * n, P))
groups = np.repeat(np.arange(K), n)         # cohort label per row
y = np.einsum("ij,ij->i", X, betas[groups]) + rng.normal(size=K * n)

m = MetaBayesianRidge().fit(X, y, groups)

m.coef_               # shared beta_0, shape (P,); use for a new cohort (corr with truth: 0.95)
m.coefs_              # cohort-specific beta_k, shape (K, P)
m.alpha_              # noise precision          0.99  (truth 1)
m.lambda0_            # precision of beta_0      26.4  (truth 1/0.2^2 = 25)
m.lambda_delta_       # precision of deviations  94.5  (truth 1/0.1^2 = 100)
m.cross_cohort_corr_  # rho = (1/lambda0) / (1/lambda0 + 1/lambda_delta): 0.78 (truth 0.8)
y_hat = m.predict(X, groups)
```

## Reproduce

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests
PYTHONPATH=. .venv/bin/python sim/simulate.py      # ~40 min on 12 cores
.venv/bin/python sim/plot.py
cd docs && tectonic metaBRR.tex
```
