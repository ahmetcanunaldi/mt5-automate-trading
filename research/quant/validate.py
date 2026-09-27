"""Validation core of the quant-math branch ("the limits of statistics").

- sharpe / probabilistic_sharpe / deflated_sharpe (Bailey & Lopez de Prado 2012, 2014)
- pbo_cscv: probability of backtest overfitting via combinatorially symmetric cross-validation (Bailey et al. 2017)
- stationary_bootstrap indices (Politis & Romano 1994) and spa_test (Hansen 2005, consistent p-value)
- placebo: null distribution of any statistic by shuffling within groups (e.g. years) or by block bootstrap
- yearly_splits: expanding-window walk-forward with purge/embargo
- TrialLedger: every configuration evaluated is recorded (reports/quant/trials.jsonl) -> honest trial counts for DSR
"""
import itertools
import json
import math
import pathlib
import time

import numpy as np
import pandas as pd
from scipy import stats

EULER = 0.5772156649015329
LEDGER = pathlib.Path(__file__).resolve().parents[2] / "reports" / "quant" / "trials.jsonl"


def sharpe(r, periods=252):
    r = np.asarray(r, float); r = r[np.isfinite(r)]
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods)) if len(r) > 2 and r.std() > 0 else 0.0


def probabilistic_sharpe(r, sr_bench=0.0):
    """P(true per-period SR > sr_bench) given sample length, skewness and kurtosis (per-period units)."""
    r = np.asarray(r, float); r = r[np.isfinite(r)]
    n = len(r)
    if n < 10 or r.std() == 0:
        return 0.0
    sr = r.mean() / r.std(ddof=1)
    g3 = stats.skew(r); g4 = stats.kurtosis(r, fisher=False)
    den = math.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
    return float(stats.norm.cdf((sr - sr_bench) * math.sqrt(n - 1) / den))


def expected_max_sharpe(n_trials, sr_var):
    """Expected maximum of n_trials per-period Sharpe ratios under the null (all true SR = 0)."""
    if n_trials <= 1:
        return 0.0
    z1 = stats.norm.ppf(1 - 1.0 / n_trials); z2 = stats.norm.ppf(1 - 1.0 / (n_trials * math.e))
    return math.sqrt(sr_var) * ((1 - EULER) * z1 + EULER * z2)


def deflated_sharpe(r, n_trials, sr_var=None):
    """DSR = PSR against the expected max SR of n_trials null strategies. sr_var = variance of the per-period SRs
    across the trials (if unknown, the sampling variance 1/(n-1) of one SR is used - conservative-ish)."""
    r = np.asarray(r, float); r = r[np.isfinite(r)]
    if sr_var is None:
        sr_var = 1.0 / max(len(r) - 1, 1)
    return probabilistic_sharpe(r, expected_max_sharpe(n_trials, sr_var))


def pbo_cscv(M, n_splits=16, metric=None):
    """M: T x N matrix (per-period returns of N configurations). Splits the T rows into n_splits blocks, uses every
    combination of half the blocks as in-sample, picks the best configuration IS and records its OOS rank.
    PBO = share of combinations where the IS-best is below the OOS median (logit <= 0)."""
    M = np.asarray(M, float)
    T, N = M.shape
    metric = metric or (lambda X: X.mean(0) / (X.std(0, ddof=1) + 1e-12))
    blocks = np.array_split(np.arange(T), n_splits)
    logits = []
    for comb in itertools.combinations(range(n_splits), n_splits // 2):
        is_idx = np.concatenate([blocks[i] for i in comb])
        oos_idx = np.concatenate([blocks[i] for i in range(n_splits) if i not in comb])
        best = int(np.argmax(metric(M[is_idx])))
        oos = metric(M[oos_idx])
        rank = (stats.rankdata(oos)[best]) / (N + 1)
        logits.append(math.log(rank / (1 - rank)))
    logits = np.array(logits)
    return {"pbo": float((logits <= 0).mean()), "logit_median": float(np.median(logits)), "n_comb": len(logits)}


def stationary_bootstrap(n, mean_block=20, rng=None):
    """Index array of one stationary-bootstrap resample of length n (geometric block lengths)."""
    rng = rng or np.random.default_rng()
    idx = np.empty(n, np.int64)
    i = int(rng.integers(n))
    p = 1.0 / mean_block
    for k in range(n):
        idx[k] = i
        i = int(rng.integers(n)) if rng.random() < p else (i + 1) % n
    return idx


def spa_test(D, B=1000, mean_block=20, seed=0):
    """Hansen's SPA (consistent). D: T x K matrix of performance differentials vs the benchmark (e.g. strategy
    returns when the benchmark is 'no trade'). H0: no strategy beats the benchmark. Returns the consistent p-value."""
    D = np.asarray(D, float)
    T, K = D.shape
    rng = np.random.default_rng(seed)
    dbar = D.mean(0)
    boot = np.empty((B, K))
    for b in range(B):
        boot[b] = D[stationary_bootstrap(T, mean_block, rng)].mean(0)
    omega = np.sqrt(T) * boot.std(0) + 1e-12
    t_obs = max(0.0, float(np.max(np.sqrt(T) * dbar / omega)))
    thr = -np.sqrt(2 * np.log(np.log(T))) * omega / np.sqrt(T)
    mu_c = np.where(dbar >= thr, dbar, 0.0)
    t_b = np.max(np.sqrt(T) * (boot - mu_c) / omega, axis=1)
    return {"t_spa": t_obs, "p_consistent": float((np.maximum(t_b, 0) >= t_obs).mean())}


def placebo(stat_fn, series: pd.Series, n=200, within="year", mean_block=20, seed=0):
    """Null distribution of stat_fn(series) when the temporal link to the signal is destroyed.
    within="year": shuffle values inside each calendar year (keeps level/vol by year);
    within="block": stationary block bootstrap of the whole series (keeps short-range dependence)."""
    rng = np.random.default_rng(seed)
    real = stat_fn(series)
    vals = series.to_numpy().copy()
    years = series.index.year.to_numpy()
    null = []
    for _ in range(n):
        if within == "year":
            v = vals.copy()
            for y in np.unique(years):
                m = years == y
                v[m] = rng.permutation(v[m])
        else:
            v = vals[stationary_bootstrap(len(vals), mean_block, rng)]
        null.append(stat_fn(pd.Series(v, index=series.index)))
    null = np.array(null)
    return {"real": float(real), "null_mean": float(np.nanmean(null)), "null_p95": float(np.nanpercentile(null, 95)),
            "p": float((null >= real).mean())}


def yearly_splits(index: pd.DatetimeIndex, first_test_year, last_test_year, min_train_years=2, embargo="0D"):
    """Expanding-window walk-forward: train = everything before Jan 1 of the test year minus the embargo."""
    for y in range(first_test_year, last_test_year + 1):
        start = pd.Timestamp(f"{y}-01-01")
        tr = index < start - pd.Timedelta(embargo)
        te = (index >= start) & (index < pd.Timestamp(f"{y + 1}-01-01"))
        if index[tr].size and (start - index[tr].min()).days >= 365 * min_train_years and te.any():
            yield y, tr, te


class TrialLedger:
    """Append-only record of every evaluated configuration, grouped by family (for DSR trial counts)."""

    def __init__(self, path=LEDGER):
        self.path = pathlib.Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, exp, family, config: dict, metrics: dict):
        rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "exp": exp, "family": family, "config": config, "metrics": metrics}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=float) + "\n")

    def count(self, family=None):
        if not self.path.exists():
            return 0
        n = 0
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if family is None or json.loads(line)["family"] == family:
                n += 1
        return n
