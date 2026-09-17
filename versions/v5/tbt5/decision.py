"""Turning a conditional distribution into the number that minimises mean APE.

The point nobody exploits
-------------------------
Every version so far fitted log(price) under squared loss and reported the
exponential of the fit. That predicts the conditional *geometric mean*. It is not
the estimate that minimises mean absolute percentage error, and the gap is not
subtle once the conditional spread is wide — which on this data it is.

Write the objective out. For a point estimate z against an outcome Y,

    APE(z) = |z - Y| / Y

so the quantity to minimise is E[|z - Y| / Y]. Differentiate:

    d/dz E[|z - Y|/Y] = E[ sign(z - Y) / Y ] = 0

The minimiser is therefore the median of Y under the measure reweighted by 1/y —
not the mean, not the median, and always *below* the median whenever the
distribution has any right skew in price.

The reason is worth stating plainly because it is counter-intuitive: APE is
asymmetric. Overpredicting by a factor of two costs 100%; underpredicting by a
factor of two costs 50%. A metric that punishes overprediction twice as hard is
minimised by an estimate that leans low.

For a lognormal conditional distribution the answer is closed form. If
log Y ~ N(mu, sigma^2) then the 1/y-reweighted measure is lognormal with location
mu - sigma^2, so

    z* = exp(mu - sigma^2)

The shrink is proportional to the *conditional variance*, which is the part that
matters: it shrinks hard exactly where the model is uncertain — big tanks, thin
segments, unusual scope — and barely at all where it is confident. A single
global fudge factor cannot do that, which is why v4's flat recalibration attempts
moved nothing.

What this costs
---------------
Aggregate bias. An estimator tuned for mean APE is deliberately low, so summing
its predictions across a portfolio will undershoot. That is the correct trade
when the objective is per-quote accuracy and the wrong one when the objective is
valuing a backlog, so both are available: `objective="mape"` for per-quote work,
`objective="unbiased"` when the sum has to come out right. v4's handoff warns
against using the model to value a backlog; with `objective="unbiased"` that
warning is addressed rather than inherited.
"""
import numpy as np
from scipy.stats import norm

OBJECTIVES = ("mape", "median", "unbiased")


def sigma_from_quantiles(logq, levels):
    """Robust conditional log-scale from a quantile fan.

    Uses the widest symmetric pair available and rescales by the normal
    interquantile range, which is far steadier than a moment estimate when the
    fan crosses or the tails are ragged.
    """
    logq = np.atleast_2d(np.asarray(logq, dtype=float))
    levels = np.asarray(levels, dtype=float)
    lo_i, hi_i = int(np.argmin(levels)), int(np.argmax(levels))
    span = norm.ppf(levels[hi_i]) - norm.ppf(levels[lo_i])
    if span <= 0:
        return np.zeros(logq.shape[0])
    return np.maximum(logq[:, hi_i] - logq[:, lo_i], 0.0) / span


def _tail_extrapolated_grid(logq, levels, n_grid=257):
    """Resample a quantile fan onto a dense probability grid.

    Interpolation is linear in the *normal score* rather than in probability, so
    the reconstruction is exact when the conditional distribution is lognormal
    and the tails extend sensibly rather than flattening at the outermost
    quantile supplied.
    """
    levels = np.asarray(levels, dtype=float)
    order = np.argsort(levels)
    levels, logq = levels[order], np.asarray(logq, dtype=float)[:, order]

    # Enforce monotone quantiles. Quantile regressions fitted independently can
    # cross, especially in thin regions; sorting is the standard repair.
    logq = np.sort(logq, axis=1)

    zs = norm.ppf(levels)
    p = np.linspace(0.005, 0.995, n_grid)
    zg = norm.ppf(p)

    out = np.empty((logq.shape[0], n_grid))
    for i in range(logq.shape[0]):
        out[i] = np.interp(zg, zs, logq[i])
        # np.interp clamps outside the range; replace the clamped tails with the
        # local linear slope so the distribution keeps widening.
        if len(zs) >= 2:
            lo_slope = (logq[i, 1] - logq[i, 0]) / max(zs[1] - zs[0], 1e-9)
            hi_slope = (logq[i, -1] - logq[i, -2]) / max(zs[-1] - zs[-2], 1e-9)
            left, right = zg < zs[0], zg > zs[-1]
            out[i, left] = logq[i, 0] + lo_slope * (zg[left] - zs[0])
            out[i, right] = logq[i, -1] + hi_slope * (zg[right] - zs[-1])
    return out


def mape_optimal(logq, levels):
    """The 1/y-weighted median, computed numerically from a quantile fan.

    Numerical rather than closed-form so that genuine skew is respected: pricing
    residuals have a fatter right tail than a lognormal, because scope creep adds
    cost and nothing symmetrically removes it.
    """
    grid = _tail_extrapolated_grid(logq, levels)
    # Equal probability mass per grid point, reweighted by 1/y = exp(-log y).
    # Subtract the row max first or exp underflows on million-dollar tanks.
    w = np.exp(-(grid - grid.max(axis=1, keepdims=True)))
    cw = np.cumsum(w, axis=1)
    cw /= cw[:, -1:]
    idx = np.argmax(cw >= 0.5, axis=1)
    return grid[np.arange(grid.shape[0]), idx]


def point_estimate(log_median, logq=None, levels=None, objective="mape",
                   sigma=None):
    """Collapse a conditional distribution to one number, in log space.

    log_median is the central fit; logq/levels the quantile fan if one was
    produced. Returns log-dollars so the caller can exponentiate once.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")
    log_median = np.asarray(log_median, dtype=float)
    if objective == "median":
        return log_median

    if sigma is None:
        sigma = (sigma_from_quantiles(logq, levels) if logq is not None
                 else np.zeros_like(log_median))
    sigma = np.clip(np.asarray(sigma, dtype=float), 0.0, 1.5)

    if objective == "unbiased":
        # Conditional mean of a lognormal: the estimate whose sum is right.
        return log_median + 0.5 * sigma ** 2

    if logq is not None and np.asarray(logq).shape[1] >= 3:
        z = mape_optimal(logq, levels)
        # Guard the numeric route against a degenerate fan by keeping it within
        # the analytic answer's neighbourhood.
        return np.clip(z, log_median - 3.0 * sigma ** 2 - 0.02, log_median + 0.02)
    return log_median - sigma ** 2


def fit_global_shift(pred, actual, objective="mape", lo=-0.25, hi=0.10, n=176):
    """One scalar multiplicative correction, chosen to suit the objective.

    Fitted on out-of-fold predictions only. This mops up whatever the per-row
    shrink left on the table — model-wide optimism, a residual distribution that
    is not quite the shape assumed above — and it is the cheapest thing in the
    pipeline that moves the headline metric.

    The criterion has to match the objective or the objective is a lie. A shift
    fitted on mean APE and applied to `objective="median"` produces something
    that is not the conditional median, and makes any comparison between the two
    meaningless — both would then carry the same global mean-APE correction and
    differ only in the per-row part.

    It is a *shift in log space*, i.e. a constant multiplier. Anything richer
    would be a recalibrator, and v4 established there is nothing for one to
    learn: the model is unbiased in sample, log-log slope 1.007. What is being
    corrected here is the asymmetry of the objective, not a miscalibration.
    """
    pred = np.asarray(pred, dtype=float)
    actual = np.asarray(actual, dtype=float)
    ok = np.isfinite(pred) & np.isfinite(actual) & (actual > 0) & (pred > 0)
    if ok.sum() < 30:
        return 0.0
    p, a = pred[ok], actual[ok]

    if objective == "unbiased":
        # Make the sum come out right — the only thing "unbiased" can mean.
        return float(np.log(a.sum() / p.sum()))

    score = ((lambda e: float(np.median(e))) if objective == "median"
             else (lambda e: float(np.mean(e))))
    cs = np.linspace(lo, hi, n)
    errs = [score(np.abs(p * np.exp(c) - a) / a) for c in cs]
    return float(cs[int(np.argmin(errs))])
