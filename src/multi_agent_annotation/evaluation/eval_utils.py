"""Shared statistical utilities for eval scripts (stdlib only, no scipy)."""
from __future__ import annotations

import math
import random
from typing import Callable, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────
# Bootstrap CI
# ─────────────────────────────────────────────────────────────

def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def bootstrap_ci(
    values: List[float],
    stat_fn: Callable[[List[float]], float] = _mean,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Percentile bootstrap confidence interval for stat_fn applied to values.

    Returns (lo, hi).  If len(values) < 2, returns (point_est, point_est).
    """
    n = len(values)
    if n < 2:
        v = stat_fn(values) if values else float("nan")
        return (v, v)

    rng = random.Random(seed)
    boot_stats: List[float] = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boot_stats.append(stat_fn(sample))

    boot_stats.sort()
    alpha = 1.0 - ci
    lo_idx = max(0, int(alpha / 2 * n_boot))
    hi_idx = min(n_boot - 1, int((1 - alpha / 2) * n_boot) - 1)
    return (boot_stats[lo_idx], boot_stats[hi_idx])


# ─────────────────────────────────────────────────────────────
# Two-sample permutation test (H0: mean(a) == mean(b))
# ─────────────────────────────────────────────────────────────

def two_sample_bootstrap_p(
    a_vals: List[float],
    b_vals: List[float],
    n_boot: int = 2000,
    seed: int = 42,
) -> Optional[float]:
    """
    Permutation p-value for H0: mean(a) == mean(b), two-sided.

    Returns None if either list is empty.
    """
    if not a_vals or not b_vals:
        return None

    obs_diff = abs(_mean(a_vals) - _mean(b_vals))
    pooled = a_vals + b_vals
    na = len(a_vals)

    rng = random.Random(seed)
    count = 0
    for _ in range(n_boot):
        rng.shuffle(pooled)
        diff = abs(_mean(pooled[:na]) - _mean(pooled[na:]))
        if diff >= obs_diff:
            count += 1

    return (count + 1) / (n_boot + 1)


# ─────────────────────────────────────────────────────────────
# Spearman rank correlation (duplicated here to keep utils self-contained)
# ─────────────────────────────────────────────────────────────

def _spearman_rho(x: List[float], y: List[float]) -> Optional[float]:
    """
    Spearman rank correlation = Pearson correlation of the average ranks.

    NOTE: do NOT use the 1 - 6*sum(d^2)/(n(n^2-1)) shortcut here. That identity
    only holds when there are NO ties, and several of our signals are binary or
    near-constant (human_type_disagree 0/1, flagged_for_review 0/1,
    rounds_used ~ always 2). With that much tying the shortcut is badly biased:
    on a binary target with ~10% positives it returns rho ~ +0.36 for pure
    noise. Pearson-on-ranks is tie-correct and matches scipy.stats.spearmanr.
    """
    n = len(x)
    if n != len(y) or n < 3:
        return None

    def _rank(vals: List[float]) -> List[float]:
        indexed = sorted(enumerate(vals), key=lambda iv: iv[1])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and indexed[j + 1][1] == indexed[j][1]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[indexed[k][0]] = avg_rank
            i = j + 1
        return ranks

    rx, ry = _rank(x), _rank(y)
    mx = sum(rx) / n
    my = sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx <= 0 or vy <= 0:          # a constant input has no rank variance
        return None
    return round(cov / math.sqrt(vx * vy), 6)


# ─────────────────────────────────────────────────────────────
# Spearman permutation test (H0: rho == 0)
# ─────────────────────────────────────────────────────────────

def permutation_p_rho(
    x: List[float],
    y: List[float],
    n_boot: int = 2000,
    seed: int = 42,
) -> Optional[float]:
    """
    Permutation p-value for H0: Spearman rho == 0, two-sided.

    Returns None if n < 3 or rho cannot be computed.
    """
    n = len(x)
    if n < 3 or n != len(y):
        return None

    obs_rho = _spearman_rho(x, y)
    if obs_rho is None:
        return None
    obs_abs = abs(obs_rho)

    rng = random.Random(seed)
    y_perm = list(y)
    count = 0
    for _ in range(n_boot):
        rng.shuffle(y_perm)
        rho_perm = _spearman_rho(x, y_perm)
        if rho_perm is not None and abs(rho_perm) >= obs_abs:
            count += 1

    return (count + 1) / (n_boot + 1)


# ─────────────────────────────────────────────────────────────
# Fisher's exact test (two-sided, pure stdlib)
# ─────────────────────────────────────────────────────────────

def _log_comb(n: int, k: int) -> float:
    """log C(n, k) using log-factorials for numerical stability."""
    if k < 0 or k > n:
        return float("-inf")
    if k == 0 or k == n:
        return 0.0
    k = min(k, n - k)
    return sum(math.log(n - i) - math.log(i + 1) for i in range(k))


def fisher_exact_pvalue(a: int, b: int, c: int, d: int) -> float:
    """
    Two-sided Fisher's exact test p-value for a 2×2 contingency table:

        [[a, b],
         [c, d]]

    Uses the hypergeometric distribution; sums all tables whose probability
    is ≤ the observed probability (the standard two-sided definition).
    Returns 1.0 for degenerate tables.
    """
    n = a + b + c + d
    K = a + c      # column-1 marginal
    nn = a + b     # row-1 marginal

    if n == 0 or K == 0 or K == n or nn == 0 or nn == n:
        return 1.0

    def _log_p(k: int) -> float:
        return _log_comb(K, k) + _log_comb(n - K, nn - k) - _log_comb(n, nn)

    try:
        log_p_obs = _log_p(a)
    except (ValueError, OverflowError):
        return 1.0

    k_min = max(0, nn + K - n)
    k_max = min(nn, K)

    p_value = 0.0
    for k in range(k_min, k_max + 1):
        try:
            lp = _log_p(k)
            if lp <= log_p_obs + 1e-10:
                p_value += math.exp(lp)
        except (ValueError, OverflowError):
            pass

    return min(1.0, p_value)


# ─────────────────────────────────────────────────────────────
# Wilson binomial CI
# ─────────────────────────────────────────────────────────────

def binomial_wilson_ci(
    successes: int,
    n: int,
    ci: float = 0.95,
) -> Tuple[float, float]:
    """
    Wilson score interval for a proportion.  More accurate than the
    Wald interval for small n or extreme proportions.

    Returns (lo, hi).  Returns (0.0, 1.0) when n == 0.
    """
    if n == 0:
        return (0.0, 1.0)

    p_hat = successes / n
    # z ≈ 1.96 for 95 % CI; computed from normal approximation
    alpha = 1.0 - ci
    # Beasley-Springer-Moro approximation of normal quantile
    z = _normal_quantile(1 - alpha / 2)

    z2 = z * z
    denom = 1 + z2 / n
    center = (p_hat + z2 / (2 * n)) / denom
    half = z * math.sqrt(p_hat * (1 - p_hat) / n + z2 / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _normal_quantile(p: float) -> float:
    """Rational approximation of the normal quantile (Abramowitz & Stegun 26.2.17)."""
    if p >= 1.0:
        return float("inf")
    if p <= 0.0:
        return float("-inf")
    if p < 0.5:
        return -_normal_quantile(1 - p)
    t = math.sqrt(-2 * math.log(1 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t)


# ─────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────

def fmt_ci(lo: float, hi: float, decimals: int = 3) -> str:
    """Format a confidence interval as '[lo, hi]'."""
    fmt = f"{{:.{decimals}f}}"
    return f"[{fmt.format(lo)}, {fmt.format(hi)}]"


def fmt_p(p: Optional[float]) -> str:
    """Format a p-value with a significance star."""
    if p is None:
        return "n/a"
    if p < 0.001:
        return f"{p:.3e} ***"
    if p < 0.01:
        return f"{p:.4f} **"
    if p < 0.05:
        return f"{p:.4f} *"
    return f"{p:.4f}"

# ─────────────────────────────────────────────────────────────
# Provenance
# ─────────────────────────────────────────────────────────────

def _sha256(path) -> Optional[str]:
    import hashlib
    from pathlib import Path as _P
    try:
        h = hashlib.sha256()
        with _P(path).open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def stamp_provenance(report: dict, agent_jsonl, **extra) -> dict:
    """Record WHICH BYTES this report scored, under report["provenance"].

    None of the eval scripts previously recorded their input, so a report could
    not be tied back to the file it came from. That is not hypothetical: one raw
    run file was edited 16 minutes AFTER its reports were written, and the
    reports silently described a file that no longer existed. Separately, three
    byte-identical copies of one run now exist under names making contradictory
    claims about the experimental arm — a filename cannot be trusted as identity,
    only a hash can.

    Cheap enough to call unconditionally: one hash of a few-MB file per report.
    """
    from pathlib import Path as _P
    import datetime as _dt
    p = _P(agent_jsonl) if agent_jsonl else None
    prov = {
        "agent_jsonl": str(p) if p else None,
        "agent_jsonl_sha256": _sha256(p) if p else None,
        "agent_jsonl_mtime": (
            _dt.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")
            if p and p.exists() else None
        ),
        "scored_at": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    # The run's own provenance, if the pipeline stamped it (records written after
    # 2026-08-04 carry run_meta). Copied from the first record so a report says
    # which experimental arm produced the data, not just which file.
    try:
        import json as _json
        with _P(agent_jsonl).open() as f:
            for line in f:
                line = line.strip()
                if line.startswith("{"):
                    rm = (_json.loads(line) or {}).get("run_meta")
                    if rm:
                        prov["run_meta"] = rm
                    break
    except Exception:
        pass
    prov.update(extra)
    report["provenance"] = prov
    return report
