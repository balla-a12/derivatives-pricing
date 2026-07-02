"""Risk-neutral Monte Carlo pricing under geometric Brownian motion.

Covers vanilla European, arithmetic-average Asian, and barrier options, each
returning a price with its standard error so the sampling uncertainty stays visible.
Barrier options use discrete monitoring at the simulation grid; in-out parity holds
exactly at that monitoring, which the tests use to validate the pricer.
"""
import numpy as np


def gbm_paths(S0, T, r, sigma, N, M, q=0.0, seed=None):
    """Simulate ``M`` GBM paths over ``N`` steps. Returns an array of shape (M, N+1)."""
    rng = np.random.default_rng(seed)
    dt = T / N
    Z = rng.standard_normal((M, N))
    increments = (r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.concatenate([np.zeros((M, 1)), np.cumsum(increments, axis=1)], axis=1)
    return S0 * np.exp(log_paths)


def _price_se(discounted_payoff):
    M = len(discounted_payoff)
    return float(discounted_payoff.mean()), float(discounted_payoff.std(ddof=1) / np.sqrt(M))


def mc_european(S0, K, T, r, sigma, M=100_000, option_type="call", q=0.0, seed=None):
    """Vanilla European option via terminal risk-neutral sampling. Returns (price, se)."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(M)
    ST = S0 * np.exp((r - q - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
    payoff = np.maximum(ST - K, 0.0) if option_type == "call" else np.maximum(K - ST, 0.0)
    return _price_se(np.exp(-r * T) * payoff)


def mc_asian(S0, K, T, r, sigma, N=50, M=50_000, option_type="call", q=0.0, seed=None):
    """Arithmetic-average Asian option over the full path. Returns (price, se)."""
    paths = gbm_paths(S0, T, r, sigma, N, M, q, seed)
    S_avg = paths.mean(axis=1)
    payoff = np.maximum(S_avg - K, 0.0) if option_type == "call" else np.maximum(K - S_avg, 0.0)
    return _price_se(np.exp(-r * T) * payoff)


def mc_barrier(S0, K, H, T, r, sigma, barrier_type="down-and-out",
               N=252, M=100_000, option_type="call", q=0.0, rebate=0.0, seed=None):
    """Barrier option via discrete monitoring on the simulation grid.

    ``barrier_type`` is one of down-and-out, down-and-in, up-and-out, up-and-in.
    A knocked-out path pays the (discounted) ``rebate``; a knock-in path pays only
    if the barrier is touched. Returns (price, se).
    """
    paths = gbm_paths(S0, T, r, sigma, N, M, q, seed)
    ST = paths[:, -1]
    payoff = np.maximum(ST - K, 0.0) if option_type == "call" else np.maximum(K - ST, 0.0)

    if barrier_type.startswith("down"):
        touched = paths.min(axis=1) <= H
    elif barrier_type.startswith("up"):
        touched = paths.max(axis=1) >= H
    else:
        raise ValueError("barrier_type must start with 'down' or 'up'")

    knocked_in = barrier_type.endswith("in")
    active = touched if knocked_in else ~touched
    value = np.where(active, payoff, 0.0)
    if not knocked_in:
        value = value + np.where(~active, rebate, 0.0)   # rebate on knock-out
    return _price_se(np.exp(-r * T) * value)


def mc_convergence(pricer, sample_sizes, **kwargs):
    """Run a Monte Carlo pricer across increasing sample sizes for a convergence plot.

    Returns three arrays: the sample sizes, the price estimates, and their standard
    errors, so the estimate can be shown tightening as the standard error falls like
    ``1/sqrt(M)``.
    """
    sizes, prices, ses = [], [], []
    for M in sample_sizes:
        price, se = pricer(M=int(M), **kwargs)
        sizes.append(int(M)); prices.append(price); ses.append(se)
    return np.array(sizes), np.array(prices), np.array(ses)
