"""Cross-validation tests. Each module is checked against another so an error in one
shows up as a disagreement rather than passing silently.
"""
import numpy as np
from derivpricing.analytic import black_scholes, bs_price, put_call_parity, implied_vol
from derivpricing.trees import binomial_tree, trinomial_tree
from derivpricing.montecarlo import mc_european, mc_barrier, gbm_paths

S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2


def test_put_call_parity():
    c = bs_price(S0, K, T, r, sigma, "call")
    p = bs_price(S0, K, T, r, sigma, "put")
    assert abs((c - p) - (S0 - K * np.exp(-r * T))) < 1e-10
    assert abs(put_call_parity(S=S0, K=K, T=T, r=r, C=c, P=None) - p) < 1e-10


def test_greeks_vs_finite_difference():
    bs = black_scholes(S0, K, T, r, sigma, "call")
    h = 1e-4
    delta_fd = (bs_price(S0 + h, K, T, r, sigma) - bs_price(S0 - h, K, T, r, sigma)) / (2 * h)
    gamma_fd = (bs_price(S0 + h, K, T, r, sigma) - 2 * bs_price(S0, K, T, r, sigma)
                + bs_price(S0 - h, K, T, r, sigma)) / h ** 2
    vega_fd = (bs_price(S0, K, T, r, sigma + h) - bs_price(S0, K, T, r, sigma - h)) / (2 * h)
    rho_fd = (bs_price(S0, K, T, r + h, sigma) - bs_price(S0, K, T, r - h, sigma)) / (2 * h)
    assert abs(bs["delta"] - delta_fd) < 1e-5
    assert abs(bs["gamma"] - gamma_fd) < 1e-4
    assert abs(bs["vega"] - vega_fd) < 1e-4
    assert abs(bs["rho"] - rho_fd) < 1e-4


def test_implied_vol_roundtrip():
    price = bs_price(S0, K, T, r, sigma, "call")
    iv, iters = implied_vol(price, S0, K, T, r, "call")
    assert abs(iv - sigma) < 1e-5 and iters < 50


def test_trees_converge_to_bs():
    bs = bs_price(S0, K, T, r, sigma, "call")
    assert abs(binomial_tree(S0, K, T, r, sigma, 2000, "call") - bs) < 1e-2
    assert abs(trinomial_tree(S0, K, T, r, sigma, 800, "call") - bs) < 1e-2


def test_mc_matches_bs():
    bs = bs_price(S0, K, T, r, sigma, "call")
    price, se = mc_european(S0, K, T, r, sigma, M=400_000, option_type="call", seed=1)
    assert abs(price - bs) < 4 * se        # within four standard errors


def test_barrier_in_out_parity():
    H, N, M, seed = 90.0, 252, 200_000, 7
    out, _ = mc_barrier(S0, K, H, T, r, sigma, "down-and-out", N, M, "call", seed=seed)
    inn, _ = mc_barrier(S0, K, H, T, r, sigma, "down-and-in", N, M, "call", seed=seed)
    # same seed => same paths => knock-out + knock-in equals vanilla on those paths
    paths = gbm_paths(S0, T, r, sigma, N, M, seed=seed)
    vanilla = float(np.exp(-r * T) * np.maximum(paths[:, -1] - K, 0.0).mean())
    assert abs((out + inn) - vanilla) < 1e-8
