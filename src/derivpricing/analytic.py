"""Closed-form Black-Scholes-Merton pricing, Greeks, parity, and implied vol.

All rates and volatilities are continuously compounded and annualized. A constant
dividend yield ``q`` is supported and defaults to zero. Option type is selected with
``option_type='call'`` or ``'put'``.
"""
import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, T, r, sigma, q=0.0):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def black_scholes(S, K, T, r, sigma, option_type="call", q=0.0):
    """Black-Scholes-Merton price and the full Greek set.

    Returns a dict with price, delta, gamma, vega, theta, and rho. Vega and rho are
    quoted per unit (1.00) change in volatility and rate; divide by 100 for the
    per-1% sensitivities often shown on trading screens. Theta is per year.
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    pdf = norm.pdf(d1)
    disc_r, disc_q = np.exp(-r * T), np.exp(-q * T)

    if option_type == "call":
        price = S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
        delta = disc_q * norm.cdf(d1)
        theta = (-S * disc_q * pdf * sigma / (2 * np.sqrt(T))
                 - r * K * disc_r * norm.cdf(d2) + q * S * disc_q * norm.cdf(d1))
        rho = K * T * disc_r * norm.cdf(d2)
    elif option_type == "put":
        price = K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)
        delta = -disc_q * norm.cdf(-d1)
        theta = (-S * disc_q * pdf * sigma / (2 * np.sqrt(T))
                 + r * K * disc_r * norm.cdf(-d2) - q * S * disc_q * norm.cdf(-d1))
        rho = -K * T * disc_r * norm.cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    gamma = disc_q * pdf / (S * sigma * np.sqrt(T))
    vega = S * disc_q * pdf * np.sqrt(T)
    return {"price": float(price), "delta": float(delta), "gamma": float(gamma),
            "vega": float(vega), "theta": float(theta), "rho": float(rho)}


def bs_price(S, K, T, r, sigma, option_type="call", q=0.0):
    """Convenience wrapper returning only the Black-Scholes price."""
    return black_scholes(S, K, T, r, sigma, option_type, q)["price"]


def put_call_parity(S=None, K=None, T=None, r=None, C=None, P=None, q=0.0):
    """Solve the parity relation ``C + K e^{-rT} = S e^{-qT} + P`` for one unknown.

    Supply every argument except the single value to solve for, left as ``None``.
    """
    disc_K = K * np.exp(-r * T) if K is not None else None
    disc_S = S * np.exp(-q * T) if S is not None else None
    if C is None:
        return float(disc_S + P - disc_K)
    if P is None:
        return float(C + disc_K - disc_S)
    if S is None:
        return float((C + disc_K - P) * np.exp(q * T))
    if K is None:
        return float((disc_S + P - C) * np.exp(r * T))
    raise ValueError("Leave exactly one of C, P, S, K as None.")


def implied_vol(price, S, K, T, r, option_type="call", q=0.0,
                tol=1e-6, max_iter=100, sigma0=0.3):
    """Newton-Raphson implied volatility.

    Solves ``BS(sigma) = market_price`` using ``f'(sigma) = vega``. Returns the
    implied vol and the iteration count. Falls back toward a bisection step if a
    Newton update would leave the feasible region.
    """
    sigma = sigma0
    lo, hi = 1e-6, 5.0
    for i in range(max_iter):
        bs = black_scholes(S, K, T, r, sigma, option_type, q)
        diff = bs["price"] - price
        if abs(diff) < tol:
            return float(sigma), i + 1
        if diff > 0:
            hi = sigma
        else:
            lo = sigma
        vega = bs["vega"]
        step = diff / vega if vega > 1e-12 else 0.0
        sigma_new = sigma - step
        if not (lo < sigma_new < hi):
            sigma_new = 0.5 * (lo + hi)   # guard: bisection when Newton escapes
        sigma = sigma_new
    return float(sigma), max_iter
