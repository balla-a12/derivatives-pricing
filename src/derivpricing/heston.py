"""Heston stochastic-volatility pricing: Monte Carlo and a characteristic-function pricer.

The model under the risk-neutral measure is

    dS = r S dt + sqrt(v) S dW1
    dv = kappa (theta - v) dt + xi sqrt(v) dW2,   corr(dW1, dW2) = rho.

``heston_mc`` reproduces the Euler full-truncation scheme from the MScFE 620 group
project. ``heston_cf_price`` prices the same model semi-analytically through the
characteristic function, which runs far faster and makes calibration tractable. The two
agree to Monte Carlo error, and both match the group project's reported prices.
"""
import numpy as np


def heston_mc(S0, K, T, r, v0, kappa, theta, xi, rho,
              N=63, M=200_000, option_type="call", seed=42):
    """Heston price by Euler simulation with full truncation on the variance.

    Returns (price, standard_error). Defaults mirror the group project's grid.
    """
    rng = np.random.default_rng(seed)
    dt = T / N
    S = np.full(M, float(S0))
    v = np.full(M, float(v0))
    for _ in range(N):
        z1 = rng.standard_normal(M)
        z2 = rng.standard_normal(M)
        zv = rho * z1 + np.sqrt(1 - rho ** 2) * z2
        vp = np.maximum(v, 0.0)
        S *= np.exp((r - 0.5 * vp) * dt + np.sqrt(vp * dt) * z1)
        v = v + kappa * (theta - vp) * dt + xi * np.sqrt(vp * dt) * zv
    payoff = np.maximum(S - K, 0.0) if option_type == "call" else np.maximum(K - S, 0.0)
    disc = np.exp(-r * T) * payoff
    return float(disc.mean()), float(disc.std(ddof=1) / np.sqrt(M))


def _cf_component(u, j, S0, v0, r, T, kappa, theta, xi, rho):
    """Heston characteristic-function component for P1 (j=1) or P2 (j=2),
    using the Albrecher 'little trap' form for branch-cut stability.
    """
    i = 1j
    x = np.log(S0)
    b = kappa - rho * xi if j == 1 else kappa
    uj = 0.5 if j == 1 else -0.5
    a = kappa * theta
    rxiu = rho * xi * i * u
    d = np.sqrt((rxiu - b) ** 2 - xi ** 2 * (2 * uj * i * u - u ** 2))
    g = (b - rxiu - d) / (b - rxiu + d)
    edt = np.exp(-d * T)
    C = r * i * u * T + (a / xi ** 2) * ((b - rxiu - d) * T
                                         - 2 * np.log((1 - g * edt) / (1 - g)))
    D = ((b - rxiu - d) / xi ** 2) * ((1 - edt) / (1 - g * edt))
    return np.exp(C + D * v0 + i * u * x)


def heston_cf_price(S0, K, T, r, v0, kappa, theta, xi, rho,
                    option_type="call", U=200.0, n=256):
    """Semi-analytic Heston price via Gil-Pelaez inversion of the characteristic function.

    The two in-the-money probabilities ``P1`` and ``P2`` are recovered by integrating
    the characteristic-function components over a Gauss-Legendre grid on ``(0, U]``, and
    the call follows from ``C = S0 P1 - K e^{-rT} P2``. Puts come from parity.
    """
    nodes, weights = np.polynomial.legendre.leggauss(n)
    u = 0.5 * U * (nodes + 1.0)
    w = 0.5 * U * weights
    i = 1j
    lnK = np.log(K)

    f1 = _cf_component(u, 1, S0, v0, r, T, kappa, theta, xi, rho)
    f2 = _cf_component(u, 2, S0, v0, r, T, kappa, theta, xi, rho)
    integ1 = np.real(np.exp(-i * u * lnK) * f1 / (i * u))
    integ2 = np.real(np.exp(-i * u * lnK) * f2 / (i * u))
    P1 = 0.5 + (1.0 / np.pi) * np.sum(w * integ1)
    P2 = 0.5 + (1.0 / np.pi) * np.sum(w * integ2)

    call = S0 * P1 - K * np.exp(-r * T) * P2
    if option_type == "call":
        return float(call)
    return float(call - S0 + K * np.exp(-r * T))   # put-call parity
