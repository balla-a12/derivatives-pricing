"""Lattice pricers: Cox-Ross-Rubinstein binomial and Boyle trinomial.

Both handle European and American exercise for calls and puts, and both converge to
the Black-Scholes price as the step count grows, which makes them a useful independent
check on the analytic and Monte Carlo modules.
"""
import numpy as np


def binomial_tree(S0, K, T, r, sigma, N=500, option_type="call", american=False, q=0.0):
    """CRR binomial price.

    Calibration: ``u = exp(sigma sqrt(dt))``, ``d = 1/u``,
    ``p = (exp((r - q) dt) - d) / (u - d)``. Returns the option value at ``t = 0``.
    """
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    i = np.arange(N + 1)
    ST = S0 * u ** i * d ** (N - i)
    V = np.maximum(ST - K, 0.0) if option_type == "call" else np.maximum(K - ST, 0.0)

    for j in range(N - 1, -1, -1):
        V = disc * (p * V[1:j + 2] + (1 - p) * V[0:j + 1])
        if american:
            Sj = S0 * u ** np.arange(j + 1) * d ** (j - np.arange(j + 1))
            exercise = (Sj - K) if option_type == "call" else (K - Sj)
            V = np.maximum(V, np.maximum(exercise, 0.0))
    return float(V[0])


def trinomial_tree(S0, K, T, r, sigma, N=300, option_type="call", american=False, q=0.0):
    """Boyle trinomial price with second-moment matching.

    ``u = exp(sigma sqrt(2 dt))``, ``d = 1/u``, and the risk-neutral probabilities
    follow from matching the first two moments of the log return.
    """
    dt = T / N
    u = np.exp(sigma * np.sqrt(2 * dt))
    d = 1.0 / u
    a = np.exp((r - q) * dt / 2)
    b = np.exp(sigma * np.sqrt(dt / 2))
    c = 1.0 / b
    pu = ((a - c) / (b - c)) ** 2
    pd = ((b - a) / (b - c)) ** 2
    pm = 1 - pu - pd
    disc = np.exp(-r * dt)

    j = np.arange(N, -N - 1, -1)
    ST = S0 * u ** j
    V = np.maximum(ST - K, 0.0) if option_type == "call" else np.maximum(K - ST, 0.0)

    for step in range(N - 1, -1, -1):
        V = disc * (pu * V[0:2 * step + 1] + pm * V[1:2 * step + 2] + pd * V[2:2 * step + 3])
        if american:
            jj = np.arange(step, -step - 1, -1)
            Sj = S0 * u ** jj
            exercise = (Sj - K) if option_type == "call" else (K - Sj)
            V = np.maximum(V, np.maximum(exercise, 0.0))
    return float(V[0])
