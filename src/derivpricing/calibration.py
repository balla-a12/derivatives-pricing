"""Calibrate the Heston parameters to a market implied-volatility surface.

The five parameters ``{v0, kappa, theta, xi, rho}`` are fit by least squares so the
model implied volatilities match the market surface, using the fast characteristic-
function pricer for each evaluation. The fit residuals are returned so the quality of
the calibration stays visible rather than assumed.
"""
import numpy as np
from scipy.optimize import least_squares

from .heston import heston_cf_price
from .analytic import implied_vol, bs_price

PARAM_NAMES = ["v0", "kappa", "theta", "xi", "rho"]
LOWER = [1e-4, 1e-2, 1e-4, 1e-2, -0.999]
UPPER = [1.0, 20.0, 1.0, 5.0, 0.999]


def implied_vol_surface(S0, r, strikes, maturities, price_fn, option_type="call"):
    """Build an implied-volatility grid from a pricing function.

    ``price_fn(K, T)`` returns an option price; each price is inverted to its
    Black-Scholes implied volatility. Returns a dict keyed by (K, T).
    """
    surface = {}
    for T in maturities:
        for K in strikes:
            price = price_fn(K, T)
            iv, _ = implied_vol(price, S0, K, T, r, option_type)
            surface[(float(K), float(T))] = float(iv)
    return surface


def _model_iv(params, S0, r, K, T, option_type):
    v0, kappa, theta, xi, rho = params
    price = heston_cf_price(S0, K, T, r, v0, kappa, theta, xi, rho, option_type)
    iv, _ = implied_vol(price, S0, K, T, r, option_type)
    return iv


def calibrate_heston(S0, r, market_surface, x0=None, option_type="call"):
    """Fit Heston parameters to a market implied-volatility surface.

    ``market_surface`` is a dict mapping (K, T) to a market implied volatility.
    Returns a dict with the fitted parameters, the per-quote residuals in vol points,
    the root-mean-square error, and the optimizer's success flag.
    """
    keys = list(market_surface.keys())
    strikes = np.array([k for k, _ in keys])
    mats = np.array([t for _, t in keys])
    target = np.array([market_surface[k] for k in keys])

    if x0 is None:
        x0 = [0.04, 2.0, 0.04, 0.4, -0.5]

    def residuals(params):
        model = np.array([_model_iv(params, S0, r, K, T, option_type)
                          for K, T in zip(strikes, mats)])
        return model - target

    result = least_squares(residuals, x0, bounds=(LOWER, UPPER), xtol=1e-8, ftol=1e-8)
    fitted = dict(zip(PARAM_NAMES, result.x))
    res = result.fun
    return {"params": fitted,
            "residuals": {k: float(r_) for k, r_ in zip(keys, res)},
            "rmse": float(np.sqrt(np.mean(res ** 2))),
            "max_abs_residual": float(np.max(np.abs(res))),
            "success": bool(result.success)}
