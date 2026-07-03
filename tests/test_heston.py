"""Heston tests: the characteristic-function pricer against Monte Carlo and the
Black-Scholes limit, and a calibration round-trip that recovers known parameters.
"""
import numpy as np
from derivpricing.heston import heston_cf_price, heston_mc
from derivpricing.analytic import bs_price
from derivpricing.calibration import implied_vol_surface, calibrate_heston

GWP2 = dict(S0=80, K=80, T=0.25, r=0.055, v0=0.032, kappa=1.85, theta=0.045, xi=0.35)


def test_cf_matches_mc():
    for rho in (-0.30, -0.70):
        cf = heston_cf_price(GWP2["S0"], GWP2["K"], GWP2["T"], GWP2["r"], GWP2["v0"],
                             GWP2["kappa"], GWP2["theta"], GWP2["xi"], rho, "call")
        mc, se = heston_mc(rho=rho, **GWP2)
        assert abs(cf - mc) < 4 * se


def test_cf_collapses_to_black_scholes():
    cf = heston_cf_price(100, 100, 1.0, 0.05, v0=0.04, kappa=2.0, theta=0.04,
                         xi=1e-3, rho=0.0, option_type="call")
    assert abs(cf - bs_price(100, 100, 1.0, 0.05, 0.2, "call")) < 1e-3


def test_calibration_recovers_parameters():
    S0, r = 80.0, 0.055
    true = dict(v0=0.032, kappa=1.85, theta=0.045, xi=0.35, rho=-0.50)
    strikes = [S0 / m for m in (0.85, 0.95, 1.00, 1.05, 1.15)]
    mats = [0.25, 0.5, 1.0]
    price_fn = lambda K, T: heston_cf_price(S0, K, T, r, option_type="call", **true)
    surface = implied_vol_surface(S0, r, strikes, mats, price_fn)
    res = calibrate_heston(S0, r, surface, x0=[0.05, 3.0, 0.05, 0.5, -0.2])
    assert res["success"] and res["rmse"] < 1e-4
    for k in true:
        assert abs(res["params"][k] - true[k]) < 1e-2
