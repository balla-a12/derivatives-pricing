# Derivatives Pricing and Heston Calibration

An options pricing library spanning analytic Black-Scholes with the full Greek set, the binomial and trinomial lattices, Monte Carlo for vanilla, Asian, and barrier options, the Heston stochastic-volatility model in both Monte Carlo and characteristic-function form, and calibration of the Heston parameters to an implied-volatility surface. Every model is cross-checked against an independent calculation.

_Not investment advice._

**Stack:** Python · numpy · scipy · matplotlib

---

## What it does

The package prices equity options across four modelling approaches and fits the Heston model to market data. It grew out of the WQU MScFE 620 Derivative Pricing coursework and the Heston/barrier group project, refactored into an installable, tested package with a consistent interface: every pricer takes explicit inputs and selects the contract with `option_type='call'` or `'put'`.

## Validation is built in

Each module is checked against another, so a mistake surfaces as a disagreement rather than passing quietly:

- The closed-form Greeks match central finite differences of the price.
- The binomial and trinomial lattices converge to the Black-Scholes value as the step count grows.
- The Monte Carlo price agrees with the closed form to within its standard error.
- Barrier prices satisfy in-out parity: a knock-out plus its matching knock-in reconstructs the vanilla option on the same paths.
- The Heston characteristic-function pricer matches the Monte Carlo version and reproduces the group project's call prices of $3.48$ at $\rho = -0.30$ and $3.50$ at $\rho = -0.70$, collapsing to Black-Scholes as vol-of-vol vanishes.
- Calibration recovers known parameters from a synthetic surface to machine precision.

Nine tests encode these checks and run with `pytest`.

## Models

| Module | Contents |
| --- | --- |
| `analytic` | Black-Scholes-Merton price, full Greeks, put-call parity, Newton-Raphson implied volatility |
| `trees` | Cox-Ross-Rubinstein binomial and Boyle trinomial, European and American |
| `montecarlo` | Vanilla, arithmetic-average Asian, and barrier options, each with a standard error, plus a convergence helper |
| `heston` | Euler full-truncation Monte Carlo and a characteristic-function pricer via Gil-Pelaez inversion |
| `calibration` | Least-squares fit of $\{v_0, \kappa, \theta, \xi, \rho\}$ to an implied-volatility surface with residual diagnostics |

## Getting started

```bash
git clone https://github.com/<user>/derivatives-pricing.git
cd derivatives-pricing
pip install -e .
pytest -q
```

```python
from derivpricing.analytic import black_scholes
from derivpricing.heston import heston_cf_price
from derivpricing.calibration import calibrate_heston

black_scholes(100, 100, 1.0, 0.05, 0.2, "call")          # price and Greeks
heston_cf_price(80, 80, 0.25, 0.055, 0.032, 1.85, 0.045, 0.35, -0.30)   # Heston call
```

## Notebooks

| Notebook | Focus |
| --- | --- |
| `01_pricing_and_greeks` | Black-Scholes pricing, Greeks against finite differences, implied-vol round-trip, lattice convergence |
| `02_monte_carlo_convergence` | Monte Carlo for vanilla, Asian, and barrier options, with the $1/\sqrt{M}$ error convergence and in-out parity |
| `03_heston_calibration` | Heston priced two ways, the volatility smile, and calibration to a surface with the fit shown against the market |

Each notebook clones the repo, installs the package, and runs on `numpy`/`scipy` with no external data or keys.

## Grounding

The Heston Monte Carlo scheme, the barrier parity checks, and the calibration parameters come from the MScFE 620 group project, and the characteristic-function pricer is validated against that project's reported prices. The lattice, analytic, and Vasicek-style models trace back to the course modules.

---

*Built as a portfolio project for the WorldQuant University MScFE program. Not investment advice.*
