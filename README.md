# Quantitative Risk Portfolio

I'm Paul Bulliard. I built this portfolio to demonstrate how I think
about measuring and managing financial risk: three self-contained
projects, each answering a business question a risk consulting team
would actually be asked, each validated rather than merely computed.

The code is written to be read — and to be defended line by line in a
technical discussion: explicit variable names, small single-purpose
functions, comments that explain the finance rather than the Python,
and pytest suites that check model properties (convergence laws,
no-arbitrage identities, statistical calibration) instead of
implementation details. Every simulation is seeded, so every number
and figure in the READMEs reproduces exactly.

## The projects

### [1 — Margin Call Liquidity Risk Simulator](01-margin-call-liquidity/)

A commodity trader long physical inventory and short futures is
perfectly hedged in value but not in cash: futures are marked to market
daily while the physical is not, so a price rally drains cash through
variation margin even though nothing has been lost economically — the
mechanism that broke several European energy traders in 2022. The
project simulates a mean-reverting commodity price (Schwartz 1997
one-factor model, exact discretisation), marks a fixed-maturity short
futures hedge to market daily, and sizes the committed credit line
needed to survive at 99% confidence (Liquidity-at-Risk). Headline
result: a value-hedged $80m position needs a ~$52m credit line — and
ignoring mean reversion (GBM) would oversize it by ~45%.
*Demonstrates: stochastic process simulation, funding vs market risk,
Monte Carlo risk measurement, parameter sensitivity analysis.*

### [2 — VaR and Expected Shortfall Engine with Backtesting](02-var-es-backtesting/)

Calculating VaR is one quantile; the substance is proving the model
delivers the coverage it promises. Three methods (historical
simulation, parametric, Monte Carlo) are backtested out of sample over
1,003 days with a 250-day rolling window and judged with the Kupiec
proportion-of-failures test, the Christoffersen independence test, and
Basel traffic light zones. All three pass on exception counts, with
historical simulation best calibrated (10 exceptions vs 10.0 expected
at 99%) — but borderline Christoffersen p-values expose the shared
weakness: exceptions cluster in volatility spikes, because
unconditional windows react too slowly. The README also covers why
FRTB moved regulatory capital from VaR to ES.
*Demonstrates: risk model validation, statistical hypothesis testing,
regulatory frameworks, honest interpretation of borderline results.*

### [3 — Independent Model Validation of Option Pricing](03-option-pricing-validation/)

Not "I built a Black-Scholes pricer": the same option is priced three
independent ways — closed form (with all seven Greeks in market
conventions), antithetic Monte Carlo (with honest confidence
intervals), and a Cox-Ross-Rubinstein tree (European and American) —
and the work is demonstrating they agree at the expected accuracy for
the expected reasons. 56/56 grid checks pass; Monte Carlo error falls
as 1/√n and the tree as ~1/n with its characteristic oscillation;
analytical and finite-difference Greeks agree to 1e-5; put-call parity
holds to machine precision; an implied-vol smile survives a round trip
through the pricer to 14 decimal places.
*Demonstrates: independent reimplementation as a validation technique,
numerical methods and their convergence signatures, Greeks and market
conventions, no-arbitrage reasoning.*

## Running everything

```bash
git clone <this-repo>
cd quant-risk-portfolio
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the test suites (17 tests across the three projects):

```bash
./run_all_tests.sh
```

(Each project is self-contained with its own `src` package, so the
suites run as separate pytest sessions — run `pytest` from inside a
project folder to test just that one.)

Reproduce any project's figures and tables:

```bash
cd 01-margin-call-liquidity   # or 02-... / 03-...
python run_analysis.py
```

No network access is needed: project 2's market data is cached in a
committed CSV (`yfinance` is only required to refresh it).

## Libraries

Pinned in [requirements.txt](requirements.txt): `numpy` and `pandas`
for computation, `scipy` for distributions, statistical tests and root
finding, `matplotlib` for figures, `pytest` for the test suites, and
`yfinance` solely to (re)download project 2's price cache. Nothing
else — no risk or pricing libraries; every model is implemented from
the equations.
