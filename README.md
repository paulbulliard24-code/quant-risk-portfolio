# Quantitative Risk Portfolio

Three self-contained projects on measuring and managing financial risk,
written to be read: explicit variable names, small single-purpose
functions, comments that explain the finance, and pytest suites that
check model properties rather than implementation details.

## Projects

| # | Project | Question it answers |
|---|---------|---------------------|
| 1 | [Margin Call Liquidity Risk Simulator](01-margin-call-liquidity/) | How large a committed credit line does a hedged commodity trader need to survive margin calls at 99% confidence? |
| 2 | [VaR and Expected Shortfall Backtesting](02-var-es-backtesting/) | Do our market risk models actually deliver the coverage they promise? |
| 3 | [Option Pricing Model Validation](03-option-pricing-validation/) | *(coming soon)* How would a validation team independently check an option pricing model? |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Each project folder has its own README with the model, parameters,
results, and limitations. To reproduce a project's results:

```bash
cd 01-margin-call-liquidity
pytest
python run_analysis.py
```

All simulations are seeded, so every number and figure in the READMEs is
exactly reproducible.

Note: run `pytest` from inside a project folder (or use
`./run_all_tests.sh` from the root). The projects are self-contained,
each with its own `src` package, so their suites run as separate pytest
sessions.

## Dependencies

`numpy`, `pandas`, `matplotlib`, `scipy`, `pytest` — plus `yfinance`,
used once in project 2 to download price data (the download is cached
and committed, so the repo runs fully offline).
