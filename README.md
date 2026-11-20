# roi.lens

A business case that says "payback in fourteen months" has hidden every uncertain thing it depends on: how many agents there are, how many tickets the tool will really deflect, whether the integration slips by a quarter. Leadership approves the number and discovers the assumptions later, as surprises.

roi.lens shows the range instead, and what it would take to narrow it. Give each uncertain input a distribution, describe the timing and the risks, and it runs a seeded simulation that reports the odds of making money, the average of the bad cases, how long payback takes, which inputs drive the answer, and what it would be worth to find out more before you commit. Standard library only, Python 3.10 to 3.13.

Capital allocation conversations get better when the question changes from "what is the ROI" to "how likely is it that we lose money, and what would change that". To put the same discipline on a single decision between building and buying, see [build.buy](https://github.com/AndMaverick/build.buy).

## What it does

A scenario is a JSON file. Inputs are ranges, benefits and costs are formulas over those inputs, and risks are events with a probability. An abridged excerpt (the full file is `examples/support_copilot.json`):

```json
{
  "name": "Support copilot for a 50-agent help desk",
  "horizon_months": 36, "build_months": 4, "ramp_months": 6, "ramp": "smooth",
  "benefit_decay_per_year": 0.08, "discount_rate": 0.10,
  "inputs": {
    "agents": {"dist": "pert", "low": 40, "mode": 50, "high": 70},
    "deflect_rate": {"dist": "ci90", "low": 0.25, "high": 0.55, "lower": 0, "upper": 1},
    "cost_per_ticket": {"dist": "ci90", "low": 4.5, "high": 9.5, "kind": "lognormal"}
  },
  "benefits": {"deflected_ticket_cost": "agents * tickets_per_agent_month * deflect_rate * cost_per_ticket"},
  "events": [{"name": "integration_slips", "probability": 0.30, "kind": "delay", "amount": {"dist": "uniform", "low": 1, "high": 5}}],
  "gates": {"min_prob_positive": 0.8, "max_payback_p90_months": 30}
}
```

```
python -m roi_lens analyze examples/support_copilot.json
```

```
Expected NPV        514,407  (standard error 5,554)
P(NPV > 0)          92.3%  (95% interval 91.5% to 93.0%)
NPV p10 / p50 / p90 35,942 / 479,004 / 1,021,323
Bad-case average    -125,861  (mean of the worst 5% of outcomes)
Payback             93.4% pay back within 36 months; median 15 months, p90 27 months

What moves the answer (rank correlation with NPV)
  cost_per_ticket             +0.53
  deflect_rate                +0.51
  sponsor_pulls_funding       -0.37

Decision gates
  PASS  max_payback_p90_months: 27 months against 30 months
  PASS  min_expected_npv: 514,407 against 0
  PASS  min_p10_npv: 35,942 against -100,000
  PASS  min_prob_positive: 92.3% against 80.0%

Fingerprint 8a57496a5500d683709181d8fc9a81b3ef49a464d5e2629780c1c28215e6795c
```

The CLI exits 0 when every gate passes and 1 when one fails, so an investment rule becomes a CI step. `examples/marginal_copilot.json` is the same project with less optimistic inputs; it fails its gates and exits 1. The ranges in both files are illustrative, not drawn from any client or real deployment.

## The pieces

- **Distributions.** Fixed, uniform, triangular, PERT (the standard estimating Beta), truncated normal, lognormal, empirical, and `ci90`: "I am 90 percent sure it is between A and B", which is the form that can be calibrated. Every distribution has `ppf`, `cdf`, `mean` and `var`, so samples can be checked against the exact answer.
- **Safe formulas.** Formulas are parsed with `ast` and restricted to arithmetic, comparisons, `a if c else b`, and `min max abs sqrt log exp ceil floor`. They never reach `eval`; attribute access, subscripts, lambdas, strings and imports are rejected before anything runs.
- **Latin hypercube sampling** by default (simple random is a flag), and **rank correlation** between inputs by a Gaussian copula that only reorders sampled values, so each input keeps exactly its distribution.
- **Cash flow model.** Build phase, the month it goes live, ramp (linear or smooth), benefit decay, discounting. Outputs NPV, net, ROI, payback month and IRR.
- **Risk events.** Delay, benefit shortfall, one time extra cost, and cancellation, each with a probability.
- **What drives it.** Rank correlation of every input and every risk with the outcome, a deterministic tornado, and the values at which the answer breaks even, with the probability the input lands on the losing side.
- **What information is worth.** EVPI (the most any study could be worth) and EVPPI per input (what learning that one input exactly would be worth), so "should we run a pilot" has a number: `python -m roi_lens sensitivity scenario.json --pilot-cost 25000`.
- **Portfolio under a budget.** Exhaustive over every feasible combination, with the optimum by mean NPV exact, and a second ranking by p10 or by probability of profit.
- **Calibration.** Feed it your past 90 percent intervals and the answers; it reports your hit rate with an interval, and the factor that would have made your intervals right.

## Verification

`python examples/evaluate.py` reproduces these figures in about six seconds. Each compares the library with an answer computed a different way.

| Check | Result |
|---|---|
| PERT quantile solver, 20,000 random shapes | largest CDF residual 6.7e-15 |
| Incomplete beta vs exact binomial sum | 2,000 of 2,000 agree to 1e-9 |
| Formula evaluator vs Python's own `eval` | 4,988 of 4,988 random formulas identical |
| EVPI of a sum of two normals | 1.2010 against closed form 1.2022 (standard error 0.0035) |
| EVPPI of each input | 0.5703 against 0.5727, and 0.9642 against 0.9634 |
| Simulated quantiles of a normal sum | largest error 0.041 on a spread of 3.61 |
| Cash flow engine against a month by month oracle | 1,500 random models, largest relative gap 7.0e-14 |
| Median interval coverage | 95.2% against 95% nominal |
| Wilson interval coverage | 96.2% against 95% nominal |
| Calibration correction reaches 90% on the sample it was fitted to | 300 of 300 |
| Latin hypercube on an additive model | variance of the mean estimate about 10,000 times lower than simple random |
| Full analysis (9 inputs, 4 risks, 5,000 trials) | about 1.3 seconds, identical on every repeat |

The test suite has 125 tests, including: Beta(3,3) at 0.25 equals 0.103515625 exactly, the break even of a linear model equals its root solved by hand, a knapsack solved by dynamic programming agrees with the portfolio optimiser on 25 random problems, results are identical across `PYTHONHASHSEED` values and across the order keys are written in, and every unknown key in a scenario file is an error.

## Reproducible

Every report ends with a fingerprint: the SHA-256 of the scenario, trial count, seed, sampling method and library version. The same fingerprint gives the same numbers on any machine, so a number in a board paper can be derived again by anyone who has the file.

## Honest limits

- NPV is over the horizon you set, with no terminal value. A project that pays back in year four looks worse on a three year horizon, by design.
- Costs and benefits are formulas of the inputs; there is no learning curve beyond ramp and decay, and no feedback between months.
- IRR can have several roots when cash flows change sign more than once; the report samples 500 trials for it and says how many had one.
- The tornado holds every other input at its median and risk events off. It shows sensitivity, not probability.
- EVPPI is estimated by binning and is biased slightly upward with few trials; EVPI is an upper bound on any study, since it assumes the uncertainty is resolved completely.
- Portfolio cases are simulated independently; correlation between projects is not modelled. The `p10` and `prob_positive` objectives rank the 64 best again subsets by mean, which can miss a subset with a much safer spread and a worse mean.
- The quality of the answer is the quality of the ranges. Use `calibrate` to find out how far to trust your own.

## Compatibility

The 0.1 interface is unchanged and its results reproduce exactly (`UseCase`, `point_estimate`, `simulate`, `rank`, `triangular`):

```python
from roi_lens import UseCase, point_estimate, simulate

case = UseCase("support-copilot", users=200, hours_saved_per_user_month=(2, 4, 6), hourly_cost=40,
               adoption=(0.5, 0.7, 0.9), monthly_run_cost=3000, build_cost=60000)

point_estimate(case)   # {'net': 172800.0, 'roi': 2.88}
simulate(case)         # {'p10': 89602.26, 'p50': 170260.0, 'p90': 257907.36, 'prob_positive': 1.0}
```

`simulate` accepts either a `UseCase` (the 0.1 behaviour) or a `Model` (the full engine).

## Tests

```
python -m unittest discover -s tests -t . -v
```

MIT licensed. Built by [AndMaverick LLC](https://www.andmaverick.com).

---

**Related repositories:** [build.buy](https://github.com/AndMaverick/build.buy) compares building with buying on one timeline. [token.econ](https://github.com/AndMaverick/token.econ) models the unit economics of an AI product.

**If this is your problem.** If your AI spend is outrunning your results, that is an architecture problem, and it is solvable. I am happy to look at it with you. [Begin a conversation](https://calendly.com/andmaverick?utm_source=github&utm_medium=repo&utm_campaign=roi.lens).
