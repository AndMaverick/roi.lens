# roi.lens

Most AI business cases show one number. roi.lens shows the range. Give each uncertain input a low, expected, and high value, and it runs a seeded Monte Carlo simulation to report the 10th, 50th, and 90th percentile net value and the probability the project makes money at all. Standard library only.

Capital allocation conversations get better when the question changes from "what is the ROI" to "how likely is it that we lose money, and what would change that". If you are deciding where to put AI budget, I run that analysis with leadership teams through an [architecture review](https://calendly.com/andmaverick/30min?utm_source=github&utm_medium=repo&utm_campaign=roi.lens).

## Use

```python
from roi_lens import UseCase, point_estimate, simulate

case = UseCase(
    "support-copilot", users=200,
    hours_saved_per_user_month=(2, 4, 6),   # low, expected, high
    hourly_cost=40,
    adoption=(0.5, 0.7, 0.9),
    monthly_run_cost=3000, build_cost=60000,
)

point_estimate(case)   # {'net': 172800.0, 'roi': 2.88}
simulate(case)         # {'p10': 89602.26, 'p50': 170260.0, 'p90': 257907.36, 'prob_positive': 1.0}
```

The inputs above are illustrative, not drawn from any client or real deployment. A weak case, such as ten users saving an hour a month, shows a negative median and zero probability of profit, which is the useful result.

## Design

- **Ranges in, ranges out.** Triangular distributions over low, expected, and high values.
- **Reproducible.** The simulation is seeded, so the same inputs give the same answer.
- **Rank a portfolio.** rank orders use cases by median net value and shows the odds of loss alongside.
- **Validated.** A range where low exceeds expected is rejected.

## Tests

```
python -m unittest discover -s tests -t . -v
```

MIT licensed. Built by [AndMaverick LLC](https://www.andmaverick.com).

---

**Related reading:** [The Meter Is Running. Nobody Built the Architecture to Stop It.](https://blog.andmaverick.com/the-meter-is-running-nobody-built-the-architecture-to-stop-it?utm_source=github&utm_medium=repo&utm_campaign=roi.lens)

**If this is your problem.** If your AI spend is outrunning your results, that is an architecture problem, and it is solvable. I am happy to look at it with you. [Begin a conversation](https://calendly.com/andmaverick?utm_source=github&utm_medium=repo&utm_campaign=roi.lens).
