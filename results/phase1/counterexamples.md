# Phase 1: Mathematical Counterexamples and Formal Demonstrations

## Dropping normalizer term underestimates actual error

```json
{
  "title": "Dropping normalizer term underestimates actual error",
  "naive_bound": 0.0,
  "actual_error": 0.09715128297332737,
  "full_certified_bound": 0.9466730130639571,
  "underestimation_factor": Infinity,
  "soundness_verified": true,
  "parameters": {
    "Z_A": 1.0,
    "L_O": 0.2,
    "U_O": 10.0,
    "Z_hat_O": 5.1,
    "Z_O_true": 9.5,
    "B_O": 0.0,
    "r": [
      0.0,
      1.0
    ],
    "y_star": [
      0.09523809523809523,
      0.9047619047619048
    ],
    "y_hat_A": [
      0.1639344262295082,
      0.8360655737704918
    ]
  }
}
```

## Bounded recurrent state cannot achieve arbitrary exact recall past capacity

```json
{
  "title": "Bounded recurrent state cannot achieve arbitrary exact recall past capacity",
  "d_k": 2,
  "d_v": 1,
  "num_associations": 6,
  "max_recall_error_sequential": 1.845091873125757,
  "mean_recall_error_sequential": 0.8372927970219627,
  "max_recall_error_optimal_least_squares": 1.2440169358562925,
  "recalled_errors_per_token": [
    0.06526428962874675,
    1.845091873125757,
    0.6015222283113841,
    0.8450918731257568,
    1.6667865179401309,
    0.0
  ],
  "conclusion": "With dim 2, state has at most 2 degrees of freedom; N=6 arbitrary associations force collision and recall loss."
}
```

## Finite softmax is an interpolator, not a hard addressable lookup

```json
{
  "title": "Finite softmax is an interpolator, not a hard addressable lookup",
  "num_items": 10,
  "kappa_sweeps": [
    {
      "kappa": 0.5,
      "target_probability": 0.15482809896025468,
      "distractor_leakage": 0.8451719010397454
    },
    {
      "kappa": 1.0,
      "target_probability": 0.23196931668407392,
      "distractor_leakage": 0.768030683315926
    },
    {
      "kappa": 2.0,
      "target_probability": 0.4508530603792838,
      "distractor_leakage": 0.5491469396207163
    },
    {
      "kappa": 5.0,
      "target_probability": 0.942825618574015,
      "distractor_leakage": 0.057174381425984966
    },
    {
      "kappa": 10.0,
      "target_probability": 0.9995915675173918,
      "distractor_leakage": 0.0004084324826082453
    }
  ],
  "conclusion": "Finite softmax always leaks probability mass to non-matching keys; retrieval mixes values for all finite kappa."
}
```

## Error reduction is not monotonic on each fetch

```json
{
  "title": "Error reduction is not monotonic on each fetch",
  "initial_bound_before_fetch": 8.341666666666667,
  "bound_after_fetching_page_1": 9.375,
  "bound_increase": 1.0333333333333332,
  "monotonicity_violated": true,
  "mechanism": "Fetching page 1 with non-prior values shifted y_hat away from r, increasing normalizer uncertainty penalty faster than unread residual reduced."
}
```

