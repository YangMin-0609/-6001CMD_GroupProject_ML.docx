# Credit Card Default Prediction – Parts E, F and G

UCI Default of Credit Card Clients (id=350). All four scripts use the same data,
the same 80/20 stratified split (`random_state=42`), the same cleaning, the same
preprocessing (RobustScaler + one-hot encoding), the same 5 stratified folds and
the same five metrics as the team notebook, so every model is compared the same way.

| Script | Member | Part | Run time (approx.) |
|---|---|---|---|
| `lr_tuning.py` | Member 1 | E – tune Logistic Regression | 2–5 min |
| `rf_tuning.py` | Member 2 | E – tune Random Forest | 15–30 min |
| `hgb_tuning.py` | Member 3 | E – tune HistGradientBoosting | 5–15 min |
| `evaluation.py` | Member 4 | F – ensemble, G – evaluation and comparison | 10–30 min |

## How to run

From the project folder, with the virtual environment active:

```
python lr_tuning.py
python rf_tuning.py
python hgb_tuning.py
python evaluation.py
```

The three tuning scripts can run in any order, but `evaluation.py` must run last,
because it reads `results/lr_best_params.json`, `results/rf_best_params.json` and
`results/hgb_best_params.json`.

## What each step does

- **Tuning scripts (Part E):** baseline cross-validation, a coarse search
  (grid search for Logistic Regression, randomised search for the two tree models),
  a finer grid around the best result, a baseline-vs-tuned table, an overfitting
  check (training vs validation ROC-AUC), a validation curve and the tuned
  parameters for Member 4. The test set is not used.
- **evaluation.py (Parts F and G):**
  1. Rebuilds the three tuned models and evaluates them on the same 5 folds.
  2. Selects the two models with the highest validation ROC-AUC; the third is excluded.
  3. Combines them with soft voting and with stacking, and keeps the better one
     (voting if they are within 0.001, because it is simpler).
  4. Checks whether the ensemble is really better: paired t-test on the 5 folds and a
     1,000-sample bootstrap on the test set.
  5. Evaluates all four final models once on the test set, at the 0.5 cut-off and at
     an F1-optimal cut-off chosen on training data only.

## Outputs

- `results/` – CSV and JSON tables for the report (`lr_*`, `rf_*`, `hgb_*`, `eval_*`)
- `figures/` – `05`–`06` Random Forest, `07`–`08` Logistic Regression,
  `09` HistGradientBoosting, `10`–`12` comparison, ROC curves and confusion matrices
