# 6001CMD Group Project

We use the UCI Default of Credit Card Clients dataset to predict next-month credit card default using Logistic Regression, Random Forest and HistGradientBoosting. Start with the [final notebook](credit_default_full.ipynb), or download the [processed dataset](data/credit_default_processed.csv). See the [dataset notes](data/README.md) for column definitions, the split and attribution.

The team submits one shared notebook, `credit_default_full.ipynb`. For individual contributions, each member shares their own Python script and the results and figures it produces. The `Logistic-Regression` branch is the example for how to organise these files.

## Who did what

| Member | Branch and commit | Script | Main outputs |
| --- | --- | --- | --- |
| **1 — Goay Wei Jun (Random Forest)** | `Random-Forest`, [tuning `ff5f931`](https://github.com/YangMin-0609/-6001CMD_GroupProject_ML.docx/commit/ff5f931712f5a4800d21df775ff3b7e1170dd4d4) | [`rf_tuning.py`](rf_tuning.py) | [Best parameters](results/rf_best_params.json), [baseline vs tuned CV results](results/rf_baseline_vs_tuned.csv) and [validation curve](figures/05_rf_validation_curve_min_samples_leaf.png) |
| **2 — Woon Foo Yang (Logistic Regression)** | [`Logistic-Regression`](https://github.com/YangMin-0609/-6001CMD_GroupProject_ML.docx/tree/Logistic-Regression), [tuning `f19b403`](https://github.com/YangMin-0609/-6001CMD_GroupProject_ML.docx/commit/f19b40390a8c40be88c0108113a3f373119d8083), [results `a010e99`](https://github.com/YangMin-0609/-6001CMD_GroupProject_ML.docx/commit/a010e992a3dd398432c1f46332eb3ecd6e5db009) | [`lr_tuning.py`](lr_tuning.py) | [Best parameters](results/lr_best_params.json) and [baseline vs tuned CV results](results/lr_baseline_vs_tuned.csv) |
| **3 — Jeremy Tan Jian Pin (HistGradientBoosting)** | `Random-Forest`, first appears in the [integration commit `3cdc57c`](https://github.com/YangMin-0609/-6001CMD_GroupProject_ML.docx/commit/3cdc57cadd2003a36834a9d44a7f19ed221e0553) | [`hgb_tuning.py`](hgb_tuning.py) | [Best parameters](results/hgb_best_params.json) and [baseline vs tuned CV results](results/hgb_baseline_vs_tuned.csv) |
| **4 — Ensemble and comparison** | `Random-Forest`, first appears in the same integration commit `3cdc57c`. [Ensemble guide](docs/member-4-ensemble.md) | [`evaluation.py`](evaluation.py) | [Ensemble CV results](results/eval_cv_comparison.csv), [final test comparison](results/eval_test_results.csv), [ROC curves](figures/11_roc_curves_test.png) and [confusion matrices](figures/12_confusion_matrices_test.png) |

## How the final version came together

1. **Parts D and E — Baselines and tuning.** Members 1–3 can run their scripts in parallel. All three use the same preprocessing, 80/20 stratified split and five CV folds. LR uses grid search and a refined grid. RF and HGB use randomised search followed by a refined grid. Each script saves a `*_best_params.json` file for Member 4.
2. **Part F — Ensemble.** Member 4 runs `evaluation.py` after all three JSON files are available. It rebuilds the tuned models and ranks them by mean validation ROC-AUC. HGB (0.7862) and RF (0.7859) were the top two. The script compares Soft Voting and Stacking and selects **Soft Voting** as the fourth model.
3. **Part G — Final comparison.** The same script compares tuned LR, tuned RF, tuned HGB and Soft Voting on the same held-out test set. The notebook brings the steps and results together.

See [RUNNING.md](RUNNING.md) for run commands and output details.
