# Ensemble and model comparison

## How to run

From the project folder, run `python evaluation.py` with the project environment active. The script reads the tuned parameters for [Random Forest](../results/rf_best_params.json), [Logistic Regression](../results/lr_best_params.json) and [HistGradientBoosting](../results/hgb_best_params.json). See [the run guide](../RUNNING.md) for setup and run order.

## Part F — Build the ensemble

[evaluation.py](../evaluation.py) rebuilds the three tuned models and ranks them by mean ROC-AUC across the same five stratified validation folds. HGB (0.7862) and Random Forest (0.7859) ranked ahead of Logistic Regression (0.7701).

Soft Voting averages the two selected models' default probabilities. Stacking uses Logistic Regression to combine out-of-fold predictions from the same two models. Soft Voting scored **0.7875** and Stacking scored **0.7873** in validation. The script uses a 0.001 ROC-AUC margin and selects the simpler Soft Voting approach when the two results are this close. See the [validation results](../results/eval_cv_comparison.csv).

## Part G — Compare the four models

The final comparison uses the same held-out test set for all four models. Recall and F1 below use a probability threshold of 0.5.

| Model | Test ROC-AUC | Default recall | Default F1 |
| --- | ---: | ---: | ---: |
| Tuned Logistic Regression | 0.7601 | 0.3444 | 0.4529 |
| Tuned Random Forest | 0.7764 | 0.6081 | 0.5396 |
| Tuned HGB | 0.7797 | 0.3677 | 0.4729 |
| Soft Voting (HGB + Random Forest) | 0.7790 | 0.4800 | 0.5286 |

HGB is the report's recommended model because it had the highest test ROC-AUC and was faster to fit and predict with than the ensemble in this run. Random Forest caught the most defaults at the 0.5 threshold, with lower precision (0.4850). Soft Voting caught more defaults than HGB at that threshold, while their test ROC-AUC scores were close. The bootstrap interval for their ROC-AUC difference included zero.

Results and figures are available in the [test comparison](../results/eval_test_results.csv), [threshold-adjusted comparison](../results/eval_test_results_tuned_threshold.csv), [ensemble significance checks](../results/eval_ensemble_significance.csv), [ROC curves](../figures/11_roc_curves_test.png) and [confusion matrices](../figures/12_confusion_matrices_test.png).
