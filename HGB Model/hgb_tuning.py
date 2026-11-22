"""
Part E - Hyperparameter Optimisation: HistGradientBoosting (Member 3)
UCI Default of Credit Card Clients (id=350)

Run from the project folder:
    python hgb_tuning.py

The data loading, train/test split, cleaning, preprocessing, folds and metrics
are identical to the team notebook and to lr_tuning.py / rf_tuning.py, so the
results can be compared directly with the other models.

Outputs
    results/  - tables for the report and for Member 4
    figures/  - 09_hgb_validation_curve_learning_rate.png

The test set is NOT used in this script; it is reserved for the final evaluation.
Expected run time: about 5-15 minutes on a typical laptop.
"""

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save figures to files without opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
    validation_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

# Number of random combinations in Stage 1 (reduce to 20 if too slow)
N_ITER = 40


# =============================================================================
# 1. Shared setup (same as the team notebook)
# =============================================================================

COLUMN_NAMES = {
    "X1": "LIMIT_BAL", "X2": "SEX", "X3": "EDUCATION", "X4": "MARRIAGE", "X5": "AGE",
    "X6": "PAY_0", "X7": "PAY_2", "X8": "PAY_3", "X9": "PAY_4", "X10": "PAY_5", "X11": "PAY_6",
    "X12": "BILL_AMT1", "X13": "BILL_AMT2", "X14": "BILL_AMT3",
    "X15": "BILL_AMT4", "X16": "BILL_AMT5", "X17": "BILL_AMT6",
    "X18": "PAY_AMT1", "X19": "PAY_AMT2", "X20": "PAY_AMT3",
    "X21": "PAY_AMT4", "X22": "PAY_AMT5", "X23": "PAY_AMT6",
}

CATEGORICAL_COLUMNS = [
    "SEX", "EDUCATION", "MARRIAGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
]


def load_data():
    """Load the UCI dataset and give columns descriptive names."""
    credit = fetch_ucirepo(id=350)
    X = credit.data.features.copy().rename(columns=COLUMN_NAMES)
    y = credit.data.targets.iloc[:, 0].copy().rename("DEFAULT_NEXT_MONTH")
    return X, y


def clean_category_codes(features):
    """Same cleaning rules as the notebook."""
    cleaned = features.copy()
    cleaned["EDUCATION"] = cleaned["EDUCATION"].replace({5: 0, 6: 0})
    for column in ["SEX", "EDUCATION", "MARRIAGE"]:
        cleaned[column] = cleaned[column].astype("category")
    return cleaned


def build_preprocessor(numeric_columns):
    """RobustScaler for numerical features, one-hot encoding for categorical ones."""
    return ColumnTransformer(transformers=[
        ("num", RobustScaler(), numeric_columns),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORICAL_COLUMNS),
    ])


def build_scoring():
    """Same five metrics as the notebook; precision/recall/F1 for the default class."""
    return {
        "Accuracy": "accuracy",
        "ROC_AUC": "roc_auc",
        "Precision": make_scorer(precision_score, pos_label=1, zero_division=0),
        "Recall": make_scorer(recall_score, pos_label=1, zero_division=0),
        "F1": make_scorer(f1_score, pos_label=1, zero_division=0),
    }


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def search_summary(search, top=10):
    """Return the top configurations of a search as a readable table."""
    results = pd.DataFrame(search.cv_results_)
    n_folds = search.n_splits_

    param_columns = [c for c in results.columns if c.startswith("param_")]
    table = results[param_columns].copy()
    table.columns = [c.replace("param_model__", "") for c in param_columns]

    val_folds = results[[f"split{i}_test_ROC_AUC" for i in range(n_folds)]]
    table["Val ROC_AUC"] = results["mean_test_ROC_AUC"]
    table["Val ROC_AUC std"] = val_folds.std(axis=1, ddof=1)
    table["Train ROC_AUC"] = results["mean_train_ROC_AUC"]
    table["Gap"] = table["Train ROC_AUC"] - table["Val ROC_AUC"]
    table["Val F1"] = results["mean_test_F1"]
    table["Val Recall"] = results["mean_test_Recall"]
    table["Fit time (s)"] = results["mean_fit_time"]
    table["Rank"] = results["rank_test_ROC_AUC"]
    return table.sort_values("Rank").head(top)


def neighbours(value, ordered_options):
    """Return the value together with the values on either side of it."""
    position = ordered_options.index(value)
    return ordered_options[max(0, position - 1): position + 2]


# =============================================================================
# 2. Main
# =============================================================================

def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    # ---------------------------------------------------------------- data
    section("Loading data and recreating the shared setup")
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train_clean = clean_category_codes(X_train)
    # X_test is not used in this script (reserved for the final evaluation)

    numeric_columns = [c for c in X_train_clean.columns if c not in CATEGORICAL_COLUMNS]
    preprocessor = build_preprocessor(numeric_columns)
    scoring = build_scoring()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("Training set:", X_train_clean.shape)

    # Baseline HistGradientBoosting (identical to the notebook)
    hgb_pipeline = Pipeline([
        ("preprocess", clone(preprocessor)),
        ("model", HistGradientBoostingClassifier(random_state=42)),
    ])
    hgb_baseline_cv = cross_validate(
        hgb_pipeline, X_train_clean, y_train,
        cv=cv, scoring=scoring, return_train_score=True,
        n_jobs=-1, error_score="raise",
    )
    print("Baseline HGB validation ROC-AUC:",
          round(hgb_baseline_cv["test_ROC_AUC"].mean(), 4))

    # ------------------------------------------------- Stage 1: random search
    section(f"Stage 1 - Randomised search ({N_ITER} combinations x 5 folds)")
    # Early stopping is switched on explicitly: each model holds back 10% of its
    # training fold and stops adding trees when that score stops improving for
    # 20 rounds, so max_iter is only an upper limit.
    hgb_param_distributions = {
        "model__learning_rate": [0.02, 0.05, 0.1, 0.2],  # contribution of each tree
        "model__max_iter": [200, 400, 800],              # upper limit on trees
        "model__max_leaf_nodes": [15, 31, 63],           # size of each tree
        "model__max_depth": [3, 5, 8, None],
        "model__min_samples_leaf": [20, 50, 100, 200],   # minimum customers per leaf
        "model__l2_regularization": [0.0, 0.1, 1.0, 10.0],
        "model__max_features": [0.5, 0.8, 1.0],          # share of features per split
        "model__class_weight": [None, "balanced"],
        "model__early_stopping": [True],
        "model__n_iter_no_change": [20],
    }
    hgb_random_search = RandomizedSearchCV(
        estimator=hgb_pipeline,
        param_distributions=hgb_param_distributions,
        n_iter=N_ITER,
        scoring=scoring,
        refit="ROC_AUC",
        cv=cv,
        return_train_score=True,
        n_jobs=-1,
        random_state=42,
        error_score="raise",
    )
    start = time.perf_counter()
    hgb_random_search.fit(X_train_clean, y_train)
    hgb_random_time = time.perf_counter() - start
    print(f"Finished in {hgb_random_time / 60:.1f} minutes")
    print("Best validation ROC-AUC:", round(hgb_random_search.best_score_, 4))
    print("\nTop 10 configurations:")
    print(search_summary(hgb_random_search, top=10).round(4).to_string())
    pd.DataFrame(hgb_random_search.cv_results_).to_csv(
        RESULTS_DIR / "hgb_random_search_cv_results.csv", index=False
    )

    # --------------------------------------------------- Stage 2: grid search
    section("Stage 2 - Grid search around the best randomised result")
    best_random = hgb_random_search.best_params_
    # Finer scales; they contain every value used in Stage 1
    rate_options = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3]
    leaf_options = [10, 20, 30, 50, 75, 100, 150, 200, 300]

    # Fix all parameters at their best Stage 1 values, then refine the learning
    # rate and min_samples_leaf, the two settings that mainly control overfitting
    hgb_param_grid = {name: [value] for name, value in best_random.items()}
    hgb_param_grid["model__learning_rate"] = neighbours(
        best_random["model__learning_rate"], rate_options)
    hgb_param_grid["model__min_samples_leaf"] = neighbours(
        best_random["model__min_samples_leaf"], leaf_options)
    print("learning_rate values:", hgb_param_grid["model__learning_rate"])
    print("min_samples_leaf values:", hgb_param_grid["model__min_samples_leaf"])

    hgb_grid_search = GridSearchCV(
        estimator=hgb_pipeline, param_grid=hgb_param_grid, scoring=scoring,
        refit="ROC_AUC", cv=cv, return_train_score=True, n_jobs=-1, error_score="raise",
    )
    start = time.perf_counter()
    hgb_grid_search.fit(X_train_clean, y_train)
    hgb_grid_time = time.perf_counter() - start
    print(f"Finished in {hgb_grid_time / 60:.1f} minutes")
    print("Stage 1 best ROC-AUC:", round(hgb_random_search.best_score_, 4))
    print("Stage 2 best ROC-AUC:", round(hgb_grid_search.best_score_, 4))
    print()
    print(search_summary(hgb_grid_search, top=9).round(4).to_string())
    pd.DataFrame(hgb_grid_search.cv_results_).to_csv(
        RESULTS_DIR / "hgb_grid_search_cv_results.csv", index=False
    )

    # ------------------------------------------ Final tuned HGB vs baseline
    section("Tuned HistGradientBoosting vs baseline (same 5 folds)")
    hgb_best_params = hgb_grid_search.best_params_
    hgb_tuned_pipeline = clone(hgb_pipeline).set_params(**hgb_best_params)
    hgb_tuned_cv = cross_validate(
        hgb_tuned_pipeline, X_train_clean, y_train,
        cv=cv, scoring=scoring, return_train_score=True,
        n_jobs=-1, error_score="raise",
    )

    comparison_rows = []
    for metric in scoring:
        baseline_scores = hgb_baseline_cv[f"test_{metric}"]
        tuned_scores = hgb_tuned_cv[f"test_{metric}"]
        comparison_rows.append({
            "Metric": metric,
            "Baseline mean": baseline_scores.mean(),
            "Baseline std": baseline_scores.std(ddof=1),
            "Tuned mean": tuned_scores.mean(),
            "Tuned std": tuned_scores.std(ddof=1),
            "Change": tuned_scores.mean() - baseline_scores.mean(),
        })
    hgb_comparison = pd.DataFrame(comparison_rows).set_index("Metric")
    print(hgb_comparison.round(4).to_string())

    hgb_fit_check = pd.DataFrame({
        "Train ROC_AUC": [hgb_baseline_cv["train_ROC_AUC"].mean(),
                          hgb_tuned_cv["train_ROC_AUC"].mean()],
        "Validation ROC_AUC": [hgb_baseline_cv["test_ROC_AUC"].mean(),
                               hgb_tuned_cv["test_ROC_AUC"].mean()],
        "Mean fit time per fold (s)": [hgb_baseline_cv["fit_time"].mean(),
                                       hgb_tuned_cv["fit_time"].mean()],
    }, index=["Baseline HGB", "Tuned HGB"])
    hgb_fit_check.insert(2, "Gap (train - val)",
                         hgb_fit_check["Train ROC_AUC"] - hgb_fit_check["Validation ROC_AUC"])
    print("\nOverfitting and training-time check")
    print(hgb_fit_check.round(4).to_string())
    hgb_comparison.to_csv(RESULTS_DIR / "hgb_baseline_vs_tuned.csv")
    hgb_fit_check.to_csv(RESULTS_DIR / "hgb_fit_check.csv")

    # Number of trees actually used after early stopping (full training set)
    hgb_final_model = clone(hgb_tuned_pipeline).fit(X_train_clean, y_train)
    n_trees = hgb_final_model.named_steps["model"].n_iter_
    print(f"\nTrees used after early stopping: {n_trees} "
          f"(limit {hgb_best_params['model__max_iter']})")

    # ----------------------------------------------------- Validation curve
    section("Validation curve - learning_rate")
    chosen_rate = hgb_best_params["model__learning_rate"]
    rate_range = sorted({0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, chosen_rate})
    train_scores, val_scores = validation_curve(
        hgb_tuned_pipeline, X_train_clean, y_train,
        param_name="model__learning_rate", param_range=rate_range,
        cv=cv, scoring="roc_auc", n_jobs=-1,
    )
    train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1, ddof=1)
    val_mean, val_std = val_scores.mean(axis=1), val_scores.std(axis=1, ddof=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rate_range, train_mean, marker="o", color="#4C78A8", label="Training ROC-AUC")
    ax.fill_between(rate_range, train_mean - train_std, train_mean + train_std,
                    color="#4C78A8", alpha=0.15)
    ax.plot(rate_range, val_mean, marker="o", color="#E45756", label="Validation ROC-AUC")
    ax.fill_between(rate_range, val_mean - val_std, val_mean + val_std,
                    color="#E45756", alpha=0.15)
    ax.axvline(chosen_rate, color="grey", linestyle="--",
               label=f"Selected learning_rate = {chosen_rate:g}")
    ax.set_xscale("log")
    ax.set_xticks(rate_range)
    ax.set_xticklabels([f"{r:g}" for r in rate_range])
    ax.set_xlabel("learning_rate (log scale)")
    ax.set_ylabel("ROC-AUC (5-fold mean ± std)")
    ax.set_title("HistGradientBoosting Validation Curve — learning_rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "09_hgb_validation_curve_learning_rate.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    curve_table = pd.DataFrame({
        "learning_rate": rate_range, "Train ROC_AUC": train_mean,
        "Validation ROC_AUC": val_mean, "Gap": train_mean - val_mean,
    }).set_index("learning_rate")
    print(curve_table.round(4).to_string())
    curve_table.to_csv(RESULTS_DIR / "hgb_validation_curve.csv")

    # ------------------------------------------------- Hand-over to Member 4
    section("Hand-over files for Member 4")
    hgb_best_params_clean = {
        name.replace("model__", ""): (value.item() if hasattr(value, "item") else value)
        for name, value in hgb_best_params.items()
    }
    with open(RESULTS_DIR / "hgb_best_params.json", "w") as f:
        json.dump(hgb_best_params_clean, f, indent=2)

    hgb_oof_proba = cross_val_predict(
        hgb_tuned_pipeline, X_train_clean, y_train,
        cv=cv, method="predict_proba", n_jobs=-1,
    )[:, 1]
    pd.DataFrame({"y_true": y_train, "hgb_proba": hgb_oof_proba},
                 index=X_train_clean.index).to_csv(RESULTS_DIR / "hgb_oof_proba.csv")

    hgb_tuned_summary = pd.DataFrame(
        {metric: [hgb_tuned_cv[f"test_{metric}"].mean()] for metric in scoring},
        index=pd.Index(["HistGradientBoosting (tuned)"], name="Model"),
    )
    hgb_tuned_summary.to_csv(RESULTS_DIR / "hgb_tuned_cv_summary.csv")

    print("Tuned HistGradientBoosting parameters:")
    print(json.dumps(hgb_best_params_clean, indent=2))
    print()
    print(hgb_tuned_summary.round(4).to_string())
    print(f"\nTotal search time: {(hgb_random_time + hgb_grid_time) / 60:.1f} minutes")


# The main guard is required on Windows when n_jobs=-1 starts worker processes
if __name__ == "__main__":
    main()
