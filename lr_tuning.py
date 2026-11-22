"""
Part E - Hyperparameter Optimisation: Logistic Regression (Member 1)
UCI Default of Credit Card Clients (id=350)

Run from the project folder:
    python lr_tuning.py

The data loading, train/test split, cleaning, preprocessing, folds and metrics
are identical to the team notebook and to rf_tuning.py / hgb_tuning.py, so the
results can be compared directly with the other models.

Outputs
    results/  - tables for the report and for Member 4
    figures/  - 07_lr_validation_curve_C.png
                08_lr_coefficients.png

The test set is NOT used in this script; it is reserved for the final evaluation.
Expected run time: about 2-5 minutes on a typical laptop.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import (
    GridSearchCV,
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
    # Merge undocumented education codes 5 and 6 into category 0
    cleaned["EDUCATION"] = cleaned["EDUCATION"].replace({5: 0, 6: 0})
    # Declare demographic codes as categorical
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

    # Baseline Logistic Regression (identical to the notebook)
    lr_baseline = Pipeline([
        ("preprocess", clone(preprocessor)),
        ("model", LogisticRegression(max_iter=2000)),
    ])
    lr_baseline_cv = cross_validate(
        lr_baseline, X_train_clean, y_train,
        cv=cv, scoring=scoring, return_train_score=True,
        n_jobs=-1, error_score="raise",
    )
    print("Baseline LR validation ROC-AUC:",
          round(lr_baseline_cv["test_ROC_AUC"].mean(), 4))

    # The tuned model uses the liblinear solver, which supports both
    # L2 (l1_ratio=0) and L1 (l1_ratio=1) regularisation for binary problems.
    lr_pipeline = Pipeline([
        ("preprocess", clone(preprocessor)),
        ("model", LogisticRegression(solver="liblinear", max_iter=2000)),
    ])

    # ------------------------------------------------- Stage 1: grid search
    # Logistic Regression has few hyperparameters and trains in about a second,
    # so every combination can be tried (grid search) instead of a random sample.
    section("Stage 1 - Grid search over C, penalty type and class weight")
    lr_param_grid = {
        # C = 1 / regularisation strength; small C = simpler model
        "model__C": [0.001, 0.01, 0.1, 1, 10, 100],
        "model__l1_ratio": [0.0, 1.0],              # 0 = L2 (ridge), 1 = L1 (lasso)
        "model__class_weight": [None, "balanced"],  # extra weight for defaults
    }
    lr_grid_search = GridSearchCV(
        estimator=lr_pipeline,
        param_grid=lr_param_grid,
        scoring=scoring,
        refit="ROC_AUC",            # Choose the best combination by validation ROC-AUC
        cv=cv,                      # Same five folds as the other models
        return_train_score=True,
        n_jobs=-1,
        error_score="raise",
    )
    start = time.perf_counter()
    lr_grid_search.fit(X_train_clean, y_train)
    lr_grid_time = time.perf_counter() - start
    print(f"Finished in {lr_grid_time:.1f} seconds")
    print("Best validation ROC-AUC:", round(lr_grid_search.best_score_, 4))
    print("\nTop 10 configurations:")
    print(search_summary(lr_grid_search, top=10).round(4).to_string())
    pd.DataFrame(lr_grid_search.cv_results_).to_csv(
        RESULTS_DIR / "lr_grid_search_cv_results.csv", index=False
    )

    # ------------------------------------------- Stage 2: finer grid for C
    section("Stage 2 - Finer grid for C around the best value")
    best_stage1 = lr_grid_search.best_params_
    best_c = best_stage1["model__C"]
    # Half a decade either side of the best C, in quarter-decade steps
    fine_c = sorted({round(float(best_c * 10 ** p), 6) for p in (-0.5, -0.25, 0, 0.25, 0.5)})
    lr_fine_grid = {name: [value] for name, value in best_stage1.items()}
    lr_fine_grid["model__C"] = fine_c
    print("C values:", fine_c)

    lr_fine_search = GridSearchCV(
        estimator=lr_pipeline, param_grid=lr_fine_grid, scoring=scoring,
        refit="ROC_AUC", cv=cv, return_train_score=True, n_jobs=-1, error_score="raise",
    )
    start = time.perf_counter()
    lr_fine_search.fit(X_train_clean, y_train)
    lr_fine_time = time.perf_counter() - start
    print("Stage 1 best ROC-AUC:", round(lr_grid_search.best_score_, 4))
    print("Stage 2 best ROC-AUC:", round(lr_fine_search.best_score_, 4))
    print()
    print(search_summary(lr_fine_search, top=5).round(4).to_string())
    pd.DataFrame(lr_fine_search.cv_results_).to_csv(
        RESULTS_DIR / "lr_fine_search_cv_results.csv", index=False
    )

    # ------------------------------------------- Final tuned LR vs baseline
    section("Tuned Logistic Regression vs baseline (same 5 folds)")
    lr_best_params = lr_fine_search.best_params_
    lr_tuned_pipeline = clone(lr_pipeline).set_params(**lr_best_params)
    lr_tuned_cv = cross_validate(
        lr_tuned_pipeline, X_train_clean, y_train,
        cv=cv, scoring=scoring, return_train_score=True,
        n_jobs=-1, error_score="raise",
    )

    comparison_rows = []
    for metric in scoring:
        baseline_scores = lr_baseline_cv[f"test_{metric}"]
        tuned_scores = lr_tuned_cv[f"test_{metric}"]
        comparison_rows.append({
            "Metric": metric,
            "Baseline mean": baseline_scores.mean(),
            "Baseline std": baseline_scores.std(ddof=1),
            "Tuned mean": tuned_scores.mean(),
            "Tuned std": tuned_scores.std(ddof=1),
            "Change": tuned_scores.mean() - baseline_scores.mean(),
        })
    lr_comparison = pd.DataFrame(comparison_rows).set_index("Metric")
    print(lr_comparison.round(4).to_string())

    lr_fit_check = pd.DataFrame({
        "Train ROC_AUC": [lr_baseline_cv["train_ROC_AUC"].mean(),
                          lr_tuned_cv["train_ROC_AUC"].mean()],
        "Validation ROC_AUC": [lr_baseline_cv["test_ROC_AUC"].mean(),
                               lr_tuned_cv["test_ROC_AUC"].mean()],
        "Mean fit time per fold (s)": [lr_baseline_cv["fit_time"].mean(),
                                       lr_tuned_cv["fit_time"].mean()],
    }, index=["Baseline LR", "Tuned LR"])
    lr_fit_check.insert(2, "Gap (train - val)",
                        lr_fit_check["Train ROC_AUC"] - lr_fit_check["Validation ROC_AUC"])
    print("\nOverfitting and training-time check")
    print(lr_fit_check.round(4).to_string())
    lr_comparison.to_csv(RESULTS_DIR / "lr_baseline_vs_tuned.csv")
    lr_fit_check.to_csv(RESULTS_DIR / "lr_fit_check.csv")

    # ----------------------------------------------------- Validation curve
    section("Validation curve - C (regularisation strength)")
    chosen_c = lr_best_params["model__C"]
    c_range = sorted({0.0001, 0.001, 0.01, 0.1, 1, 10, 100, chosen_c})
    train_scores, val_scores = validation_curve(
        lr_tuned_pipeline, X_train_clean, y_train,
        param_name="model__C", param_range=c_range,
        cv=cv, scoring="roc_auc", n_jobs=-1,
    )
    train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1, ddof=1)
    val_mean, val_std = val_scores.mean(axis=1), val_scores.std(axis=1, ddof=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(c_range, train_mean, marker="o", color="#4C78A8", label="Training ROC-AUC")
    ax.fill_between(c_range, train_mean - train_std, train_mean + train_std,
                    color="#4C78A8", alpha=0.15)
    ax.plot(c_range, val_mean, marker="o", color="#E45756", label="Validation ROC-AUC")
    ax.fill_between(c_range, val_mean - val_std, val_mean + val_std,
                    color="#E45756", alpha=0.15)
    ax.axvline(chosen_c, color="grey", linestyle="--", label=f"Selected C = {chosen_c:g}")
    ax.set_xscale("log")
    ax.set_xlabel("C (log scale; larger C = weaker regularisation)")
    ax.set_ylabel("ROC-AUC (5-fold mean ± std)")
    ax.set_title("Logistic Regression Validation Curve — C")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "07_lr_validation_curve_C.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    curve_table = pd.DataFrame({
        "C": c_range, "Train ROC_AUC": train_mean,
        "Validation ROC_AUC": val_mean, "Gap": train_mean - val_mean,
    }).set_index("C")
    print(curve_table.round(4).to_string())
    curve_table.to_csv(RESULTS_DIR / "lr_validation_curve.csv")

    # ---------------------------------------------------------- Coefficients
    section("Coefficients of the tuned Logistic Regression")
    # Fit once on the full training set
    lr_final_model = clone(lr_tuned_pipeline).fit(X_train_clean, y_train)
    feature_names = lr_final_model.named_steps["preprocess"].get_feature_names_out()
    feature_names = [name.replace("num__", "").replace("cat__", "") for name in feature_names]
    coefficients = pd.Series(lr_final_model.named_steps["model"].coef_[0], index=feature_names)

    # A positive coefficient raises the predicted probability of default.
    # Numerical features were scaled with RobustScaler, so their coefficients are
    # per interquartile range. One-hot columns keep every category, so compare
    # the coefficients of categories within the same column (e.g. PAY_0_2 vs PAY_0_0).
    top = coefficients.reindex(coefficients.abs().sort_values(ascending=False).index).head(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#E45756" if value > 0 else "#4C78A8" for value in top.iloc[::-1]]
    top.iloc[::-1].plot.barh(ax=ax, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Tuned Logistic Regression — 15 Largest Coefficients")
    ax.set_xlabel("Coefficient (red = higher default risk, blue = lower)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "08_lr_coefficients.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    n_zero = int((coefficients == 0).sum())
    print(top.round(4).to_frame("Coefficient").to_string())
    print(f"\nCoefficients set to exactly zero (only with L1): {n_zero} of {len(coefficients)}")
    coefficients.sort_values().to_csv(RESULTS_DIR / "lr_coefficients.csv", header=["Coefficient"])

    # ------------------------------------------------- Hand-over to Member 4
    section("Hand-over files for Member 4")
    lr_best_params_clean = {
        name.replace("model__", ""): (value.item() if hasattr(value, "item") else value)
        for name, value in lr_best_params.items()
    }
    lr_best_params_clean["solver"] = "liblinear"
    lr_best_params_clean["max_iter"] = 2000
    with open(RESULTS_DIR / "lr_best_params.json", "w") as f:
        json.dump(lr_best_params_clean, f, indent=2)

    lr_oof_proba = cross_val_predict(
        lr_tuned_pipeline, X_train_clean, y_train,
        cv=cv, method="predict_proba", n_jobs=-1,
    )[:, 1]
    pd.DataFrame({"y_true": y_train, "lr_proba": lr_oof_proba},
                 index=X_train_clean.index).to_csv(RESULTS_DIR / "lr_oof_proba.csv")

    lr_tuned_summary = pd.DataFrame(
        {metric: [lr_tuned_cv[f"test_{metric}"].mean()] for metric in scoring},
        index=pd.Index(["Logistic Regression (tuned)"], name="Model"),
    )
    lr_tuned_summary.to_csv(RESULTS_DIR / "lr_tuned_cv_summary.csv")

    print("Tuned Logistic Regression parameters:")
    print(json.dumps(lr_best_params_clean, indent=2))
    print()
    print(lr_tuned_summary.round(4).to_string())
    print(f"\nTotal search time: {(lr_grid_time + lr_fine_time):.1f} seconds")


# The main guard is required on Windows when n_jobs=-1 starts worker processes
if __name__ == "__main__":
    main()
