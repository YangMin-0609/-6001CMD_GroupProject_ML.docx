"""
Part E - Hyperparameter Optimisation: Random Forest (Member 2)
UCI Default of Credit Card Clients (id=350)

Run from the project folder:
    python rf_tuning.py

The data loading, train/test split, cleaning, preprocessing, folds and metrics
are identical to the team notebook (credit_default.ipynb), so the results can
be compared directly with the other models.

Outputs
    results/  - tables for the report and for Member 4
    figures/  - 05_rf_validation_curve_min_samples_leaf.png
                06_rf_feature_importance.png

The test set is NOT used in this script; it is reserved for the final evaluation.
Expected run time: about 15-30 minutes on a typical laptop.
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
from sklearn.ensemble import RandomForestClassifier
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

# Show wide tables in the terminal
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


# =============================================================================
# 2. Helper: readable summary of a search
# =============================================================================

def search_summary(search, top=10):
    """Return the top configurations of a search as a readable table."""
    results = pd.DataFrame(search.cv_results_)
    n_folds = search.n_splits_

    # Hyperparameter columns, shown without the "param_model__" prefix
    param_columns = [c for c in results.columns if c.startswith("param_")]
    table = results[param_columns].copy()
    table.columns = [c.replace("param_model__", "") for c in param_columns]

    # Sample standard deviation (ddof=1), consistent with the baseline tables
    val_folds = results[[f"split{i}_test_ROC_AUC" for i in range(n_folds)]]

    table["Val ROC_AUC"] = results["mean_test_ROC_AUC"]
    table["Val ROC_AUC std"] = val_folds.std(axis=1, ddof=1)
    table["Train ROC_AUC"] = results["mean_train_ROC_AUC"]
    # Gap = training score minus validation score; a large gap indicates overfitting
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
# 3. Main
# =============================================================================

def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    # ---------------------------------------------------------------- data
    section("Loading data and recreating the shared setup")
    X, y = load_data()

    # Same 80/20 stratified split as the notebook
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
    print("Numeric features:", len(numeric_columns),
          "| Categorical features:", len(CATEGORICAL_COLUMNS))

    # Baseline Random Forest (identical to the notebook)
    rf_pipeline = Pipeline([
        ("preprocess", clone(preprocessor)),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)),
    ])

    # Baseline, with training scores so overfitting can be compared
    rf_baseline_cv = cross_validate(
        rf_pipeline, X_train_clean, y_train,
        cv=cv, scoring=scoring, return_train_score=True,
        n_jobs=-1, error_score="raise",
    )
    print("Baseline RF validation ROC-AUC:",
          round(rf_baseline_cv["test_ROC_AUC"].mean(), 4))

    # ------------------------------------------------- Stage 1: random search
    section(f"Stage 1 - Randomised search ({N_ITER} combinations x 5 folds)")

    # The prefix "model__" points to the "model" step inside the pipeline
    rf_param_distributions = {
        "model__n_estimators": [100, 200, 300, 500],
        "model__max_depth": [8, 12, 16, 20, 30, None],      # None = unlimited depth
        "model__min_samples_leaf": [1, 2, 5, 10, 20],
        "model__min_samples_split": [2, 5, 10, 20],
        "model__max_features": ["sqrt", "log2", 0.2, 0.3],  # 0.3 = 30% of features
        "model__class_weight": [None, "balanced", "balanced_subsample"],
        "model__max_samples": [None, 0.5, 0.8],             # bootstrap sample size
    }

    rf_random_search = RandomizedSearchCV(
        estimator=rf_pipeline,
        param_distributions=rf_param_distributions,
        n_iter=N_ITER,
        scoring=scoring,
        refit="ROC_AUC",            # Choose the best combination by validation ROC-AUC
        cv=cv,                      # Same five folds as the other models
        return_train_score=True,    # Record training scores to check overfitting
        n_jobs=-1,                  # Use all CPU cores
        random_state=42,
        error_score="raise",
    )

    start = time.perf_counter()
    rf_random_search.fit(X_train_clean, y_train)
    rf_random_time = time.perf_counter() - start

    print(f"Finished in {rf_random_time / 60:.1f} minutes")
    print("Best validation ROC-AUC:", round(rf_random_search.best_score_, 4))
    print("\nTop 10 configurations:")
    print(search_summary(rf_random_search, top=10).round(4).to_string())

    pd.DataFrame(rf_random_search.cv_results_).to_csv(
        RESULTS_DIR / "rf_random_search_cv_results.csv", index=False
    )

    # --------------------------------------------------- Stage 2: grid search
    section("Stage 2 - Grid search around the best randomised result")

    best_random = rf_random_search.best_params_

    # Finer scales; they contain every value used in Stage 1
    depth_options = [6, 8, 10, 12, 14, 16, 18, 20, 25, 30, None]
    leaf_options = [1, 2, 3, 5, 7, 10, 15, 20, 30]

    # Fix all parameters at their best Stage 1 values,
    # then refine the two complexity parameters
    rf_param_grid = {name: [value] for name, value in best_random.items()}
    rf_param_grid["model__max_depth"] = neighbours(
        best_random["model__max_depth"], depth_options
    )
    rf_param_grid["model__min_samples_leaf"] = neighbours(
        best_random["model__min_samples_leaf"], leaf_options
    )

    print("max_depth values:", rf_param_grid["model__max_depth"])
    print("min_samples_leaf values:", rf_param_grid["model__min_samples_leaf"])

    rf_grid_search = GridSearchCV(
        estimator=rf_pipeline,
        param_grid=rf_param_grid,
        scoring=scoring,
        refit="ROC_AUC",
        cv=cv,
        return_train_score=True,
        n_jobs=-1,
        error_score="raise",
    )

    start = time.perf_counter()
    rf_grid_search.fit(X_train_clean, y_train)
    rf_grid_time = time.perf_counter() - start

    print(f"Finished in {rf_grid_time / 60:.1f} minutes")
    print("Stage 1 best ROC-AUC:", round(rf_random_search.best_score_, 4))
    print("Stage 2 best ROC-AUC:", round(rf_grid_search.best_score_, 4))
    print()
    print(search_summary(rf_grid_search, top=9).round(4).to_string())

    pd.DataFrame(rf_grid_search.cv_results_).to_csv(
        RESULTS_DIR / "rf_grid_search_cv_results.csv", index=False
    )

    # ------------------------------------------- Final tuned RF vs baseline
    section("Tuned Random Forest vs baseline (same 5 folds)")

    rf_best_params = rf_grid_search.best_params_
    rf_tuned_pipeline = clone(rf_pipeline).set_params(**rf_best_params)

    rf_tuned_cv = cross_validate(
        rf_tuned_pipeline, X_train_clean, y_train,
        cv=cv, scoring=scoring, return_train_score=True,
        n_jobs=-1, error_score="raise",
    )

    comparison_rows = []
    for metric in scoring:
        baseline_scores = rf_baseline_cv[f"test_{metric}"]
        tuned_scores = rf_tuned_cv[f"test_{metric}"]
        comparison_rows.append({
            "Metric": metric,
            "Baseline mean": baseline_scores.mean(),
            "Baseline std": baseline_scores.std(ddof=1),
            "Tuned mean": tuned_scores.mean(),
            "Tuned std": tuned_scores.std(ddof=1),
            "Change": tuned_scores.mean() - baseline_scores.mean(),
        })
    rf_comparison = pd.DataFrame(comparison_rows).set_index("Metric")
    print(rf_comparison.round(4).to_string())

    # Overfitting and training-time check
    rf_fit_check = pd.DataFrame({
        "Train ROC_AUC": [
            rf_baseline_cv["train_ROC_AUC"].mean(),
            rf_tuned_cv["train_ROC_AUC"].mean(),
        ],
        "Validation ROC_AUC": [
            rf_baseline_cv["test_ROC_AUC"].mean(),
            rf_tuned_cv["test_ROC_AUC"].mean(),
        ],
        "Mean fit time per fold (s)": [
            rf_baseline_cv["fit_time"].mean(),
            rf_tuned_cv["fit_time"].mean(),
        ],
    }, index=["Baseline RF", "Tuned RF"])
    rf_fit_check.insert(
        2, "Gap (train - val)",
        rf_fit_check["Train ROC_AUC"] - rf_fit_check["Validation ROC_AUC"],
    )
    print("\nOverfitting and training-time check")
    print(rf_fit_check.round(4).to_string())

    rf_comparison.to_csv(RESULTS_DIR / "rf_baseline_vs_tuned.csv")
    rf_fit_check.to_csv(RESULTS_DIR / "rf_fit_check.csv")

    # ----------------------------------------------------- Validation curve
    section("Validation curve - min_samples_leaf")

    # Include the selected value so it appears on the curve
    chosen_leaf = rf_best_params["model__min_samples_leaf"]
    leaf_range = sorted({1, 2, 5, 10, 20, 40, 80, chosen_leaf})

    train_scores, val_scores = validation_curve(
        rf_tuned_pipeline, X_train_clean, y_train,
        param_name="model__min_samples_leaf", param_range=leaf_range,
        cv=cv, scoring="roc_auc", n_jobs=-1,
    )
    train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1, ddof=1)
    val_mean, val_std = val_scores.mean(axis=1), val_scores.std(axis=1, ddof=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(leaf_range, train_mean, marker="o", color="#4C78A8", label="Training ROC-AUC")
    ax.fill_between(leaf_range, train_mean - train_std, train_mean + train_std,
                    color="#4C78A8", alpha=0.15)
    ax.plot(leaf_range, val_mean, marker="o", color="#E45756", label="Validation ROC-AUC")
    ax.fill_between(leaf_range, val_mean - val_std, val_mean + val_std,
                    color="#E45756", alpha=0.15)
    ax.axvline(chosen_leaf, color="grey", linestyle="--",
               label=f"Selected min_samples_leaf = {chosen_leaf}")
    ax.set_xscale("log")
    ax.set_xticks(leaf_range)
    ax.set_xticklabels(leaf_range)
    ax.set_xlabel("min_samples_leaf (log scale)")
    ax.set_ylabel("ROC-AUC (5-fold mean ± std)")
    ax.set_title("Random Forest Validation Curve — min_samples_leaf")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "05_rf_validation_curve_min_samples_leaf.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    curve_table = pd.DataFrame({
        "min_samples_leaf": leaf_range,
        "Train ROC_AUC": train_mean,
        "Validation ROC_AUC": val_mean,
        "Gap": train_mean - val_mean,
    }).set_index("min_samples_leaf")
    print(curve_table.round(4).to_string())
    curve_table.to_csv(RESULTS_DIR / "rf_validation_curve.csv")

    # ---------------------------------------------------- Feature importance
    section("Feature importance of the tuned Random Forest")

    # Fit once on the full training set
    rf_final_model = clone(rf_tuned_pipeline).fit(X_train_clean, y_train)
    fitted_preprocessor = rf_final_model.named_steps["preprocess"]
    forest = rf_final_model.named_steps["model"]

    # Map each transformed column back to its original feature
    # (ColumnTransformer outputs numerical columns first, then one-hot columns)
    encoder = fitted_preprocessor.named_transformers_["cat"]
    original_feature = list(numeric_columns)
    for column, categories in zip(CATEGORICAL_COLUMNS, encoder.categories_):
        original_feature += [column] * len(categories)
    assert len(original_feature) == len(forest.feature_importances_)

    rf_importance = (
        pd.Series(forest.feature_importances_, index=original_feature)
        .groupby(level=0).sum()
        .sort_values(ascending=False)
    )

    top_n = 15
    fig, ax = plt.subplots(figsize=(8, 6))
    rf_importance.head(top_n).iloc[::-1].plot.barh(ax=ax, color="#59A14F")
    ax.set_title(f"Tuned Random Forest — Top {top_n} Features (impurity importance)")
    ax.set_xlabel("Mean decrease in impurity (summed over one-hot columns)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "06_rf_feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(rf_importance.round(4).to_frame("Importance").to_string())
    rf_importance.to_csv(RESULTS_DIR / "rf_feature_importance.csv", header=["Importance"])

    # ------------------------------------------------- Hand-over to Member 4
    section("Hand-over files for Member 4")

    # 1) Tuned hyperparameters (JSON keeps None as null)
    rf_best_params_clean = {
        name.replace("model__", ""): (value.item() if hasattr(value, "item") else value)
        for name, value in rf_best_params.items()
    }
    with open(RESULTS_DIR / "rf_best_params.json", "w") as f:
        json.dump(rf_best_params_clean, f, indent=2)

    # 2) Out-of-fold probability of default (same five folds)
    rf_oof_proba = cross_val_predict(
        rf_tuned_pipeline, X_train_clean, y_train,
        cv=cv, method="predict_proba", n_jobs=-1,
    )[:, 1]
    pd.DataFrame(
        {"y_true": y_train, "rf_proba": rf_oof_proba},
        index=X_train_clean.index,
    ).to_csv(RESULTS_DIR / "rf_oof_proba.csv")

    # 3) One-row summary in the same format as the baseline table
    rf_tuned_summary = pd.DataFrame(
        {metric: [rf_tuned_cv[f"test_{metric}"].mean()] for metric in scoring},
        index=pd.Index(["Random Forest (tuned)"], name="Model"),
    )
    rf_tuned_summary.to_csv(RESULTS_DIR / "rf_tuned_cv_summary.csv")

    print("Tuned Random Forest parameters:")
    print(json.dumps(rf_best_params_clean, indent=2))
    print()
    print(rf_tuned_summary.round(4).to_string())
    print("\nSaved to results/: rf_best_params.json, rf_oof_proba.csv, "
          "rf_tuned_cv_summary.csv, rf_baseline_vs_tuned.csv, rf_fit_check.csv, ...")
    print("Saved to figures/: 05_rf_validation_curve_min_samples_leaf.png, "
          "06_rf_feature_importance.png")
    print(f"\nTotal search time: {(rf_random_time + rf_grid_time) / 60:.1f} minutes")


# The main guard is required on Windows when n_jobs=-1 starts worker processes
if __name__ == "__main__":
    main()
