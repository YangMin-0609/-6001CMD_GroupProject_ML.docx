"""Tune Logistic Regression model.

Run from the project folder with:
    python lr_tuning.py

The script follows the split, cleaning, preprocessing, CV and metrics in
credit_default.ipynb. The held-out test set is used only after tuning.
"""

from __future__ import annotations

import json
import warnings
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save plots without requiring an interactive display.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

# requests is not one of the group's pinned dependencies, so support its
# network exceptions when it is present without making it a hard dependency.
try:
    from requests.exceptions import RequestException as RequestsRequestException
except ImportError:
    RequestsRequestException = None

API_CONNECTION_ERRORS = (ImportError, ConnectionError, TimeoutError, OSError)
if RequestsRequestException is not None:
    API_CONNECTION_ERRORS += (RequestsRequestException,)


RANDOM_STATE = 42
MAX_ITER = 5000
N_JOBS = 2
SEARCH_TOL = 1e-3
BASELINE_TOL = 1e-4
TARGET_NAME = "DEFAULT_NEXT_MONTH"
METRIC_NAMES = ["ROC_AUC", "Accuracy", "Precision", "Recall", "F1"]


def load_and_prepare_data():
    """Try UCI API first, then load the bundled official UCI workbook/export."""
    column_names = {
        "X1": "LIMIT_BAL", "X2": "SEX", "X3": "EDUCATION", "X4": "MARRIAGE",
        "X5": "AGE", "X6": "PAY_0", "X7": "PAY_2", "X8": "PAY_3",
        "X9": "PAY_4", "X10": "PAY_5", "X11": "PAY_6",
        "X12": "BILL_AMT1", "X13": "BILL_AMT2", "X14": "BILL_AMT3",
        "X15": "BILL_AMT4", "X16": "BILL_AMT5", "X17": "BILL_AMT6",
        "X18": "PAY_AMT1", "X19": "PAY_AMT2", "X20": "PAY_AMT3",
        "X21": "PAY_AMT4", "X22": "PAY_AMT5", "X23": "PAY_AMT6",
    }
    expected_features = list(column_names.values())
    data_dir = Path(__file__).resolve().parent / "data"
    fallback_reason = None
    source = "UCI API: ucimlrepo.fetch_ucirepo(id=350)"

    try:
        from ucimlrepo import fetch_ucirepo
        credit = fetch_ucirepo(id=350)
        X = credit.data.features.copy()
        y = credit.data.targets.iloc[:, 0].copy()
        if list(X.columns) != list(column_names):
            raise ValueError(f"Unexpected API feature columns: {list(X.columns)}")
        X = X.rename(columns=column_names)
    except API_CONNECTION_ERRORS as error:
        fallback_reason = f"{type(error).__name__}: {error}"
        xls_path = data_dir / "default of credit card clients.xls"
        csv_path = data_dir / "credit_default_data.csv"
        source = None
        if xls_path.exists():
            try:
                local = pd.read_excel(xls_path, header=1)
                source = "Official UCI Excel file: data/default of credit card clients.xls"
            except ImportError:
                if not csv_path.exists():
                    raise RuntimeError("Reading the bundled .xls needs xlrd; no CSV fallback was bundled.")
        if source is None:
            if not csv_path.exists():
                raise FileNotFoundError(f"No local UCI fallback found in {data_dir}")
            local = pd.read_csv(csv_path, header=1)
            source = "CSV exported from the official UCI Excel file: data/credit_default_data.csv"
        local.columns = [str(c).strip() for c in local.columns]
        target_column = "default payment next month"
        required = {"ID", target_column, *expected_features}
        if not required.issubset(local.columns):
            raise ValueError(f"Local UCI file is missing columns: {sorted(required - set(local.columns))}")
        X = local[expected_features].copy()
        y = local[target_column].copy()

    X.columns = [str(c).strip() for c in X.columns]
    if list(X.columns) != expected_features:
        raise ValueError(f"Feature order/names differ from notebook: {list(X.columns)}")
    if X.shape != (30000, 23) or len(y) != 30000:
        raise ValueError(f"Expected 30,000 rows and 23 features; got X={X.shape}, y={y.shape}.")
    y = pd.Series(y, index=X.index, name=TARGET_NAME).astype(int)
    if X.isna().to_numpy().any() or y.isna().any():
        raise ValueError("The selected dataset contains missing values; inspect before modelling.")
    target_counts = y.value_counts().sort_index().to_dict()
    expected_target_counts = {0: 23364, 1: 6636}
    if target_counts != expected_target_counts:
        raise ValueError(f"Unexpected target distribution {target_counts}; expected {expected_target_counts}.")

    # Match the notebook: split first, then apply its fixed cleaning to both sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    education_codes_before = {
        "5": int(X_train["EDUCATION"].eq(5).sum() + X_test["EDUCATION"].eq(5).sum()),
        "6": int(X_train["EDUCATION"].eq(6).sum() + X_test["EDUCATION"].eq(6).sum()),
    }
    def clean_category_codes(features):
        cleaned = features.copy()
        cleaned["EDUCATION"] = cleaned["EDUCATION"].replace({5: 0, 6: 0})
        for column in ["SEX", "EDUCATION", "MARRIAGE"]:
            cleaned[column] = cleaned[column].astype("category")
        return cleaned
    X_train = clean_category_codes(X_train)
    X_test = clean_category_codes(X_test)

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_folds = []
    for fold_number, (fold_train, fold_validation) in enumerate(splitter.split(X_train, y_train), start=1):
        cv_folds.append({
            "fold": fold_number, "train_rows": len(fold_train), "validation_rows": len(fold_validation),
            "validation_target_counts": {str(k): int(v) for k, v in y_train.iloc[fold_validation].value_counts().sort_index().items()},
        })
    summary = {
        "source": source, "api_attempted": True,
        "api_fallback_reason": fallback_reason,
        "records": int(len(X)), "predictor_count": int(X.shape[1]),
        "feature_columns": expected_features, "target_name": TARGET_NAME,
        "target_counts": {str(k): int(v) for k, v in target_counts.items()},
        "missing_values": 0, "train_rows": len(X_train), "test_rows": len(X_test),
        "train_target_counts": {str(k): int(v) for k, v in y_train.value_counts().sort_index().items()},
        "test_target_counts": {str(k): int(v) for k, v in y_test.value_counts().sort_index().items()},
        "split": {"test_size": 0.20, "random_state": RANDOM_STATE, "stratified": True},
        "cleaning": "EDUCATION codes 5 and 6 mapped to 0; SEX, EDUCATION, MARRIAGE cast to category",
        "education_codes_before_cleaning": education_codes_before,
        "cross_validation": {"method": "StratifiedKFold", "n_splits": 5, "shuffle": True,
                             "random_state": RANDOM_STATE, "folds": cv_folds},
    }
    print(f"Dataset source: {source}")
    if fallback_reason:
        print(f"API unavailable; local fallback selected ({fallback_reason})")
    print(f"Verified data: X={X.shape}, y={y.shape}, targets={target_counts}")
    print(f"Split: train={len(X_train)} {y_train.value_counts().sort_index().to_dict()}, test={len(X_test)} {y_test.value_counts().sort_index().to_dict()}")
    return X_train, X_test, y_train, y_test, summary


def make_pipeline():
    """Build preprocessing + Logistic Regression so all transforms are fold-local."""
    categorical_columns = [
        "SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3",
        "PAY_4", "PAY_5", "PAY_6",
    ]
    numeric_columns = [
        "LIMIT_BAL", "AGE", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
        "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", "PAY_AMT1", "PAY_AMT2",
        "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    ]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", RobustScaler(), numeric_columns),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_columns),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LogisticRegression(max_iter=MAX_ITER, tol=SEARCH_TOL, random_state=RANDOM_STATE)),
        ]
    )


def get_scoring():
    """Use the notebook's metrics, with default (class 1) precision/recall/F1."""
    return {
        "ROC_AUC": "roc_auc",
        "Accuracy": "accuracy",
        "Precision": make_scorer(precision_score, pos_label=1, zero_division=0),
        "Recall": make_scorer(recall_score, pos_label=1, zero_division=0),
        "F1": make_scorer(f1_score, pos_label=1, zero_division=0),
    }


def get_parameter_grid():
    """Return 70 valid configurations using current sklearn's l1_ratio API.

    l1_ratio=0 is L2, l1_ratio=1 is L1, and intermediate values are
    Elastic Net. The solver grids include only compatible values.
    """
    c_values = [0.01, 0.1, 1.0, 10.0, 100.0]
    weights = [None, "balanced"]
    # scikit-learn 1.9 deprecates the explicit penalty parameter. Use its
    # equivalent l1_ratio values; liblinear supports L1 and L2, but not Elastic Net.
    liblinear = {
        "model__solver": ["liblinear"],
        "model__l1_ratio": [0.0, 1.0],
        "model__C": c_values,
        "model__class_weight": weights,
    }
    # saga supports L1, L2 and Elastic Net (the three intermediate ratios).
    saga = {
        "model__solver": ["saga"],
        "model__l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
        "model__C": c_values,
        "model__class_weight": weights,
    }
    return [liblinear, saga]


def summarize_cv_results(cv_results, label):
    """Produce a compact mean and standard-deviation row from fold scores."""
    row = {"Model": label}
    for metric in METRIC_NAMES:
        scores = np.asarray(cv_results[f"test_{metric}"])
        row[f"{metric}_mean"] = float(scores.mean())
        row[f"{metric}_std"] = float(scores.std(ddof=1))
    return row


def evaluate_test_set(model, X_test, y_test):
    """Calculate final held-out test metrics and confusion-matrix counts."""
    prediction = model.predict(X_test)
    probability = model.predict_proba(X_test)[:, 1]
    matrix = confusion_matrix(y_test, prediction, labels=[0, 1])
    return {
        "ROC_AUC": float(roc_auc_score(y_test, probability)),
        "Accuracy": float(accuracy_score(y_test, prediction)),
        "Precision": float(precision_score(y_test, prediction, pos_label=1, zero_division=0)),
        "Recall": float(recall_score(y_test, prediction, pos_label=1, zero_division=0)),
        "F1": float(f1_score(y_test, prediction, pos_label=1, zero_division=0)),
        "confusion_matrix_labels": ["actual_0", "actual_1"],
        "confusion_matrix": matrix.tolist(),
    }


def save_comparison_plot(comparison, figures_dir):
    """Plot cross-validation ROC-AUC and its fold standard deviation."""
    names = comparison["Model"].tolist()
    means = comparison["ROC_AUC_mean"].to_numpy()
    errors = comparison["ROC_AUC_std"].to_numpy()
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(names, means, yerr=errors, capsize=6, color=["#4C78A8", "#E45756"])
    ax.set_ylabel("ROC-AUC (5-fold mean ± standard deviation)")
    ax.set_title("Baseline vs Tuned Logistic Regression")
    ax.set_ylim(max(0, means.min() - 0.05), min(1, means.max() + 0.05))
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{mean:.4f}",
                ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(figures_dir / "lr_baseline_vs_tuned_roc_auc.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix_figure(metrics, figures_dir):
    """Save a labeled confusion matrix for the tuned held-out evaluation."""
    matrix = np.asarray(metrics["tuned"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, label="Customers")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["No default", "Default"],
           yticklabels=["No default", "Default"], xlabel="Predicted", ylabel="Actual",
           title="Tuned LR — Test Confusion Matrix")
    threshold = matrix.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center",
                    color="white" if matrix[i, j] > threshold else "black", fontsize=12)
    fig.tight_layout()
    fig.savefig(figures_dir / "lr_tuned_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def main():
    project_dir = Path(__file__).resolve().parent
    results_dir = project_dir / "results"
    figures_dir = project_dir / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    X_train, X_test, y_train, y_test, dataset_summary = load_and_prepare_data()
    pipeline = make_pipeline()
    scoring = get_scoring()

    # Materialize the folds once, then pass this exact list to both evaluations.
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_splits = list(splitter.split(X_train, y_train))

    # Confirm that the baseline matches the notebook's default LR settings.
    baseline = make_pipeline()
    baseline.set_params(model__C=1.0, model__l1_ratio=0.0, model__solver="lbfgs",
                        model__class_weight=None, model__max_iter=2000, model__tol=BASELINE_TOL)

    caught_warnings = []
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", ConvergenceWarning)
        baseline_cv = cross_validate(
            baseline, X_train, y_train, cv=cv_splits, scoring=scoring,
            n_jobs=N_JOBS, error_score="raise", return_train_score=False,
        )
        for warning in recorded:
            if issubclass(warning.category, ConvergenceWarning):
                caught_warnings.append(f"BASELINE: {warning.message}")

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=get_parameter_grid(),
        scoring=scoring,
        refit="ROC_AUC",
        cv=cv_splits,
        n_jobs=N_JOBS,
        error_score="raise",
        return_train_score=False,
        verbose=1,
    )
    tuning_started = time.perf_counter()
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", ConvergenceWarning)
        search.fit(X_train, y_train)
        for warning in recorded:
            if issubclass(warning.category, ConvergenceWarning):
                caught_warnings.append(f"GRID SEARCH: {warning.message}")
    tuning_runtime_seconds = time.perf_counter() - tuning_started


    # Baseline and selected candidate summaries use the same five held-in folds.
    best_index = search.best_index_
    tuned_cv = {
        f"test_{metric}": np.asarray([
            search.cv_results_[f"split{fold}_test_{metric}"][best_index]
            for fold in range(len(cv_splits))
        ])
        for metric in METRIC_NAMES
    }
    comparison = pd.DataFrame([
        summarize_cv_results(baseline_cv, "Baseline Logistic Regression"),
        summarize_cv_results(tuned_cv, "Tuned Logistic Regression"),
    ])

    # Save detailed search evidence and a concise parameter record.
    with (results_dir / "lr_dataset_summary.json").open("w", encoding="utf-8") as output:
        json.dump(dataset_summary, output, indent=2)
    pd.DataFrame(search.cv_results_).to_csv(results_dir / "lr_grid_search_cv_results.csv", index=False)
    comparison.to_csv(results_dir / "lr_cv_comparison.csv", index=False)
    with (results_dir / "lr_best_parameters.json").open("w", encoding="utf-8") as output:
        json.dump({
            "primary_metric": "ROC_AUC",
            "best_cv_roc_auc": float(search.best_score_),
            "best_parameters": search.best_params_,
            "baseline_parameters": {
                "C": 1.0, "penalty": "l2", "l1_ratio": 0.0, "solver": "lbfgs",
                "class_weight": None, "max_iter": 2000,
                "tol": BASELINE_TOL,
            },
            "candidate_count": int(len(search.cv_results_["params"])),
            "grid_search_runtime_seconds": float(tuning_runtime_seconds),
            "max_iter_for_search_candidates": MAX_ITER,
            "tolerance_for_search_candidates": SEARCH_TOL,
            "cv": {"method": "StratifiedKFold", "n_splits": 5,
                   "shuffle": True, "random_state": RANDOM_STATE},
        }, output, indent=2)

    # The test split is first used here, after CV has selected the parameters.
    tuned_test = evaluate_test_set(search.best_estimator_, X_test, y_test)
    baseline_final = make_pipeline()
    baseline_final.set_params(model__C=1.0, model__l1_ratio=0.0, model__solver="lbfgs",
                              model__class_weight=None, model__max_iter=2000, model__tol=BASELINE_TOL)
    baseline_final.fit(X_train, y_train)
    baseline_test = evaluate_test_set(baseline_final, X_test, y_test)
    with (results_dir / "lr_test_metrics.json").open("w", encoding="utf-8") as output:
        json.dump({"baseline": baseline_test, "tuned": tuned_test}, output, indent=2)
    pd.DataFrame([
        {"Model": "Baseline Logistic Regression", **{k: v for k, v in baseline_test.items()
                                                       if k in METRIC_NAMES}},
        {"Model": "Tuned Logistic Regression", **{k: v for k, v in tuned_test.items()
                                                    if k in METRIC_NAMES}},
    ]).to_csv(results_dir / "lr_test_comparison.csv", index=False)

    warning_text = "\n".join(caught_warnings) if caught_warnings else "No ConvergenceWarning was raised."
    (results_dir / "lr_convergence_warnings.txt").write_text(warning_text + "\n", encoding="utf-8")
    save_comparison_plot(comparison, figures_dir)
    save_confusion_matrix_figure({"tuned": tuned_test}, figures_dir)
    (results_dir / "lr_runtime.txt").write_text(f"GridSearchCV runtime: {tuning_runtime_seconds:.2f} seconds\n", encoding="utf-8")

    print("\nBaseline and tuned 5-fold CV summary:")
    print(comparison.round(4).to_string(index=False))
    print(f"\nGridSearchCV runtime: {tuning_runtime_seconds:.2f} seconds")
    print("\nBest parameters:", search.best_params_)
    print("\nHeld-out test metrics (computed after parameter selection):")
    print(json.dumps({"baseline": baseline_test, "tuned": tuned_test}, indent=2))
    print(f"\nConvergence warning report: {results_dir / 'lr_convergence_warnings.txt'}")
    print(f"Results saved in: {results_dir}")
    print(f"Plot saved in: {figures_dir / 'lr_baseline_vs_tuned_roc_auc.png'}")


if __name__ == "__main__":
    main()


