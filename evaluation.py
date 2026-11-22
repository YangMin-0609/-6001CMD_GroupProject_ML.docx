"""
Part F - Ensemble Model and Part G - Evaluation and Comparison (Member 4)
UCI Default of Credit Card Clients (id=350)

Run AFTER the three tuning scripts, from the same project folder:
    python lr_tuning.py      (Member 1)
    python rf_tuning.py      (Member 2)
    python hgb_tuning.py     (Member 3)
    python evaluation.py     (Member 4)

What this script does
    1. Rebuilds the three tuned models from results/*_best_params.json.
    2. Evaluates them the same way: same data, same 5 folds, same metrics.
    3. Picks the two best models by validation ROC-AUC (training data only).
    4. Combines them with soft voting and with stacking, and keeps the better one.
    5. Tests whether the ensemble is really better than the best single model.
    6. Evaluates all four final models ONCE on the reserved test set, at the
       usual 0.5 cut-off and at an F1-optimal cut-off chosen on training data.

Outputs
    results/  - eval_*.csv tables for the report
    figures/  - 10_cv_comparison.png
                11_roc_curves_test.png
                12_confusion_matrices_test.png

Expected run time: about 10-30 minutes on a typical laptop (mostly stacking).
"""

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save figures to files without opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from ucimlrepo import fetch_ucirepo

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

# If two ensembles are within this ROC-AUC of each other, keep the simpler one (voting)
TIE_MARGIN = 0.001
N_BOOTSTRAP = 1000


# =============================================================================
# 1. Shared setup (same as the team notebook and the three tuning scripts)
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


# =============================================================================
# 2. Shared evaluation code - every model is evaluated by these two functions
# =============================================================================

def load_params(prefix):
    """Read the tuned hyperparameters saved by a tuning script."""
    path = RESULTS_DIR / f"{prefix}_best_params.json"
    if not path.exists():
        script = {"lr": "lr_tuning.py", "rf": "rf_tuning.py", "hgb": "hgb_tuning.py"}[prefix]
        raise SystemExit(f"Missing {path}. Run 'python {script}' first.")
    with open(path) as f:
        return json.load(f)


def evaluate_cv(model, X, y, cv, scoring):
    """
    5-fold validation scores, training scores and fit times for one model,
    plus its out-of-fold probabilities of default. Each customer's probability
    comes from the fold model that did not see that customer during training,
    so no extra fitting is needed.
    """
    results = cross_validate(
        model, X, y, cv=cv, scoring=scoring, return_train_score=True,
        return_estimator=True, return_indices=True, n_jobs=-1, error_score="raise",
    )
    oof = np.zeros(len(y))
    for fitted, val_idx in zip(results["estimator"], results["indices"]["test"]):
        oof[val_idx] = fitted.predict_proba(X.iloc[val_idx])[:, 1]
    return results, oof


def best_f1_threshold(y_true, proba):
    """Threshold that maximises F1 on out-of-fold (training) predictions."""
    thresholds = np.arange(0.05, 0.951, 0.01)
    scores = [f1_score(y_true, (proba >= t).astype(int), zero_division=0) for t in thresholds]
    return float(round(thresholds[int(np.argmax(scores))], 2))


def cv_row(name, results, scoring):
    """One summary row: mean and std of each metric, plus the overfitting gap."""
    row = {"Model": name}
    for metric in scoring:
        row[metric] = results[f"test_{metric}"].mean()
        row[f"{metric} std"] = results[f"test_{metric}"].std(ddof=1)
    row["Train ROC_AUC"] = results["train_ROC_AUC"].mean()
    row["Gap (train - val)"] = row["Train ROC_AUC"] - row["ROC_AUC"]
    row["Fit time per fold (s)"] = results["fit_time"].mean()
    return row


def test_metrics(y_true, proba, threshold=0.5):
    """Test-set metrics for one model; precision/recall/F1 for the default class."""
    predicted = (proba >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, predicted),
        "ROC_AUC": roc_auc_score(y_true, proba),
        "Precision": precision_score(y_true, predicted, pos_label=1, zero_division=0),
        "Recall": recall_score(y_true, predicted, pos_label=1, zero_division=0),
        "F1": f1_score(y_true, predicted, pos_label=1, zero_division=0),
    }


# =============================================================================
# 3. Main
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
    X_test_clean = clean_category_codes(X_test)   # same fixed rules, nothing learned

    numeric_columns = [c for c in X_train_clean.columns if c not in CATEGORICAL_COLUMNS]
    preprocessor = build_preprocessor(numeric_columns)
    scoring = build_scoring()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("Training set:", X_train_clean.shape, "| Test set:", X_test_clean.shape)

    # ------------------------------------------ Rebuild the three tuned models
    section("Tuned models from Members 1-3")
    lr_params, rf_params, hgb_params = load_params("lr"), load_params("rf"), load_params("hgb")
    for name, params in [("Logistic Regression", lr_params),
                         ("Random Forest", rf_params),
                         ("HistGradientBoosting", hgb_params)]:
        print(f"{name}: {params}")

    tuned_models = {
        "Logistic Regression": Pipeline([
            ("preprocess", clone(preprocessor)),
            ("model", LogisticRegression(**lr_params)),
        ]),
        "Random Forest": Pipeline([
            ("preprocess", clone(preprocessor)),
            ("model", RandomForestClassifier(random_state=42, n_jobs=1, **rf_params)),
        ]),
        "HistGradientBoosting": Pipeline([
            ("preprocess", clone(preprocessor)),
            ("model", HistGradientBoostingClassifier(random_state=42, **hgb_params)),
        ]),
    }

    # ------------------------------- Part G (1): validation comparison, 3 models
    section("Validation comparison of the three tuned models (same 5 folds)")
    cv_results, oof_proba = {}, {}
    for name, model in tuned_models.items():
        print(f"Evaluating {name}...")
        cv_results[name], oof_proba[name] = evaluate_cv(
            model, X_train_clean, y_train, cv, scoring)

    base_table = pd.DataFrame(
        [cv_row(name, res, scoring) for name, res in cv_results.items()]
    ).set_index("Model").sort_values("ROC_AUC", ascending=False)
    print(base_table.round(4).to_string())

    # How similar are the models' predictions? Lower correlation = more diversity
    oof_corr = pd.DataFrame(oof_proba).corr()
    print("\nCorrelation of out-of-fold default probabilities")
    print(oof_corr.round(3).to_string())

    # ---------------------------------- Part F (1): choose the two best models
    section("Part F - Selecting two models for the ensemble")
    ranking = base_table["ROC_AUC"]
    selected = list(ranking.index[:2])
    excluded = ranking.index[2]
    print(f"Selected by validation ROC-AUC: {selected[0]} ({ranking.iloc[0]:.4f}) "
          f"and {selected[1]} ({ranking.iloc[1]:.4f})")
    print(f"Excluded: {excluded} ({ranking.iloc[2]:.4f}, "
          f"{ranking.iloc[1] - ranking.iloc[2]:.4f} below the second model)")
    print(f"Correlation between the two selected models: "
          f"{oof_corr.loc[selected[0], selected[1]]:.3f}")

    short = {"Logistic Regression": "lr", "Random Forest": "rf", "HistGradientBoosting": "hgb"}
    members = [(short[name], clone(tuned_models[name])) for name in selected]

    # --------------------------------------- Part F (2): build two ensembles
    section("Part F - Soft voting vs stacking (same 5 folds)")
    ensembles = {
        # Soft voting: average the two predicted probabilities
        "Soft Voting": VotingClassifier(estimators=members, voting="soft", n_jobs=1),
        # Stacking: a Logistic Regression learns how much to trust each model,
        # using out-of-fold predictions from an inner 5-fold split
        "Stacking": StackingClassifier(
            estimators=members,
            final_estimator=LogisticRegression(max_iter=2000),
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            stack_method="predict_proba",
            n_jobs=1,
        ),
    }
    ensemble_results, ensemble_oof = {}, {}
    for name, model in ensembles.items():
        print(f"Evaluating {name}...")
        start = time.perf_counter()
        ensemble_results[name], ensemble_oof[name] = evaluate_cv(
            model, X_train_clean, y_train, cv, scoring)
        print(f"  done in {(time.perf_counter() - start) / 60:.1f} minutes")

    ensemble_table = pd.DataFrame(
        [cv_row(name, res, scoring) for name, res in ensemble_results.items()]
    ).set_index("Model")
    print(ensemble_table.round(4).to_string())

    voting_auc = ensemble_table.loc["Soft Voting", "ROC_AUC"]
    stacking_auc = ensemble_table.loc["Stacking", "ROC_AUC"]
    if stacking_auc > voting_auc + TIE_MARGIN:
        chosen_technique = "Stacking"
        reason = "higher validation ROC-AUC by more than the tie margin"
    else:
        chosen_technique = "Soft Voting"
        reason = ("higher or practically equal validation ROC-AUC, and it is simpler "
                  "and faster to train")
    print(f"\nChosen ensemble technique: {chosen_technique} ({reason})")
    ensemble_name = f"Ensemble ({chosen_technique}: {selected[0]} + {selected[1]})"

    # The 0.5 cut-off is not ideal when only 22% of customers default, and the
    # models were tuned with different class weights. As an extra comparison,
    # each model also gets the cut-off that maximises F1 on its out-of-fold
    # (training) predictions. The test set is not used to choose it.
    thresholds = {name: best_f1_threshold(y_train, p) for name, p in oof_proba.items()}
    thresholds[ensemble_name] = best_f1_threshold(y_train, ensemble_oof[chosen_technique])
    print("\nF1-optimal thresholds chosen on out-of-fold training predictions:")
    for name, t in thresholds.items():
        print(f"  {name}: {t:.2f}")

    # ------------- Part G (2): is the ensemble better than the best single model?
    section("Does the ensemble significantly improve on the best single model?")
    best_single = selected[0]
    ens_folds = ensemble_results[chosen_technique]["test_ROC_AUC"]
    best_folds = cv_results[best_single]["test_ROC_AUC"]
    fold_diff = ens_folds - best_folds
    t_stat, p_value = stats.ttest_rel(ens_folds, best_folds)
    print(pd.DataFrame({"Ensemble": ens_folds, best_single: best_folds,
                        "Difference": fold_diff},
                       index=[f"Fold {i + 1}" for i in range(len(fold_diff))]).round(4).to_string())
    print(f"\nMean difference: {fold_diff.mean():+.4f} | paired t-test p = {p_value:.3f}")
    print("Note: with only 5 folds the test has low power; the bootstrap on the "
          "test set below is a second check.")

    all_cv = pd.concat([base_table, ensemble_table])
    all_cv.to_csv(RESULTS_DIR / "eval_cv_comparison.csv")
    pd.DataFrame(oof_proba).assign(**{chosen_technique: ensemble_oof[chosen_technique]}).to_csv(
        RESULTS_DIR / "eval_oof_proba.csv", index=False)
    oof_corr.to_csv(RESULTS_DIR / "eval_oof_correlation.csv")

    # Figure: validation ROC-AUC of all five candidates
    order = all_cv["ROC_AUC"].sort_values().index
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#F58518" if name in ensembles else "#4C78A8" for name in order]
    ax.barh(order, all_cv.loc[order, "ROC_AUC"], xerr=all_cv.loc[order, "ROC_AUC std"],
            color=colors, capsize=4)
    low = (all_cv["ROC_AUC"] - all_cv["ROC_AUC std"]).min()
    high = (all_cv["ROC_AUC"] + all_cv["ROC_AUC std"]).max()
    left = max(0.5, low - 0.02)
    ax.set_xlim(left, min(1.0, high + 0.01))
    for i, name in enumerate(order):  # value printed inside the start of each bar
        ax.text(left + 0.002, i, f"{all_cv.loc[name, 'ROC_AUC']:.4f}",
                va="center", fontsize=9, color="white", fontweight="bold")
    ax.set_xlabel("Validation ROC-AUC (5-fold mean ± std)")
    ax.set_title("Tuned Models and Ensembles — Validation ROC-AUC")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "10_cv_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ------------------------- Part G (3): final evaluation on the test set
    section("Final evaluation on the reserved test set (used once)")
    final_models = {name: clone(model) for name, model in tuned_models.items()}
    final_models[ensemble_name] = clone(ensembles[chosen_technique])

    test_rows, test_proba, confusion, tuned_threshold_rows = [], {}, {}, []
    for name, model in final_models.items():
        start = time.perf_counter()
        model.fit(X_train_clean, y_train)            # fit on the full training set
        fit_time = time.perf_counter() - start

        start = time.perf_counter()
        proba = model.predict_proba(X_test_clean)[:, 1]
        predict_time = time.perf_counter() - start

        test_proba[name] = proba
        confusion[name] = confusion_matrix(y_test, (proba >= 0.5).astype(int))
        tuned_threshold_rows.append({"Model": name, "Threshold": thresholds[name],
                                     **test_metrics(y_test, proba, thresholds[name])})
        row = {"Model": name, **test_metrics(y_test, proba)}
        row["Train ROC_AUC"] = roc_auc_score(
            y_train, model.predict_proba(X_train_clean)[:, 1])
        row["Gap (train - test)"] = row["Train ROC_AUC"] - row["ROC_AUC"]
        row["Fit time (s)"] = fit_time
        row["Predict time per 1,000 customers (ms)"] = predict_time / len(X_test_clean) * 1e6
        test_rows.append(row)

    test_table = pd.DataFrame(test_rows).set_index("Model")
    print("Threshold 0.5")
    print(test_table.round(4).to_string())
    test_table.to_csv(RESULTS_DIR / "eval_test_results.csv")

    tuned_threshold_table = pd.DataFrame(tuned_threshold_rows).set_index("Model")
    print("\nF1-optimal threshold (chosen on training data)")
    print(tuned_threshold_table.round(4).to_string())
    tuned_threshold_table.to_csv(RESULTS_DIR / "eval_test_results_tuned_threshold.csv")

    confusion_rows = []
    for name, matrix in confusion.items():
        tn, fp, fn, tp = matrix.ravel()
        confusion_rows.append({"Model": name, "TN": tn, "FP": fp, "FN": fn, "TP": tp})
    confusion_table = pd.DataFrame(confusion_rows).set_index("Model")
    print("\nConfusion matrices (threshold 0.5)")
    print(confusion_table.to_string())
    confusion_table.to_csv(RESULTS_DIR / "eval_confusion_matrices.csv")

    # Bootstrap: resample test customers to see how stable the ROC-AUC difference
    # between the ensemble and each of its two members is
    rng = np.random.default_rng(42)
    y_test_array = y_test.to_numpy()
    samples = []
    while len(samples) < N_BOOTSTRAP:
        idx = rng.integers(0, len(y_test_array), len(y_test_array))
        if y_test_array[idx].min() != y_test_array[idx].max():  # needs both classes
            samples.append(idx)

    significance_rows = []
    for member in selected:
        ens_p, mem_p = test_proba[ensemble_name], test_proba[member]
        diffs = np.array([roc_auc_score(y_test_array[i], ens_p[i])
                          - roc_auc_score(y_test_array[i], mem_p[i]) for i in samples])
        ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
        observed = test_table.loc[ensemble_name, "ROC_AUC"] - test_table.loc[member, "ROC_AUC"]
        member_folds = cv_results[member]["test_ROC_AUC"]
        significance_rows.append({
            "Comparison": f"Ensemble - {member}",
            "CV mean difference": (ens_folds - member_folds).mean(),
            "CV paired t-test p": stats.ttest_rel(ens_folds, member_folds).pvalue,
            "Test difference": observed,
            "Bootstrap 2.5%": ci_low,
            "Bootstrap 97.5%": ci_high,
            "Ensemble better in % of resamples": np.mean(diffs > 0) * 100,
            "Clear improvement": bool(ci_low > 0),
        })
    significance = pd.DataFrame(significance_rows).set_index("Comparison")
    print(f"\nEnsemble vs its members on the test set ({N_BOOTSTRAP} bootstrap resamples)")
    print(significance.round(4).to_string())
    print("'Clear improvement' = the 95% bootstrap interval of the difference is above zero.")
    significance.to_csv(RESULTS_DIR / "eval_ensemble_significance.csv")

    # Figure: ROC curves on the test set
    fig, ax = plt.subplots(figsize=(7, 6))
    palette = ["#4C78A8", "#59A14F", "#B279A2", "#F58518"]
    for color, (name, proba) in zip(palette, test_proba.items()):
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(fpr, tpr, color=color, linewidth=2 if name == ensemble_name else 1.4,
                label=f"{name} (AUC = {roc_auc_score(y_test, proba):.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random guess")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC Curves — Test Set")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "11_roc_curves_test.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure: confusion matrices on the test set
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, (name, matrix) in zip(axes.ravel(), confusion.items()):
        ConfusionMatrixDisplay(matrix, display_labels=["No default", "Default"]).plot(
            ax=ax, colorbar=False, values_format="d", cmap="Blues")
        ax.set_title(name if len(name) < 40 else name.replace(": ", ":\n"), fontsize=10)
    fig.suptitle("Confusion Matrices — Test Set (threshold 0.5)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "12_confusion_matrices_test.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------- Summary
    section("Summary for the report")
    best_test = test_table["ROC_AUC"].idxmax()
    print(f"Ensemble members: {selected[0]} + {selected[1]} ({chosen_technique}); "
          f"excluded: {excluded}")
    print(f"Highest test ROC-AUC: {best_test} ({test_table.loc[best_test, 'ROC_AUC']:.4f})")
    for comparison, row in significance.iterrows():
        print(f"{comparison}: {row['Test difference']:+.4f} test ROC-AUC; "
              + ("clear improvement." if row["Clear improvement"] else "no clear improvement."))
    print(f"Ensemble training time: {test_table.loc[ensemble_name, 'Fit time (s)']:.1f} s vs "
          f"{test_table.loc[best_single, 'Fit time (s)']:.1f} s for {best_single}")
    print("\nSaved to results/: eval_cv_comparison.csv, eval_oof_correlation.csv, "
          "eval_test_results.csv, eval_test_results_tuned_threshold.csv,\n"
          "                   eval_confusion_matrices.csv, eval_ensemble_significance.csv, "
          "eval_oof_proba.csv")
    print("Saved to figures/: 10_cv_comparison.png, 11_roc_curves_test.png, "
          "12_confusion_matrices_test.png")


# The main guard is required on Windows when n_jobs=-1 starts worker processes
if __name__ == "__main__":
    main()
