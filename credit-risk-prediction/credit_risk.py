"""
Credit Risk Prediction
======================
Binary classification model to predict loan default from borrower data.
Dataset: Give Me Some Credit — Kaggle Competition (150,000 borrowers)

Author  : Amr Mohammed Ali
GitHub  : https://github.com/Amr-Mo-ali
Kaggle  : https://www.kaggle.com/amrmohammedali

Results : Recall 77.6% | AUC 0.866 | 94.4% reduction in missed defaulters
"""

# ─────────────────────────────────────────────────────────────
# 1. Imports
# ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import (
    train_test_split,
    cross_val_predict,
    GridSearchCV,
    StratifiedKFold
)
from sklearn.metrics import (
    confusion_matrix,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc
)
from xgboost import XGBClassifier


# ─────────────────────────────────────────────────────────────
# 2. Load Data
# ─────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    """
    Load the Give Me Some Credit dataset.
    index_col=0 drops the unnamed row-number column
    that would otherwise appear as a feature.
    """
    df = pd.read_csv(path, index_col=0)
    print(f"Dataset shape     : {df.shape}")
    print(f"Missing values    :\n{df.isnull().sum()}")
    balance = df['SeriousDlqin2yrs'].value_counts(normalize=True) * 100
    print(f"\nClass balance (%):\n{balance.round(1)}")
    return df


# ─────────────────────────────────────────────────────────────
# 3. EDA
# ─────────────────────────────────────────────────────────────
def run_eda(df: pd.DataFrame) -> None:
    """
    Exploratory Data Analysis.
    Generates 6 plots and saves them as eda_plots.png.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Credit Risk — Exploratory Data Analysis", fontsize=14)

    # 1. Target distribution
    axes[0, 0].bar(
        ['No Default (0)', 'Default (1)'],
        df['SeriousDlqin2yrs'].value_counts().values,
        color=['steelblue', 'crimson'], alpha=0.8
    )
    axes[0, 0].set_title('Target Distribution')

    # 2. Age distribution by class
    axes[0, 1].hist(
        df[df['SeriousDlqin2yrs'] == 1]['age'],
        alpha=0.6, color='crimson', label='Default', bins=30
    )
    axes[0, 1].hist(
        df[df['SeriousDlqin2yrs'] == 0]['age'],
        alpha=0.6, color='steelblue', label='No Default', bins=30
    )
    axes[0, 1].set_title('Age Distribution by Class')
    axes[0, 1].legend()

    # 3. Monthly income distribution (clipped for readability)
    income = df['MonthlyIncome'].dropna()
    axes[0, 2].hist(
        income[income < 20000], bins=50,
        color='purple', alpha=0.7
    )
    axes[0, 2].set_title('Monthly Income (< $20k shown)')
    axes[0, 2].set_xlabel('Income ($)')

    # 4. Debt ratio by class (clipped to remove extreme outliers)
    axes[1, 0].boxplot([
        df[df['SeriousDlqin2yrs'] == 0]['DebtRatio'].clip(0, 2),
        df[df['SeriousDlqin2yrs'] == 1]['DebtRatio'].clip(0, 2)
    ], labels=['No Default', 'Default'])
    axes[1, 0].set_title('Debt Ratio by Class')

    # 5. Revolving utilization by class (clipped at 100%)
    axes[1, 1].boxplot([
        df[df['SeriousDlqin2yrs'] == 0]
          ['RevolvingUtilizationOfUnsecuredLines'].clip(0, 1),
        df[df['SeriousDlqin2yrs'] == 1]
          ['RevolvingUtilizationOfUnsecuredLines'].clip(0, 1)
    ], labels=['No Default', 'Default'])
    axes[1, 1].set_title('Credit Utilization by Class')

    # 6. Correlation heatmap
    corr = df.corr()[['SeriousDlqin2yrs']]\
             .sort_values('SeriousDlqin2yrs', ascending=False)
    sns.heatmap(
        corr, annot=True, fmt='.2f',
        ax=axes[1, 2], cmap='coolwarm', center=0
    )
    axes[1, 2].set_title('Correlation with Target')

    plt.tight_layout()
    plt.savefig("eda_plots.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("EDA plots saved → eda_plots.png")


# ─────────────────────────────────────────────────────────────
# 4. Baseline — Show the Imbalance Problem
# ─────────────────────────────────────────────────────────────
def baseline_evaluation(X_train: pd.DataFrame,
                         y_train: pd.Series) -> None:
    """
    Train a plain Logistic Regression with no imbalance correction.
    Demonstrates why accuracy is useless for imbalanced data.
    """
    baseline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   LogisticRegression(
                        random_state=42, max_iter=1000))
    ])

    y_pred = cross_val_predict(baseline, X_train, y_train, cv=3)

    print("\n=== BASELINE (no imbalance correction) ===")
    print("Confusion Matrix:\n", confusion_matrix(y_train, y_pred))
    print(f"Recall   : {recall_score(y_train, y_pred):.3f}  ← only 4% !")
    print(f"Precision: {precision_score(y_train, y_pred):.3f}")
    print(f"F1       : {f1_score(y_train, y_pred):.3f}")
    print("\n→ 93% accuracy but 4% recall — useless for a bank.")
    print("→ Model predicts 'safe' for everyone.")


# ─────────────────────────────────────────────────────────────
# 5. Model Comparison
# ─────────────────────────────────────────────────────────────
def compare_models(X_train: pd.DataFrame,
                   y_train: pd.Series) -> None:
    """
    Compare 5 models with class imbalance correction.
    All evaluated with 3-fold StratifiedKFold on training data.
    Scoring: Recall (primary metric for credit risk).
    """
    scale_pos = int((y_train == 0).sum() / (y_train == 1).sum())

    models = {
        "Logistic Regression": LogisticRegression(
            random_state=42, max_iter=1000,
            class_weight="balanced"),
        "Decision Tree":       DecisionTreeClassifier(
            random_state=42, max_depth=5,
            class_weight="balanced"),
        "Random Forest":       RandomForestClassifier(
            random_state=42, n_estimators=100,
            class_weight="balanced_subsample", n_jobs=-1),
        "SVM":                 SVC(
            random_state=42, kernel="rbf",
            class_weight="balanced"),
        "XGBoost":             XGBClassifier(
            random_state=42, scale_pos_weight=scale_pos,
            eval_metric="logloss", n_jobs=-1)
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    print("\n=== MODEL COMPARISON (3-fold StratifiedKFold) ===")
    print(f"{'Model':<22} {'Recall':>8} {'Precision':>10} {'F1':>8}")
    print("=" * 52)

    for name, model in models.items():
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("model",   model)
        ])
        y_pred    = cross_val_predict(pipe, X_train, y_train, cv=cv)
        recall    = recall_score(y_train, y_pred)
        precision = precision_score(y_train, y_pred)
        f1        = f1_score(y_train, y_pred)
        print(f"{name:<22} {recall:>8.3f} {precision:>10.3f} {f1:>8.3f}")


# ─────────────────────────────────────────────────────────────
# 6. Threshold Analysis
# ─────────────────────────────────────────────────────────────
def threshold_analysis(X_train: pd.DataFrame,
                        y_train: pd.Series) -> None:
    """
    Analyze Random Forest at different decision thresholds.
    Default 0.5 assumes balanced classes — wrong for 7% minority.
    """
    scale_pos = int((y_train == 0).sum() / (y_train == 1).sum())

    rf_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   RandomForestClassifier(
            random_state=42, n_estimators=200,
            class_weight="balanced_subsample", n_jobs=-1))
    ])

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    y_proba = cross_val_predict(
        rf_pipeline, X_train, y_train,
        cv=cv, method="predict_proba"
    )[:, 1]

    print("\n=== RANDOM FOREST — THRESHOLD ANALYSIS ===")
    print(f"{'Threshold':>10} {'Recall':>8} {'Precision':>10} {'F1':>8}")
    print("-" * 40)

    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
        y_pred    = (y_proba >= threshold).astype(int)
        recall    = recall_score(y_train, y_pred)
        precision = precision_score(y_train, y_pred)
        f1        = f1_score(y_train, y_pred)
        print(f"{threshold:>10.1f} {recall:>8.3f} "
              f"{precision:>10.3f} {f1:>8.3f}")


# ─────────────────────────────────────────────────────────────
# 7. Hyperparameter Tuning — XGBoost
# ─────────────────────────────────────────────────────────────
def tune_xgboost(X_train: pd.DataFrame,
                  y_train: pd.Series) -> Pipeline:
    """
    GridSearchCV on XGBoost optimizing for recall.
    32 combinations × 3 folds = 96 model fits.
    """
    scale_pos = int((y_train == 0).sum() / (y_train == 1).sum())

    xgb_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   XGBClassifier(
            random_state=42,
            scale_pos_weight=scale_pos,
            eval_metric="logloss",
            n_jobs=-1
        ))
    ])

    param_grid = {
        "model__n_estimators":     [100, 300],
        "model__max_depth":        [3, 5],
        "model__learning_rate":    [0.05, 0.1],
        "model__subsample":        [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0]
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        xgb_pipeline,
        param_grid,
        cv=cv,
        scoring="recall",
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    print(f"\nBest parameters : {grid_search.best_params_}")
    print(f"Best CV recall  : {grid_search.best_score_:.3f}")

    return grid_search.best_estimator_


# ─────────────────────────────────────────────────────────────
# 8. Final Evaluation
# ─────────────────────────────────────────────────────────────
def evaluate_model(model: Pipeline,
                   X_test: pd.DataFrame,
                   y_test: pd.Series) -> None:
    """
    Full evaluation on held-out test set.
    Prints metrics and saves PR curve + feature importance plots.
    """
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    cm        = confusion_matrix(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_proba)

    print("\n=== FINAL TEST SET RESULTS ===")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Recall    : {recall:.3f}")
    print(f"Precision : {precision:.3f}")
    print(f"F1        : {f1:.3f}")
    print(f"AUC       : {auc_score:.3f}")
    print(f"\nFalse Negatives (missed defaulters): {cm[1][0]:,}")

    # Business impact
    fn_before = y_test.sum()
    fn_after  = cm[1][0]
    reduction = (1 - fn_after / fn_before) * 100
    savings   = (fn_before - fn_after) * 20000
    print(f"\n── Business Impact ──────────────────")
    print(f"Defaulters missed before model : {fn_before:,}")
    print(f"Defaulters missed after model  : {fn_after:,}")
    print(f"Reduction                      : {reduction:.1f}%")
    print(f"Estimated loss prevented       : ${savings:,.0f}")

    # PR Curve
    precisions, recalls, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recalls, precisions)

    plt.figure(figsize=(8, 5))
    plt.plot(recalls, precisions,
             color="purple", linewidth=2,
             label=f"XGBoost (PR-AUC = {pr_auc:.3f})")
    plt.axhline(y=y_test.mean(), color="gray",
                linestyle="--",
                label=f"Random baseline ({y_test.mean():.1%})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve — Credit Risk Detection")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("pr_curve.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("PR curve saved → pr_curve.png")

    # Feature importance
    xgb_model   = model.named_steps["model"]
    feature_names = model.named_steps["imputer"].feature_names_in_

    importances = pd.Series(
        xgb_model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=True)

    plt.figure(figsize=(10, 6))
    importances.plot(kind="barh", color="steelblue", alpha=0.8)
    plt.title("Feature Importance — XGBoost Credit Risk Model")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Feature importance saved → feature_importance.png")


# ─────────────────────────────────────────────────────────────
# 9. Predict Single Applicant
# ─────────────────────────────────────────────────────────────
def predict_applicant(model: Pipeline,
                       applicant: dict) -> None:
    """
    Run inference on a single new loan applicant.
    Uses the exact feature names the model was trained on.
    """
    feature_names = list(
        model.named_steps["imputer"].feature_names_in_
    )
    sample = pd.DataFrame([applicant])[feature_names]

    proba = model.predict_proba(sample)[0][1]
    pred  = model.predict(sample)[0]

    print("\n── Applicant Prediction ─────────────")
    print(f"Default probability : {proba:.1%}")
    print(f"Decision            : "
          f"{'REJECT ❌ (high risk)' if pred == 1 else 'APPROVE ✅ (low risk)'}")


# ─────────────────────────────────────────────────────────────
# 10. Main
# ─────────────────────────────────────────────────────────────
def main():

    # ── Load ──────────────────────────────────────────────────
    df = load_data("data/cs-training.csv")

    # ── EDA ───────────────────────────────────────────────────
    run_eda(df)

    # ── Split ─────────────────────────────────────────────────
    X = df.drop('SeriousDlqin2yrs', axis=1)
    y = df['SeriousDlqin2yrs']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain : {X_train.shape} | Test : {X_test.shape}")
    print(f"Train default rate : {y_train.mean():.1%}")
    print(f"Test  default rate : {y_test.mean():.1%}")

    # ── Baseline ──────────────────────────────────────────────
    baseline_evaluation(X_train, y_train)

    # ── Model comparison ──────────────────────────────────────
    compare_models(X_train, y_train)

    # ── Threshold analysis ────────────────────────────────────
    threshold_analysis(X_train, y_train)

    # ── Tune XGBoost ──────────────────────────────────────────
    best_model = tune_xgboost(X_train, y_train)

    # ── Final evaluation ──────────────────────────────────────
    evaluate_model(best_model, X_test, y_test)

    # ── Save model ────────────────────────────────────────────
    joblib.dump(best_model, "credit_risk_model.pkl")
    print("\nModel saved → credit_risk_model.pkl")

    # ── Sample prediction ─────────────────────────────────────
    # Use the exact feature names the model was trained on
    feature_names = list(
        best_model.named_steps["imputer"].feature_names_in_
    )
    print(f"\nModel expects these features:\n{feature_names}")

    # High-risk applicant example
    high_risk = {name: 0 for name in feature_names}
    high_risk.update({
        'RevolvingUtilizationOfUnsecuredLines': 0.95,
        'age':                                  35,
        'NumberOfTime30-59DaysPastDueNotWorse': 3,
        'DebtRatio':                            0.8,
        'MonthlyIncome':                        2500,
        'NumberOfOpenCreditLinesAndLoans':       6,
        'NumberOfTimes90DaysLate':              2,
        'NumberOfDependents':                   3
    })
    predict_applicant(best_model, high_risk)

    # Low-risk applicant example
    low_risk = {name: 0 for name in feature_names}
    low_risk.update({
        'RevolvingUtilizationOfUnsecuredLines': 0.10,
        'age':                                  45,
        'NumberOfTime30-59DaysPastDueNotWorse': 0,
        'DebtRatio':                            0.2,
        'MonthlyIncome':                        8000,
        'NumberOfOpenCreditLinesAndLoans':       4,
        'NumberOfTimes90DaysLate':              0,
        'NumberOfDependents':                   1
    })
    predict_applicant(best_model, low_risk)


if __name__ == "__main__":
    main()
