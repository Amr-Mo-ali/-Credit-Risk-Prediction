# 💳 Credit Risk Prediction
### Predicting Loan Default Using Machine Learning

---

| | |
|---|---|
| **Author** | Amr Mohammed Ali |
| **Dataset** | Give Me Some Credit — Kaggle Competition |
| **Goal** | Predict whether a borrower will default within 2 years |
| **Best Model** | XGBoost — Recall: 77.6% / AUC: 0.866 |
| **Kaggle** | [Notebook](https://www.kaggle.com/code/amrmohammedali/heart-disease-detection) |
| **GitHub** | [Repository](https://github.com/Amr-Mo-ali) |

---

## 🏦 Business Context

A bank wants to predict whether a loan applicant will default
before approving their loan. Every missed defaulter costs the
bank $10,000–$50,000+ in losses.

| Mistake | Real-world consequence |
|---|---|
| **False Negative** (approve a defaulter) | Bank loses the loan amount |
| **False Positive** (reject a safe borrower) | Bank loses a customer |

A False Negative is far more costly → **we optimize for Recall.**

### Success Criteria
- Primary: **Recall ≥ 75%** — catch at least 3 in 4 defaulters
- Secondary: **AUC ≥ 0.85** — strong overall discrimination
- Evaluation: **PR Curve** — because classes are severely imbalanced (93% / 7%)

---

## 📊 Dataset

- **Source:** [Give Me Some Credit — Kaggle](https://www.kaggle.com/datasets/brycecf/give-me-some-credit-dataset)
- **Size:** 150,000 borrowers × 11 features
- **Target:** `SeriousDlqin2yrs` — 1 = defaulted within 2 years
- **Class balance:** 93% no default / 7% default — severely imbalanced

| Feature | Description |
|---|---|
| RevolvingUtilizationOfUnsecuredLines | Credit card usage ratio |
| age | Borrower age |
| NumberOfTime30-59DaysPastDueNotWorse | Times 30-59 days late |
| DebtRatio | Monthly debt / monthly income |
| MonthlyIncome | Monthly income |
| NumberOfOpenCreditLinesAndLoans | Open credit lines |
| NumberOfTimes90DaysLate | Times 90+ days late |
| NumberRealEstateLoansOrLines | Real estate loans |
| NumberOfTime60-89DaysPastDueNotWorse | Times 60-89 days late |
| NumberOfDependents | Number of dependents |

---

## 🔬 Project Workflow

```
1. Problem Framing     → Recall priority, imbalanced data strategy
2. EDA                 → Distributions, correlations, class balance
3. Data Cleaning       → Pipeline + SimpleImputer (no leakage)
4. Class Imbalance     → scale_pos_weight (93/7 ratio = 14x weight)
5. Model Comparison    → 5 algorithms compared on recall
6. Threshold Analysis  → Random Forest diagnosis (0.5 → 0.1)
7. GridSearchCV        → 96 fits optimizing for recall
8. Final Evaluation    → Confusion matrix, PR curve, AUC
9. Feature Importance  → Business interpretation
10. Model Saving       → joblib export for production
```

---

## 🚨 Key Challenge — Class Imbalance

Without correction, Logistic Regression achieved:
- Accuracy: 93% ← meaningless
- Recall: **4%** ← catching only 4 in 100 defaulters

**Why?** The model learned to predict "no default" for everyone —
getting 93% accuracy while being completely useless for the bank.

**Fix:** `scale_pos_weight = 14` in XGBoost — tells the model a
defaulter is 14× more important than a non-defaulter.

**Result:** Recall jumped from **4% → 77.6%**

---

## 📈 Results

### Model Comparison (3-fold CV on training set)

| Model | Recall | Precision | F1 |
|---|---|---|---|
| Logistic Regression | 64.8% | 17.4% | 27.4% |
| Decision Tree | 77.8% | 18.9% | 30.4% |
| Random Forest* | 66.7% | 25.0% | 36.3% |
| SVM | 65.1% | 24.8% | 35.9% |
| **XGBoost** | **76.3%** | — | — |

*Required threshold tuning: 0.5 → 0.1

### Final Test Set Performance (XGBoost tuned)

| Metric | Score |
|---|---|
| **Recall** | **77.6%** |
| Precision | 21.5% |
| F1 | 33.7% |
| **AUC** | **0.866** |

### Confusion Matrix

```
                    Predicted Safe   Predicted Default
Actual Safe              22,310            5,685
Actual Default              448            1,557
```

### 💰 Business Impact

| | Before Model | After Model |
|---|---|---|
| Defaulters missed | 8,021 | 448 |
| Reduction | — | **94.4%** |
| Estimated loss prevented | — | **~$150M per 150k loans** |

---

## 🏆 Best Hyperparameters (GridSearchCV)

```python
{
    'n_estimators':    100,
    'max_depth':       3,      # shallow trees → less overfitting
    'learning_rate':   0.05,   # small steps → better generalization
    'subsample':       0.8,    # 80% samples per tree → randomness
    'colsample_bytree': 1.0    # use all features per tree
}
```

GridSearchCV tested **32 combinations × 3 folds = 96 model fits**,
scoring on recall at every step.

---

## 🔍 Feature Importance

Top 3 most predictive features:

1. **RevolvingUtilizationOfUnsecuredLines** — High credit card usage signals cash-flow stress
2. **NumberOfTimes90DaysLate** — Severe past delinquency is the strongest behavioral signal
3. **NumberOfTime30-59DaysPastDueNotWorse** — Even mild delays indicate repayment difficulty

> **Key insight:** All top features are behavioral (payment history),
> not demographic (age, income). Past behavior is the strongest predictor
> of future default — consistent with FICO score methodology.

---

## 🛠 Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.x | Core language |
| Pandas + NumPy | Data manipulation |
| Matplotlib + Seaborn | Visualization |
| Scikit-learn | Pipelines, models, evaluation |
| XGBoost | Final champion model |
| Joblib | Model serialization |
| Imbalanced-learn | SMOTE (explored) |

---

## 📁 Project Structure

```
credit-risk-prediction/
│
├── credit_risk.ipynb          ← Full notebook with analysis
├── credit_risk.py             ← Clean Python script
├── credit_risk_model.pkl      ← Saved trained model
├── README.md                  ← This file
└── data/
    └── cs-training.csv        ← Dataset
```

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/Amr-Mo-ali/credit-risk-prediction.git
cd credit-risk-prediction

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib imbalanced-learn

# Run the notebook
jupyter notebook credit_risk.ipynb
```

---

## 🔮 Load the Model & Predict

```python
import joblib
import pandas as pd

model = joblib.load("credit_risk_model.pkl")

applicant = pd.DataFrame([{
    'RevolvingUtilizationOfUnsecuredLines': 0.85,
    'age': 35,
    'NumberOfTime30-59DaysPastDueNotWorse': 2,
    'DebtRatio': 0.6,
    'MonthlyIncome': 3000,
    'NumberOfOpenCreditLinesAndLoans': 5,
    'NumberOfTimes90DaysLate': 1,
    'NumberRealEstateLoansOrLines': 0,
    'NumberOfTime60-89DaysPastDueNotWorse': 1,
    'NumberOfDependents': 3
}])

proba = model.predict_proba(applicant)[0][1]
pred  = model.predict(applicant)[0]

print(f"Default probability: {proba:.1%}")
print(f"Decision: {'REJECT ❌' if pred==1 else 'APPROVE ✅'}")
```

---

## 📚 Key Learnings

1. **Accuracy is useless for imbalanced data** — 93% accuracy with 4% recall is a failure
2. **Default threshold (0.5) assumes balanced classes** — always tune for imbalanced problems
3. **Past payment behavior beats demographics** — behavioral signals dominate
4. **Pipeline prevents data leakage** — imputer fits on train only, transforms test
5. **Business impact > metrics** — 94.4% reduction in missed defaults is the real story

---

## 👤 Author

**Amr Mohammed Ali**
- 🎓 CS & AI Graduate — Beni Suef University (2024)
- 💼 ML Engineer | AI Automation Developer
- 🔗 [GitHub](https://github.com/Amr-Mo-ali)
- 📊 [Kaggle](https://www.kaggle.com/amrmohammedali)

---

## 📄 License

MIT License — free to use and modify with attribution.
