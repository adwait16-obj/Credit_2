#  Credit Risk Analysis with LightGBM, XGBoost, and SHAP

This project presents a detailed implementation of **credit default prediction** using gradient-boosted decision trees. It leverages the [Home Credit Default Risk dataset](https://www.kaggle.com/competitions/home-credit-default-risk) to build interpretable models capable of identifying potentially risky loan applicants.

---

##  Objective

- Predict the probability of **loan default** from loan application and credit history.
- Engineer features from auxiliary data (e.g., `bureau.csv`) to capture behavioral signals.
- Train **LightGBM** and **XGBoost** models and evaluate via **AUC-ROC**.
- Apply **SHAP** to explain global feature importance and local applicant risk.

---

##  Methodology

### 1. Data Sources

- `application_train.csv` — Main data with features and binary target
- `bureau.csv` — Applicant credit records from other institutions
- `application_test.csv` — Hold-out data for simulation/testing

### 2. Data Preprocessing

- Merged `bureau.csv` aggregates:  
  - `BUREAU_OVERDUE_MEAN`, `BUREAU_OVERDUE_MAX`, `BUREAU_CREDIT_SUM`, etc.
- Missing values handled via:
  - Median imputation for numerics
  - Mode imputation + `LabelEncoder` for categoricals
- Removed ID columns (`SK_ID_CURR`) post-merge

### 3. Feature Construction

- Final feature matrix `X` included over 100 features
- Target variable: `y = application_train['TARGET']`
- All object columns converted to numeric labels

### 4. Model Training

#### LightGBM

LightGBM AUC-ROC = 0.76
XGBoost AUC-ROC = 0.75

---

##  Models and Results

| Model    | AUC (Validation) |
| -------- | ---------------- |
| LightGBM | 0.76             |
| XGBoost  | 0.75             |

- Used `StratifiedShuffleSplit` with 20% validation
- Evaluated with AUC-ROC on unseen validation data
- `application_test.csv` used for simulated production prediction
- Features like `EXT_SOURCE_1` and `CREDIT_DAY_OVERDUE` were top predictors

---

##  Interpretability with SHAP

SHAP was used to analyze both global and individual predictions:

- `EXT_SOURCE_1`: Strong negative correlation with default
- `DAYS_CREDIT_ENDDATE`: Longer pending credits increase risk
- Visualized summary plots and individual waterfall explanations

---

##  Sample Output

```python
predict_risk(sk_id=100001)
# → Shows SHAP waterfall and risk score: 0.87 (high risk)
```

---



##  Author

Adwait Shelke · [GitHub](https://github.com/adwaitshelke) · [LinkedIn](https://www.linkedin.com)

---



