#  Credit Risk Analysis with LightGBM, XGBoost, and SHAP

This project presents a detailed implementation of **credit default prediction** using gradient-boosted decision trees. It leverages the [Home Credit Default Risk dataset](https://www.kaggle.com/competitions/home-credit-default-risk) to build interpretable models capable of identifying potentially risky loan applicants.

---

## Important terminologies
## Technical Primer: Algorithms and Interpretability

### What is XGBoost?

XGBoost (Extreme Gradient Boosting) is an efficient and scalable implementation of gradient-boosted decision trees. It is particularly effective for structured/tabular datasets and is widely used in machine learning competitions.

Key characteristics:
- Regularization (L1 and L2) to reduce overfitting
- Tree boosting via gradient descent optimization
- Row and column subsampling for variance reduction
- Parallel and distributed training for speed

XGBoost builds models iteratively, where each new tree corrects the residuals of the previous ones.

---

### What is LightGBM?

LightGBM (Light Gradient Boosting Machine) is a high-performance gradient boosting framework developed by Microsoft. It is optimized for both speed and efficiency, especially on large datasets with categorical features.

Key features:
- Histogram-based split finding
- Leaf-wise tree growth with depth constraints
- Efficient handling of categorical variables
- Lower memory usage and faster training compared to XGBoost

LightGBM is particularly suitable for high-dimensional, sparse data and is often preferred in large-scale production pipelines.

---

### What is SHAP?

SHAP (SHapley Additive exPlanations) is a unified framework for interpreting machine learning predictions based on game theory. It attributes to each feature the contribution it made to a particular prediction, ensuring fairness and consistency.

Key principles:
- SHAP values represent the average marginal contribution of a feature across all possible coalitions (feature combinations).
- The sum of SHAP values for all features equals the difference between the actual model output and the expected output.
- Provides both global interpretability (feature importance across all samples) and local interpretability (why a model predicted a specific value for one sample).

SHAP visualizations used in this project include:
- Summary bar plots to rank global feature importance
- Waterfall plots to explain individual predictions

---

### Why Interpretability Matters

In credit risk modeling, interpretability is crucial due to:

- Regulatory compliance (e.g., explaining loan rejection decisions)
- Transparency for stakeholders and clients
- Trustworthiness and fairness in automated decision-making systems
- Model debugging and feature selection insights

By combining SHAP with high-performing models like XGBoost and LightGBM, we ensure that our predictions are not only accurate but also explainable and reliable.


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
### SHAP Summary Plot — LightGBM

![SHAP LightGBM](outputs/shap_summary_lgbm.png)

### SHAP Summary Plot — XGBoost

![SHAP XGBoost](outputs/shap_summary_xgb.png)
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



