# 🧠 Credit Risk Analysis using XGBoost and LightGBM

This project predicts the probability of loan default using the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) dataset. It leverages ensemble models like **LightGBM** and **XGBoost**, with a strong focus on interpretability using **SHAP**.

---

##  Project Structure

```
credit-risk-project/
├── notebooks/                # Jupyter notebooks with EDA, modeling, SHAP
├── outputs/                  # Model predictions (not pushed to GitHub)
├── feature_engineering.py    # Custom feature transformations
├── requirements.txt          # Reproducible dependencies
├── README.md
└── .gitignore
```

---

##  Models and Results

| Model    | AUC (Validation) |
| -------- | ---------------- |
| LightGBM | 0.79             |
| XGBoost  | 0.76             |

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

##  Future Work

- Streamlit dashboard for live scoring and interpretability
- Integrate auxiliary tables (bureau, previous applications)
- Hyperparameter tuning and model ensembling

---

##  Author

Adwait Shelke · [GitHub](https://github.com/adwaitshelke) · [LinkedIn](https://www.linkedin.com)

---



