# Credit Card Default Risk Lab

## a. Problem statement
Predict whether a credit-card client will **default on the next month's payment**
(binary classification: 0 = no default, 1 = default) from demographic details,
credit limit, six months of repayment status, bill amounts and payment amounts.
Five classifiers are trained on the same split and compared with six metrics; the
trained models are served through an interactive Streamlit app.

## b. Dataset description
- **Source:** UCI Machine Learning Repository / Kaggle - *Default of Credit Card Clients* (Taiwan, 2005).
- **Instances:** 30,000 | **Features:** 23 (after dropping `ID`) | **Target:** `default_next_month` (0/1).
- **Feature groups:** `LIMIT_BAL`; demographics (`SEX`, `EDUCATION`, `MARRIAGE`, `AGE`);
  repayment status `PAY_1`-`PAY_6`; bill statements `BILL_AMT1`-`BILL_AMT6`;
  previous payments `PAY_AMT1`-`PAY_AMT6`.
- **Class balance:** ~22 % defaulters (imbalanced) - <FILL: exact value printed by train.py>.
- **Preprocessing:** `ID` dropped; `PAY_0` renamed `PAY_1`; undocumented codes in
  `EDUCATION` (0, 5, 6) merged into 4 and `MARRIAGE` 0 merged into 3; `StandardScaler`
  inside the pipeline for LR, KNN, NB and SVM.
- **Split:** stratified 80/20, `random_state=42`. The 20 % hold-out is `test_data.csv`.

## c. GitHub repository link
<FILL: https://github.com/sathyapv92/creditcard-default-detection>

Live app: <FILL: https://creditcard-default-detection.streamlit.app>

## d. Models used

Metrics on the 20 % hold-out set (copy from `metrics.csv`):

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | | | | | | |
| Decision Tree | | | | | | |
| KNN | | | | | | |
| Naive Bayes | | | | | | |
| Support Vector Machine | | | | | | |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | <FILL - write from YOUR numbers> |
| Decision Tree | <FILL> |
| KNN | <FILL> |
| Naive Bayes | <FILL> |
| Support Vector Machine | <FILL> |
| **Overall winner for this dataset** | <FILL: model + justification (AUC / F1 / MCC)> |

## Repository layout
```
app.py            Streamlit front-end (loads saved models, never retrains)
evaluation.py     metric helper shared by train.py and app.py
train.py          trains + saves the five pipelines, writes metrics.csv / test_data.csv
requirements.txt  deployment dependencies
test_data.csv     held-out test set (features + target) to upload in the app
metrics.csv       training-time metrics for all models
model/            five .joblib pipelines
```

## Run locally
```
pip install -r requirements.txt
python train.py --data data/UCI_Credit_Card.csv
streamlit run app.py
```
