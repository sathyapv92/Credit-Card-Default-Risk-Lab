# Credit Card Default Risk Lab

## a. Problem statement
Predict whether a credit-card client will **default on the next month's payment**
(binary classification: 0 = no default, 1 = default) from demographic details,
credit limit, six months of repayment status, bill amounts and payment amounts.
Five classifiers are trained on the same split and compared with six metrics; the
trained models are served through an interactive Streamlit app.

## b. Dataset description
- **Source:** UCI Machine Learning Repository / Kaggle - *Default of Credit Card Clients* (Taiwan, 2005).
  Kaggle link: https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset
- **Instances:** 30,000 | **Features:** 23 (after dropping `ID`) | **Target:** `default_next_month` (0/1).
  (Meets the assignment minimums of 12 features and 500 instances.)
- **Feature groups:** `LIMIT_BAL`; demographics (`SEX`, `EDUCATION`, `MARRIAGE`, `AGE`);
  repayment status `PAY_1`-`PAY_6`; bill statements `BILL_AMT1`-`BILL_AMT6`;
  previous payments `PAY_AMT1`-`PAY_AMT6`.
- **Class balance:** imbalanced - 23,364 non-defaulters (77.88 %) vs 6,636 defaulters (22.12 %);
  default rate = 0.221 as printed by `train.py`. The test split keeps the same ratio
  (1,327 defaulters out of 6,000 rows = 22.12 %).
- **Preprocessing:** `ID` dropped; `PAY_0` renamed `PAY_1`; undocumented codes in
  `EDUCATION` (0, 5, 6) merged into 4 and `MARRIAGE` 0 merged into 3; `StandardScaler`
  inside the pipeline for LR, KNN, NB and SVM (the Decision Tree is scale-invariant, so it has no scaler).
- **Split:** stratified 80/20 (24,000 train / 6,000 test), `random_state=42`. The 20 % hold-out is `test_data.csv`.

## c. GitHub repository link
https://github.com/sathyapv92/Credit-Card-Default-Risk-Lab

Live app: https://sathyapv92-credit-card-default-risk-lab-app-odaors.streamlit.app
## d. Models used

Metrics on the 20 % hold-out set (6,000 rows), produced by `train.py` and saved in `metrics.csv`:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8083 | 0.7079 | 0.6903 | 0.2419 | 0.3583 | 0.3276 |
| Decision Tree | 0.8167 | 0.7508 | 0.6697 | 0.3376 | 0.4489 | 0.3828 |
| KNN | 0.8082 | 0.7429 | 0.6302 | 0.3210 | 0.4254 | 0.3512 |
| Naive Bayes | 0.7518 | 0.7248 | 0.4504 | 0.5539 | 0.4968 | 0.3376 |
| Support Vector Machine | 0.8162 | 0.7186 | 0.6662 | 0.3384 | 0.4488 | 0.3814 |

Model settings: Logistic Regression (`max_iter=2000`); Decision Tree (`max_depth=6`, `min_samples_leaf=50`);
KNN (`n_neighbors=25`); Gaussian Naive Bayes (defaults); SVM (RBF kernel, `C=1.0`, `gamma="scale"`).
All use `random_state=42` where applicable.

Confusion matrices on the test set (TN / FP / FN / TP):

| Model | TN | FP | FN | TP |
|---|---|---|---|---|
| Logistic Regression | 4529 | 144 | 1006 | 321 |
| Decision Tree | 4452 | 221 | 879 | 448 |
| KNN | 4423 | 250 | 901 | 426 |
| Naive Bayes | 3776 | 897 | 592 | 735 |
| Support Vector Machine | 4448 | 225 | 878 | 449 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Reaches 80.8 % accuracy, but that is only slightly above the 77.9 % obtained by always predicting "no default". It has the highest precision (0.69) yet the lowest recall (0.24): it catches only 321 of 1,327 defaulters. It also has the lowest AUC (0.708) and F1 (0.358), because a linear boundary cannot capture non-linear patterns in the repayment-status features. It is a conservative, interpretable baseline. |
| Decision Tree | Best accuracy (0.8167), best AUC (0.7508) and best MCC (0.3828). The depth cap (6) and minimum leaf size (50) keep it from overfitting. It can pick out strong rules such as recent late-payment status (`PAY_1` and the other `PAY_*` columns). Precision (0.67) and recall (0.34) are balanced better than for LR or KNN, and the model is easy to explain. |
| KNN | Second-best AUC (0.7429) with accuracy 0.8082. Its precision (0.63) is lower than LR's and the tree's, and recall (0.32) is modest. It sits in the middle on F1 (0.425) and MCC (0.351). It needs scaled features, stores the whole training set (the largest model file after SVM) and is slower at prediction time, without outperforming the tree. |
| Naive Bayes | Lowest accuracy (0.7518) and lowest precision (0.45), but the highest recall (0.554) and highest F1 (0.497); it flags 735 of 1,327 defaulters. The price is 897 false alarms. The independence assumption is violated because the six bill amounts are highly correlated, so its probabilities are poorly calibrated, yet it errs towards flagging risk. It suits cases where missing a defaulter costs more than a false alarm. |
| Support Vector Machine | Nearly tied with the Decision Tree on accuracy (0.8162), F1 (0.4488) and MCC (0.3814), and almost identical TP/FP counts. Its AUC (0.7186) is lower than the tree's, KNN's and NB's, so its ranking of risk is weaker. Training took about 26 s, versus under 1 s for the other models, and it gives no natural probabilities. Its extra complexity brings no advantage over the tree. |
| **Overall winner for this dataset** | **Decision Tree.** It ranks first on AUC (0.7508), Accuracy (0.8167) and MCC (0.3828, the most balanced single metric on imbalanced data), and is a close second on F1 (0.4489, behind Naive Bayes at 0.4968), while being small, fast and interpretable. SVM is a close runner-up on MCC, and Naive Bayes is the best choice if recall (catching defaulters) is the priority. Note that the margins are small and come from a single stratified split, so differences of about 0.01 should not be over-interpreted. |

## Streamlit app features
- Upload a test CSV (features + `default_next_month` target) or use the bundled `test_data.csv`
- Model selection dropdown (all five classifiers)
- Evaluation metrics: Accuracy, AUC, Precision, Recall, F1, MCC
- Confusion matrix and classification report
- Saved models are only loaded, never retrained on the uploaded data

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
