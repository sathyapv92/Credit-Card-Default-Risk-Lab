# Setup guide (read me first; not part of the graded README)

## 0. Honest status of this package
The code was tested end-to-end on a SYNTHETIC dataset with the same schema
(30,000 x 25). train.py ran for all five models; app.py's logic was exercised
with a stubbed Streamlit for all five models and both validation errors.
Not tested here: the real Kaggle file (no internet in my sandbox) and a real
Streamlit server. Your first run on the real data is the real test.
No model files or metrics are included: you generate them from the real data.

## 1. Get the data
Download `UCI_Credit_Card.csv` from Kaggle ("Default of Credit Card Clients Dataset")
and put it in `data/`. (UCI's .xls also works: `--data file.xls`.)

## 2. On BITS Virtual Lab (this earns the screenshot mark)
```
pip install -r requirements.txt
python train.py
streamlit run app.py
```
Take ONE screenshot showing the terminal output of train.py (metrics table) and,
ideally, the lab environment/URL bar visible.

## 3. Pin scikit-learn
train.py prints a line like `scikit-learn==1.x.y`. Replace `scikit-learn>=1.3`
in requirements.txt with it. Models saved with one version may not load on another.

## 4. Fill the README
Paste numbers from metrics.csv, write your own observations, add both links.

## 5. GitHub
Create a public repo; upload (in several commits over a few days):
app.py, evaluation.py, train.py, requirements.txt, README.md, test_data.csv,
metrics.csv, model/*.joblib. Do NOT upload the big raw dataset (data/ is git-ignored).
If an upload is >25 MB via the browser, use git from the command line.

## 6. Streamlit Community Cloud
share.streamlit.io -> sign in with GitHub -> New app -> pick repo, branch main,
main file `app.py` -> Deploy. Open the URL, tick "Use bundled test_data.csv" or
upload test_data.csv, and check each model. If the build fails, read the logs
(usually a version mismatch; see step 3).

## 7. Personalize (the checker flags identical structure/variable names)
- Rename variables/files, change hyperparameters (try GridSearchCV) and say why.
- Restyle the app (titles, colours, tabs, extra plots).
- Write the observations yourself, from your results.

## 8. Final PDF (this order)
GitHub link -> Streamlit link -> BITS Lab screenshot -> README content.
Deadline: 08-Nov-2026 23:59, no drafts.
