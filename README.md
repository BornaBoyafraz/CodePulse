# CodePulse — Predictive Code Risk from Git History

CodePulse mines the full commit history of any public GitHub repository and predicts which files are most likely to introduce bugs in the next 30 days — before any new code is written. It extracts 22 per-file signals from git history (churn, coupling, authorship entropy, cyclomatic complexity), trains an XGBoost classifier on a strict temporal split, and explains every risk score with SHAP values so the output is never a black box.

## Why This Is Different From Static Analysis

Static analysis reads the code. CodePulse reads the commit history. Static tools find bugs that already exist in the source. CodePulse predicts bugs that have not been written yet, based on where historical failures have concentrated. A file that has been modified 40 times in the last 30 days by 6 different authors, always alongside another notoriously unstable file, has a story that no linter can see.

## Tech Stack

| Purpose | Library |
|---|---|
| Git mining | pydriller |
| Data processing | pandas, numpy |
| Cyclomatic complexity | radon |
| Machine learning | scikit-learn, xgboost |
| Explainability | shap |
| File coupling graphs | networkx |
| Visualization | plotly |
| Backend API | fastapi, uvicorn |
| Frontend | HTML, CSS, JavaScript, Plotly.js |
| Caching | joblib |

## How to Run Locally

```bash
git clone https://github.com/BornaBoyafraz/CodePulse.git
cd CodePulse
pip install -r requirements.txt
uvicorn api.main:app --reload
```

Open `http://localhost:8000` in your browser, enter any public GitHub repository URL, and click Analyze. The first run mines the full commit history and may take 60–120 seconds depending on repository size. Subsequent runs on the same repository are served from cache in a few seconds.

## How It Works

1. **Mine** — PyDriller walks the full commit history, extracting every file touched, its author, lines changed, and commit message.
2. **Label** — Commits whose messages match bug-fix keywords (fix, bug, error, defect, patch, crash, etc.) are labeled positive.
3. **Feature engineering** — For each file, 22 features are computed across four dimensions:
   - **Churn** — rolling lines changed and commit counts over 30, 60, and 90-day windows; spike detection
   - **Coupling** — co-change matrix built with NetworkX; files that always change together share risk
   - **Authorship** — bus factor score, Shannon entropy of contributor distribution
   - **Complexity** — radon cyclomatic complexity grade and rank
4. **Temporal split** — Training data = all file activity before a cutoff 6 months before the most recent commit. Test data = the 6 months after. No random shuffling is used; this prevents data leakage and reflects how the model would actually be used in practice.
5. **Train** — XGBoost binary classifier with `scale_pos_weight` to handle class imbalance. Early stopping on AUC.
6. **Explain** — SHAP TreeExplainer produces a ranked list of the specific features driving each file's risk score upward.

## Repos Tested On

- `pallets/flask`
- `psf/requests`
- `scikit-learn/scikit-learn`
- `django/django`

## Disclaimer

Risk scores are statistical estimates based on historical commit patterns. They reflect where bugs have concentrated in the past, not where they will certainly appear in the future. This tool is for informational and exploratory purposes.

---

*Solo hackathon project. Built with Python, FastAPI, XGBoost, and SHAP.*
