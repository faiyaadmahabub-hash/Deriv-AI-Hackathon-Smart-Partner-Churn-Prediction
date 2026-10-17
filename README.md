# Smart Partner Churn Prediction

An independent **Deriv AI Hackathon prototype** for relationship managers exploring partner churn: who appears at risk, what signals might explain it, and how a retention intervention could be prioritized.

The project combines a synthetic-data generator, an XGBoost training pipeline and a Streamlit dashboard with risk, portfolio, business-impact, competitor and intervention views. **Every dataset and business scenario is synthetic.** This is a collaborative hackathon project, not an official Deriv product or a deployed decision system.

![Local synthetic partner dashboard](docs/assets/dashboard-local.png)

*Real local Streamlit capture, 17 October 2026. The dashboard uses its separate 200-row synthetic table and hand-coded risk rules. Displayed dollars, risk levels and interventions are illustrative; they are not measured commercial outcomes or trained-model predictions.*

## What is implemented

- Schema-driven generation of partner engagement, financial, relationship and competitive signals, with a simulated decline cascade and churn labels.
- Cleaning, feature engineering, encoding/scaling, stratified splitting and an XGBoost training/evaluation entry point.
- Library modules for SHAP explanations, CUSUM warnings, priority tiers, financial exposure, intervention ROI, competitor grouping and feedback analysis.
- A working Streamlit demonstration with a home view and five analysis pages.
- Optional Gemini recommendation code; service access and generated advice were not exercised in this verification.

```mermaid
flowchart LR
    Gen[Synthetic generator] --> Prep[Preprocessing]
    Prep --> Train[XGBoost training and evaluation]
    Train --> Model[Saved model artifact]
    Gen --> Modules[Risk / exposure / competitor / feedback modules]
    Sample[Separate synthetic dashboard table] --> UI[Streamlit views]
    Modules -. partial integration .-> UI
    Model -. future dashboard integration .-> UI
```

## Working status and evidence

The local verification uses 160 newly generated synthetic rows, then checks preprocessing, risk-tier boundaries and financial-exposure arithmetic without retraining a model. Streamlit checks exercise actual application pages. Exact commands, results and limitations are in [verification notes](docs/VERIFICATION.md).

No new accuracy, ROC-AUC, SHAP quality, early-warning lead time or retention-impact result is claimed. Saved binary models and datasets are historical project artifacts. Full training and model/service integration need separate validation.

## Quick start

Python 3.10 was used locally. The source is nested inside two archive-style directories; start from the repository root and enter the actual application folder:

```powershell
Set-Location 'Deriv-AI-Hackathon-Smart-Partner-Churn-Prediction-master (2)/Deriv-AI-Hackathon-Smart-Partner-Churn-Prediction-master'
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install pandas numpy scikit-learn xgboost joblib python-dotenv streamlit plotly
python -m streamlit run dashboard/app.py --server.address 127.0.0.1
```

Open `http://localhost:8501`. The dashboard can run without a model or API key. Dependency versions are not locked; this was tested with existing local packages, not a clean installation.

For the fuller synthetic pipeline, `python run_demo.py --mode train` generates data, preprocesses and trains/evaluates a model; it overwrites generated artifacts. `--mode analyze` runs the orchestrator, which can use synthetic probabilities without a trained model. Optional SHAP and Gemini modules need additional dependencies. See the [full setup and code guide](docs/PROJECT_GUIDE.md).

From the repository root, repeat the bounded local check with:

```powershell
python docs/verify_local.py
```

## Limitations that matter

The dashboard is not wired to the trained XGBoost/SHAP pipeline: its explanatory bars, trends and intervention outcomes are demonstrations. Generator labels are rule-produced, and the committed data has a strongly skewed platinum cohort. Pre-split feature engineering uses dataset-wide statistics; validation on independent real partner history is absent.

Definitions differ across modules, including risk thresholds, region names and feature columns. The orchestrator has a model-probability shape mismatch and an ROI-key mismatch. CUSUM has toy demonstrations, not a validated warning lead time. Financial assumptions are fixed placeholders. The Gemini code and model identifiers require review before service use; no API success is claimed here. Keep credentials outside Git, and load pickle/joblib artifacts only from trusted sources.

## Provenance and deeper material

Repository history includes contributions from Emmanuel Nara, Faiyad Mahabub and `takif0`; the included archive identifies Team35. This overview describes the combined checked-in work without assigning unsupported individual ownership. The [preserved original guide](docs/PROJECT_GUIDE.md) contains module details, issue analysis and the source layout. Strategy documents, the challenge brief and supplementary category datasets remain in their original folders. There is no license file; reuse rights are not specified.
