<div align="center">

# LearnMate Analytics AI

### Student Employability & Placement Intelligence Platform

**From Student Data to Employability Intelligence**

![Image Alt](https://github.com/tejasvimunjal17-source/LEARNMATE-ANALYTICS-AI/blob/main/LearnMate%20Analytics%20AI%20Blueprint%20Architecture.png)

<br>

![Python](https://img.shields.io/badge/Python-0B3D2E?style=for-the-badge&logo=python&logoColor=14B8A6)
![Streamlit](https://img.shields.io/badge/Streamlit-1F2937?style=for-the-badge&logo=streamlit&logoColor=14B8A6)
![Pandas](https://img.shields.io/badge/Pandas-0B3D2E?style=for-the-badge&logo=pandas&logoColor=14B8A6)
![NumPy](https://img.shields.io/badge/NumPy-1F2937?style=for-the-badge&logo=numpy&logoColor=14B8A6)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-0B3D2E?style=for-the-badge&logo=scikit-learn&logoColor=14B8A6)
![Plotly](https://img.shields.io/badge/Plotly-1F2937?style=for-the-badge&logo=plotly&logoColor=14B8A6)
![Matplotlib](https://img.shields.io/badge/Matplotlib-0B3D2E?style=for-the-badge&logo=python&logoColor=14B8A6)

</div>

<br>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" alt="" style="max-width: 100%; display: inline-block;" data-target="animated-image.originalImage">

Live at - https://learnmate-analytics-ai-student-employability.streamlit.app/ 

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" alt="" style="max-width: 100%; display: inline-block;" data-target="animated-image.originalImage">


> A Streamlit-based student employability and placement intelligence platform built for the AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026 (BharatCares). It turns a real campus-placement dataset into exploratory analytics, classification, regression, clustering, grounded AI insights, and interactive scenario simulation — with every number calculated live from the data, not hard-coded.

<br>

## 📑 Table of Contents

| | | |
|---|---|---|
| [1. Project Overview](#1-project-overview) | [7. Project Architecture](#7-project-architecture) | [13. Running the Application](#13-running-the-application) |
| [2. Problem Statement](#2-problem-statement) | [8. Application Pages](#8-application-pages) | [14. Project Structure](#14-project-structure) |
| [3. Key Features](#3-key-features) | [9. Key Findings](#9-key-findings) | [15. Testing](#15-testing) |
| [4. Technology Stack](#4-technology-stack) | [10. Machine Learning Results](#10-machine-learning-results) | [16. Future Scope](#16-future-scope) |
| [5. Dataset](#5-dataset) | [11. Responsible AI / Limitations](#11-responsible-ai--limitations) | [17. Internship Context](#17-internship-context) |
| [6. Analytics & ML Pipeline](#6-analytics--ml-pipeline) | [12. Installation](#12-installation) | [18. Author / Student Information](#18-author--student-information) |

<br>

---

## 1. Project Overview

LearnMate Analytics AI combines Data Analytics, Exploratory Data Analysis, Classification, Regression, Clustering, explainable model evaluation, grounded AI insights, and scenario simulation into a single application. It is an **educational demonstration**, not a production employability predictor — it does not guarantee any individual's placement or salary outcome.

---

## 2. Problem Statement

Student placement datasets contain academic, educational, work-experience and specialisation information, but raw tables do not directly provide actionable insights. This project transforms the dataset through the pipeline:

<div align="center">

**DATA → INSIGHTS → PREDICTIONS → SEGMENTS → SCENARIOS → ACTIONABLE INTERPRETATION**

</div>

---

## 3. Key Features

| Module | What it delivers |
|---|---|
| **Overview dashboard** | Live KPI cards and academic/profile distributions |
| **Data Explorer** | Schema, data-quality report, cleaning summary, automated leakage-validation checks |
| **Exploratory Data Analysis** | Placement analysis, academic performance (with a chart-type selector), salary analysis, correlation/relationship analysis, deterministic FACT/INTERPRETATION insight cards, and a "From Data to Action" business-insights section |
| **Student Segmentation** | K-Means clustering (K=2–6) with Elbow and Silhouette diagnostics, an interactive K selector, cluster profiles, a PCA 2D visualization, and clearly-labeled post-clustering descriptive outcomes |
| **AI Insight Copilot** | A evidence-grounded question-answering assistant with quick-question cards and free-text input; works fully offline with a deterministic fallback engine, with an optional (not required) external LLM layer |
| **What-If Simulator** | Compares a baseline student profile against a hypothetical one through the *existing* trained classification model, plus single-variable and categorical "model response" experiments |
| **Placement Prediction** | Logistic Regression and Decision Tree classifiers with a live prediction form |
| **Salary Prediction** | Linear Regression and Random Forest Regressor with a live prediction form |
| **Model Evaluation** | Full classification and regression metric suites, confusion matrices, ROC curves, coefficients/feature importances, and a depth-limited decision tree visualization |
| **Project Summary** | A consolidated, submission-ready overview of the whole project |

---

## 4. Technology Stack

| Category | Technologies |
|---|---|
| **Application Framework** | Streamlit |
| **Data Handling** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn — Logistic Regression, Decision Tree, Linear Regression, Random Forest, K-Means, PCA, preprocessing, metrics |
| **Visualization** | Plotly (all interactive charts), Matplotlib (used only for the depth-limited decision tree visualization on the Model Evaluation page) |

> No other third-party packages are used. The optional AI Insight Copilot LLM layer uses only Python's standard library (`urllib`, `json`) — no extra dependency is required for it either.

---

## 5. Dataset

**"Factors Affecting Campus Placement"** (Kaggle, dataset author: benroshan)
Source: https://www.kaggle.com/datasets/benroshan/factors-affecting-campus-placement

- **215 rows, 15 original columns**: `sl_no, gender, ssc_p, ssc_b, hsc_p, hsc_b, hsc_s, degree_p, degree_t, workex, etest_p, specialisation, mba_p, status, salary`
- `sl_no` — row identifier, excluded from every model
- `gender` — sensitive attribute, excluded from every predictive model (shown only in descriptive context if at all)
- `status` — placement classification target (Placed / Not Placed)
- `salary` — salary regression target; present **only** for placed students (148 of 215); **never** imputed for non-placed students and **never** used as a placement-prediction feature

> **[VERIFY BEFORE SUBMISSION]**: this build's copy of the CSV was obtained from a public mirror of the dataset (verified against the documented Kaggle schema and row count) rather than downloaded directly via the Kaggle API, since this development environment has no direct Kaggle access. Re-downloading the original CSV from the Kaggle link above and replacing `data/campus_placement.csv` is recommended before final submission, if a direct Kaggle download is required by your internship program.

---

## 6. Analytics & ML Pipeline

```
DATA → DATA QUALITY → EDA → CLASSIFICATION → REGRESSION → CLUSTERING → AI INSIGHTS → WHAT-IF SIMULATION
```

| Stage | What it does |
|---|---|
| Data Quality | Validates schema, checks duplicates/missing values, prepares a model-ready dataframe |
| EDA | Explores placement, academic performance, salary, and relationships |
| Classification | Predicts placement with Logistic Regression + Decision Tree |
| Regression | Estimates salary (placed students only) with Linear Regression + Random Forest Regressor |
| Clustering | Groups students with K-Means, diagnosed via Elbow/Silhouette |
| AI Insights | Answers analytical questions from verified evidence, with a no-API fallback |
| What-If Simulation | Compares hypothetical profiles through the existing trained classification model |

---

## 7. Project Architecture

```
app.py                       # Streamlit UI — all pages, routing, caching
utils/
├── data_processing.py       # Load, clean, validate, feature/target split, preprocessing pipelines
├── analytics.py              # Deterministic KPI/group/correlation calculations + insight text
├── visualizations.py         # All Plotly chart builders (themed)
├── ml_models.py               # Classification, regression, K-Means segmentation, What-If helpers
└── ai_insights.py             # Evidence registry + deterministic fallback + optional LLM layer
```

> **Architecture choice:** a practical single-`app.py` + `utils/` layout (per the internship's own "keep it practical" guidance) rather than a deeper package structure, since the project stayed manageable at this size.

---

## 8. Application Pages

<div align="center">

`Overview` · `Data Explorer` · `Exploratory Data Analysis` · `Student Segmentation` · `AI Insight Copilot` · `What-If Simulator` · `Placement Prediction` · `Salary Prediction` · `Model Evaluation` · `Project Summary`

</div>

---

## 9. Key Findings

*All figures below are calculated live by the application from the current dataset.*

- **215 students**, **148 placed**, **67 not placed** — **68.8%** observed placement rate
- Average salary among placed students: **₹288,655** (median **₹265,000**), based on 148 salary records
- Work experience: **86.5%** observed placement rate (n=74) vs **59.6%** without (n=141) — a **+26.9 percentage point** observed difference
- Academic-feature correlations with salary were all weak (**|r| ≤ 0.18**)

> These are **observed dataset patterns**, not causal relationships.

---

## 10. Machine Learning Results

**Placement Classification** (80/20 stratified split, `random_state=42`, 172 train / 43 test):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| Logistic Regression | 0.860 | 0.929 | 0.867 | 0.897 | 0.938 |
| Decision Tree | 0.744 | 0.806 | 0.833 | 0.820 | 0.622 |

> Neither model is labeled "the best" — compare the metrics directly for your use case.

**Salary Regression** (148 salary records, 118 train / 30 test):

| Model | MAE | RMSE | R² |
|---|:---:|:---:|:---:|
| Linear Regression | ₹70,572 | ₹98,188 | **−0.140** |
| Random Forest Regressor | ₹74,254 | ₹104,408 | **−0.289** |

> A negative R² means the model did not outperform a simple mean-salary baseline on this test split — reported honestly rather than hidden, and consistent with the weak salary correlations found during EDA.

**Student Segmentation** (K=2–6 tested, K=2 recommended by silhouette score):

| K | 2 | 3 | 4 | 5 | 6 |
|---|:---:|:---:|:---:|:---:|:---:|
| Silhouette | **0.180** | 0.133 | 0.128 | 0.116 | 0.117 |

Cluster 0 — "Emerging Academic Profile" (110 students, 51.2%); Cluster 1 — "Higher Academic Profile" (105 students, 48.8%). Post-clustering descriptive outcomes (not used to build the clusters): Cluster 0 placement rate 50.0%, avg salary ₹266,200 (n=55); Cluster 1 placement rate 88.6%, avg salary ₹301,935 (n=93).

---

## 11. Responsible AI / Limitations

- Small dataset (215 records; only 148 with a recorded salary)
- Observational data — associations are not causal evidence
- Model metrics come from a single train/test split and may vary with a different one
- Salary regression is weak on this dataset (negative R²)
- Clustering separation is modest (silhouette 0.12–0.18)
- `gender` and `sl_no` are excluded from every predictive model
- The AI Insight Copilot's fallback engine is rule-based keyword matching, not full NLU — some phrasings fall through to a safe generic response rather than a specific answer
- No guarantee of placement or salary is made anywhere in this application
- A larger, richer dataset (company, role, location, industry) would be needed for any real-world deployment consideration

---

## 12. Installation

```bash
pip install -r requirements.txt
```

---

## 13. Running the Application

```bash
streamlit run app.py
```

> **[VERIFY BEFORE SUBMISSION]**: this application has not been deployed to Streamlit Community Cloud or any other host as part of this build — it has only been run and tested locally/in a sandboxed environment. No deployment URL exists unless you deploy it yourself.

---

## 14. Project Structure

```
LearnMate-Analytics-AI/
├── app.py
├── requirements.txt
├── README.md
├── PROJECT_REPORT.md
├── tests_stage2_smoke.py
├── data/
│   └── campus_placement.csv
└── utils/
    ├── data_processing.py
    ├── analytics.py
    ├── visualizations.py
    ├── ml_models.py
    └── ai_insights.py
```

---

## 15. Testing

<details>
<summary><strong>Compile checks</strong></summary>
<br>

All Python files compile cleanly (`python -m py_compile`)

</details>

<details>
<summary><strong>Smoke tests</strong> (<code>tests_stage2_smoke.py</code>)</summary>
<br>

Stubs Streamlit + Plotly and executes the real `app.py` logic for every page (10/10 pages), catching real runtime errors even though visual rendering can't be verified in a headless sandbox

</details>

<details>
<summary><strong>Classification verification</strong></summary>
<br>

Feature exclusion (`salary`/`gender`/`sl_no`/`status`), binary target encoding, valid prediction outputs, and metrics re-derived from the live model

</details>

<details>
<summary><strong>Regression verification</strong></summary>
<br>

148-record salary-only subset, feature exclusion, valid positive predictions, MAE/RMSE/R² re-derived live (including the negative R²)

</details>

<details>
<summary><strong>Segmentation verification</strong></summary>
<br>

Feature exclusion, no-NaN transformed matrix, K=2/3/4 label validity, profile counts summing to 215, Elbow/Silhouette computed for K=2–6, PCA projection row count

</details>

<details>
<summary><strong>AI Copilot verification</strong></summary>
<br>

Evidence registry schema validation, 9+ supported question categories, a dedicated hallucination-resistance test ("Which company offered the highest salary?") run through the actual `app.py` UI code path with no API key configured

</details>

<details>
<summary><strong>What-If Simulator verification</strong></summary>
<br>

22-item checklist including both-model comparison, single-variable and categorical experiments, change detection (including the zero-change edge case), and min/max input edge cases

</details>

<br>

> All of the above were actually executed against the real dataset during development, not assumed. This is not a claim of 100% code coverage — it is targeted verification of the leakage-prevention, correctness, and no-fabrication requirements that mattered most for this project.

---

## 16. Future Scope

- Larger, multi-year datasets
- Company-level, job-role, location, and industry information
- Internship-quality and richer student profile data
- Improved NLP for the AI Insight Copilot (beyond rule-based keyword matching)
- Model monitoring and periodic re-evaluation
- Fairness evaluation across sensitive attributes
- Deployment with authentication, if required by the use case

> None of the above have been implemented in this build.

---

## 17. Internship Context

Built for the **AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026 (BharatCares)**. Development was Claude-assisted (Python/Streamlit), honestly documented as such since direct desktop access to IBM BOB was not available during development. AI-assisted development is disclosed here rather than misrepresented.

---

## 18. Author / Student Information

| | |
|---|---|
| **Student** | [TO BE FILLED BY STUDENT] |
| **Institution** | [TO BE FILLED BY STUDENT] |
| **Year** | 2026 |

<div align="center">
<br>

—

<sub>LearnMate Analytics AI · Student Employability & Placement Intelligence Platform</sub>

</div>
