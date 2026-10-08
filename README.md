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

Live At: https://learnmate-analytics-ai-student-employability.streamlit.app/ 

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" alt="" style="max-width: 100%; display: inline-block;" data-target="animated-image.originalImage">


> A Streamlit-based student employability and placement intelligence platform built for the AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026 (BharatCares). It turns a real campus-placement dataset into exploratory analytics, classification, regression, clustering, grounded AI insights, and interactive scenario simulation — with every number calculated live from the data, not hard-coded.
>
> The platform is also **dataset-adaptive**: it accepts reasonably structured CSV/XLSX datasets from other domains, profiles the *active* dataset, and lets the AI Insight Copilot answer natural-language questions grounded in verified evidence from that dataset (via OpenRouter when an API key is configured, with a deterministic fallback otherwise). The campus-placement workflow remains as a specialized mode.

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

LearnMate Analytics AI combines Data Analytics, Exploratory Data Analysis, Classification, Regression, Clustering, explainable model evaluation, grounded AI insights, and scenario simulation into a single dataset-adaptive application.

The platform supports the original campus-placement benchmark workflow while also providing generic analytics capabilities for compatible student and campus-related CSV/XLSX datasets.

It is an **educational demonstration**, not a production employability predictor — it does not guarantee any individual's placement or salary outcome.

LearnMate Analytics AI is therefore both:

1. **A specialized student employability / placement intelligence platform** — when the active dataset matches the campus-placement schema (the bundled benchmark, or an upload containing the same columns), the full Placement / Salary / Segmentation / What-If / Model Evaluation workflow is available.
2. **An adaptive analytics platform for reasonably structured CSV/XLSX datasets** — for any other dataset, the application profiles the *active dataset* instead of assuming the benchmark schema, and provides adaptive analytics plus a dataset-grounded AI Insight Copilot.

The benchmark placement functionality is a **specialized capability**, not a requirement for using the platform.

```
                      CSV / XLSX  (or the bundled benchmark CSV)
                                  │
                        Active Dataset Loading
                                  │
                        Automatic Profiling
                                  │
                      Adaptive Analytics Layer
                                  │
                        AI Insight Copilot
                       ┌──────────┴───────────┐
          OpenRouter key configured      No key / request fails /
          → openai/gpt-4o-mini           unusable response
          → AI-generated, grounded       → deterministic fallback
            in active-dataset evidence     (clearly labeled)

   If the dataset matches the campus-placement schema, in addition:
        Specialized Placement Mode → Placement · Salary · Segmentation ·
                                     What-If · Model Evaluation · Prediction
```

> **Scope:** the adaptive capability is intended for *reasonably structured* tabular CSV/XLSX files (examples: student, sales, employee, finance, marketing, attendance or customer data). It is not claimed to handle every possible Excel workbook, and not every dataset supports every analysis — the application detects what the active dataset can support instead of forcing a model.

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
| **Dataset upload & profiling** | Upload a CSV, XLSX or TSV file from the sidebar; the application profiles the active dataset (row/column counts, detected column types, missing values, duplicates, identifier-like and high-cardinality columns) and detects whether it matches the campus-placement benchmark schema. A "Reset to Benchmark Dataset" button restores the bundled benchmark |
| **Overview dashboard** | Live KPI cards and academic/profile distributions (benchmark mode); for other datasets, a generic dataset summary with a notice that the full placement dashboard needs the benchmark schema |
| **Data Explorer** | Schema, data-quality report, cleaning summary, automated leakage-validation checks; works as a generic profile (preview, detected column types, missing values, duplicates) for any valid tabular dataset |
| **Exploratory Data Analysis** | Placement analysis, academic performance (with a chart-type selector), salary analysis, correlation/relationship analysis, deterministic FACT/INTERPRETATION insight cards, and a "From Data to Action" business-insights section. For non-benchmark datasets: adaptive EDA (Distributions / Relationships / Group Summaries tabs) built from whatever numeric and categorical columns were detected |
| **Student Segmentation** | K-Means clustering (K=2–6) with Elbow and Silhouette diagnostics, an interactive K selector, cluster profiles, a PCA 2D visualization, and clearly-labeled post-clustering descriptive outcomes. For non-benchmark datasets: generic K-Means where the user chooses the feature columns (identifier-like columns pre-excluded) |
| **AI Insight Copilot** | A dataset-grounded question-answering assistant with quick-question cards and free-text input. **Benchmark mode** answers from the verified benchmark evidence registry; **generic mode** profiles the active dataset, builds verified evidence from it, and sends natural-language questions through OpenRouter (`openai/gpt-4o-mini`) when an API key is configured. When no key is configured or a request fails, a clearly-labeled deterministic fallback answers instead. The answer source is shown in the UI |
| **What-If Simulator** | Compares a baseline student profile against a hypothetical one through the *existing* trained classification model, plus single-variable and categorical "model response" experiments (available when the benchmark classification model can be used; otherwise a notice points to Predictive Analysis) |
| **Predictive Analysis** | Target-driven classification/regression for any dataset: the user picks a target column and task, the application validates suitability before training, excludes identifier-like columns and warns about likely leakage columns |
| **Placement Prediction** | Logistic Regression and Decision Tree classifiers with a live prediction form (benchmark schema) |
| **Salary Prediction** | Linear Regression and Random Forest Regressor with a live prediction form (benchmark schema) |
| **Model Evaluation** | Full classification and regression metric suites, confusion matrices, ROC curves, coefficients/feature importances, and a depth-limited decision tree visualization (benchmark schema) |
| **Project Summary** | A consolidated, submission-ready overview of the whole project (with a generic variant for non-benchmark datasets) |
| **AI Data Mentor (chatbot)** | An interactive analytics-focused assistant panel with quick prompts and conversation support, integrated into the interface (not a separate route). It is switched ON/OFF with a sidebar toggle and starts closed; a launcher opens/closes the panel. In benchmark mode it reuses the grounded Copilot evidence (and the OpenRouter layer when a key is configured); in generic mode it uses the deterministic generic engine |
| **Government Services** | Provides integrated student and government-support resources through a dedicated application page |
---

## 4. Technology Stack

| Category | Technologies |
|---|---|
| **Application Framework** | Streamlit |
| **Data Handling** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn — Logistic Regression, Decision Tree, Linear Regression, Random Forest, K-Means, PCA, preprocessing, metrics |
| **Visualization** | Plotly (all interactive charts), Matplotlib (used only for the depth-limited decision tree visualization on the Model Evaluation page) |
| **Excel input** | openpyxl (reads `.xlsx` uploads through `pandas.read_excel`) |
| **LLM access (optional)** | OpenRouter chat-completions API (`openai/gpt-4o-mini`), called with Python's standard library (`urllib`, `json`) — no OpenAI/OpenRouter SDK is installed |

> `requirements.txt` lists `streamlit`, `pandas`, `numpy`, `scikit-learn`, `plotly`, `matplotlib` and `openpyxl` (unpinned). No other third-party packages are used. The OpenRouter LLM layer uses only Python's standard library (`urllib`, `json`) — no extra dependency is required for it either.

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

### Uploaded and other datasets (adaptive support)

The benchmark above is the project's **specialized campus-placement workflow**; it is not the only dataset the application accepts.

- **Accepted uploads:** `.csv`, `.xlsx` and `.tsv` (`.xls` is not supported — save as `.xlsx` or `.csv`). For `.xlsx`, `pandas.read_excel` reads the workbook's default (first) sheet.
- **Schema detection:** the specialized placement mode is enabled only if **every** benchmark column is present (matched after name normalization). The check is deliberately strict so benchmark-specific analytics never run silently on the wrong data.
- **Any other dataset:** the application profiles the active dataset and offers adaptive analytics (Data Explorer, adaptive EDA, generic Segmentation, target-driven Predictive Analysis, generic Project Summary, Government Services) and the generic AI Insight Copilot. Pages that need the benchmark schema (Placement Prediction, Salary Prediction, Model Evaluation, the full placement Overview, and the What-If Simulator when the benchmark classification model is not available) show an explanatory notice instead of failing.
- **Reset:** the sidebar's *Reset to Benchmark Dataset* button restores the bundled `data/campus_placement.csv`.

---

## 6. Analytics & ML Pipeline

```
DATA → DATA QUALITY → EDA → CLASSIFICATION → REGRESSION → CLUSTERING → AI INSIGHTS → WHAT-IF SIMULATION
```


| Stage | What it does |
|---|---|
| **Data Quality** | Loads, validates, profiles and prepares the active dataset |
| **EDA** | Generates descriptive statistics, relationships, correlations and analytical insights |
| **Classification** | Performs placement classification when a compatible target and feature structure are available |
| **Regression** | Performs numerical prediction when a compatible regression target is available |
| **Clustering** | Segments student records using unsupervised learning where appropriate |
| **AI Insights** | Provides evidence-grounded interpretations from the active dataset |
| **What-If Simulation** | Runs scenario experiments when the required trained classification model is available |

### Dataset-Adaptive Workflow

The application can load compatible CSV and XLSX datasets, profile their structure, detect available analytical capabilities, and provide generic analytical functionality where the required fields are available. The original campus-placement dataset remains a supported specialized benchmark mode.

**Active-dataset profiling** (`utils/data_processing.py` → `profile_dataset`) detects, for the dataset currently loaded: row and column counts, column names, numeric / categorical / datetime-like columns, columns with missing values, duplicate rows, identifier-like columns (name pattern *and* near-unique values) and high-cardinality (likely free-text) columns. Detection is heuristic and is labeled as such in the UI.

**Generic AI evidence** (`utils/generic_insights.py` → `build_generic_evidence`) builds a JSON-serialisable evidence object from the *active dataframe only*:

| Evidence section | Contents |
|---|---|
| Dataset | Source name, row/column counts, column names, duplicate rows |
| Column groups & schema | Numeric, categorical, boolean, datetime, high-cardinality text, identifier-like, empty and personal-data-like columns; per-column dtype, missing count/percentage, unique count |
| Numeric statistics | Count, mean, median, min, max, standard deviation and sum per numeric column |
| Distributions | Value counts/percentages for reasonable-cardinality categorical, boolean and low-cardinality numeric columns |
| Group summaries | Count, sum and mean of numeric metrics by categorical group (truncated, with a note, when there are many groups) |
| Correlations | Strongest Pearson correlations between numeric columns where there are enough rows |
| Time trends | Yearly / monthly / daily record counts and metric sums for detectable date columns |
| Data quality | Missing cells and rows, duplicates, completely empty or constant columns, numbers stored as text, possible outliers (1.5×IQR rule), category labels differing only by case/whitespace, small-sample warning |
| Concept presence | Which common concepts (e.g. company, salary, placement status, city, date) have a matching column *name*, and which do not |
| Analysis support | Whether descriptive statistics, correlation, group comparison and time analysis are possible, plus *heuristic* candidate classification/regression targets |

Safeguards in the profiler: personal-data-like columns (e.g. name, email, phone, address columns) and identifier-like columns are reported by **name only** — their values are not placed in the evidence and no raw rows are sent; high-cardinality text columns are not expanded; evidence size is capped; and a single odd column (mixed types, empty, unhashable cells, unusual names) cannot crash profiling. Original column names are preserved in user-facing answers.

---

## 7. Project Architecture

```
├── app.py                         # Streamlit UI, 12-route navigation/router, sidebar, CSS theme layer, session state and caching
│
├── .streamlit/
│   └── config.toml                # Streamlit theme settings (dark base, emerald primary colour)
│
├── data/
│   └── campus_placement.csv       # Bundled benchmark dataset
│
└── utils/
    ├── data_processing.py         # Load/validate/clean/profile datasets, benchmark-schema detection, semantic candidates, feature/target preparation
    ├── analytics.py               # Deterministic KPI, group, correlation and analytical calculations
    ├── visualizations.py          # Plotly/Matplotlib chart builders and visualization helpers
    ├── ml_models.py               # Benchmark classification/regression/K-Means/What-If helpers + generic target-driven models
    ├── ai_insights.py             # Benchmark evidence registry, deterministic fallback engine, shared OpenRouter HTTP helper
    ├── generic_insights.py        # Active-dataset profiling → verified generic evidence, grounded prompt, OpenRouter routing, fallback
    ├── chatbot_ui.py              # AI Data Mentor panel UI (launcher, open/close state, history, quick prompts)
    └── government_services.py     # Government and student-support services page
```

> **Architecture choice:** a practical single-`app.py` + `utils/` layout (per the internship's own "keep it practical" guidance) rather than a deeper package structure, since the project stayed manageable at this size.

### AI Insight Copilot architecture

```
Question (Copilot page)
        │
        ├── Benchmark schema active ─────────────► build_evidence() → validate_evidence()
        │                                          → generate_response()  (ai_insights.py)
        │                                               key + successful call → AI-generated
        │                                               otherwise → deterministic intent engine
        │
        └── Any other active dataset ───────────► build_generic_evidence()  (generic_insights.py)
                                                  → answer_generic_question()
                                                       key + successful call → AI-generated
                                                       otherwise → unsupported_answer() / deterministic fallback
```

Benchmark compatibility does **not** decide whether the Copilot can answer: generic datasets reach the same OpenRouter layer as the benchmark.

| Module | Responsibility |
|---|---|
| `utils/ai_insights.py` | Benchmark-oriented insight logic: evidence registry (`build_evidence`, `validate_evidence`), question classification and intent handlers, deterministic fallback (`generate_fallback_insight`), `QUICK_QUESTIONS`, the OpenRouter configuration constants (`OPENROUTER_BASE_URL`, `OPENROUTER_MODEL`), `get_configured_api_key`, and the **shared OpenRouter HTTP helper** `_openrouter_chat` (used by both the benchmark `call_llm` and the generic path) |
| `utils/generic_insights.py` | Active-dataset profiling and evidence (`build_generic_evidence`), the schema-driven check for information the dataset cannot supply (`missing_concepts_for_question`, `unsupported_answer`), the grounded `GENERIC_SYSTEM_PROMPT`, message construction (`build_generic_messages`), generic routing (`answer_generic_question`) and an evidence-based digest used by the fallback (`generic_digest_markdown`) |
| `app.py` | Copilot page wiring, the answer-source label, the keyword-based `generic_copilot_answer` deterministic engine used as the generic fallback and by the AI Data Mentor in generic mode |

**Model configuration**

| Setting | Value |
|---|---|
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` (requests go to `<base>/chat/completions`) |
| `OPENROUTER_MODEL` | `openai/gpt-4o-mini` |
| `OPENROUTER_API_KEY` | Read from Streamlit secrets or an environment variable; never hard-coded. See [Running the Application](#13-running-the-application) |

**Grounding and hallucination protection.** The generic Copilot receives only evidence computed from the active dataframe, plus a system prompt that instructs it to analyze only that dataset, never to invent columns, values, categories, companies, salaries, placement outcomes or relationships, to say plainly when the dataset cannot answer a question, to quote numbers from the evidence, to distinguish association from causation, and to treat the evidence block as data rather than instructions. In addition, a schema-driven check compares what a question asks about (e.g. company, salary, placement status, gender, age, city, state/region) with the column names of the active dataset; when no column can supply it, a `SCHEMA CHECK` line is added to the prompt, and the deterministic fallback returns an explicit "not available in this dataset" answer that lists the real columns. For example, with no company column, *"Which companies hired these students?"* is answered by saying company/hiring information is not available — not by naming companies. A dataset that *does* have, say, a City column is not told it lacks city information. This is a grounding/safety behavior, **not** a guarantee of perfect AI accuracy.

**Fallback behavior.** The deterministic fallback is used when no API key is configured, the OpenRouter request fails or times out, or the response is malformed/empty. It is never the primary AI engine: with a key configured and a working API, generic questions go to OpenRouter regardless of whether the dataset is the benchmark. The UI states the source — **"Answer source: AI-generated • Grounded in active dataset"** or **"Answer source: Fallback engine • Deterministic"** — and notes when an AI request was unavailable. API keys and request headers are never displayed.

**Privacy note.** When a key is configured, the question and the aggregate evidence (column names, statistics, group/category labels and counts) are sent to OpenRouter. Personal-data-like columns are withheld by a name/content heuristic that may not catch every case, so do not upload data you are not permitted to share with a third-party API.

---

## 8. Application Pages

<div align="center">

`Overview` · `Data Explorer` · `Exploratory Data Analysis` · `Student Segmentation` · `AI Insight Copilot` · `What-If Simulator` · `Predictive Analysis` · `Placement Prediction` · `Salary Prediction` · `Model Evaluation` · `Project Summary` · `Government Services`

> **Note:** The AI Data Mentor (chatbot) is integrated into the application interface rather than being a separate analytical page.
</div>

The primary navigation has **12 routes**, shown in the sidebar in three groups (one `st.sidebar.radio` feeds the single router; the groups and button appearance are CSS only):

| Group | Routes |
|---|---|
| **INTELLIGENCE** | Overview · Data Explorer · Exploratory Data Analysis · Student Segmentation |
| **AI & PREDICTION** | AI Insight Copilot · What-If Simulator · Predictive Analysis · Placement Prediction · Salary Prediction · Model Evaluation |
| **PROJECT** | Project Summary · Government Services |

### Interface (UI / UX)

- **Visual identity:** dark green / emerald theme (theme values in `.streamlit/config.toml`, extended by a CSS layer in `app.py`).
- **Sidebar:** LearnMate brand block, grouped navigation shown as full-width outlined buttons with a gradient active state, the AI Data Mentor ON/OFF toggle, the dataset upload and *Reset to Benchmark Dataset* controls, upload warnings/errors, and an Active Dataset card with mode captions. The sidebar uses Streamlit's native collapse/expand control, restyled with a 💻 icon.
- **Styling layer:** tabs, buttons, form-submit buttons, expanders, inputs and cards share one outlined/rounded style; the AI Data Mentor keeps its own styling. The CSS also hides Streamlit's default toolbar/menu/footer chrome where the deployed Streamlit version exposes matching elements, and includes small-screen rules.
- **Limits:** the styling depends on Streamlit's internal DOM, and `requirements.txt` is unpinned, so details can vary between Streamlit versions. Controls injected by the hosting platform (for example Streamlit Community Cloud's own toolbar) are outside the app and cannot be removed by application CSS.

---

## 9. Key Findings

*The figures below represent results from the original 215-row campus-placement benchmark dataset. When another compatible dataset is uploaded, the application's generic analytics are recalculated for the active dataset.*

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
- The AI Insight Copilot's deterministic fallback engine is rule-based keyword matching, not full NLU — some phrasings fall through to a safe generic response (or an evidence-based dataset digest in generic mode) rather than a specific answer
- AI-generated Copilot answers come from a language model (`openai/gpt-4o-mini` via OpenRouter). They are constrained by verified evidence, a grounded prompt and a schema check, but a language model can still misread or mis-summarize; treat them as assistance, not as verified facts
- Generic-mode evidence is size-capped (columns, categories, groups and periods are truncated, with a note), and target/analysis suggestions are name- and shape-based heuristics, not verified targets
- Adaptive analytics are intended for reasonably structured CSV/XLSX files; not every dataset supports every analysis, and no model is forced onto an unsuitable dataset
- When an API key is configured, aggregate evidence from the active dataset is sent to a third-party API (see the privacy note under *AI Insight Copilot architecture*)
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

> The application is deployed on Streamlit Community Cloud. The live application is available at the project URL shown at the top of this README.

**Using datasets.** On start-up the bundled benchmark (`data/campus_placement.csv`) is active and the specialized placement mode is on. Use the sidebar uploader (CSV, XLSX or TSV) to analyze another dataset; if it does not match the benchmark schema, the adaptive/generic mode is used. *Reset to Benchmark Dataset* restores the benchmark.

### Optional: OpenRouter configuration (AI-generated Copilot answers)

The application runs fully **without** any API key (deterministic fallback). To enable AI-generated answers, provide an OpenRouter key using **either** Streamlit secrets **or** an environment variable. Use placeholders below — never commit a real key.

`.streamlit/secrets.toml` (local) or *Settings → Secrets* (Streamlit Community Cloud):

```toml
OPENROUTER_API_KEY = "your-openrouter-api-key"
```

or an environment variable:

```bash
export OPENROUTER_API_KEY="your-openrouter-api-key"
```

`OPENROUTER_BASE_URL` (default `https://openrouter.ai/api/v1`) and `OPENROUTER_MODEL` (default `openai/gpt-4o-mini`) are read from environment variables of the same names and fall back to these defaults, so they normally need not be set.

> The repository includes `.streamlit/secrets.toml` and `env.txt` as **placeholder templates**. Replace the placeholder only in your local copy and keep real keys out of version control. The application does not load a `.env` file automatically (`python-dotenv` is not used), so export the variables in your shell if you use the `env.txt` template.

---

## 14. Project Structure

```
LearnMate-Analytics-AI/
│
├── app.py
├── requirements.txt
├── README.md
├── PROJECT_REPORT.md
├── LearnMate Analytics AI Blueprint Architecture.png
│
├── tests_stage2_smoke.py          # 24 page-runs: 12 routes x (benchmark + generic mode)
├── tests_generic_copilot.py       # 150 checks for the generic AI Insight Copilot
│
├── env.txt                        # placeholder template for environment variables (no real key)
│
├── .streamlit/
│   ├── config.toml                # theme settings
│   └── secrets.toml               # placeholder template (replace locally; do not commit a real key)
│
├── data/
│   └── campus_placement.csv       # bundled benchmark dataset
│
└── utils/
    ├── data_processing.py
    ├── analytics.py
    ├── visualizations.py
    ├── ml_models.py
    ├── ai_insights.py
    ├── generic_insights.py
    ├── chatbot_ui.py
    └── government_services.py
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

Stubs Streamlit + Plotly and executes the real `app.py` logic for every page. Current result: **24/24** — the 12 routes, each run in benchmark mode and in generic (non-benchmark) mode — catching real runtime errors even though visual rendering can't be verified in a headless sandbox

```bash
python tests_stage2_smoke.py
```

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
<summary><strong>Generic AI Insight Copilot tests</strong> (<code>tests_generic_copilot.py</code>)</summary>
<br>

Current result: **150/150 checks pass**.

```bash
python tests_generic_copilot.py
```

The tests stub Streamlit/Plotly like the smoke suite, run the real `app.py` Copilot page, and **mock the network**: `urllib.request.urlopen` is replaced by a fake OpenRouter that records each request. They use three datasets generated inside the test (seeded synthetic data written to and read back through the real loader): a 500-row × 7-column student workbook (`.xlsx`) with no company/salary/placement columns, a 400-row sales CSV (Product, Category, Region, Units Sold, Revenue, Date), and a small unusual/mixed-type CSV. Areas covered:

- **OpenRouter routing:** a non-benchmark dataset reaches OpenRouter when a key is present; request URL, model (`openai/gpt-4o-mini`), `Authorization` header, and the question/evidence in the prompt; the key never appears in the body or the UI
- **Active-dataset evidence:** evidence figures cross-checked against pandas (row/column counts, department/year/city counts, CGPA statistics, top product/region, total revenue, monthly trend); no benchmark column names in generic prompts; personal-data values (names, emails, IDs) withheld
- **Unusual data:** empty columns, numbers stored as text, mixed types, unicode/spaces/punctuation in column names profile without error
- **UI source label:** "AI-generated • Grounded in active dataset" on success; "Fallback engine • Deterministic" with no key, on network failure and on a malformed response
- **Unsupported questions / hallucination guards:** company, salary, placement-rate, gender and city questions on datasets lacking those columns return an explicit "not available" answer listing the real columns (fallback) and carry a `SCHEMA CHECK` in the AI prompt; supported questions are not wrongly refused
- **Benchmark mode unchanged:** no network call without a key; with a key the existing benchmark prompt path is used, not the generic prompt

> **Important:** the OpenRouter tests use **mocked responses**; the offline test environment had no network access, so **no live OpenRouter / GPT-4o-mini call was made** and the content of real model replies has not been validated. The tests verify what the application sends and how it labels and falls back — not what the model says. The real `student_data_500_rows.xlsx` was not used by the automated tests (the synthetic stand-in above was).

</details>

<details>
<summary><strong>What-If Simulator verification</strong></summary>
<br>

22-item checklist including both-model comparison, single-variable and categorical experiments, change detection (including the zero-change edge case), and min/max input edge cases

</details>

<br>

> All of the above were actually executed during development, not assumed (the benchmark checks against the real benchmark dataset; the generic Copilot tests against generated datasets with a mocked OpenRouter). This is not a claim of 100% code coverage — it is targeted verification of the leakage-prevention, correctness, and no-fabrication requirements that mattered most for this project.

---

## 16. Future Scope

- Larger, multi-year datasets
- Company-level, job-role, location, and industry information
- Internship-quality and richer student profile data
- Improved NLP for the AI Insight Copilot's deterministic fallback and the generic-mode AI Data Mentor (beyond rule-based keyword matching)
- Live validation of the OpenRouter integration against real model responses (the current automated tests use mocked calls)
- Pinned dependency versions in `requirements.txt` for reproducible deployments
- Model monitoring and periodic re-evaluation
- Fairness evaluation across sensitive attributes
- Deployment with authentication, if required by the use case

> None of the above have been implemented in this build. (The OpenRouter-backed Copilot described earlier *is* implemented; live validation of it is listed above as future work.)

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
