# LearnMate Analytics AI
### Student Employability & Placement Intelligence Platform

**AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026 — BharatCares**

Student: **[PLACEHOLDER — TO BE FILLED BY STUDENT]**
Institution: **[PLACEHOLDER — TO BE FILLED BY STUDENT]**
Year: 2026

---

## Table of Contents

1. Abstract
2. Introduction
3. Problem Statement
4. Objectives
5. Dataset Description
6. Data Preprocessing
7. Exploratory Data Analysis
8. Placement Prediction
9. Salary Prediction
10. Student Segmentation
11. AI Insight Copilot
12. What-If Simulator
13. Results & Discussion
14. Limitations
15. Future Scope
16. Conclusion
17. References / Data Source
18. Submission Checklist

---

## 1. Abstract

LearnMate Analytics AI is a student employability and placement intelligence platform built as a Data Analytics with AI internship project. Using a real campus-placement dataset of 215 students, the project applies a full analytics-to-machine-learning pipeline: data quality validation, exploratory data analysis, supervised classification of placement outcomes, regression-based salary estimation for placed students, unsupervised K-Means segmentation of student profiles, a grounded AI question-answering assistant, and an interactive "What-If" scenario simulator built on the trained classification model.

The project's placement classifiers (Logistic Regression and a Decision Tree) achieved accuracies of 0.860 and 0.744 respectively on a held-out 20% test split, with Logistic Regression showing a notably higher ROC-AUC (0.938 vs 0.622). Salary regression, by contrast, was found to be weak on this dataset — both Linear Regression and a Random Forest Regressor produced negative R² values on the test split, indicating that the available academic and employability features do not meaningfully explain salary variation in this small sample. K-Means segmentation identified two moderately-separated student profiles (silhouette score 0.180) differing mainly in academic average and work-experience share.

Throughout, the project deliberately excludes sensitive and leakage-prone fields (`gender`, `sl_no`, and, for placement prediction, `salary`/`status`) from all predictive models, and reports negative or weak results honestly rather than concealing them. The result is not a production-ready predictor but a transparent, verifiable demonstration of an end-to-end Data Analytics + Machine Learning workflow, with clearly documented limitations and a defined path for future improvement with richer data.

## 2. Introduction

Data analytics is the process of examining raw data to draw conclusions and support decision-making; when applied to student and career outcomes, it is often called employability analytics. Educational institutions increasingly hold rich records of student academic performance, work experience, and placement outcomes, but these records rarely translate directly into actionable insight without structured analysis.

This project's purpose is to demonstrate, on a real dataset, how such records can be turned into (a) descriptive insight — what patterns exist in the data; (b) predictive insight — whether placement and salary outcomes can be estimated from available features; (c) structural insight — whether meaningful student groupings emerge from the data; and (d) interactive insight — how a trained model responds to hypothetical changes in a student's profile. Each of these is implemented as a distinct, testable stage of the application rather than a single opaque model.

## 3. Problem Statement

Student placement datasets contain academic, educational, work-experience, and specialisation information, but raw tables do not directly answer the questions institutions and students actually have: What patterns are associated with placement outcomes? Which student characteristics differ between placed and non-placed students? Can placement outcomes be predicted from available features? What factors are associated with salary among placed students? Can students be grouped into meaningful data-driven profiles? This project addresses these questions directly, using real calculated statistics and trained models rather than assumptions.

## 4. Objectives

1. Load, validate, and clean the real campus placement dataset without fabricating or imputing values inappropriately (particularly salary for non-placed students).
2. Perform exploratory data analysis to surface verified, calculated patterns in placement, academic performance, and salary.
3. Build and evaluate supervised classification models (Logistic Regression, Decision Tree) to predict placement outcomes, with strict leakage prevention.
4. Build and evaluate supervised regression models (Linear Regression, Random Forest Regressor) to estimate salary for placed students, reporting performance honestly including negative results.
5. Apply unsupervised K-Means clustering to discover student profile segments, using diagnostic methods (Elbow, Silhouette) rather than an arbitrarily chosen K.
6. Build a grounded AI Insight Copilot that answers analytical questions using only verified, pre-calculated evidence, with a fully functional offline fallback.
7. Build an interactive What-If Simulator that reuses the trained classification model to compare hypothetical student profiles without retraining or manual preprocessing.
8. Document all limitations, exclusions, and responsible-AI considerations transparently, including model uncertainty and the observational (non-causal) nature of the data.

## 5. Dataset Description

**Source:** "Factors Affecting Campus Placement" (Kaggle, dataset author: benroshan) — https://www.kaggle.com/datasets/benroshan/factors-affecting-campus-placement

**Size:** 215 rows, 15 original columns.

**Fields:** `sl_no, gender, ssc_p, ssc_b, hsc_p, hsc_b, hsc_s, degree_p, degree_t, workex, etest_p, specialisation, mba_p, status, salary`

**Field treatment:**
- `sl_no` — row identifier; excluded from every model.
- `gender` — a sensitive demographic attribute; excluded from every predictive model to avoid using it as a basis for individual recommendations.
- `ssc_p, hsc_p, degree_p, etest_p, mba_p` — numeric academic/employability percentages; used as predictors.
- `ssc_b, hsc_b, hsc_s, degree_t, workex, specialisation` — categorical predictors.
- `status` — placement classification target (Placed / Not Placed); never used as an input feature.
- `salary` — salary regression target, present only for placed students (148 of 215); never imputed for non-placed students and never used as a placement-prediction input, since it is only observed *after* the placement outcome.

**[VERIFY BEFORE SUBMISSION]**: This build's copy of the dataset was obtained from a public mirror of the file (verified against the documented Kaggle schema, column names, and row count) rather than a direct Kaggle API download, since the development environment used to build this project had no direct Kaggle access. It is recommended to re-download the canonical CSV from the Kaggle link above before final submission if your program requires a verified direct-source copy.

## 6. Data Preprocessing

**Validation:** the raw CSV is checked against an expected-schema list; row count, column count, data types, and per-column missing-value counts are computed and displayed (Data Explorer page). An automated validation function additionally confirms, on every run, that `salary`, `status`, `sl_no`, and `gender` are absent from the classification feature set, and that `status`, `sl_no`, and `gender` are absent from the regression feature set — this is a hard, machine-checked guarantee, not a documentation claim.

**Missing values:** the dataset has exactly one column with missing values — `salary`, missing for all 67 non-placed students. This is treated as *structurally* missing (salary does not apply to a non-placed student), not as a data-quality defect, and is never imputed. No other column had missing values in the source file, so no other imputation was needed or performed.

**Duplicates:** checked via `DataFrame.duplicated()`; the dataset contains 0 duplicate rows.

**Categorical encoding:** `OneHotEncoder(handle_unknown="ignore")`, fit only on the training split for every supervised model, and on the full feature set for the unsupervised segmentation (which has no train/test split, since there is no target to leak).

**Scaling:** `StandardScaler` applied to numeric features for Logistic Regression, Linear Regression, K-Means, and (harmlessly, since tree splits are scale-invariant) the Decision Tree and Random Forest, for pipeline consistency.

**Train/test splitting:** classification uses an 80/20 stratified split (`random_state=42`) — 172 train / 43 test. Regression uses an 80/20 split (`random_state=42`) on the 148-record salary-only subset — 118 train / 30 test.

**Salary subset handling:** all salary-based analysis (EDA, regression, segmentation post-clustering outcomes) is restricted to the 148 placed students with a recorded salary; the 67 non-placed students are excluded from these calculations, never assigned a synthetic salary value.

## 7. Exploratory Data Analysis

Key calculated findings: 68.8% overall observed placement rate (148/215 placed); students with work experience showed an 86.5% observed placement rate (n=74) versus 59.6% without (n=141) — a 26.9 percentage point observed difference; average salary among placed students was ₹288,655 (median ₹265,000); all academic-feature correlations with salary were weak (|r| ≤ 0.18); academic-feature correlations among themselves ranged from weak to moderate (0.22–0.54). These are reported throughout the application as observed dataset patterns, explicitly not as causal relationships, since the data is observational.

## 8. Placement Prediction

Two classifiers were trained on an 11-feature set (5 numeric + 6 categorical, `gender`/`sl_no`/`status`/`salary` excluded): **Logistic Regression** (`max_iter=1000`, `random_state=42`) and a **Decision Tree** (`max_depth=4`, `random_state=42`), both via identical `ColumnTransformer` preprocessing (median-impute + scale for numeric, most-frequent-impute + one-hot for categorical) fit only on the training split.

| Metric | Logistic Regression | Decision Tree |
|---|---|---|
| Accuracy | 0.860 | 0.744 |
| Precision | 0.929 | 0.806 |
| Recall | 0.867 | 0.833 |
| F1 Score | 0.897 | 0.820 |
| ROC-AUC | 0.938 | 0.622 |

Confusion matrices, ROC curves, Logistic Regression coefficients (labeled as model-specific associations, not causal effects), and a depth-limited Decision Tree visualization are all available on the Model Evaluation page. **Neither model is labeled as "the best"** — the two show different trade-offs (Logistic Regression is notably better at ranking/separating the classes per ROC-AUC), and results come from a single 80/20 split on a small dataset, so they should be interpreted as an educational demonstration rather than a production benchmark.

## 9. Salary Prediction

**Linear Regression** and a **Random Forest Regressor** (`n_estimators=300, max_depth=6, random_state=42`) were trained on the same 11-feature set, restricted to the 148 placed students with a recorded salary (118 train / 30 test, `random_state=42`).

| Metric | Linear Regression | Random Forest Regressor |
|---|---|---|
| MAE | ₹70,572 | ₹74,254 |
| RMSE | ₹98,188 | ₹104,408 |
| R² | **−0.140** | **−0.289** |

**Negative R² interpretation:** a negative R² means that, on this test split, the model performed *worse* than a naive baseline that simply predicts the average salary for every student. This is not a sign of broken code — it is a genuine, informative finding: the available academic and employability variables do not provide enough signal to accurately explain salary variation in this small sample, which is directly consistent with the weak salary correlations (|r| ≤ 0.18) already found during EDA. This result is reported plainly rather than hidden or reframed as a success.

## 10. Student Segmentation

**Algorithm:** K-Means, `random_state=42`, features: `ssc_p, hsc_p, degree_p, etest_p, mba_p` (StandardScaler) and `ssc_b, hsc_b, hsc_s, degree_t, workex, specialisation` (OneHotEncoder). Excluded: `sl_no, gender, status, salary` — placement/salary outcomes are never used to *create* the clusters.

**K=2–6 diagnostics:**

| K | Inertia (Elbow) | Silhouette |
|---|---|---|
| 2 | 1337.7 | **0.180** |
| 3 | 1204.0 | 0.133 |
| 4 | 1113.4 | 0.128 |
| 5 | 1056.9 | 0.116 |
| 6 | 1002.6 | 0.117 |

K=2 was recommended (highest silhouette in range) — a diagnostic suggestion, not an absolute truth; the application lets the user pick any K from 2–6.

**Cluster profiles (K=2):** Cluster 0 — "Emerging Academic Profile" (110 students, 51.2%; academic avg 61.8%, work experience 22.7%); Cluster 1 — "Higher Academic Profile" (105 students, 48.8%; academic avg 72.2%, work experience 46.7%). Labels are algorithm-generated descriptions calculated from cluster statistics, not fixed rankings — cluster numbers are arbitrary identifiers.

**PCA:** used only to project the clustering result into 2 dimensions for visualization; clustering itself happens in the full preprocessed feature space, not the PCA-reduced space.

**Post-clustering descriptive outcomes** (calculated *after* clustering, not used to build it): Cluster 0 — 50.0% placement rate, ₹266,200 average salary (n=55 with recorded salary); Cluster 1 — 88.6% placement rate, ₹301,935 average salary (n=93). This is a real, notable descriptive pattern, but membership in a cluster is not claimed to *cause* either outcome.

**Limitations:** silhouette scores across the tested range are modest (0.12–0.18), meaning the clusters are real but not sharply separated — an expected result for observational student data at this sample size.

## 11. AI Insight Copilot

The Copilot follows an evidence-first architecture: `USER QUESTION → INTENT → VERIFIED PYTHON EVIDENCE → EXPLANATION`. A structured evidence registry is built entirely from numbers already calculated elsewhere in the project (the same `ClassificationResults`/`RegressionResults`/segmentation objects used on the Model Evaluation and Segmentation pages) — the AI layer never independently calculates or invents a statistic.

A rule-based, keyword-driven **deterministic fallback engine** works with zero API key and zero internet access, answering questions across 12 topic categories (dataset overview, academic-placement correlation, placement by work experience/degree/specialisation, salary overview and by specialisation, salary-model performance including the negative-R² explanation, classification performance, segmentation overview and differences, main findings, and next-steps) in a FACT / INSIGHT / POSSIBLE ACTION / LIMITATION format. Questions about information the dataset does not contain (e.g. company, location, interview details) receive an explicit "the current dataset does not contain X-level information" response rather than a fabricated answer — verified directly with a dedicated hallucination-resistance test.

An **optional external LLM layer** (Anthropic Messages API, called via Python's standard library only, no extra dependency) can be configured via an API key, but is never described as "connected" unless a real call actually succeeds at runtime; any failure (missing key, no network, timeout) falls back to the deterministic engine silently. In this development environment (no outbound network access), the optional layer has never actually reached an external server — this is disclosed honestly rather than claimed.

## 12. What-If Simulator

The simulator reuses the **existing** Stage 3 classification pipeline — no new model is trained, and no manual scaling or encoding is performed; the already-fitted `ColumnTransformer` + classifier pipeline's own `predict_proba()` produces every probability shown. A baseline profile (dataset median for numeric features, mode for categorical) is compared against a user-adjustable "What-If" profile across the same 11 features used in Stage 3. Users can compare both trained models side-by-side, see which inputs changed, and run single-variable (numeric sweep or categorical comparison) experiments that hold all other inputs fixed. Every output is explicitly labeled a "model-estimated probability," and the interpretation text explicitly states that a change in model output does not establish that changing the underlying attribute would causally produce a placement outcome in the real world.

## 13. Results & Discussion

The project's strongest, most reliable result is the placement classification: Logistic Regression in particular achieves a high ROC-AUC (0.938), indicating the available academic/employability features do meaningfully separate placed from non-placed students in this dataset. By contrast, the salary regression result is a genuine negative finding: neither tested model beats a naive mean-salary baseline, which — combined with the weak EDA salary correlations — suggests that predicting *how much* a placed student earns requires information this dataset does not contain (e.g. company, role, location, industry). The segmentation result sits in between: real, interpretable structure exists (two profiles differing in academic average and work-experience share, with a large observed placement-rate gap between them), but the modest silhouette scores mean this structure is not sharply defined. Together, these results paint a consistent, honest picture: this dataset can meaningfully inform *whether* a student is likely to be placed, but not *how much* they will earn if placed, nor does it support sharply-defined student typologies.

## 14. Limitations

- The dataset is small (215 rows total; only 148 with a recorded salary), so all model metrics should be read as an educational demonstration, not a statistically robust benchmark.
- The salary regression signal is weak (negative R² for both models tested).
- Clustering separation is modest (silhouette 0.12–0.18 across all tested K).
- All classification/regression metrics come from a single fixed train/test split and may vary with a different split or random seed.
- The data is observational, not experimental — none of the associations reported (work experience, academic scores, cluster membership) should be read as causal.
- The feature set is limited to academic percentages, board/stream/degree-type categories, work experience, and MBA specialisation; no company, job-role, location, or industry information is available.
- The AI Insight Copilot's fallback engine uses rule-based keyword matching rather than true natural-language understanding, so some valid but unusually-phrased questions fall through to a generic safe response instead of a specific one.

## 15. Future Scope

- Larger, multi-year placement datasets for more statistically robust modelling.
- Company-level, job-role, location, and industry information to improve salary prediction specifically.
- Richer student profile data (internship history, project portfolios, soft-skill assessments).
- Improved natural-language understanding for the AI Insight Copilot, beyond rule-based keyword matching.
- Ongoing model monitoring and periodic re-evaluation as new cohorts are added.
- A dedicated fairness evaluation across sensitive attributes before any real-world use.
- Deployment with authentication and access controls, if the tool were ever used beyond an educational context.

None of the above have been implemented in this build; they are documented here as legitimate next steps, not current features.

## 16. Conclusion

LearnMate Analytics AI demonstrates a complete, transparent Data Analytics + Machine Learning workflow on a real student placement dataset — from data validation and exploratory analysis, through supervised classification and regression, unsupervised segmentation, a grounded (non-hallucinating) AI assistant, and an interactive scenario simulator built on the trained model rather than a separate one. Its most useful finding — that available features can meaningfully predict placement but not salary — was arrived at honestly, including reporting a negative R² without concealment. This project is **not production-ready** and makes no claim of real-world deployment readiness; it is an educational demonstration whose value lies in showing how each analytical claim in the application traces back to a specific, reproducible calculation on the actual dataset, with limitations and exclusions (sensitive attributes, leakage-prone features) documented and enforced in code, not just prose.

## 17. References / Data Source

- Dataset: "Factors Affecting Campus Placement," Kaggle (author: benroshan). https://www.kaggle.com/datasets/benroshan/factors-affecting-campus-placement
- No other academic or external references were used in the construction of this project's analysis or models; all statistics and findings in this report are calculated directly from the dataset above by the project's own code.

## 18. Submission Checklist

- [x] app.py
- [x] requirements.txt
- [x] README.md
- [x] PROJECT_REPORT.md
- [x] campus_placement.csv
- [x] utils/data_processing.py
- [x] utils/analytics.py
- [x] utils/visualizations.py
- [x] utils/ml_models.py
- [x] utils/ai_insights.py
- [x] tests_stage2_smoke.py

- [ ] GitHub repository created — **[TO BE FILLED BY STUDENT]**
- [ ] Dataset source verified (direct Kaggle download recommended — see Section 5) — **[VERIFY BEFORE SUBMISSION]**
- [ ] Student name added — **[TO BE FILLED BY STUDENT]**
- [ ] Institution added — **[TO BE FILLED BY STUDENT]**
- [ ] Screenshots added to report, if required by your program — **[TO BE FILLED BY STUDENT]**
- [x] Final application tested (compile checks + full smoke suite across all 10 pages + targeted per-stage verification — see README.md Section 15 for details)
- [ ] README reviewed by student — **[VERIFY BEFORE SUBMISSION]**
- [ ] Report reviewed by student — **[VERIFY BEFORE SUBMISSION]**
