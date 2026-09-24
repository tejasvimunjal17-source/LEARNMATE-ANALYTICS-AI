"""
app.py
------
LearnMate Analytics AI - Student Employability & Placement Intelligence Platform

STAGE 2 build: Overview dashboard + Data Explorer (Stage 1, preserved) +
Exploratory Data Analysis. Predictive models, clustering, AI Copilot and
What-If simulator are NOT built yet (Stage 3+).

Run with:  streamlit run app.py
"""

import pandas as pd
import streamlit as st

from utils.data_processing import (
    DATA_DICTIONARY,
    NUMERIC_FEATURES,
    REGRESSION_TARGET,
    load_data,
    data_quality_report,
    clean_data,
    validate_pipeline,
)
from utils.analytics import (
    kpi_summary,
    placement_rate_by_group,
    salary_stats,
    salary_by_group,
    correlation_matrix,
    insight_group_comparison,
    insight_correlation,
    generate_business_insights,
)
from utils.visualizations import (
    PRIMARY,
    SECONDARY,
    bar_chart,
    pie_chart,
    histogram_chart,
    box_chart,
    scatter_chart,
    heatmap_chart,
    numeric_distribution_chart,
    confusion_matrix_chart,
    roc_curve_chart,
    coefficient_bar_chart,
    actual_vs_predicted_chart,
    residual_chart,
    plot_elbow_curve,
    plot_silhouette_scores,
    plot_cluster_distribution,
    plot_cluster_profiles,
    plot_cluster_workex,
    plot_cluster_pca,
    plot_scenario_probability_comparison,
    plot_single_feature_response,
    plot_categorical_scenario_comparison,
    CHART_TYPE_OPTIONS,
)
from utils.ml_models import (
    MODEL_REGISTRY,
    MODEL_EXPLANATIONS,
    METRIC_EXPLANATIONS,
    train_and_evaluate,
    comparison_table,
    predict_single,
    REGRESSION_MODEL_REGISTRY,
    REGRESSION_MODEL_EXPLANATIONS,
    REGRESSION_METRIC_EXPLANATIONS,
    SMALL_SAMPLE_WARNING,
    train_and_evaluate_salary,
    regression_comparison_table,
    predict_salary,
    CLUSTER_NUMERIC_FEATURES,
    CLUSTER_CATEGORICAL_FEATURES,
    CLUSTER_EXCLUDED,
    CLUSTER_K_RANGE,
    RANDOM_STATE,
    SEGMENTATION_SMALL_SAMPLE_WARNING,
    compute_segmentation_diagnostics,
    recommend_k,
    run_segmentation,
    post_clustering_outcomes,
    generate_segmentation_insights,
    WHATIF_FEATURE_LABELS,
    WHATIF_DATASET_SIZE_NOTE,
    WHATIF_LIMITATION_NOTE,
    baseline_profile,
    detect_profile_changes,
    compare_scenarios,
    simulate_single_numeric_feature,
    simulate_categorical_feature,
)
from utils.ai_insights import (
    build_evidence,
    validate_evidence,
    generate_response,
    answer_intent,
    render_response_markdown,
    get_configured_api_key,
    QUICK_QUESTIONS,
)

st.set_page_config(
    page_title="LearnMate Analytics AI",
    page_icon="\U0001F393",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Premium visual theme (CSS)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp { background-color: #F8FAFC; }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #F3F0FF 100%);
        border-right: 1px solid #ECE9FE;
    }
    h1, h2, h3 { color: #111827; font-weight: 700; }
    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #ECE9FE;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        box-shadow: 0 1px 2px rgba(17,24,39,0.04);
    }
    [data-testid="stMetricLabel"] { color: #6B7280; font-weight: 500; }
    [data-testid="stMetricValue"] { color: #111827; }
    .lm-badge {
        display: inline-block; padding: 0.15rem 0.6rem; border-radius: 999px;
        background: #EFEBFF; color: #7C5CFF; font-size: 0.75rem; font-weight: 600;
        margin-bottom: 0.4rem;
    }
    .lm-section-title { color: #111827; font-weight: 700; margin-top: 0.25rem; }
    .lm-card {
        background: white; border: 1px solid #ECE9FE; border-radius: 14px;
        padding: 1rem 1.2rem; margin-bottom: 0.8rem;
    }
    .lm-insight {
        border-left: 4px solid #7C5CFF; background: #FAFAFF; border-radius: 8px;
        padding: 0.75rem 1rem; margin: 0.5rem 0 1.2rem 0;
    }
    .lm-insight b { color: #111827; }
    .lm-action {
        border-left: 4px solid #22D3B0; background: #F2FEFB; border-radius: 8px;
        padding: 0.9rem 1.1rem; margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.markdown("### \U0001F393 LearnMate Analytics AI")
st.sidebar.caption("Student Employability & Placement Intelligence Platform")
st.sidebar.caption("ANALYTICS")
page = st.sidebar.radio(
    "Navigate",
    [
        "Overview", "Data Explorer", "Exploratory Data Analysis",
        "\U0001F9E9 Student Segmentation",
        "\U0001F9E0 AI Insight Copilot",
        "\U0001F3AF What-If Simulator",
        "\U0001F916 Placement Prediction", "\U0001F916 Salary Prediction",
        "\U0001F916 Model Evaluation",
        "\U0001F4D8 Project Summary",
    ],
    label_visibility="collapsed",
)
st.sidebar.divider()

# ---------------------------------------------------------------------------
# Load + clean data once (shared by every page)
# ---------------------------------------------------------------------------
try:
    raw_df = load_data()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

quality = data_quality_report(raw_df)
cleaned_df, cleaning_log = clean_data(raw_df)
validation = validate_pipeline(raw_df)


@st.cache_resource
def get_classification_results(_df):
    """Trains both models once per session. Leading underscore on the
    parameter tells Streamlit not to try to hash the dataframe."""
    return train_and_evaluate(_df)


@st.cache_resource
def get_regression_results(_df):
    return train_and_evaluate_salary(_df)


@st.cache_resource
def get_segmentation_diagnostics(_df):
    """Fits preprocessing once and scores K=2-6 for the Elbow/Silhouette
    diagnostics. Expensive-ish (6 KMeans fits), so cached per dataframe."""
    return compute_segmentation_diagnostics(_df)


@st.cache_resource
def get_segmentation_result(_df, k: int, _diagnostics):
    """Cached PER K, so picking a different K on the slider always
    recalculates (cache miss) while re-selecting a previous K reuses it."""
    return run_segmentation(_df, k=k, diagnostics=_diagnostics)


@st.cache_resource
def get_copilot_evidence(_df, _cls_results, _reg_results, _seg_diagnostics):
    """Builds the AI Insight Copilot's evidence registry once per session
    from the SAME trained results used elsewhere in the app."""
    return build_evidence(_df, _cls_results, _reg_results, _seg_diagnostics)


CATEGORY_OPTIONS = {
    col: sorted(cleaned_df[col].dropna().unique().tolist())
    for col in ["ssc_b", "hsc_b", "hsc_s", "degree_t", "workex", "specialisation"]
}
NUMERIC_RANGES = {
    col: (float(cleaned_df[col].min()), float(cleaned_df[col].max()), float(cleaned_df[col].median()))
    for col in ["ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p"]
}


def insight_card(fact: str, interpretation: str) -> None:
    st.markdown(
        f"""<div class="lm-insight">
        <b>FACT:</b> {fact}<br/>
        <b>INTERPRETATION:</b> {interpretation}
        </div>""",
        unsafe_allow_html=True,
    )


def business_insight_card(bi) -> None:
    st.markdown(
        f"""<div class="lm-action">
        <b>FACT:</b> {bi.fact}<br/><br/>
        <b>INSIGHT:</b> {bi.insight}<br/><br/>
        <b>OPPORTUNITY:</b> {bi.opportunity}<br/><br/>
        <b>ACTION:</b> {bi.action}
        </div>""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# PAGE: OVERVIEW
# ---------------------------------------------------------------------------

def render_overview() -> None:
    st.markdown('<span class="lm-badge">LIVE DATA - CALCULATED FROM CSV</span>', unsafe_allow_html=True)
    st.title("Overview")
    st.caption("From Student Data to Employability Intelligence")

    kpi = kpi_summary(cleaned_df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students Analysed", f"{kpi.total_students}")
    c2.metric("Placement Rate", f"{kpi.placement_rate_pct}%")
    c3.metric("Placed Students", f"{kpi.placed_students}")
    avg_salary_display = f"INR {kpi.avg_salary_placed:,.0f}" if kpi.avg_salary_placed else "N/A"
    c4.metric("Avg. Salary (Placed)", avg_salary_display,
              help=f"Based on {kpi.salary_sample_size} placed students with a recorded salary.")

    st.divider()

    st.markdown('<h3 class="lm-section-title">Placement Overview</h3>', unsafe_allow_html=True)
    p1, p2 = st.columns([1, 1])
    with p1:
        status_counts = cleaned_df["status"].value_counts().reset_index()
        status_counts.columns = ["status", "count"]
        st.plotly_chart(
            pie_chart(status_counts, names="status", values="count", title="Placed vs Not Placed"),
            use_container_width=True,
        )
    with p2:
        st.markdown("##### Key Data Snapshot")
        st.markdown(
            f"""
            - **{kpi.placed_students}** of **{kpi.total_students}** students were placed
              (**{kpi.placement_rate_pct}%** placement rate).
            - **{kpi.not_placed_students}** students were not placed.
            - Average salary among placed students: **INR {kpi.avg_salary_placed:,.0f}**
              (median: INR {kpi.median_salary_placed:,.0f}), based on
              **{kpi.salary_sample_size}** salary records.
            - Salary is only defined for placed students; it is never estimated
              for the {kpi.not_placed_students} students without a placement.
            """
        )

    st.divider()

    st.markdown('<h3 class="lm-section-title">Academic Overview</h3>', unsafe_allow_html=True)
    academic_cols = st.columns(len(NUMERIC_FEATURES))
    academic_titles = {
        "ssc_p": "SSC %", "hsc_p": "HSC %", "degree_p": "Degree %",
        "etest_p": "E-test %", "mba_p": "MBA %",
    }
    for col_widget, feat in zip(academic_cols, NUMERIC_FEATURES):
        with col_widget:
            st.plotly_chart(
                histogram_chart(cleaned_df, feat, academic_titles.get(feat, feat), nbins=15),
                use_container_width=True,
            )

    st.divider()

    st.markdown('<h3 class="lm-section-title">Student Profile</h3>', unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    with s1:
        we = cleaned_df["workex"].value_counts().reset_index()
        we.columns = ["workex", "count"]
        st.plotly_chart(bar_chart(we, x="workex", y="count", title="Work Experience"),
                         use_container_width=True)
    with s2:
        dt = cleaned_df["degree_t"].value_counts().reset_index()
        dt.columns = ["degree_t", "count"]
        st.plotly_chart(bar_chart(dt, x="degree_t", y="count", title="Degree Type"),
                         use_container_width=True)
    with s3:
        sp = cleaned_df["specialisation"].value_counts().reset_index()
        sp.columns = ["specialisation", "count"]
        st.plotly_chart(bar_chart(sp, x="specialisation", y="count", title="Specialisation"),
                         use_container_width=True)


# ---------------------------------------------------------------------------
# PAGE: DATA EXPLORER (Stage 1, preserved, presentation lightly improved)
# ---------------------------------------------------------------------------

def render_data_explorer() -> None:
    st.title("\U0001F4CA Data Explorer")
    st.caption("Every number below is calculated live from data/campus_placement.csv.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", quality.n_rows)
    c2.metric("Columns", quality.n_cols)
    c3.metric("Duplicate rows", quality.duplicate_rows)
    c4.metric("Schema matches expected", "Yes" if quality.schema_matches_expected else "No")

    st.divider()
    st.subheader("Dataset preview")
    st.dataframe(raw_df.head(15), use_container_width=True)

    st.divider()
    st.subheader("Columns & data types")
    schema_df = pd.DataFrame({
        "column": quality.columns,
        "dtype": [quality.dtypes[c] for c in quality.columns],
        "missing_values": [quality.missing_counts[c] for c in quality.columns],
        "description": [DATA_DICTIONARY.get(c, "") for c in quality.columns],
    })
    st.dataframe(schema_df, use_container_width=True, hide_index=True)
    st.markdown(f"**Numeric columns:** {', '.join(quality.numeric_columns) or 'none'}")
    st.markdown(f"**Categorical columns:** {', '.join(quality.categorical_columns) or 'none'}")

    st.divider()
    st.subheader("Data Quality")
    total_missing = sum(quality.missing_counts.values())
    st.write(f"Total missing values across all columns: **{total_missing}**")
    missing_df = pd.DataFrame(
        [(c, n) for c, n in quality.missing_counts.items() if n > 0],
        columns=["column", "missing_count"],
    )
    if not missing_df.empty:
        st.dataframe(missing_df, use_container_width=True, hide_index=True)
    else:
        st.info("No missing values in the raw file.")

    st.subheader("Cleaning summary")
    checklist = [
        (True, "Dataset loaded"),
        (cleaning_log.duplicates_checked, "Duplicate check completed"
            + (f" ({cleaning_log.duplicates_removed} removed)" if cleaning_log.duplicates_removed else " (0 found)")),
        (cleaning_log.missing_value_analysis_done, "Missing-value analysis completed"),
        (cleaning_log.dtypes_validated, "Data types validated"),
        (cleaning_log.model_ready, "Model-ready dataset prepared"),
    ]
    for done, label in checklist:
        st.write(("\u2705 " if done else "\u274C ") + label)
    for note in cleaning_log.notes:
        st.caption(f"Note: {note}")

    st.divider()
    st.subheader("Pipeline validation (leakage checks)")
    if validation.passed:
        st.success("All checks passed - no data leakage detected.")
    else:
        st.error("Some checks failed - see details below.")
    for name, ok in validation.checks.items():
        st.write(("\u2705 " if ok else "\u274C ") + name.replace("_", " "))


# ---------------------------------------------------------------------------
# PAGE: EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------------------------

def render_eda() -> None:
    st.title("\U0001F50E Exploratory Data Analysis")

    tab_a, tab_b, tab_c, tab_d = st.tabs(
        ["Placement Analysis", "Academic Performance", "Salary Analysis", "Relationships"]
    )

    # A. Placement Analysis --------------------------------------------
    with tab_a:
        st.markdown("##### Placement Status Distribution")
        status_counts = cleaned_df["status"].value_counts().reset_index()
        status_counts.columns = ["status", "count"]
        st.plotly_chart(
            bar_chart(status_counts, x="status", y="count", title="Placement Status Distribution"),
            use_container_width=True,
        )

        for col, label in [("workex", "work experience"), ("degree_t", "degree type"),
                            ("specialisation", "specialisation")]:
            st.markdown(f"##### Placement by {label.title()}")
            rates = placement_rate_by_group(cleaned_df, col)
            st.plotly_chart(
                bar_chart(rates, x=col, y="placement_rate_pct",
                          title=f"Placement Rate (%) by {label.title()}"),
                use_container_width=True,
            )
            ins = insight_group_comparison(cleaned_df, col, label)
            insight_card(ins.fact, ins.interpretation)

    # B. Academic Performance --------------------------------------------
    with tab_b:
        st.caption("Choose a metric and a chart type. Box plots also split by placement status.")
        metric_labels = {
            "ssc_p": "SSC %", "hsc_p": "HSC %", "degree_p": "Degree %",
            "etest_p": "E-test %", "mba_p": "MBA %",
        }
        col_sel, type_sel = st.columns([2, 1])
        with col_sel:
            metric = st.selectbox("Metric", list(metric_labels.keys()),
                                   format_func=lambda k: metric_labels[k], key="eda_academic_metric")
        with type_sel:
            chart_type = st.radio("Chart type", CHART_TYPE_OPTIONS, horizontal=True, key="eda_academic_chart")

        color_by = "status" if chart_type == "Box" else None
        fig = numeric_distribution_chart(
            cleaned_df, column=metric, title=f"{metric_labels[metric]} Distribution",
            chart_type=chart_type, color=color_by,
        )
        st.plotly_chart(fig, use_container_width=True)

        desc = cleaned_df[metric].describe()
        st.caption(
            f"n={int(desc['count'])}, mean={desc['mean']:.1f}, median={cleaned_df[metric].median():.1f}, "
            f"min={desc['min']:.1f}, max={desc['max']:.1f}"
        )

    # C. Salary Analysis (placed rows only) -------------------------------
    with tab_c:
        stats = salary_stats(cleaned_df)
        if stats.get("count", 0) == 0:
            st.warning("No placed students with a recorded salary - salary analysis unavailable.")
        else:
            st.caption(
                f"Based on {stats['count']} placed students with a recorded salary. "
                f"Non-placed students are excluded, not imputed."
            )
            m1, m2, m3 = st.columns(3)
            m1.metric("Mean salary", f"INR {stats['mean']:,.0f}")
            m2.metric("Median salary", f"INR {stats['median']:,.0f}")
            m3.metric("Salary range", f"INR {stats['min']:,.0f} - {stats['max']:,.0f}")

            placed_df = cleaned_df[cleaned_df[REGRESSION_TARGET].notna()]

            st.markdown("##### Salary Distribution")
            st.plotly_chart(
                histogram_chart(placed_df, REGRESSION_TARGET, "Salary Distribution (Placed Students)", nbins=25),
                use_container_width=True,
            )

            st.markdown("##### Salary by Degree Type")
            st.plotly_chart(
                box_chart(placed_df, x="degree_t", y=REGRESSION_TARGET, title="Salary by Degree Type"),
                use_container_width=True,
            )

            st.markdown("##### Salary by Specialisation")
            st.plotly_chart(
                box_chart(placed_df, x="specialisation", y=REGRESSION_TARGET, title="Salary by Specialisation"),
                use_container_width=True,
            )

            sc1, sc2 = st.columns(2)
            with sc1:
                st.markdown("##### Salary vs Degree %")
                st.plotly_chart(
                    scatter_chart(placed_df, x="degree_p", y=REGRESSION_TARGET, title="Salary vs Degree %"),
                    use_container_width=True,
                )
                ins = insight_correlation(cleaned_df, "degree_p", REGRESSION_TARGET, "Degree %", "Salary")
                insight_card(ins.fact, ins.interpretation)
            with sc2:
                st.markdown("##### Salary vs E-test %")
                st.plotly_chart(
                    scatter_chart(placed_df, x="etest_p", y=REGRESSION_TARGET, title="Salary vs E-test %"),
                    use_container_width=True,
                )
                ins = insight_correlation(cleaned_df, "etest_p", REGRESSION_TARGET, "E-test %", "Salary")
                insight_card(ins.fact, ins.interpretation)

    # D. Relationship Analysis --------------------------------------------
    with tab_d:
        st.markdown("##### Correlation Heatmap - Academic Features (all students)")
        corr_academic = correlation_matrix(cleaned_df, NUMERIC_FEATURES)
        st.plotly_chart(heatmap_chart(corr_academic, "Academic Feature Correlations"), use_container_width=True)

        st.markdown("##### Correlation Heatmap - Academic Features + Salary (placed students only)")
        corr_salary = correlation_matrix(cleaned_df, NUMERIC_FEATURES + [REGRESSION_TARGET])
        st.plotly_chart(heatmap_chart(corr_salary, "Academic Features vs Salary"), use_container_width=True)
        st.caption(
            "This second heatmap automatically uses only the rows where salary is present "
            "(placed students), since rows with a missing value in any selected column are dropped."
        )

        st.markdown("##### Scatter: Degree % vs MBA %, colored by placement status")
        st.plotly_chart(
            scatter_chart(cleaned_df, x="degree_p", y="mba_p", color="status",
                          title="Degree % vs MBA % by Placement Status"),
            use_container_width=True,
        )

        st.markdown("##### Box: SSC % by Placement Status")
        st.plotly_chart(
            box_chart(cleaned_df, x="status", y="ssc_p", title="SSC % by Placement Status"),
            use_container_width=True,
        )

    st.divider()
    st.markdown('<h3 class="lm-section-title">From Data to Action</h3>', unsafe_allow_html=True)
    st.caption("FACT -> INSIGHT -> OPPORTUNITY -> ACTION. All facts are calculated from the dataset.")
    for bi in generate_business_insights(cleaned_df):
        business_insight_card(bi)


# ---------------------------------------------------------------------------
# PAGE: MODEL EVALUATION
# ---------------------------------------------------------------------------

def render_model_evaluation() -> None:
    st.title("\U0001F916 Model Evaluation")
    st.info(
        f"Dataset size: {quality.n_rows} records. Model metrics should be interpreted as an "
        f"educational demonstration and may vary with different train/test splits. "
        f"This is not a production-ready model."
    )

    results = get_classification_results(cleaned_df)
    st.caption(
        f"80/20 stratified train-test split -> {results.n_train} training rows, "
        f"{results.n_test} test rows. Fixed random_state={results.random_state} for reproducibility."
    )

    st.markdown('<h3 class="lm-section-title">Model performance comparison</h3>', unsafe_allow_html=True)
    st.dataframe(comparison_table(results), use_container_width=True, hide_index=True)

    with st.expander("What do these metrics mean?"):
        for metric, explanation in METRIC_EXPLANATIONS.items():
            st.markdown(f"**{metric}:** {explanation}")

    st.caption(
        "No single model is labeled 'best' here - the table above lets you compare the actual "
        "trade-offs (e.g. one model may have higher precision, another higher recall)."
    )

    st.divider()

    for model_name, ev in results.evaluations.items():
        st.markdown(f'<h3 class="lm-section-title">{model_name}</h3>', unsafe_allow_html=True)
        with st.expander(f"What is {model_name}?"):
            st.write(MODEL_EXPLANATIONS[model_name])

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                confusion_matrix_chart(ev.confusion, ["Not Placed", "Placed"],
                                       f"Confusion Matrix - {model_name}"),
                use_container_width=True,
            )
        with c2:
            if ev.roc_fpr is not None:
                st.plotly_chart(
                    roc_curve_chart(ev.roc_fpr, ev.roc_tpr, ev.roc_auc, f"ROC Curve - {model_name}"),
                    use_container_width=True,
                )
            else:
                st.info("ROC curve unavailable (test split did not contain both classes).")

        if ev.coefficients:
            st.plotly_chart(
                coefficient_bar_chart(ev.coefficients, "Logistic Regression Coefficients (top 10 by magnitude)"),
                use_container_width=True,
            )
            st.caption(
                "These are coefficients within this fitted Logistic Regression model, not proof "
                "of a causal effect - the dataset is observational, not experimental."
            )
        if ev.feature_importances:
            st.plotly_chart(
                coefficient_bar_chart(ev.feature_importances, "Decision Tree Feature Importance (top 10)"),
                use_container_width=True,
            )
            st.caption(
                "Feature importance reflects how much this specific Decision Tree relied on each "
                "feature to split the data - not a causal claim."
            )

            if model_name == "Decision Tree":
                with st.expander("View a simplified Decision Tree (depth-limited for readability)"):
                    try:
                        import matplotlib.pyplot as plt
                        from sklearn.tree import plot_tree, DecisionTreeClassifier as _DTC
                        from utils.data_processing import get_classification_data as _gcd

                        X, y_raw = _gcd(cleaned_df)
                        y = (y_raw.str.lower() == "placed").astype(int)
                        preprocess = results.pipelines[model_name].named_steps["preprocess"]
                        X_enc = preprocess.transform(X)
                        feat_names = list(preprocess.get_feature_names_out())
                        display_tree = _DTC(max_depth=3, random_state=results.random_state)
                        display_tree.fit(X_enc, y)

                        fig, ax = plt.subplots(figsize=(16, 8))
                        plot_tree(display_tree, feature_names=feat_names,
                                  class_names=["Not Placed", "Placed"], filled=True,
                                  fontsize=8, ax=ax)
                        st.pyplot(fig)
                        st.caption(
                            "Display-only tree limited to depth 3 for readability. The actual "
                            "evaluated model above (max_depth=4) is what produced the metrics."
                        )
                    except Exception as e:
                        st.warning(f"Tree visualization unavailable: {e}")

        st.divider()

    st.markdown('<h3 class="lm-section-title">Fairness note</h3>', unsafe_allow_html=True)
    st.caption(
        "Gender was excluded from predictive modelling to avoid using a sensitive demographic "
        "attribute as a model input."
    )


# ---------------------------------------------------------------------------
# PAGE: PLACEMENT PREDICTION
# ---------------------------------------------------------------------------

def render_placement_prediction() -> None:
    st.title("\U0001F916 Placement Prediction")
    st.caption(
        "Enter a student's profile below. Predictions come from models trained on the "
        f"{quality.n_rows}-record dataset and are a model estimate, not a guarantee of an "
        "actual placement outcome."
    )

    results = get_classification_results(cleaned_df)

    with st.form("placement_prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            ssc_p = st.slider("SSC %", 0.0, 100.0, NUMERIC_RANGES["ssc_p"][2])
            hsc_p = st.slider("HSC %", 0.0, 100.0, NUMERIC_RANGES["hsc_p"][2])
            degree_p = st.slider("Degree %", 0.0, 100.0, NUMERIC_RANGES["degree_p"][2])
            etest_p = st.slider("E-test %", 0.0, 100.0, NUMERIC_RANGES["etest_p"][2])
            mba_p = st.slider("MBA %", 0.0, 100.0, NUMERIC_RANGES["mba_p"][2])
        with col2:
            ssc_b = st.selectbox("SSC Board", CATEGORY_OPTIONS["ssc_b"])
            hsc_b = st.selectbox("HSC Board", CATEGORY_OPTIONS["hsc_b"])
            hsc_s = st.selectbox("HSC Specialisation (stream)", CATEGORY_OPTIONS["hsc_s"])
            degree_t = st.selectbox("Degree Type", CATEGORY_OPTIONS["degree_t"])
            workex = st.selectbox("Work Experience", CATEGORY_OPTIONS["workex"])
            specialisation = st.selectbox("MBA Specialisation", CATEGORY_OPTIONS["specialisation"])

        model_name = st.selectbox("Model", list(MODEL_REGISTRY.keys()))
        submitted = st.form_submit_button("Predict placement")

    if submitted:
        input_values = {
            "ssc_p": ssc_p, "hsc_p": hsc_p, "degree_p": degree_p,
            "etest_p": etest_p, "mba_p": mba_p,
            "ssc_b": ssc_b, "hsc_b": hsc_b, "hsc_s": hsc_s,
            "degree_t": degree_t, "workex": workex, "specialisation": specialisation,
        }
        result = predict_single(results, model_name, input_values)

        st.divider()
        st.markdown('<h3 class="lm-section-title">Prediction result</h3>', unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3)
        r1.metric("Predicted outcome", result["predicted_status"])
        r2.metric("Estimated probability of placement",
                  f"{result['predicted_probability_placed'] * 100:.1f}%")
        r3.metric("Model used", result["model_name"])
        st.caption("This is a model-based prediction and not a guarantee of an actual placement outcome.")


# ---------------------------------------------------------------------------
# PAGE: SALARY PREDICTION
# ---------------------------------------------------------------------------

def render_salary_prediction() -> None:
    st.title("\U0001F916 Salary Prediction")
    st.warning(SMALL_SAMPLE_WARNING)

    reg_results = get_regression_results(cleaned_df)
    st.caption(
        f"Trained on {reg_results.n_salary_records} placed-student records with an observed "
        f"salary -> {reg_results.n_train} training rows, {reg_results.n_test} test rows "
        f"(80/20 split, random_state={reg_results.random_state})."
    )

    st.markdown('<h3 class="lm-section-title">Model performance comparison</h3>', unsafe_allow_html=True)
    st.dataframe(regression_comparison_table(reg_results), use_container_width=True, hide_index=True)

    with st.expander("What do these metrics mean?"):
        for metric, explanation in REGRESSION_METRIC_EXPLANATIONS.items():
            st.markdown(f"**{metric}:** {explanation}")

    st.divider()

    for model_name, ev in reg_results.evaluations.items():
        st.markdown(f'<h3 class="lm-section-title">{model_name}</h3>', unsafe_allow_html=True)
        with st.expander(f"What is {model_name}?"):
            st.write(REGRESSION_MODEL_EXPLANATIONS[model_name])

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                actual_vs_predicted_chart(ev.y_test, ev.y_pred, f"Actual vs Predicted Salary - {model_name}"),
                use_container_width=True,
            )
        with c2:
            st.plotly_chart(
                residual_chart(ev.y_pred, ev.residuals, f"Residuals - {model_name}"),
                use_container_width=True,
            )

        if ev.coefficients:
            st.plotly_chart(
                coefficient_bar_chart(ev.coefficients, "Linear Regression Coefficients (top 10 by magnitude, INR per unit)"),
                use_container_width=True,
            )
            st.caption(
                "Within this fitted model, each bar shows that feature's coefficient direction and "
                "magnitude - not a causal claim about what increases salary."
            )
        if ev.feature_importances:
            st.plotly_chart(
                coefficient_bar_chart(ev.feature_importances, "Random Forest Feature Importance (top 10)"),
                use_container_width=True,
            )
            st.caption(
                "Feature importance reflects this model's contribution measure and does not "
                "establish causation."
            )
        st.divider()

    st.markdown('<h3 class="lm-section-title">Try a prediction</h3>', unsafe_allow_html=True)
    st.caption("This is a model-based salary estimate from the available dataset. "
               "It is not a guaranteed salary offer.")

    with st.form("salary_prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            ssc_p = st.slider("SSC %", 0.0, 100.0, NUMERIC_RANGES["ssc_p"][2], key="sal_ssc")
            hsc_p = st.slider("HSC %", 0.0, 100.0, NUMERIC_RANGES["hsc_p"][2], key="sal_hsc")
            degree_p = st.slider("Degree %", 0.0, 100.0, NUMERIC_RANGES["degree_p"][2], key="sal_degree")
            etest_p = st.slider("E-test %", 0.0, 100.0, NUMERIC_RANGES["etest_p"][2], key="sal_etest")
            mba_p = st.slider("MBA %", 0.0, 100.0, NUMERIC_RANGES["mba_p"][2], key="sal_mba")
        with col2:
            ssc_b = st.selectbox("SSC Board", CATEGORY_OPTIONS["ssc_b"], key="sal_ssc_b")
            hsc_b = st.selectbox("HSC Board", CATEGORY_OPTIONS["hsc_b"], key="sal_hsc_b")
            hsc_s = st.selectbox("HSC Specialisation (stream)", CATEGORY_OPTIONS["hsc_s"], key="sal_hsc_s")
            degree_t = st.selectbox("Degree Type", CATEGORY_OPTIONS["degree_t"], key="sal_degree_t")
            workex = st.selectbox("Work Experience", CATEGORY_OPTIONS["workex"], key="sal_workex")
            specialisation = st.selectbox("MBA Specialisation", CATEGORY_OPTIONS["specialisation"], key="sal_spec")

        reg_model_name = st.selectbox("Model", list(REGRESSION_MODEL_REGISTRY.keys()))
        submitted = st.form_submit_button("Predict salary")

    if submitted:
        input_values = {
            "ssc_p": ssc_p, "hsc_p": hsc_p, "degree_p": degree_p,
            "etest_p": etest_p, "mba_p": mba_p,
            "ssc_b": ssc_b, "hsc_b": hsc_b, "hsc_s": hsc_s,
            "degree_t": degree_t, "workex": workex, "specialisation": specialisation,
        }
        result = predict_salary(reg_results, reg_model_name, input_values)

        st.divider()
        st.markdown('<h3 class="lm-section-title">Prediction result</h3>', unsafe_allow_html=True)
        r1, r2 = st.columns(2)
        r1.metric("Predicted salary", f"\u20B9{result['predicted_salary']:,.0f}")
        r2.metric("Model used", result["model_name"])
        st.caption("This is a model-based salary estimate from the available dataset. "
                   "It is not a guaranteed salary offer.")


# ---------------------------------------------------------------------------
# PAGE: STUDENT SEGMENTATION
# ---------------------------------------------------------------------------

def render_student_segmentation() -> None:
    st.title("\U0001F9E9 Student Segmentation")
    st.caption("Discover student profiles through unsupervised learning.")
    st.info(SEGMENTATION_SMALL_SAMPLE_WARNING)

    with st.expander("Methodology"):
        m1, m2, m3 = st.columns(3)
        m1.markdown("**Algorithm**\n\nK-Means Clustering")
        m2.markdown("**Scaling**\n\nStandardScaler")
        m3.markdown("**Categorical Encoding**\n\nOneHotEncoder")
        m4, m5 = st.columns(2)
        m4.markdown(f"**Excluded features**\n\n{', '.join(CLUSTER_EXCLUDED)}")
        m5.markdown(f"**Random State**\n\n{RANDOM_STATE}")
        st.caption(
            f"Clustering features - numeric: {', '.join(CLUSTER_NUMERIC_FEATURES)}. "
            f"categorical: {', '.join(CLUSTER_CATEGORICAL_FEATURES)}."
        )

    diagnostics = get_segmentation_diagnostics(cleaned_df)
    suggested_k = recommend_k(diagnostics)

    st.markdown('<h3 class="lm-section-title">K diagnostics</h3>', unsafe_allow_html=True)
    st.caption(
        "Higher silhouette score generally indicates better cluster separation/cohesion. "
        "Treat this as one diagnostic, not proof that the segmentation is meaningful."
    )

    d1, d2 = st.columns(2)
    with d1:
        st.plotly_chart(plot_elbow_curve(diagnostics.elbow_scores, highlighted_k=suggested_k),
                         use_container_width=True)
    with d2:
        st.plotly_chart(plot_silhouette_scores(diagnostics.silhouette_scores, highlighted_k=suggested_k),
                         use_container_width=True)

    st.markdown('<h3 class="lm-section-title">Select number of student segments</h3>', unsafe_allow_html=True)
    st.caption(
        f"Diagnostic recommendation: K={suggested_k} (highest silhouette score in the tested range "
        f"K={diagnostics.k_range[0]}-{diagnostics.k_range[-1]}). This is a suggestion, not an absolute "
        f"truth - pick any K below."
    )
    k = st.select_slider("Select number of student segments", options=diagnostics.k_range, value=suggested_k)

    seg = get_segmentation_result(cleaned_df, k, diagnostics)

    st.divider()
    st.markdown('<h3 class="lm-section-title">Cluster overview</h3>', unsafe_allow_html=True)
    sizes = [p.count for p in seg.profiles]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Number of clusters", seg.k)
    c2.metric("Total students", sum(sizes))
    c3.metric("Largest cluster", max(sizes))
    c4.metric("Smallest cluster", min(sizes))
    st.caption("Cluster numbers are algorithm-generated identifiers and do not represent rankings.")

    e1, e2 = st.columns(2)
    with e1:
        st.plotly_chart(plot_cluster_distribution(seg.profiles), use_container_width=True)
    with e2:
        st.plotly_chart(plot_cluster_workex(seg.profiles), use_container_width=True)

    st.divider()
    st.markdown('<h3 class="lm-section-title">Cluster profiles</h3>', unsafe_allow_html=True)
    st.plotly_chart(plot_cluster_profiles(seg.profiles), use_container_width=True)

    for p in seg.profiles:
        with st.expander(f"Cluster {p.cluster_id} - {p.label} (n={p.count}, {p.pct_of_total}%)"):
            pc1, pc2 = st.columns(2)
            with pc1:
                st.markdown(
                    f"- SSC avg: **{p.ssc_avg}%**\n"
                    f"- HSC avg: **{p.hsc_avg}%**\n"
                    f"- Degree avg: **{p.degree_avg}%**\n"
                    f"- E-test avg: **{p.etest_avg}%**\n"
                    f"- MBA avg: **{p.mba_avg}%**"
                )
            with pc2:
                st.markdown(
                    f"- Work experience: **{p.workex_yes_pct}%**\n"
                    f"- Most common HSC stream: **{p.top_hsc_stream}**\n"
                    f"- Most common degree type: **{p.top_degree_type}**\n"
                    f"- Most common MBA specialisation: **{p.top_specialisation}**"
                )
            st.caption(
                f"Overall dataset averages for comparison: academic avg {seg.overall_academic_avg}%, "
                f"work experience {seg.overall_workex_pct}%."
            )

    st.divider()
    st.markdown('<h3 class="lm-section-title">2D cluster visualization (PCA)</h3>', unsafe_allow_html=True)
    st.plotly_chart(plot_cluster_pca(seg.pca_df), use_container_width=True)
    st.caption(
        "Clusters are formed in the full preprocessed feature space. PCA is used only to visualize "
        "the multidimensional data in two dimensions."
    )

    st.divider()
    st.markdown('<h3 class="lm-section-title">Post-clustering descriptive outcomes</h3>', unsafe_allow_html=True)
    st.caption(
        "These outcomes were NOT used to create the clusters - placement status and salary play no "
        "role in the clustering features. Shown here only as a descriptive comparison after the fact."
    )
    outcomes_df = post_clustering_outcomes(cleaned_df, seg.labels)
    st.dataframe(outcomes_df, use_container_width=True, hide_index=True)
    st.caption("Cluster membership is not claimed to cause any placement or salary outcome.")

    st.divider()
    st.markdown('<h3 class="lm-section-title">Segmentation Insights</h3>', unsafe_allow_html=True)
    insights = generate_segmentation_insights(seg.profiles)
    if not insights:
        st.info("Not enough separation between clusters to generate a segmentation insight.")
    for ins in insights:
        st.markdown(
            f"""<div class="lm-action">
            <b>FACT:</b> {ins.fact}<br/><br/>
            <b>INSIGHT:</b> {ins.insight}<br/><br/>
            <b>POSSIBLE ACTION:</b> {ins.possible_action}
            </div>""",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# PAGE: AI INSIGHT COPILOT
# ---------------------------------------------------------------------------

def render_ai_copilot() -> None:
    st.title("\U0001F9E0 AI Insight Copilot")
    st.caption("Ask questions about your student data, models, and insights.")

    cls_results = get_classification_results(cleaned_df)
    reg_results = get_regression_results(cleaned_df)
    seg_diagnostics = get_segmentation_diagnostics(cleaned_df)
    evidence = get_copilot_evidence(cleaned_df, cls_results, reg_results, seg_diagnostics)

    valid, issues = validate_evidence(evidence)
    if not valid:
        st.error("Evidence validation failed - the Copilot cannot safely answer right now:")
        for issue in issues:
            st.write(f"- {issue}")
        return

    secrets_key = None
    try:
        secrets_key = st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        secrets_key = None
    api_key = get_configured_api_key(secrets_key)

    provider_col, badge_col = st.columns([3, 1])
    with provider_col:
        if api_key:
            st.caption(
                "An optional AI provider key is configured. Each answer below still starts from the "
                "same verified evidence; the app falls back automatically if the API call fails."
            )
        else:
            st.caption(
                "AI Provider: **Fallback Mode** - a built-in, deterministic analytical engine. "
                "No external API key is configured, and none is required for this Copilot to work."
            )
    with badge_col:
        st.markdown('<span class="lm-badge">GROUNDED IN PROJECT DATA</span>', unsafe_allow_html=True)

    st.markdown('<h3 class="lm-section-title">Quick questions</h3>', unsafe_allow_html=True)
    cols = st.columns(3)
    clicked_intent = None
    for i, (label, intent) in enumerate(QUICK_QUESTIONS):
        with cols[i % 3]:
            if st.button(label, use_container_width=True, key=f"quick_{intent}"):
                clicked_intent = intent

    st.markdown('<h3 class="lm-section-title">Ask a question</h3>', unsafe_allow_html=True)
    with st.form("copilot_question_form"):
        question = st.text_input(
            "Your question",
            placeholder="e.g. Does work experience relate to placement?",
            label_visibility="collapsed",
        )
        asked = st.form_submit_button("Ask")

    response = None
    if clicked_intent is not None:
        response = answer_intent(clicked_intent, evidence)
    elif asked and question.strip():
        response = generate_response(question.strip(), evidence, api_key=api_key)

    if response is not None:
        st.divider()
        source_label = "AI-generated (grounded in evidence below)" if response.get("source") == "ai" else "Fallback engine (deterministic, no API used)"
        st.caption(f"Answer source: {source_label}")

        if response.get("source") == "ai" and response.get("ai_markdown"):
            st.markdown(response["ai_markdown"])
            with st.expander("Verified evidence used for this answer"):
                st.markdown(render_response_markdown({k: v for k, v in response.items()
                                                        if k in ("fact", "insight", "action", "limitation")}))
        else:
            st.markdown(render_response_markdown(response))


# ---------------------------------------------------------------------------
# PAGE: WHAT-IF SIMULATOR
# ---------------------------------------------------------------------------

def render_whatif_simulator() -> None:
    st.title("\U0001F3AF What-If Simulator")
    st.caption("Explore how hypothetical student profiles change model-estimated placement probabilities.")
    st.info(WHATIF_DATASET_SIZE_NOTE)

    results = get_classification_results(cleaned_df)
    base = baseline_profile(cleaned_df)

    st.caption(
        "Baseline values are initialized from the dataset distribution - numeric defaults use the "
        "median, categorical defaults use the most frequent category. This does not represent a real "
        "individual student."
    )

    model_choice = st.selectbox("Model", list(MODEL_REGISTRY.keys()) + ["Compare Both Models"],
                                 key="whatif_model")

    numeric_feats = CLUSTER_NUMERIC_FEATURES
    categorical_feats = CLUSTER_CATEGORICAL_FEATURES
    whatif_keys = [f"whatif_{f}" for f in numeric_feats + categorical_feats]

    if st.button("\U0001F504 Reset What-If"):
        for k in whatif_keys:
            st.session_state.pop(k, None)
        st.rerun()

    st.markdown('<h3 class="lm-section-title">Scenario A - Baseline (fixed)</h3>', unsafe_allow_html=True)
    a1, a2 = st.columns(2)
    with a1:
        for f in numeric_feats:
            st.caption(f"{WHATIF_FEATURE_LABELS[f]}: {base[f]:.1f}")
    with a2:
        for f in categorical_feats:
            st.caption(f"{WHATIF_FEATURE_LABELS[f]}: {base[f]}")

    st.markdown('<h3 class="lm-section-title">Scenario B - What-If (adjust below)</h3>', unsafe_allow_html=True)
    whatif = {}
    b1, b2 = st.columns(2)
    with b1:
        for f in numeric_feats:
            whatif[f] = st.slider(WHATIF_FEATURE_LABELS[f], 0.0, 100.0, base[f], key=f"whatif_{f}")
    with b2:
        for f in categorical_feats:
            options = CATEGORY_OPTIONS[f]
            default_idx = options.index(base[f]) if base[f] in options else 0
            whatif[f] = st.selectbox(WHATIF_FEATURE_LABELS[f], options, index=default_idx, key=f"whatif_{f}")

    st.divider()

    models_to_run = list(MODEL_REGISTRY.keys()) if model_choice == "Compare Both Models" else [model_choice]
    comparisons = {m: compare_scenarios(results, m, base, whatif) for m in models_to_run}

    st.markdown('<h3 class="lm-section-title">Results</h3>', unsafe_allow_html=True)
    for m, cmp in comparisons.items():
        st.markdown(f"##### {m}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Baseline probability", f"{cmp.prob_baseline * 100:.1f}%")
        c2.metric("What-If probability", f"{cmp.prob_whatif * 100:.1f}%")
        c3.metric("Change", f"{cmp.diff_pp:+.1f} pp")
        c4.metric("Predicted class (A -> B)", f"{cmp.class_baseline} -> {cmp.class_whatif}")
        st.plotly_chart(
            plot_scenario_probability_comparison(cmp.prob_baseline, cmp.prob_whatif,
                                                  title=f"{m}: Scenario A vs Scenario B"),
            use_container_width=True,
        )

    changes = detect_profile_changes(base, whatif)
    st.markdown('<h3 class="lm-section-title">Changed inputs</h3>', unsafe_allow_html=True)
    if not changes:
        st.info("No changes were made to the What-If scenario.")
    else:
        for f, old, new in changes:
            label = WHATIF_FEATURE_LABELS.get(f, f)
            if isinstance(old, float):
                st.write(f"- **{label}**: {old:.1f} \u2192 {new:.1f}")
            else:
                st.write(f"- **{label}**: {old} \u2192 {new}")

    st.markdown('<h3 class="lm-section-title">Model interpretation</h3>', unsafe_allow_html=True)
    for m, cmp in comparisons.items():
        direction = "higher" if cmp.diff_pp > 0 else ("lower" if cmp.diff_pp < 0 else "the same")
        multi_note = f" ({len(cmp.changes)} inputs changed together)" if len(cmp.changes) > 1 else ""
        st.markdown(
            f"Under **{m}**, the What-If profile has a {direction} model-estimated placement probability "
            f"than the baseline profile (observed model change: **{cmp.diff_pp:+.1f} percentage points**){multi_note}. "
            f"This is a difference in model output for these two input profiles - it should not be "
            f"interpreted as evidence that changing these attributes would causally produce a placement "
            f"outcome in the real world."
        )

    st.divider()
    st.markdown('<h3 class="lm-section-title">Single Variable Experiment</h3>', unsafe_allow_html=True)
    exp_model = model_choice if model_choice != "Compare Both Models" else list(MODEL_REGISTRY.keys())[0]
    exp_feature = st.selectbox(
        "Feature to vary", numeric_feats + categorical_feats,
        format_func=lambda f: WHATIF_FEATURE_LABELS.get(f, f), key="whatif_exp_feature",
    )
    if exp_feature in numeric_feats:
        curve_df = simulate_single_numeric_feature(results, exp_model, whatif, exp_feature)
        st.plotly_chart(
            plot_single_feature_response(curve_df, WHATIF_FEATURE_LABELS[exp_feature],
                                          current_value=whatif[exp_feature]),
            use_container_width=True,
        )
    else:
        options = CATEGORY_OPTIONS[exp_feature]
        cat_df = simulate_categorical_feature(results, exp_model, whatif, exp_feature, options)
        st.plotly_chart(
            plot_categorical_scenario_comparison(cat_df, WHATIF_FEATURE_LABELS[exp_feature]),
            use_container_width=True,
        )
    st.caption(
        f"Model used for this experiment: {exp_model}. All other inputs are held at the current "
        f"What-If scenario values shown above."
    )

    st.divider()
    st.markdown("### Important interpretation")
    st.warning(WHATIF_LIMITATION_NOTE)


# ---------------------------------------------------------------------------
# PAGE: PROJECT SUMMARY
# ---------------------------------------------------------------------------

def render_project_summary() -> None:
    st.title("\U0001F4D8 Project Summary")
    st.caption("From Student Data to Employability Intelligence")

    cls_results = get_classification_results(cleaned_df)
    reg_results = get_regression_results(cleaned_df)
    seg_diagnostics = get_segmentation_diagnostics(cleaned_df)
    evidence = get_copilot_evidence(cleaned_df, cls_results, reg_results, seg_diagnostics)
    d = evidence["dataset"]
    we = {r["workex"]: r for r in evidence["placement"]["by_workex"]}
    corr = evidence["academic"]["correlation_with_placement"]
    seg = evidence["segmentation"]

    # ---- 1. Project overview -------------------------------------------
    st.markdown('<h3 class="lm-section-title">1. Project Overview</h3>', unsafe_allow_html=True)
    st.markdown(
        "LearnMate Analytics AI is a student employability and placement intelligence platform "
        "that combines data analytics, exploratory data analysis, classification, regression, "
        "clustering, explainable model evaluation, grounded AI insights, and scenario simulation "
        "on a real campus-placement dataset. It does not guarantee any individual's employment or "
        "salary outcome - it demonstrates how raw student records can be turned into transparent, "
        "verifiable analytics and model-based estimates."
    )

    # ---- 2. Problem statement -------------------------------------------
    st.markdown('<h3 class="lm-section-title">2. Problem Statement</h3>', unsafe_allow_html=True)
    st.markdown(
        "Student placement datasets contain academic, educational, work-experience and "
        "specialisation information, but raw tables do not directly provide actionable insights. "
        "This project transforms the dataset through the pipeline:\n\n"
        "**DATA \u2192 INSIGHTS \u2192 PREDICTIONS \u2192 SEGMENTS \u2192 SCENARIOS \u2192 ACTIONABLE INTERPRETATION**"
    )

    # ---- 3. Dataset -------------------------------------------------------
    st.markdown('<h3 class="lm-section-title">3. Dataset</h3>', unsafe_allow_html=True)
    st.markdown(
        f"**{quality.n_rows} students, {quality.n_cols} original columns** "
        f"(sl_no, gender, ssc_p, ssc_b, hsc_p, hsc_b, hsc_s, degree_p, degree_t, workex, etest_p, "
        f"specialisation, mba_p, status, salary)."
    )
    st.markdown(
        "- `sl_no` is a row identifier - excluded from every model.\n"
        "- `gender` is a sensitive attribute - excluded from every predictive model.\n"
        "- `status` is the placement classification target.\n"
        "- `salary` is used only for salary regression, and only for placed students.\n"
        "- `salary` is missing for non-placed students and is **never** used as a placement feature."
    )

    # ---- 4. Analytics pipeline -------------------------------------------
    st.markdown('<h3 class="lm-section-title">4. Analytics Pipeline</h3>', unsafe_allow_html=True)
    pipeline_steps = [
        ("DATA", "Load the real campus placement CSV as-is."),
        ("DATA QUALITY", "Validate schema, check duplicates and missing values, prepare a model-ready dataframe."),
        ("EDA", "Explore placement, academic performance, salary, and relationships between variables."),
        ("CLASSIFICATION", "Predict placement (Placed / Not Placed) with Logistic Regression and a Decision Tree."),
        ("REGRESSION", "Estimate salary for placed students with Linear Regression and a Random Forest Regressor."),
        ("CLUSTERING", "Group students into profiles with K-Means, diagnosed via Elbow and Silhouette analysis."),
        ("AI INSIGHTS", "Answer analytical questions from verified evidence, with a deterministic no-API fallback."),
        ("WHAT-IF SIMULATION", "Compare hypothetical student profiles through the existing trained classification model."),
    ]
    for i, (step, desc) in enumerate(pipeline_steps):
        st.markdown(f"**{step}** - {desc}")
        if i < len(pipeline_steps) - 1:
            st.markdown("<div style='text-align:center;color:#94A3B8;'>\u2193</div>", unsafe_allow_html=True)

    # ---- 5. Key verified findings -----------------------------------------
    st.markdown('<h3 class="lm-section-title">5. Key Verified Findings</h3>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total students", d["n_total"])
    k2.metric("Placed", d["n_placed"])
    k3.metric("Not placed", d["n_not_placed"])
    k4.metric("Placement rate", f"{d['placement_rate_pct']}%")
    sal = evidence["salary"]["stats"]
    st.markdown(
        f"- Average salary among placed students: **INR {sal['mean']:,.0f}** "
        f"(median **INR {sal['median']:,.0f}**), based on {sal['count']} salary records.\n"
        + (f"- Work experience: **{we['Yes']['placement_rate_pct']}%** observed placement rate "
           f"(n={int(we['Yes']['total'])}) vs **{we['No']['placement_rate_pct']}%** without "
           f"(n={int(we['No']['total'])}) - a **{we['Yes']['placement_rate_pct'] - we['No']['placement_rate_pct']:+.1f} "
           f"percentage point** observed difference.\n" if "Yes" in we and "No" in we else "")
        + (f"- Academic-feature correlations with salary were all weak (|r| \u2264 "
           f"{max(abs(v) for v in corr.values()):.2f} for placement; the equivalent salary correlations "
           f"found during EDA were all |r| \u2264 0.18).\n" if corr else "")
    )
    st.caption("These are observed dataset patterns, not causal relationships.")

    # ---- 6. Placement classification --------------------------------------
    st.markdown('<h3 class="lm-section-title">6. Placement Classification</h3>', unsafe_allow_html=True)
    st.dataframe(comparison_table(cls_results), use_container_width=True, hide_index=True)
    st.caption(
        "These results come from this project's fixed 80/20 train/test evaluation setup "
        "(random_state=42) and should not be interpreted as universal real-world performance. "
        "Neither model is labeled as \"the best\" - compare the metrics directly."
    )

    # ---- 7. Salary regression -----------------------------------------------
    st.markdown('<h3 class="lm-section-title">7. Salary Regression</h3>', unsafe_allow_html=True)
    st.markdown(f"Salary records: **{reg_results.n_salary_records}** \u2192 Training: **{reg_results.n_train}**, Test: **{reg_results.n_test}**")
    st.dataframe(regression_comparison_table(reg_results), use_container_width=True, hide_index=True)
    st.markdown(
        "A **negative R\u00b2** means that, on this test split, the model did not outperform a simple "
        "mean-salary baseline. This is consistent with the weak salary correlations found during EDA - "
        "the available academic and employability variables do not provide enough signal to accurately "
        "explain salary variation in this small sample. This result is reported as-is, not hidden."
    )

    # ---- 8. Student segmentation --------------------------------------------
    st.markdown('<h3 class="lm-section-title">8. Student Segmentation</h3>', unsafe_allow_html=True)
    st.markdown(
        f"Features used: {', '.join(CLUSTER_NUMERIC_FEATURES + CLUSTER_CATEGORICAL_FEATURES)}. "
        f"Excluded: {', '.join(CLUSTER_EXCLUDED)}.\n\n"
        f"Tested K = 2-6. Diagnostic-recommended K = **{seg['k']}** "
        f"(silhouette = **{seg['silhouette_by_k'][seg['k']]:.3f}**)."
    )
    for p in seg["profiles"]:
        st.markdown(f"- **Cluster {p['cluster_id']} - {p['label']}**: {p['count']} students ({p['pct_of_total']}%)")
    st.caption(
        "Silhouette scores across K=2-6 are modest (0.12-0.18), meaning these clusters are real but "
        "not sharply separated. Labels are algorithm-generated descriptions, not absolute rankings."
    )
    outcomes = seg.get("post_clustering_outcomes", [])
    if outcomes:
        st.markdown("**Post-clustering descriptive outcomes** (NOT used to create the clusters):")
        for o in outcomes:
            salary_txt = f", avg salary INR {o['avg_salary_placed']:,.0f} (n={o['salary_sample_size']})" if o.get("avg_salary_placed") else ""
            st.markdown(f"- Cluster {o['cluster']}: placement rate {o['placement_rate_pct']}%{salary_txt}")
    st.caption("These outcomes are descriptive post-clustering analysis and do not establish causation.")

    # ---- 9. AI Insight Copilot ---------------------------------------------
    st.markdown('<h3 class="lm-section-title">9. AI Insight Copilot</h3>', unsafe_allow_html=True)
    st.markdown(
        "The Copilot builds a structured evidence registry from this project's own calculated "
        "statistics and model results first, then answers questions about dataset overview, academic "
        "relationships, work experience and placement, placement by degree/specialisation, salary "
        "overview and by specialisation, salary-model performance, classification performance, "
        "segmentation, main findings, and next investigations - grounded in that evidence.\n\n"
        "The deterministic fallback engine works with **no API key and no internet access**. An "
        "optional external LLM layer exists but is not described as connected unless it actually "
        "succeeds at runtime."
    )

    # ---- 10. What-If Simulator ----------------------------------------------
    st.markdown('<h3 class="lm-section-title">10. What-If Simulator</h3>', unsafe_allow_html=True)
    st.markdown(
        "The simulator reuses the existing Stage 3 classification pipeline - no new model is trained. "
        "It compares a baseline profile against a hypothetical profile using `predict_proba()` from "
        "the already-fitted pipeline. The output is a **model-estimated probability**, not a "
        "guaranteed placement probability, and does not establish causality."
    )

    # ---- 11. Responsible data science ----------------------------------------
    st.markdown('<h3 class="lm-section-title">11. Responsible Data Science</h3>', unsafe_allow_html=True)
    st.warning(
        "- Small dataset size (215 records; only 148 with a recorded salary).\n"
        "- Observational data - no experimental control, so associations are not causal evidence.\n"
        "- Model uncertainty - metrics come from a single train/test split and may vary with a different one.\n"
        "- The salary regression models are weak on this dataset (negative R\u00b2).\n"
        "- Clustering separation is modest (silhouette 0.12-0.18).\n"
        "- Sensitive/non-predictive fields (`gender`, `sl_no`) are excluded from every predictive model.\n"
        "- No guarantee of placement or salary is made anywhere in this application.\n"
        "- A larger, richer dataset (e.g. with company, role, or location data) would be needed for any "
        "real-world deployment consideration."
    )

    st.divider()
    st.caption(
        "LearnMate Analytics AI - built for the AICTE | IBM SkillsBuild Data Analytics with AI "
        "Internship 2026, BharatCares. AI-assisted development was used; see README.md for details."
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

if page == "Overview":
    render_overview()
elif page == "Data Explorer":
    render_data_explorer()
elif page == "Exploratory Data Analysis":
    render_eda()
elif page == "\U0001F9E9 Student Segmentation":
    render_student_segmentation()
elif page == "\U0001F9E0 AI Insight Copilot":
    render_ai_copilot()
elif page == "\U0001F3AF What-If Simulator":
    render_whatif_simulator()
elif page == "\U0001F916 Placement Prediction":
    render_placement_prediction()
elif page == "\U0001F916 Salary Prediction":
    render_salary_prediction()
elif page == "\U0001F916 Model Evaluation":
    render_model_evaluation()
elif page == "\U0001F4D8 Project Summary":
    render_project_summary()
