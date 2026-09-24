"""
visualizations.py
------------------
STAGE 2 - Reusable Plotly chart builders for LearnMate Analytics AI.

Keeping chart construction here (separate from app.py) means every page
uses the same color theme and the same chart "grammar", and charts can be
swapped for a different type without duplicating styling code.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

PRIMARY = "#7C5CFF"
SECONDARY = "#22D3B0"
DARK = "#111827"
LIGHT = "#F8FAFC"
MUTED = "#94A3B8"

DISCRETE_PALETTE = [PRIMARY, SECONDARY, "#F59E0B", "#EF4444", "#3B82F6", "#A855F7"]

BASE_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Segoe UI, Helvetica, Arial, sans-serif", color=DARK, size=13),
    title_font=dict(size=16, color=DARK),
    margin=dict(l=40, r=20, t=50, b=40),
    plot_bgcolor="white",
    paper_bgcolor="rgba(0,0,0,0)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)


def _apply_theme(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(**BASE_LAYOUT, height=height)
    return fig


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def bar_chart(df: pd.DataFrame, x: str, y: str, title: str, color: str | None = None,
              text_auto: bool = True) -> go.Figure:
    fig = px.bar(
        df, x=x, y=y, color=color, title=title, text_auto=text_auto,
        color_discrete_sequence=DISCRETE_PALETTE,
    )
    return _apply_theme(fig)


def pie_chart(df: pd.DataFrame, names: str, values: str, title: str) -> go.Figure:
    fig = px.pie(
        df, names=names, values=values, title=title, hole=0.45,
        color_discrete_sequence=DISCRETE_PALETTE,
    )
    fig.update_traces(textinfo="label+percent")
    return _apply_theme(fig)


def histogram_chart(df: pd.DataFrame, column: str, title: str, nbins: int = 20,
                     color: str | None = None) -> go.Figure:
    fig = px.histogram(
        df, x=column, nbins=nbins, title=title, color=color,
        color_discrete_sequence=DISCRETE_PALETTE, marginal="box",
    )
    return _apply_theme(fig)


def box_chart(df: pd.DataFrame, x: str, y: str, title: str, color: str | None = None) -> go.Figure:
    fig = px.box(
        df, x=x, y=y, title=title, color=color or x,
        color_discrete_sequence=DISCRETE_PALETTE, points="outliers",
    )
    fig.update_layout(showlegend=False)
    return _apply_theme(fig)


def scatter_chart(df: pd.DataFrame, x: str, y: str, title: str,
                   color: str | None = None) -> go.Figure:
    fig = px.scatter(
        df, x=x, y=y, color=color, title=title,
        color_discrete_sequence=DISCRETE_PALETTE, opacity=0.75,
    )
    return _apply_theme(fig)


def heatmap_chart(corr_df: pd.DataFrame, title: str) -> go.Figure:
    fig = px.imshow(
        corr_df, text_auto=True, title=title, color_continuous_scale="Purples",
        zmin=-1, zmax=1, aspect="auto",
    )
    return _apply_theme(fig, height=420)


def confusion_matrix_chart(cm, labels: list[str], title: str) -> go.Figure:
    """cm is a 2x2 numpy array from sklearn.metrics.confusion_matrix
    (rows = actual, columns = predicted)."""
    fig = px.imshow(
        cm, text_auto=True, title=title, color_continuous_scale="Purples",
        x=[f"Predicted: {l}" for l in labels], y=[f"Actual: {l}" for l in labels],
        aspect="auto",
    )
    fig.update_coloraxes(showscale=False)
    return _apply_theme(fig, height=360)


def roc_curve_chart(fpr, tpr, auc: float | None, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC curve",
                              line=dict(color=PRIMARY, width=3)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess",
                              line=dict(color=MUTED, width=1, dash="dash")))
    subtitle = f"{title} (AUC = {auc:.3f})" if auc is not None else title
    fig.update_layout(title=subtitle, xaxis_title="False Positive Rate",
                       yaxis_title="True Positive Rate")
    return _apply_theme(fig, height=380)


def coefficient_bar_chart(values: dict, title: str, top_n: int = 10) -> go.Figure:
    """values: {feature_name: coefficient_or_importance}. Shows the top_n by
    absolute magnitude, largest at the top."""
    items = sorted(values.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
    items = items[::-1]  # so the largest ends up at the top of a horizontal bar chart
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    colors = [PRIMARY if v >= 0 else "#EF4444" for v in vals]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=colors))
    fig.update_layout(title=title, xaxis_title="Value")
    return _apply_theme(fig, height=max(320, 28 * len(items)))


def actual_vs_predicted_chart(y_true, y_pred, title: str) -> go.Figure:
    """Scatter of actual vs predicted values with a y=x reference line -
    points on the line are perfect predictions."""
    lo = float(min(min(y_true), min(y_pred)))
    hi = float(max(max(y_true), max(y_pred)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_true, y=y_pred, mode="markers", name="Test records",
        marker=dict(color=PRIMARY, size=9, opacity=0.75),
    ))
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect prediction (y = x)",
        line=dict(color=MUTED, width=1, dash="dash"),
    ))
    fig.update_layout(title=title, xaxis_title="Actual Salary (INR)", yaxis_title="Predicted Salary (INR)")
    return _apply_theme(fig, height=400)


def residual_chart(y_pred, residuals, title: str) -> go.Figure:
    """Residual = actual - predicted. Plotted against the prediction so
    patterns (e.g. a funnel shape) are easy to spot."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_pred, y=residuals, mode="markers", name="Residuals",
        marker=dict(color=SECONDARY, size=9, opacity=0.75),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color=MUTED)
    fig.update_layout(title=title, xaxis_title="Predicted Salary (INR)", yaxis_title="Residual (Actual - Predicted, INR)")
    return _apply_theme(fig, height=380)


# ---------------------------------------------------------------------------
# STAGE 5 - Student Segmentation charts
# ---------------------------------------------------------------------------

def plot_elbow_curve(elbow_scores: dict, highlighted_k: int | None = None) -> go.Figure:
    ks = list(elbow_scores.keys())
    inertias = list(elbow_scores.values())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ks, y=inertias, mode="lines+markers", name="Inertia",
                              line=dict(color=PRIMARY, width=3), marker=dict(size=9)))
    if highlighted_k is not None and highlighted_k in elbow_scores:
        fig.add_trace(go.Scatter(x=[highlighted_k], y=[elbow_scores[highlighted_k]], mode="markers",
                                  name=f"Selected K={highlighted_k}",
                                  marker=dict(color=SECONDARY, size=16, symbol="star")))
    fig.update_layout(title="Elbow Method - Inertia by Number of Clusters (K)",
                       xaxis_title="Number of Clusters (K)", yaxis_title="Inertia")
    fig.update_xaxes(dtick=1)
    return _apply_theme(fig, height=380)


def plot_silhouette_scores(silhouette_scores: dict, highlighted_k: int | None = None) -> go.Figure:
    ks = [k for k, v in silhouette_scores.items() if v is not None]
    scores = [silhouette_scores[k] for k in ks]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ks, y=scores, mode="lines+markers", name="Silhouette Score",
                              line=dict(color=SECONDARY, width=3), marker=dict(size=9)))
    if highlighted_k is not None and highlighted_k in silhouette_scores and silhouette_scores[highlighted_k] is not None:
        fig.add_trace(go.Scatter(x=[highlighted_k], y=[silhouette_scores[highlighted_k]], mode="markers",
                                  name=f"Selected K={highlighted_k}",
                                  marker=dict(color=PRIMARY, size=16, symbol="star")))
    fig.update_layout(title="Silhouette Score by Number of Clusters (K)",
                       xaxis_title="Number of Clusters (K)", yaxis_title="Silhouette Score")
    fig.update_xaxes(dtick=1)
    return _apply_theme(fig, height=380)


def plot_cluster_distribution(profiles, title: str = "Students per Cluster") -> go.Figure:
    """profiles: list of ClusterProfile-like objects with cluster_id, count, pct_of_total."""
    labels = [f"Cluster {p.cluster_id}" for p in profiles]
    counts = [p.count for p in profiles]
    pct = [p.pct_of_total for p in profiles]
    fig = go.Figure(go.Bar(
        x=labels, y=counts, text=[f"{c} ({p}%)" for c, p in zip(counts, pct)],
        textposition="outside", marker_color=DISCRETE_PALETTE[:len(labels)],
    ))
    fig.update_layout(title=title, xaxis_title="Cluster", yaxis_title="Number of Students")
    return _apply_theme(fig, height=380)


def plot_cluster_profiles(profiles, title: str = "Academic Profile by Cluster") -> go.Figure:
    """Grouped bar chart comparing SSC/HSC/Degree/E-test/MBA averages (%) across clusters."""
    metrics = ["ssc_avg", "hsc_avg", "degree_avg", "etest_avg", "mba_avg"]
    metric_labels = ["SSC %", "HSC %", "Degree %", "E-test %", "MBA %"]
    fig = go.Figure()
    for i, p in enumerate(profiles):
        values = [getattr(p, m) for m in metrics]
        fig.add_trace(go.Bar(
            name=f"Cluster {p.cluster_id}", x=metric_labels, y=values,
            marker_color=DISCRETE_PALETTE[i % len(DISCRETE_PALETTE)],
        ))
    fig.update_layout(title=title, barmode="group", yaxis_title="Average (%)", yaxis_range=[0, 100])
    return _apply_theme(fig, height=420)


def plot_cluster_workex(profiles, title: str = "Work Experience Share by Cluster") -> go.Figure:
    labels = [f"Cluster {p.cluster_id}" for p in profiles]
    values = [p.workex_yes_pct for p in profiles]
    fig = go.Figure(go.Bar(
        x=labels, y=values, text=[f"{v}%" for v in values], textposition="outside",
        marker_color=DISCRETE_PALETTE[:len(labels)],
    ))
    fig.update_layout(title=title, xaxis_title="Cluster", yaxis_title="Students with Work Experience (%)",
                       yaxis_range=[0, 100])
    return _apply_theme(fig, height=360)


def plot_cluster_pca(pca_df: pd.DataFrame, title: str = "Student Clusters (PCA 2D Projection)") -> go.Figure:
    fig = px.scatter(
        pca_df, x="pc1", y="pc2", color="cluster", title=title,
        color_discrete_sequence=DISCRETE_PALETTE,
        hover_data={"ssc_p": True, "hsc_p": True, "degree_p": True, "workex": True,
                    "specialisation": True, "pc1": False, "pc2": False},
    )
    fig.update_traces(marker=dict(size=9, opacity=0.8))
    fig.update_layout(xaxis_title="Principal Component 1", yaxis_title="Principal Component 2")
    return _apply_theme(fig, height=460)


# ---------------------------------------------------------------------------
# STAGE 7 - What-If Simulator charts
# ---------------------------------------------------------------------------

def plot_scenario_probability_comparison(prob_baseline: float, prob_whatif: float,
                                          title: str = "Scenario A vs Scenario B") -> go.Figure:
    labels = ["Scenario A - Baseline", "Scenario B - What-If"]
    values = [prob_baseline * 100, prob_whatif * 100]
    colors = [MUTED, PRIMARY]
    fig = go.Figure(go.Bar(
        x=labels, y=values, text=[f"{v:.1f}%" for v in values], textposition="outside",
        marker_color=colors,
    ))
    fig.update_layout(title=title, yaxis_title="Model-Estimated Placement Probability (%)",
                       yaxis_range=[0, 105])
    return _apply_theme(fig, height=380)


def plot_single_feature_response(response_df: pd.DataFrame, feature_label: str,
                                  current_value: float | None = None) -> go.Figure:
    """response_df has columns 'value' and 'probability' (0-1). Marks the
    baseline's current value on the curve if given."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=response_df["value"], y=response_df["probability"] * 100, mode="lines+markers",
        name="Model-estimated probability", line=dict(color=PRIMARY, width=3), marker=dict(size=7),
    ))
    if current_value is not None:
        fig.add_vline(x=current_value, line_dash="dash", line_color=MUTED,
                       annotation_text="Baseline value", annotation_position="top")
    fig.update_layout(
        title="Model response curve - not a causal relationship",
        xaxis_title=feature_label, yaxis_title="Model-Estimated Placement Probability (%)",
        yaxis_range=[0, 105],
    )
    return _apply_theme(fig, height=400)


def plot_categorical_scenario_comparison(cat_df: pd.DataFrame, feature_label: str) -> go.Figure:
    """cat_df has columns 'category' and 'probability' (0-1)."""
    fig = go.Figure(go.Bar(
        x=cat_df["category"], y=cat_df["probability"] * 100,
        text=[f"{v:.1f}%" for v in cat_df["probability"] * 100], textposition="outside",
        marker_color=DISCRETE_PALETTE[:len(cat_df)],
    ))
    fig.update_layout(
        title=f"Model comparison by {feature_label} - not causal evidence",
        xaxis_title=feature_label, yaxis_title="Model-Estimated Placement Probability (%)",
        yaxis_range=[0, 105],
    )
    return _apply_theme(fig, height=380)


CHART_TYPE_OPTIONS = ["Histogram", "Box"]


def numeric_distribution_chart(df: pd.DataFrame, column: str, title: str,
                                chart_type: str, color: str | None = None) -> go.Figure:
    """Used by the Academic Performance chart-selector: lets the user pick
    Histogram vs Box for the same numeric column."""
    if chart_type == "Box":
        return box_chart(df, x=color if color else None, y=column, title=title, color=color)
    return histogram_chart(df, column=column, title=title, color=color)
