"""
analytics.py
------------
STAGE 2 - Deterministic analytics engine for LearnMate Analytics AI.

Every function here takes a (cleaned) dataframe and returns NUMBERS or TEXT
that is calculated directly from that dataframe. Nothing is hard-coded.

This module deliberately contains NO Streamlit and NO Plotly code -- it is
pure pandas/numpy so it can be unit-tested and, later, reused by the AI
Insight Copilot (Stage 7) without dragging the UI along with it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from utils.data_processing import (
    CLASSIFICATION_TARGET,
    REGRESSION_TARGET,
    NUMERIC_FEATURES,
)

PLACED_LABEL = "Placed"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _placed_mask(df: pd.DataFrame) -> pd.Series:
    return df[CLASSIFICATION_TARGET].str.lower() == PLACED_LABEL.lower()


def _placed_only(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with a real, non-null salary. This is the only valid slice for
    any salary-based analysis."""
    mask = _placed_mask(df) & df[REGRESSION_TARGET].notna()
    return df.loc[mask]


# ---------------------------------------------------------------------------
# 1. KPI SUMMARY (Overview page)
# ---------------------------------------------------------------------------

@dataclass
class KPISummary:
    total_students: int
    placed_students: int
    not_placed_students: int
    placement_rate_pct: float
    avg_salary_placed: float | None
    median_salary_placed: float | None
    salary_sample_size: int


def kpi_summary(df: pd.DataFrame) -> KPISummary:
    total = len(df)
    placed = int(_placed_mask(df).sum())
    not_placed = total - placed
    rate = round(100 * placed / total, 1) if total else 0.0

    placed_df = _placed_only(df)
    avg_salary = float(placed_df[REGRESSION_TARGET].mean()) if len(placed_df) else None
    median_salary = float(placed_df[REGRESSION_TARGET].median()) if len(placed_df) else None

    return KPISummary(
        total_students=total,
        placed_students=placed,
        not_placed_students=not_placed,
        placement_rate_pct=rate,
        avg_salary_placed=avg_salary,
        median_salary_placed=median_salary,
        salary_sample_size=len(placed_df),
    )


# ---------------------------------------------------------------------------
# 2. GROUPED PLACEMENT RATES  (e.g. by work experience, degree type, ...)
# ---------------------------------------------------------------------------

def placement_rate_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """For a categorical column, return one row per group with:
    total count, placed count, and placement rate (%)."""
    g = df.groupby(group_col)[CLASSIFICATION_TARGET].apply(
        lambda s: pd.Series({
            "total": len(s),
            "placed": int((s.str.lower() == PLACED_LABEL.lower()).sum()),
        })
    ).unstack()
    g["placement_rate_pct"] = (100 * g["placed"] / g["total"]).round(1)
    g = g.reset_index().sort_values("placement_rate_pct", ascending=False)
    g["total"] = g["total"].astype(int)
    g["placed"] = g["placed"].astype(int)
    return g


# ---------------------------------------------------------------------------
# 3. SALARY ANALYSIS (placed students only)
# ---------------------------------------------------------------------------

def salary_stats(df: pd.DataFrame) -> dict:
    placed_df = _placed_only(df)
    if placed_df.empty:
        return {"count": 0}
    s = placed_df[REGRESSION_TARGET]
    return {
        "count": int(s.count()),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std()),
        "min": float(s.min()),
        "max": float(s.max()),
    }


def salary_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    placed_df = _placed_only(df)
    if placed_df.empty:
        return pd.DataFrame(columns=[group_col, "mean_salary", "median_salary", "count"])
    g = placed_df.groupby(group_col)[REGRESSION_TARGET].agg(
        mean_salary="mean", median_salary="median", count="count"
    ).reset_index()
    g["mean_salary"] = g["mean_salary"].round(0)
    g["median_salary"] = g["median_salary"].round(0)
    return g.sort_values("mean_salary", ascending=False)


# ---------------------------------------------------------------------------
# 4. CORRELATIONS
# ---------------------------------------------------------------------------

def correlation_matrix(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Pearson correlation among the given numeric columns.
    Rows with missing values in ANY of the requested columns are dropped
    (this is what makes the salary-inclusive matrix automatically use only
    placed students, since salary is NaN for everyone else)."""
    valid_cols = [c for c in columns if c in df.columns]
    subset = df[valid_cols].apply(pd.to_numeric, errors="coerce").dropna()
    if subset.empty or len(valid_cols) < 2:
        return pd.DataFrame()
    return subset.corr().round(2)


# ---------------------------------------------------------------------------
# 5. AUTOMATIC "DATA INSIGHT" CARDS (deterministic, no LLM)
# ---------------------------------------------------------------------------

@dataclass
class Insight:
    fact: str
    interpretation: str


def insight_group_comparison(df: pd.DataFrame, group_col: str, group_label: str) -> Insight:
    """Builds a FACT/INTERPRETATION pair comparing placement rate across the
    two (or more) values of a categorical column."""
    rates = placement_rate_by_group(df, group_col)
    if rates.empty or len(rates) < 2:
        return Insight(
            fact=f"Not enough groups in '{group_label}' to compare.",
            interpretation="No interpretation possible.",
        )
    top = rates.iloc[0]
    bottom = rates.iloc[-1]
    fact = (
        f"Among students grouped by {group_label}, '{top[group_col]}' shows the highest "
        f"observed placement rate at {top['placement_rate_pct']:.1f}% (n={int(top['total'])}), "
        f"while '{bottom[group_col]}' shows the lowest at {bottom['placement_rate_pct']:.1f}% "
        f"(n={int(bottom['total'])})."
    )
    interpretation = (
        f"This is an observed association in this dataset, not proof that {group_label} "
        f"causes a placement outcome. Group sizes are small, so treat the gap as a pattern "
        f"worth investigating rather than a settled conclusion."
    )
    return Insight(fact=fact, interpretation=interpretation)


def insight_correlation(df: pd.DataFrame, col_a: str, col_b: str, label_a: str, label_b: str) -> Insight:
    corr_df = correlation_matrix(df, [col_a, col_b])
    if corr_df.empty:
        return Insight(
            fact=f"Not enough overlapping data to correlate {label_a} and {label_b}.",
            interpretation="No interpretation possible.",
        )
    r = corr_df.loc[col_a, col_b]
    strength = (
        "a strong" if abs(r) >= 0.7 else
        "a moderate" if abs(r) >= 0.4 else
        "a weak" if abs(r) >= 0.2 else
        "almost no"
    )
    direction = "positive" if r >= 0 else "negative"
    fact = f"The Pearson correlation between {label_a} and {label_b} is {r:.2f}."
    interpretation = (
        f"This indicates {strength} {direction} linear relationship in the observed data. "
        f"Correlation does not establish that one causes the other."
    )
    return Insight(fact=fact, interpretation=interpretation)


# ---------------------------------------------------------------------------
# 6. "FROM DATA TO ACTION" BUSINESS INSIGHTS
# ---------------------------------------------------------------------------

@dataclass
class BusinessInsight:
    fact: str
    insight: str
    opportunity: str
    action: str


def generate_business_insights(df: pd.DataFrame) -> list[BusinessInsight]:
    insights: list[BusinessInsight] = []

    # -- Work experience vs placement rate -----------------------------
    we_rates = placement_rate_by_group(df, "workex")
    if len(we_rates) >= 2:
        yes_row = we_rates[we_rates["workex"].str.lower() == "yes"]
        no_row = we_rates[we_rates["workex"].str.lower() == "no"]
        if not yes_row.empty and not no_row.empty:
            yes_rate = yes_row.iloc[0]["placement_rate_pct"]
            no_rate = no_row.iloc[0]["placement_rate_pct"]
            gap = round(yes_rate - no_rate, 1)
            insights.append(BusinessInsight(
                fact=(f"Students with work experience show a {yes_rate:.1f}% observed placement "
                      f"rate vs {no_rate:.1f}% for those without (a {gap:+.1f} point difference)."),
                insight=("The dataset shows an observed difference in placement rate associated "
                         "with prior work experience; this is not proof of a causal effect."),
                opportunity=("Institutions could investigate whether structured internships or "
                              "practical exposure programs correlate with better outcomes for their "
                              "own student population."),
                action=("Consider offering or promoting internship opportunities earlier in the "
                        "academic program, and track whether placement outcomes change."),
            ))

    # -- Specialisation vs average salary (placed only) -----------------
    spec_salary = salary_by_group(df, "specialisation")
    if len(spec_salary) >= 2:
        top = spec_salary.iloc[0]
        bottom = spec_salary.iloc[-1]
        insights.append(BusinessInsight(
            fact=(f"Among placed students, '{top['specialisation']}' has the highest average "
                  f"salary (INR {top['mean_salary']:,.0f}, n={int(top['count'])}), while "
                  f"'{bottom['specialisation']}' has the lowest (INR {bottom['mean_salary']:,.0f}, "
                  f"n={int(bottom['count'])})."),
            insight=("This reflects observed salary outcomes for this cohort's placed students "
                      "only; sample sizes per specialisation are small."),
            opportunity=("Career counseling could reference this pattern when discussing "
                          "specialisation trade-offs, alongside the student's own interests and "
                          "aptitude."),
            action=("Present this comparison to students as context (not a guarantee) during "
                    "specialisation selection advising."),
        ))

    # -- Degree type vs placement rate -----------------------------------
    deg_rates = placement_rate_by_group(df, "degree_t")
    if len(deg_rates) >= 2:
        top = deg_rates.iloc[0]
        bottom = deg_rates.iloc[-1]
        insights.append(BusinessInsight(
            fact=(f"'{top['degree_t']}' graduates show the highest observed placement rate "
                  f"({top['placement_rate_pct']:.1f}%, n={int(top['total'])}); "
                  f"'{bottom['degree_t']}' graduates show the lowest "
                  f"({bottom['placement_rate_pct']:.1f}%, n={int(bottom['total'])})."),
            insight=("An observed association between degree field and placement rate exists in "
                      "this dataset; it may reflect industry demand, cohort size, or other "
                      "unmeasured factors."),
            opportunity=("Institutions could examine whether curriculum or industry-partnership "
                          "differences across degree fields explain part of the gap."),
            action=("Flag lower-placement-rate degree fields for additional placement-support "
                    "resources, pending further investigation."),
        ))

    return insights


# ---------------------------------------------------------------------------
# 7. Manual test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from utils.data_processing import load_data, clean_data

    raw = load_data()
    clean, _ = clean_data(raw)

    print("=== KPI SUMMARY ===")
    print(kpi_summary(clean))

    print("\n=== PLACEMENT RATE BY WORKEX ===")
    print(placement_rate_by_group(clean, "workex"))

    print("\n=== SALARY STATS ===")
    print(salary_stats(clean))

    print("\n=== SALARY BY SPECIALISATION ===")
    print(salary_by_group(clean, "specialisation"))

    print("\n=== CORRELATION (academic) ===")
    print(correlation_matrix(clean, NUMERIC_FEATURES))

    print("\n=== CORRELATION (with salary, placed only) ===")
    print(correlation_matrix(clean, NUMERIC_FEATURES + [REGRESSION_TARGET]))

    print("\n=== BUSINESS INSIGHTS ===")
    for bi in generate_business_insights(clean):
        print(bi)
        print()
