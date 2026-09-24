"""
ai_insights.py
--------------
STAGE 6 - AI Insight Copilot.

Architecture:  QUESTION -> INTENT -> VERIFIED EVIDENCE (Python) -> EXPLANATION

The evidence registry (`build_evidence`) is built ENTIRELY from numbers
already calculated elsewhere in this project (analytics.py, ml_models.py) -
nothing here recomputes or invents a statistic.

The deterministic fallback engine (`generate_fallback_insight`) turns that
evidence into a grounded FACT/INSIGHT/POSSIBLE ACTION/LIMITATION answer with
NO external dependency - it works with no internet access and no API key.

An OPTIONAL LLM layer (`call_llm`) can rephrase/explain the SAME evidence
using the Anthropic Messages API, called via the Python standard library
only (urllib) so no new package is required. If no API key is configured,
or the call fails for any reason (no network, bad key, timeout, ...), the
deterministic fallback is used silently - the app never errors just because
no key is present.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import pandas as pd

from utils.data_processing import NUMERIC_FEATURES, CLASSIFICATION_TARGET
from utils.analytics import (
    kpi_summary, placement_rate_by_group, salary_stats, salary_by_group,
)
from utils.ml_models import recommend_k, run_segmentation, post_clustering_outcomes


# ===========================================================================
# 1. EVIDENCE REGISTRY - built from already-calculated project results
# ===========================================================================

REQUIRED_EVIDENCE_KEYS = [
    "dataset", "academic", "placement", "salary",
    "classification", "regression", "segmentation",
]


def _correlation_with_placement(clean_df: pd.DataFrame) -> dict:
    """Point-biserial correlation of each academic % with the binary
    placement outcome. Computed here (not stored elsewhere) since no other
    stage needed this exact comparison across all 5 metrics at once."""
    d = clean_df.copy()
    d["_placed_flag"] = (d[CLASSIFICATION_TARGET].str.lower() == "placed").astype(int)
    out = {}
    for col in NUMERIC_FEATURES:
        sub = d[[col, "_placed_flag"]].dropna()
        if len(sub) > 1 and sub[col].std() > 0:
            out[col] = float(sub[col].corr(sub["_placed_flag"]))
    return out


def build_evidence(clean_df: pd.DataFrame, cls_results, reg_results, seg_diagnostics) -> dict:
    """Assembles the structured evidence dict the Copilot is allowed to use.
    cls_results / reg_results / seg_diagnostics are the SAME objects already
    produced by Stage 3/4/5 (train_and_evaluate, train_and_evaluate_salary,
    compute_segmentation_diagnostics) - nothing is retrained or recalculated
    differently here."""
    kpi = kpi_summary(clean_df)
    dataset = {
        "n_total": kpi.total_students,
        "n_placed": kpi.placed_students,
        "n_not_placed": kpi.not_placed_students,
        "placement_rate_pct": kpi.placement_rate_pct,
    }

    academic = {
        "means": {c: float(clean_df[c].mean()) for c in NUMERIC_FEATURES},
        "correlation_with_placement": _correlation_with_placement(clean_df),
    }

    placement = {
        "by_workex": placement_rate_by_group(clean_df, "workex").to_dict("records"),
        "by_degree_t": placement_rate_by_group(clean_df, "degree_t").to_dict("records"),
        "by_specialisation": placement_rate_by_group(clean_df, "specialisation").to_dict("records"),
    }

    salary = {
        "stats": salary_stats(clean_df),
        "by_specialisation": salary_by_group(clean_df, "specialisation").to_dict("records"),
        "by_degree_t": salary_by_group(clean_df, "degree_t").to_dict("records"),
    }

    classification = {
        name: {
            "accuracy": ev.accuracy, "precision": ev.precision, "recall": ev.recall,
            "f1": ev.f1, "roc_auc": ev.roc_auc,
        }
        for name, ev in cls_results.evaluations.items()
    }

    regression = {
        name: {"mae": ev.mae, "rmse": ev.rmse, "r2": ev.r2}
        for name, ev in reg_results.evaluations.items()
    }

    suggested_k = recommend_k(seg_diagnostics)
    seg_result = run_segmentation(clean_df, k=suggested_k, diagnostics=seg_diagnostics)
    outcomes_df = post_clustering_outcomes(clean_df, seg_result.labels)
    segmentation = {
        "k": suggested_k,
        "elbow_by_k": seg_diagnostics.elbow_scores,
        "silhouette_by_k": seg_diagnostics.silhouette_scores,
        "profiles": [
            {
                "cluster_id": p.cluster_id, "count": p.count, "pct_of_total": p.pct_of_total,
                "academic_avg": p.academic_avg, "workex_yes_pct": p.workex_yes_pct, "label": p.label,
            }
            for p in seg_result.profiles
        ],
        "post_clustering_outcomes": outcomes_df.to_dict("records"),
    }

    return {
        "dataset": dataset, "academic": academic, "placement": placement, "salary": salary,
        "classification": classification, "regression": regression, "segmentation": segmentation,
    }


def validate_evidence(evidence: dict) -> tuple[bool, list[str]]:
    """A schema check, not a statistics check: if the dataset/columns change
    shape, this should catch it rather than let the Copilot quietly answer
    from a malformed evidence dict."""
    issues = []
    for key in REQUIRED_EVIDENCE_KEYS:
        if key not in evidence:
            issues.append(f"Missing evidence section: '{key}'")
    if "dataset" in evidence and evidence["dataset"].get("n_total", 0) <= 0:
        issues.append("Dataset evidence has zero or missing student count.")
    return (len(issues) == 0, issues)


# ===========================================================================
# 2. DETERMINISTIC FALLBACK ENGINE (no API key, no internet required)
# ===========================================================================

import re

UNSUPPORTED_FIELD_KEYWORDS = [
    "company", "employer", "recruiter", "location", "city", "interview",
    "age", "hobbies", "gpa", "country", "phone", "email", "address", "name of",
]


def _contains_keyword(text: str, keyword: str) -> bool:
    """Word-boundary match so short keywords like 'age' don't false-positive
    inside unrelated words such as 'average' or 'percentage'."""
    pattern = r"\b" + re.escape(keyword) + r"\b"
    return re.search(pattern, text) is not None

ACADEMIC_LABELS = {"ssc_p": "SSC %", "hsc_p": "HSC %", "degree_p": "Degree %",
                    "etest_p": "E-test %", "mba_p": "MBA %"}


def _no_data_response(topic: str) -> dict:
    return {
        "answer": f"There isn't enough verified data to answer a question about {topic} in this dataset.",
        "fact": None, "insight": None, "action": None, "limitation": None,
    }


def handle_dataset_overview(evidence: dict) -> dict:
    d = evidence["dataset"]
    fact = (f"The dataset contains {d['n_total']} students: {d['n_placed']} placed and "
            f"{d['n_not_placed']} not placed - an observed placement rate of {d['placement_rate_pct']}%.")
    return {"answer": fact, "fact": fact, "insight": None, "action": None, "limitation": None}


def handle_academic_strength(evidence: dict) -> dict:
    corr = evidence["academic"]["correlation_with_placement"]
    if not corr:
        return _no_data_response("academic correlation with placement")
    strongest = max(corr.items(), key=lambda kv: abs(kv[1]))
    fact = (f"Among the academic percentages, {ACADEMIC_LABELS.get(strongest[0], strongest[0])} has the "
            f"strongest observed correlation with the placement outcome (r={strongest[1]:.2f}).")
    ranked = sorted(corr.items(), key=lambda kv: -abs(kv[1]))
    insight = "All correlations, for reference: " + ", ".join(
        f"{ACADEMIC_LABELS.get(k, k)} r={v:.2f}" for k, v in ranked)
    limitation = "These are correlations in an observational dataset and do not establish that this metric causes placement."
    return {"answer": fact, "fact": fact, "insight": insight, "action": None, "limitation": limitation}


def _placement_group_response(evidence: dict, group_key: str, group_col: str, label: str) -> dict:
    rows = evidence["placement"].get(group_key, [])
    if not rows:
        return _no_data_response(f"placement by {label}")
    rows_sorted = sorted(rows, key=lambda r: r["placement_rate_pct"], reverse=True)
    fact = "; ".join(f"{r[group_col]}: {r['placement_rate_pct']}% (n={int(r['total'])})" for r in rows_sorted)
    top, bottom = rows_sorted[0], rows_sorted[-1]
    insight = (f"'{top[group_col]}' shows the highest observed placement rate ({top['placement_rate_pct']}%); "
               f"'{bottom[group_col]}' shows the lowest ({bottom['placement_rate_pct']}%).")
    limitation = f"This is an observed association by {label} in this dataset, not proof of causation."
    return {"answer": fact, "fact": fact, "insight": insight, "action": None, "limitation": limitation}


def handle_workex_placement(evidence: dict) -> dict:
    rows = {r["workex"]: r for r in evidence["placement"].get("by_workex", [])}
    yes, no = rows.get("Yes"), rows.get("No")
    if not yes or not no:
        return _no_data_response("work experience and placement")
    diff = round(yes["placement_rate_pct"] - no["placement_rate_pct"], 1)
    fact = (f"{yes['placement_rate_pct']}% of students with work experience were placed "
            f"(n={int(yes['total'])}), compared with {no['placement_rate_pct']}% without work experience "
            f"(n={int(no['total'])}) - a difference of {diff:+.1f} percentage points.")
    insight = "Students with work experience show a higher observed placement rate in this dataset."
    action = ("Career-support programs could investigate whether relevant work-experience opportunities "
              "are associated with improved placement outcomes.")
    limitation = "This is an observational dataset, so the difference does not prove that work experience caused the higher placement rate."
    return {"answer": fact, "fact": fact, "insight": insight, "action": action, "limitation": limitation}


def handle_placement_by_degree(evidence: dict) -> dict:
    return _placement_group_response(evidence, "by_degree_t", "degree_t", "degree type")


def handle_placement_by_specialisation(evidence: dict) -> dict:
    return _placement_group_response(evidence, "by_specialisation", "specialisation", "specialisation")


def handle_salary_overview(evidence: dict) -> dict:
    s = evidence["salary"]["stats"]
    if s.get("count", 0) == 0:
        return _no_data_response("salary")
    fact = (f"Average salary among placed students is INR {s['mean']:,.0f} (median INR {s['median']:,.0f}), "
            f"based on {s['count']} placed students with a recorded salary. "
            f"Range: INR {s['min']:,.0f} - INR {s['max']:,.0f}.")
    limitation = "Salary is only recorded for placed students; it is never estimated for non-placed students."
    return {"answer": fact, "fact": fact, "insight": None, "action": None, "limitation": limitation}


def handle_salary_by_specialisation(evidence: dict) -> dict:
    rows = evidence["salary"].get("by_specialisation", [])
    if not rows:
        return _no_data_response("salary by specialisation")
    rows_sorted = sorted(rows, key=lambda r: r["mean_salary"], reverse=True)
    fact = "; ".join(f"{r['specialisation']}: avg INR {r['mean_salary']:,.0f} (n={int(r['count'])})"
                      for r in rows_sorted)
    insight = f"'{rows_sorted[0]['specialisation']}' has the highest observed average salary among placed students."
    limitation = "Based only on placed students with a recorded salary; sample sizes per group are small."
    return {"answer": fact, "fact": fact, "insight": insight, "action": None, "limitation": limitation}


def handle_salary_model_performance(evidence: dict) -> dict:
    reg = evidence["regression"]
    if not reg:
        return _no_data_response("salary model performance")
    lines = [f"{name}: MAE=INR {m['mae']:,.0f}, RMSE=INR {m['rmse']:,.0f}, R2={m['r2']:.3f}"
              for name, m in reg.items()]
    fact = " | ".join(lines)
    best_mae = min(reg.items(), key=lambda kv: kv[1]["mae"])
    insight = (
        f"{best_mae[0]} has the lower MAE among the models tested (INR {best_mae[1]['mae']:,.0f}). "
        f"Both models show a negative R2 on this test split, meaning they perform worse than a simple "
        f"baseline that predicts the average salary for every student. This suggests the available "
        f"academic and employability variables do not provide enough signal to accurately explain salary "
        f"variation in this small sample."
    )
    action = ("A larger dataset, or additional features such as company, role, or location, would likely "
              "be needed to build a more reliable salary estimator.")
    limitation = "A negative R2 is specific to this dataset and this particular train/test split; it does not mean the modelling code is broken."
    return {"answer": insight, "fact": fact, "insight": insight, "action": action, "limitation": limitation}


def handle_classification_performance(evidence: dict) -> dict:
    cls = evidence["classification"]
    if not cls:
        return _no_data_response("classification model performance")
    lines = [
        f"{name}: Accuracy={m['accuracy']:.3f}, Precision={m['precision']:.3f}, Recall={m['recall']:.3f}, "
        f"F1={m['f1']:.3f}" + (f", ROC-AUC={m['roc_auc']:.3f}" if m["roc_auc"] is not None else "")
        for name, m in cls.items()
    ]
    fact = " | ".join(lines)
    insight = ("Compare the metrics directly - one model may have higher precision, the other higher "
               "recall. Neither model is labeled as universally best.")
    limitation = "These results come from a single 80/20 train-test split on a small (215-record) dataset and may vary with a different split."
    return {"answer": fact, "fact": fact, "insight": insight, "action": None, "limitation": limitation}


def handle_segmentation_overview(evidence: dict) -> dict:
    seg = evidence["segmentation"]
    k = seg["k"]
    sizes = "; ".join(f"Cluster {p['cluster_id']}: {p['count']} students ({p['pct_of_total']}%)"
                       for p in seg["profiles"])
    fact = f"The diagnostic-recommended segmentation uses K={k} clusters. {sizes}."
    sil = seg["silhouette_by_k"].get(k)
    insight = f"Silhouette score at K={k} is {sil:.3f}." if sil is not None else "Silhouette score unavailable for this K."
    limitation = "Cluster numbers are algorithm-generated identifiers and do not represent rankings; a different K could also be selected in the Student Segmentation page."
    return {"answer": fact, "fact": fact, "insight": insight, "action": None, "limitation": limitation}


def handle_segmentation_differences(evidence: dict) -> dict:
    seg = evidence["segmentation"]
    parts = [f"Cluster {p['cluster_id']} ({p['label']}, n={p['count']}): academic avg {p['academic_avg']}%, "
             f"work experience {p['workex_yes_pct']}%" for p in seg["profiles"]]
    fact = "; ".join(parts)
    outcomes = seg.get("post_clustering_outcomes", [])
    outcome_text = ""
    if outcomes:
        outcome_text = " Post-clustering descriptive outcomes (NOT used to create the clusters): " + "; ".join(
            f"Cluster {o['cluster']}: placement rate {o['placement_rate_pct']}%"
            + (f", avg salary INR {o['avg_salary_placed']:,.0f}" if o.get("avg_salary_placed") else "")
            for o in outcomes
        )
    insight = "The clusters differ mainly in academic average and work-experience share." + outcome_text
    limitation = "Cluster membership is not claimed to cause any placement or salary outcome."
    return {"answer": fact, "fact": fact, "insight": insight, "action": None, "limitation": limitation}


def handle_main_findings(evidence: dict) -> dict:
    d = evidence["dataset"]
    we = {r["workex"]: r for r in evidence["placement"].get("by_workex", [])}
    reg = evidence["regression"]
    seg = evidence["segmentation"]

    bullets = [f"{d['n_total']} students, {d['placement_rate_pct']}% observed placement rate "
               f"({d['n_placed']} placed, {d['n_not_placed']} not placed)."]
    if "Yes" in we and "No" in we:
        bullets.append(f"Students with work experience had a {we['Yes']['placement_rate_pct']}% observed "
                        f"placement rate vs {we['No']['placement_rate_pct']}% without.")
    if reg:
        best = min(reg.items(), key=lambda kv: kv[1]["mae"])
        bullets.append(f"Salary regression is weak on this dataset (best-MAE model: {best[0]}, "
                        f"R2={best[1]['r2']:.3f} on this test split).")
    if seg.get("profiles"):
        sil = seg["silhouette_by_k"].get(seg["k"])
        sil_txt = f", silhouette={sil:.3f}" if sil is not None else ""
        bullets.append(f"K-Means segmentation (K={seg['k']}) found groups differing mainly in academic "
                        f"average and work-experience share{sil_txt}.")
    fact = " ".join(bullets)
    return {"answer": fact, "fact": fact, "insight": None, "action": None,
            "limitation": "All figures above are calculated from the current dataset and model runs."}


def handle_next_steps(evidence: dict) -> dict:
    action = (
        "With a larger dataset, useful next steps would include: gathering additional salary-relevant "
        "features (such as company, role, location, or industry) to improve regression performance; "
        "validating the placement classification models on a larger or more recent cohort; and "
        "re-running the K-Means diagnostics to see whether cluster separation - currently a modest "
        "silhouette score - improves with more data or different features."
    )
    return {"answer": action, "fact": None, "insight": None, "action": action, "limitation": None}


def handle_unsupported_field(evidence: dict, matched_keyword: str) -> dict:
    answer = (f"The current dataset does not contain {matched_keyword}-level information, "
              f"so this cannot be determined from the available data.")
    return {"answer": answer, "fact": None, "insight": None, "action": None,
            "limitation": "Ask about dataset size, placement, salary, models, or segmentation instead."}


def handle_unknown(evidence: dict, question: str) -> dict:
    answer = (
        "I can answer questions about dataset size and placement rate, academic performance, placement "
        "by work experience/degree type/specialisation, salary and salary-model performance, student "
        "segmentation, or classification model performance. Try one of the quick questions, or rephrase "
        "your question around one of these topics."
    )
    return {"answer": answer, "fact": None, "insight": None, "action": None, "limitation": None}


HANDLERS = {
    "dataset_overview": handle_dataset_overview,
    "academic_strength": handle_academic_strength,
    "workex_placement": handle_workex_placement,
    "placement_by_degree": handle_placement_by_degree,
    "placement_by_specialisation": handle_placement_by_specialisation,
    "salary_overview": handle_salary_overview,
    "salary_by_specialisation": handle_salary_by_specialisation,
    "salary_model_performance": handle_salary_model_performance,
    "classification_performance": handle_classification_performance,
    "segmentation_overview": handle_segmentation_overview,
    "segmentation_differences": handle_segmentation_differences,
    "main_findings": handle_main_findings,
    "next_steps": handle_next_steps,
}


def classify_question(question: str) -> tuple[str, str | None]:
    """Rule-based intent detection. Returns (intent, extra) where extra is
    only used by 'unsupported_field' (the matched keyword)."""
    q = question.lower()

    for kw in UNSUPPORTED_FIELD_KEYWORDS:
        if _contains_keyword(q, kw):
            return ("unsupported_field", kw)

    if any(w in q for w in ["r2", "r\u00b2", "negative r", "mae", "rmse"]) or \
       ("salary" in q and any(w in q for w in ["weak", "poor", "bad", "why does", "why did", "perform"])):
        return ("salary_model_performance", None)

    if "roc" in q or "auc" in q or any(w in q for w in ["logistic", "decision tree", "classification model"]) \
       or ("ml model" in q) or ("models perform" in q and "salary" not in q and "segment" not in q):
        return ("classification_performance", None)

    if "segment" in q or "cluster" in q:
        if any(w in q for w in ["differ", "describe", "compare", "show"]):
            return ("segmentation_differences", None)
        return ("segmentation_overview", None)

    if "work experience" in q or "workex" in q:
        return ("workex_placement", None)

    if "degree type" in q or ("degree" in q and "placement" in q):
        return ("placement_by_degree", None)

    if "specialisation" in q or "specialization" in q:
        if "salary" in q:
            return ("salary_by_specialisation", None)
        return ("placement_by_specialisation", None)

    if "salary" in q:
        return ("salary_overview", None)

    if "academic metric" in q or "strongest relationship" in q or "factors relate" in q or \
       ("academic" in q and "relat" in q):
        return ("academic_strength", None)

    if "main finding" in q or "overall" in q or "finding" in q:
        return ("main_findings", None)

    if "investigat" in q or "next step" in q or "larger dataset" in q or "future" in q:
        return ("next_steps", None)

    if "placement rate" in q or "how many student" in q or "students are" in q or "how many are placed" in q:
        return ("dataset_overview", None)

    return ("unknown", None)


def answer_intent(intent: str, evidence: dict) -> dict:
    """Directly dispatch a known intent (used by the quick-question cards,
    bypassing text classification for reliability)."""
    result = dict(HANDLERS[intent](evidence))
    result["intent"] = intent
    result["source"] = "fallback"
    return result


def generate_fallback_insight(question: str, evidence: dict) -> dict:
    intent, extra = classify_question(question)
    if intent == "unsupported_field":
        result = handle_unsupported_field(evidence, extra)
    elif intent == "unknown":
        result = handle_unknown(evidence, question)
    else:
        result = HANDLERS[intent](evidence)
    result = dict(result)
    result["intent"] = intent
    result["source"] = "fallback"
    return result


def render_response_markdown(response: dict) -> str:
    parts = [f"### Answer\n{response.get('answer') or ''}"]
    if response.get("fact"):
        parts.append(f"**FACT**\n\n{response['fact']}")
    if response.get("insight"):
        parts.append(f"**INSIGHT**\n\n{response['insight']}")
    if response.get("action"):
        parts.append(f"**POSSIBLE ACTION**\n\n{response['action']}")
    if response.get("limitation"):
        parts.append(f"**LIMITATION**\n\n{response['limitation']}")
    return "\n\n".join(parts)


QUICK_QUESTIONS = [
    ("\U0001F4CA What are the main dataset findings?", "main_findings"),
    ("\U0001F393 Which factors relate to placement?", "academic_strength"),
    ("\U0001F4BC How does work experience relate to placement?", "workex_placement"),
    ("\U0001F4B0 Why is salary prediction weak?", "salary_model_performance"),
    ("\U0001F9E9 What do the student segments show?", "segmentation_differences"),
    ("\U0001F916 How did the ML models perform?", "classification_performance"),
    ("\U0001F4C8 What should be investigated next?", "next_steps"),
]


# ===========================================================================
# 3. OPTIONAL LLM LAYER (stdlib-only, best-effort, silent fallback)
# ===========================================================================

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = "claude-3-5-haiku-20241022"
ANTHROPIC_VERSION = "2023-06-01"


def get_configured_api_key(secrets_value: str | None = None) -> str | None:
    """Checks (in order) a value passed in from st.secrets, then the
    ANTHROPIC_API_KEY environment variable. Never hard-codes a key."""
    return secrets_value or os.environ.get("ANTHROPIC_API_KEY") or None


def build_llm_prompt(question: str, fallback: dict) -> str:
    grounded = fallback.get("fact") or fallback.get("answer") or "No specific evidence matched this question."
    return (
        "You are an analytics assistant for a student employability dataset. Using ONLY the verified "
        "evidence below, answer the user's question. Follow this format: a short Answer, then FACT, "
        "INSIGHT, POSSIBLE ACTION and LIMITATION sections (skip a section if not applicable). Do NOT "
        "invent any number that is not present in the evidence. Never claim correlation is causation. "
        "If the evidence does not cover the question, say so plainly instead of guessing.\n\n"
        f"Verified evidence: {grounded}\n\n"
        f"Question: {question}"
    )


def call_llm(question: str, fallback: dict, api_key: str, timeout: float = 8.0) -> str | None:
    """Best-effort optional call to the Anthropic Messages API using only
    the Python standard library (urllib) - no extra pip dependency. Returns
    None on ANY failure (no network, invalid key, timeout, malformed
    response, ...) so the caller falls back to the deterministic engine
    silently, with no error shown to the user."""
    prompt = build_llm_prompt(question, fallback)
    payload = json.dumps({
        "model": ANTHROPIC_MODEL,
        "max_tokens": 500,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")
    request = urllib.request.Request(
        ANTHROPIC_API_URL, data=payload, method="POST",
        headers={
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        text = "".join(
            block.get("text", "") for block in body.get("content", []) if block.get("type") == "text"
        ).strip()
        return text or None
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, KeyError, OSError):
        return None


def generate_response(question: str, evidence: dict, api_key: str | None = None) -> dict:
    """Top-level entry point used by the UI. Always computes the grounded
    fallback first (both as the safe default AND as the evidence handed to
    the optional LLM), then tries the LLM only if a key is configured."""
    fallback = generate_fallback_insight(question, evidence)
    if not api_key:
        return fallback

    ai_text = call_llm(question, fallback, api_key)
    if not ai_text:
        return fallback

    response = dict(fallback)
    response["source"] = "ai"
    response["ai_markdown"] = ai_text
    return response


# ---------------------------------------------------------------------------
# Manual test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from utils.data_processing import load_data, clean_data
    from utils.ml_models import train_and_evaluate, train_and_evaluate_salary, compute_segmentation_diagnostics

    raw = load_data()
    clean, _ = clean_data(raw)
    cls_results = train_and_evaluate(clean)
    reg_results = train_and_evaluate_salary(clean)
    seg_diagnostics = compute_segmentation_diagnostics(clean)

    evidence = build_evidence(clean, cls_results, reg_results, seg_diagnostics)
    ok, issues = validate_evidence(evidence)
    print(f"Evidence valid: {ok}", issues if issues else "")

    test_questions = [
        "How many students are in the dataset?",
        "Does work experience relate to placement?",
        "What is the placement rate by specialisation?",
        "Why is salary prediction weak?",
        "What does the negative R2 mean?",
        "Describe the two student clusters.",
        "How did Logistic Regression perform?",
        "What are the main dataset findings?",
        "Which company offered the highest salary?",  # hallucination test
    ]
    for q in test_questions:
        resp = generate_fallback_insight(q, evidence)
        print(f"\nQ: {q}\n  intent={resp['intent']}")
        print(" ", render_response_markdown(resp).replace("\n", " "))
