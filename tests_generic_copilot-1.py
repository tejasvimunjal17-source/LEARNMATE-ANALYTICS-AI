"""
Focused tests for the GENERIC AI Insight Copilot (not part of the shipped app).

Streamlit/Plotly are stubbed exactly like tests_stage2_smoke.py, and the
network is mocked: urllib.request.urlopen is replaced by a fake OpenRouter
that records the request. That lets us verify, with no network and no real
key, that:
  * a NON-benchmark dataset reaches OpenRouter when a key is configured,
  * the request targets the configured base URL + model,
  * the prompt carries evidence from the ACTIVE dataframe only (no benchmark
    column names / numbers), withholds PII values, and never contains the key,
  * the UI reports "AI-generated • Grounded in active dataset" on success and
    "Fallback engine • Deterministic" when there is no key or the call fails,
  * the benchmark dataset behaviour is unchanged.
It cannot verify what the real GPT-4o-mini says - only what it is sent.
"""
import io
import json
import runpy
import sys
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

import tests_stage2_smoke as base

COPILOT = "\U0001F9E0 AI Insight Copilot"
FAKE_KEY = "sk-or-v1-TEST-KEY-DO-NOT-LEAK-123456"
results: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    results.append((name, bool(cond), detail))
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"  [{detail}]" if detail and not cond else ""))


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def run_copilot(df_bytes: bytes | None, filename: str, question: str, api_key: str | None,
                llm: str = "ok"):
    """Runs app.py's Copilot page once with the question 'submitted'.
    llm: 'ok' -> fake OpenRouter answers; 'fail' -> network error; 'bad' -> malformed JSON."""
    for mod in list(sys.modules):
        if mod.startswith("utils") or mod == "app":
            del sys.modules[mod]
    upload = base.FakeUploadedFile(df_bytes, filename) if df_bytes is not None else None
    st = base.make_streamlit_stub(COPILOT, uploaded_file=upload)
    st.form_submit_button.return_value = True
    st.text_input.return_value = question
    st.secrets.get.side_effect = lambda k, d=None: api_key if k == "OPENROUTER_API_KEY" else None
    plotly_mod, express_mod, go_mod = base.make_plotly_stub()
    sys.modules.update({"streamlit": st, "plotly": plotly_mod, "plotly.express": express_mod,
                        "plotly.graph_objects": go_mod})

    captured: list[dict] = []
    real_urlopen = urllib.request.urlopen

    def fake_urlopen(req, timeout=None):
        captured.append({"url": req.full_url, "headers": dict(req.header_items()),
                         "body": json.loads(req.data.decode("utf-8")), "timeout": timeout})
        if llm == "fail":
            raise urllib.error.URLError("simulated network failure")
        if llm == "bad":
            return _Resp(b"<html>not json</html>")
        return _Resp(json.dumps({"choices": [{"message": {"content": "FAKE-LLM-ANSWER"}}]}).encode())

    urllib.request.urlopen = fake_urlopen
    try:
        runpy.run_path("app.py", run_name="__main__")
    finally:
        urllib.request.urlopen = real_urlopen
    captions = [str(c.args[0]) for c in st.caption.call_args_list if c.args]
    markdown = [str(c.args[0]) for c in st.markdown.call_args_list if c.args]
    return {"captions": captions, "markdown": markdown, "requests": captured, "st": st}


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------
def make_students_xlsx() -> tuple[bytes, pd.DataFrame]:
    rng = np.random.default_rng(7)
    n = 500
    df = pd.DataFrame({
        "Student ID": [f"S{i:04d}" for i in range(n)],
        "Student Name": [f"Zed Quillfeather {i}" for i in range(n)],
        "Department": rng.choice(["CSE", "ECE", "MECH", "CIVIL", "EEE"], n),
        "Year": rng.choice([2021, 2022, 2023, 2024], n),
        "City": rng.choice(["Pune", "Delhi", "Chennai", "Mumbai", "Jaipur", "Kochi"], n),
        "CGPA": np.round(rng.normal(7.5, 1.0, n), 2),
        "Email": [f"zed{i}@college.edu" for i in range(n)],
    })
    df.loc[::40, "CGPA"] = np.nan
    buf = io.BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    return buf.getvalue(), df


def make_sales_csv() -> tuple[bytes, pd.DataFrame]:
    rng = np.random.default_rng(11)
    n = 400
    df = pd.DataFrame({
        "Product": rng.choice([f"Widget{c}" for c in "ABCDEFGHIJKLMNOPQ"], n),
        "Category": rng.choice(["Hardware", "Software", "Services"], n),
        "Region": rng.choice(["North", "South", "East", "West"], n),
        "Units Sold": rng.integers(1, 50, n),
        "Revenue": np.round(rng.uniform(100, 5000, n), 2),
        "Date": pd.date_range("2022-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
    })
    return df.to_csv(index=False).encode("utf-8"), df


def make_unusual_csv() -> tuple[bytes, pd.DataFrame]:
    df = pd.DataFrame({
        "a b!": [1, 2, None, 4, 5, 6], "€ col": ["x", "y", "x", None, "y", "x"],
        "mixed": [1, "two", 3.0, None, 5, "six"], "empty": [None] * 6,
        "nums_txt": ["1,000", "2", "3", "4", "5", "6"],
        "when": ["2020-01-01", "2020-02-01", "2020-03-01", "2021-01-01", "2021-02-01", "2021-03-01"],
    })
    return df.to_csv(index=False).encode("utf-8"), df


BENCHMARK_ONLY_TOKENS = ["ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p", "campus_placement", "workex"]


def prompt_text(req: dict) -> str:
    return "\n".join(m["content"] for m in req["body"]["messages"])


def main() -> None:
    students_bytes, students = make_students_xlsx()
    sales_bytes, sales = make_sales_csv()
    odd_bytes, odd = make_unusual_csv()

    print("=== A. Student XLSX (non-benchmark) + key configured -> OpenRouter ===")
    qs = [
        "Give me a concise overview of this dataset.",
        "What are the most important patterns in this dataset?",
        "Which category/group has the highest value?",
        "What are the main data-quality issues?",
        "Which companies hired these students?",
    ]
    for q in qs:
        r = run_copilot(students_bytes, "student_data_500_rows.xlsx", q, FAKE_KEY)
        reqs = r["requests"]
        check(f"[students] OpenRouter called for: {q[:48]}", len(reqs) == 1)
        if not reqs:
            continue
        req, txt = reqs[0], prompt_text(reqs[0])
        check("  url is OPENROUTER_BASE_URL + /chat/completions", req["url"] == "https://openrouter.ai/api/v1/chat/completions", req["url"])
        check("  model is openai/gpt-4o-mini", req["body"]["model"] == "openai/gpt-4o-mini")
        check("  key only in Authorization header", req["headers"].get("Authorization") == f"Bearer {FAKE_KEY}")
        check("  key NOT in request body / prompt", FAKE_KEY not in json.dumps(req["body"]))
        check("  prompt has active-dataset columns", all(c in txt for c in ["Department", "City", "CGPA", "Year"]))
        check("  prompt has NO benchmark tokens", not any(t in txt for t in BENCHMARK_ONLY_TOKENS))
        check("  PII values withheld (name/email/id)", not any(t in txt for t in ["Zed Quillfeather", "@college.edu", "S0001"]))
        check("  question forwarded", q in txt)
        check("  system prompt grounds + forbids invention", "Never invent" in txt and "Analyze ONLY" in txt)
        check("  UI: AI-generated • Grounded in active dataset",
              any("AI-generated" in c and "Grounded in active dataset" in c for c in r["captions"]))
        check("  UI: NOT the old deterministic label", not any("no API used" in c or "Fallback engine" in c for c in r["captions"]))
        check("  UI shows the model answer", any("FAKE-LLM-ANSWER" in m for m in r["markdown"]))
        check("  key not rendered anywhere in UI", FAKE_KEY not in json.dumps(r["captions"] + r["markdown"]))
    ev = json.loads(prompt_text(reqs[0]).split("(JSON, computed from the uploaded file):\n")[1].split("\n\nUSER QUESTION:")[0])
    check("  evidence: no company/employer column", ev["concept_presence_by_column_name"]["company_or_employer"] == [])
    check("  evidence: city column present", ev["concept_presence_by_column_name"]["city_or_location"] == ["City"])

    print("\n=== B. Evidence facts match pandas on the student data ===")
    dept = students["Department"].value_counts()
    got = {v[0]: v[1] for v in ev["distributions"]["Department"]["values"]}
    check("rows / columns", ev["dataset"]["rows"] == 500 and ev["dataset"]["columns"] == 7)
    check("department counts == pandas", got == {k: int(v) for k, v in dept.items()})
    yr = {v[0]: v[1] for v in ev["distributions"]["Year"]["values"]}
    check("year distribution == pandas", yr == {int(k): int(v) for k, v in students["Year"].value_counts().items()})
    city = {v[0]: v[1] for v in ev["distributions"]["City"]["values"]}
    check("city counts == pandas", city == {k: int(v) for k, v in students["City"].value_counts().items()})
    check("CGPA missing == pandas", ev["numeric_statistics"]["CGPA"]["count"] == int(students["CGPA"].notna().sum()))
    check("CGPA mean == pandas", abs(ev["numeric_statistics"]["CGPA"]["mean"] - students["CGPA"].mean()) < 1e-3)
    check("PII columns reported by name only",
          set(ev["column_groups"]["possible_personal_data_values_withheld"]) == {"Student Name", "Email"})
    check("ID column excluded from stats", "Student ID" not in ev["numeric_statistics"])

    print("\n=== C. Sales CSV (different domain) ===")
    for q in ["Which product generated the highest revenue?", "Which region had the most sales?",
              "What is the overall revenue?", "What are the main trends in the sales data?"]:
        r = run_copilot(sales_bytes, "sales.csv", q, FAKE_KEY)
        check(f"[sales] OpenRouter called: {q}", len(r["requests"]) == 1)
        txt = prompt_text(r["requests"][0])
        check("  sales columns in prompt, no benchmark tokens",
              all(c in txt for c in ["Product", "Revenue", "Region"]) and not any(t in txt for t in BENCHMARK_ONLY_TOKENS))
        check("  UI AI label", any("AI-generated" in c for c in r["captions"]))
    ev2 = json.loads(prompt_text(r["requests"][0]).split("(JSON, computed from the uploaded file):\n")[1].split("\n\nUSER QUESTION:")[0])
    gs = {(g["group_by"], g["metric"]): g for g in ev2["group_summaries"]}
    top_prod = sales.groupby("Product")["Revenue"].sum().idxmax()
    check("evidence: top revenue product == pandas", gs[("Product", "Revenue")]["groups"][0][0] == top_prod)
    top_reg = sales.groupby("Region")["Units Sold"].sum().idxmax()
    check("evidence: top region by units == pandas", gs[("Region", "Units Sold")]["groups"][0][0] == top_reg)
    check("evidence: total revenue == pandas",
          abs(ev2["numeric_statistics"]["Revenue"]["sum"] - round(sales["Revenue"].sum(), 2)) < 0.01)
    check("evidence: date column + monthly trend", ev2["time_trends"] and ev2["time_trends"][0]["granularity"] == "month")

    print("\n=== D. Unusual / mixed-type CSV does not crash and still reaches AI ===")
    r = run_copilot(odd_bytes, "odd.csv", "What are the main data-quality issues?", FAKE_KEY)
    check("[odd] OpenRouter called", len(r["requests"]) == 1)
    ev3 = json.loads(prompt_text(r["requests"][0]).split("(JSON, computed from the uploaded file):\n")[1].split("\n\nUSER QUESTION:")[0])
    obs = " ".join(ev3["data_quality"]["observations"])
    check("  empty column + numbers-as-text flagged", "empty" in obs and "stored as text" in obs)
    check("  column names preserved (spaces/punctuation/unicode)", "a b!" in ev3["dataset"]["column_names"] and "€ col" in ev3["dataset"]["column_names"])

    print("\n=== E. Fallback behaviour (no key / API failure / malformed response) ===")
    for label, key, mode in [("no key", None, "ok"), ("API failure", FAKE_KEY, "fail"), ("malformed response", FAKE_KEY, "bad")]:
        r = run_copilot(students_bytes, "student_data_500_rows.xlsx", "Which companies hired these students?", key, mode)
        expect_calls = 0 if key is None else 1
        check(f"[{label}] OpenRouter calls == {expect_calls}", len(r["requests"]) == expect_calls)
        check(f"[{label}] UI: Fallback engine • Deterministic", any("Fallback engine" in c and "Deterministic" in c for c in r["captions"]))
        check(f"[{label}] unsupported question gets honest non-fabricated answer",
              any("does not contain company" in m for m in r["markdown"]))
    r = run_copilot(students_bytes, "student_data_500_rows.xlsx", "Which cities have the highest number of records?", None)
    check("[no key] dataset WITH a City column is not told it lacks city info",
          not any("does not contain city" in m for m in r["markdown"]))
    r = run_copilot(students_bytes, "student_data_500_rows.xlsx", "Tell me something interesting", None)
    check("[no key] unrecognised phrasing -> evidence digest, not 'phrasing not recognized'",
          any("500 rows" in m for m in r["markdown"]) and not any("does not recognize the phrasing" in m for m in r["markdown"]))
    r = run_copilot(students_bytes, "student_data_500_rows.xlsx", "x", FAKE_KEY, "fail")
    check("[API failure] user is told the AI request was unavailable", any("unavailable" in c for c in r["captions"]))

    print("\n=== E2. Unsupported-information handling is schema-driven (any dataset) ===")
    unsupported_cases = [
        ("students", students_bytes, "student_data_500_rows.xlsx", "Which companies hired these students?", "company"),
        ("students", students_bytes, "student_data_500_rows.xlsx", "What is the average salary?", "salary"),
        ("students", students_bytes, "student_data_500_rows.xlsx", "What is the placement rate?", "placement"),
        ("students", students_bytes, "student_data_500_rows.xlsx", "How many female students are there?", "gender"),
        ("sales", sales_bytes, "sales.csv", "Which employers bought the most?", "company"),
        ("sales", sales_bytes, "sales.csv", "What is the average salary of the sales team?", "salary"),
        ("sales", sales_bytes, "sales.csv", "Which city sold the most?", "city"),
    ]
    for dname, b, fn, q, token in unsupported_cases:
        r = run_copilot(b, fn, q, None)  # deterministic path
        ans = " ".join(m for m in r["markdown"] if "Answer" in m)
        check(f"[{dname}/fallback] '{q}' -> clear not-available answer",
              "does not contain" in ans and token in ans.lower(), ans[:160])
        check("  lists the REAL columns, invents nothing",
              all(c in ans for c in (["Department", "City"] if dname == "students" else ["Product", "Revenue"])))
        check("  fallback label shown", any("Fallback engine" in c for c in r["captions"]))
        r = run_copilot(b, fn, q, FAKE_KEY)  # AI path: prompt carries the schema check
        txt = prompt_text(r["requests"][0])
        check(f"[{dname}/AI] schema check flags missing info in prompt", "SCHEMA CHECK" in txt and "NO column" in txt)
    for dname, b, fn, q in [
        ("students", students_bytes, "student_data_500_rows.xlsx", "Which cities have the highest number of records?"),
        ("students", students_bytes, "student_data_500_rows.xlsx", "What is the average age distribution by Year?"),
        ("sales", sales_bytes, "sales.csv", "Which category has the highest value?"),
        ("sales", sales_bytes, "sales.csv", "Which region had the most sales?"),
        ("sales", sales_bytes, "sales.csv", "What is the overall revenue?"),
        ("students", students_bytes, "student_data_500_rows.xlsx", "Give me a concise overview of the uploaded dataset."),
    ]:
        r = run_copilot(b, fn, q, FAKE_KEY)
        txt = prompt_text(r["requests"][0])
        refused = "SCHEMA CHECK" in txt
        expect_refused = dname == "students" and "age" in q.lower().split()
        check(f"[{dname}/AI] no false 'missing info' flag: {q[:50]}", refused == expect_refused)
    # deterministic fallback must still answer supported questions with real facts
    r = run_copilot(sales_bytes, "sales.csv", "Which region had the most sales?", None)
    ans = " ".join(m for m in r["markdown"] if "Answer" in m)
    check("[sales/fallback] supported question gets dataset-grounded answer (not a refusal)",
          "does not contain" not in ans and "400 rows" in ans, ans[:160])

    print("\n=== F. Benchmark dataset unchanged ===")
    r = run_copilot(None, "", "Does work experience relate to placement?", None)
    check("[benchmark] no key -> no network call", len(r["requests"]) == 0)
    r2 = run_copilot(None, "", "Does work experience relate to placement?", FAKE_KEY)
    check("[benchmark] key -> existing benchmark LLM path still used", len(r2["requests"]) == 1)
    txt = prompt_text(r2["requests"][0])
    check("[benchmark] benchmark prompt still uses benchmark evidence (not generic prompt)", "VERIFIED EVIDENCE FOR THE ACTIVE DATASET" not in txt)
    check("[benchmark] AI label shown on success", any("AI-generated" in c for c in r2["captions"]))

    failed = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("FAILED:\n  " + "\n  ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
