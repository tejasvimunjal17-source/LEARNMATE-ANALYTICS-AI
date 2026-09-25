"""
Smoke-test harness (NOT part of the shipped app).

Streamlit/Plotly cannot be pip-installed in this sandbox (no package-index
network access), so this script stubs both libraries with permissive mocks
and then executes app.py's actual code path for each page. This catches
real bugs (KeyError, AttributeError, wrong column names, etc.) in the
business logic even though it can't verify visual rendering.

Supports TWO modes:
  - Benchmark mode (default): no file uploaded, app.py falls back to
    data/campus_placement.csv exactly as before.
  - Generic mode: a synthetic in-memory CSV is injected as if the user had
    uploaded it via st.sidebar.file_uploader, exercising the dataset-adaptive
    (non-benchmark) code paths end-to-end.
"""
import io
import sys
import types
import runpy
from unittest.mock import MagicMock


class CtxMock(MagicMock):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeUploadedFile(io.BytesIO):
    """Minimal stand-in for Streamlit's UploadedFile: a file-like object
    with a .name attribute, which is all load_dataset_from_upload() needs."""
    def __init__(self, content_bytes: bytes, name: str):
        super().__init__(content_bytes)
        self.name = name


def make_streamlit_stub(sidebar_choice: str, selectbox_value: str = "ssc_p",
                         radio_value: str = "Histogram", uploaded_file=None):
    st_mod = MagicMock(name="streamlit")

    def columns(spec):
        n = spec if isinstance(spec, int) else len(spec)
        return tuple(CtxMock() for _ in range(n))

    def tabs(names):
        return tuple(CtxMock() for _ in names)

    st_mod.columns.side_effect = columns
    st_mod.tabs.side_effect = tabs

    def selectbox_side_effect(*args, **kwargs):
        """Returns the FIRST real option offered, so every selectbox call
        (model pickers, category pickers, feature pickers, target pickers,
        ...) gets a value that is actually valid for that specific widget."""
        options = args[1] if len(args) >= 2 else kwargs.get("options")
        if options:
            try:
                return list(options)[0]
            except Exception:
                pass
        return selectbox_value

    def multiselect_side_effect(*args, **kwargs):
        """Returns the widget's own 'default' (or all options) so generic
        pages that need a non-empty feature selection actually get one."""
        if "default" in kwargs and kwargs["default"] is not None:
            return list(kwargs["default"])
        options = args[1] if len(args) >= 2 else kwargs.get("options")
        return list(options) if options else []

    st_mod.selectbox.side_effect = selectbox_side_effect
    st_mod.multiselect.side_effect = multiselect_side_effect
    st_mod.radio.return_value = radio_value
    st_mod.slider.return_value = 65.0
    st_mod.select_slider.return_value = 3  # a valid K in range(2, 7) for segmentation
    st_mod.form_submit_button.return_value = False  # don't exercise the predict/ask branch here
    st_mod.button.return_value = False  # no quick-question card / train button "clicked" by default
    st_mod.text_input.return_value = ""
    st_mod.secrets = MagicMock()
    st_mod.secrets.get.return_value = None  # simulate no AI provider key configured
    st_mod.cache_resource.side_effect = lambda f: f  # pass-through decorator: run the REAL function
    st_mod.session_state = {}

    st_mod.sidebar = MagicMock()
    st_mod.sidebar.radio.return_value = sidebar_choice
    st_mod.sidebar.caption = MagicMock()
    st_mod.sidebar.file_uploader.return_value = uploaded_file  # None = no upload -> benchmark fallback
    st_mod.sidebar.button.return_value = False  # "Reset to Benchmark Dataset" not clicked

    def _stop():
        raise SystemExit("st.stop() called")
    st_mod.stop.side_effect = _stop

    return st_mod


def make_plotly_stub():
    plotly_mod = types.ModuleType("plotly")
    express_mod = MagicMock(name="plotly.express")
    go_mod = MagicMock(name="plotly.graph_objects")
    plotly_mod.express = express_mod
    plotly_mod.graph_objects = go_mod
    return plotly_mod, express_mod, go_mod


def run_page(page_name: str, uploaded_file=None, label: str = "") -> None:
    # fresh stubs + fresh module cache each run so `page = st.sidebar.radio(...)`
    # picks up the right value and utils modules re-import cleanly.
    for mod in list(sys.modules):
        if mod.startswith("utils") or mod == "app":
            del sys.modules[mod]

    st_stub = make_streamlit_stub(page_name, uploaded_file=uploaded_file)
    plotly_mod, express_mod, go_mod = make_plotly_stub()
    sys.modules["streamlit"] = st_stub
    sys.modules["plotly"] = plotly_mod
    sys.modules["plotly.express"] = express_mod
    sys.modules["plotly.graph_objects"] = go_mod

    tag = f" [{label}]" if label else ""
    print(f"--- Running page: {page_name}{tag} ---")
    runpy.run_path("app.py", run_name="__main__")
    print(f"--- OK: {page_name}{tag} completed without exception ---\n")


BENCHMARK_PAGES = [
    "Overview", "Data Explorer", "Exploratory Data Analysis",
    "\U0001F9E9 Student Segmentation",
    "\U0001F9E0 AI Insight Copilot",
    "\U0001F3AF What-If Simulator",
    "\U0001F9EC Predictive Analysis",
    "\U0001F916 Placement Prediction", "\U0001F916 Salary Prediction",
    "\U0001F916 Model Evaluation",
    "\U0001F4D8 Project Summary",
]

# Generic (non-benchmark) pages only - the benchmark-only pages are expected
# to show a graceful notice rather than run their specialized logic, which
# is exercised separately in the targeted Stage 8.5 test script.
GENERIC_PAGES = [
    "Overview", "Data Explorer", "Exploratory Data Analysis",
    "\U0001F9E9 Student Segmentation",
    "\U0001F9E0 AI Insight Copilot",
    "\U0001F3AF What-If Simulator",
    "\U0001F9EC Predictive Analysis",
    "\U0001F916 Placement Prediction", "\U0001F916 Salary Prediction",
    "\U0001F916 Model Evaluation",
    "\U0001F4D8 Project Summary",
]


if __name__ == "__main__":
    print("=" * 70)
    print("MODE A: BENCHMARK (no upload)")
    print("=" * 70)
    for page in BENCHMARK_PAGES:
        run_page(page, uploaded_file=None, label="benchmark")

    print("=" * 70)
    print("MODE B: GENERIC (synthetic uploaded CSV, placement+salary-like)")
    print("=" * 70)
    generic_csv = (
        "student_id,age,department,cgpa,internship_months,placement_result,starting_salary\n"
        + "\n".join(
            f"STU{i:04d},{20 + i % 5},{'CSE' if i % 2 == 0 else 'ECE'},{6.0 + (i % 40) / 10:.2f},"
            f"{i % 12},{'Placed' if i % 3 != 0 else 'Not Placed'},"
            f"{'' if i % 3 == 0 else 300000 + (i % 10) * 20000}"
            for i in range(60)
        )
    ).encode("utf-8")
    for page in GENERIC_PAGES:
        run_page(page, uploaded_file=FakeUploadedFile(generic_csv, "generic_students.csv"), label="generic")

    print("ALL SMOKE TESTS (BENCHMARK + GENERIC) EXECUTED WITHOUT ERROR")
