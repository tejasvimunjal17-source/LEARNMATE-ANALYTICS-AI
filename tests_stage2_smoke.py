"""
Smoke-test harness (NOT part of the shipped app).

Streamlit/Plotly cannot be pip-installed in this sandbox (no package-index
network access), so this script stubs both libraries with permissive mocks
and then executes app.py's actual code path for each page. This catches
real bugs (KeyError, AttributeError, wrong column names, etc.) in the
business logic even though it can't verify visual rendering.
"""
import sys
import types
import runpy
from unittest.mock import MagicMock


class CtxMock(MagicMock):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def make_streamlit_stub(sidebar_choice: str, selectbox_value: str = "ssc_p",
                         radio_value: str = "Histogram"):
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
        (model pickers, category pickers, feature pickers, ...) gets a
        value that is actually valid for that specific widget, instead of
        one fixed string reused everywhere."""
        options = args[1] if len(args) >= 2 else kwargs.get("options")
        if options:
            try:
                return list(options)[0]
            except Exception:
                pass
        return selectbox_value

    st_mod.selectbox.side_effect = selectbox_side_effect
    st_mod.radio.return_value = radio_value
    st_mod.slider.return_value = 65.0
    st_mod.select_slider.return_value = 3  # a valid K in range(2, 7) for segmentation
    st_mod.form_submit_button.return_value = False  # don't exercise the predict/ask branch here
    st_mod.button.return_value = False  # no quick-question card "clicked" by default
    st_mod.text_input.return_value = ""
    st_mod.secrets = MagicMock()
    st_mod.secrets.get.return_value = None  # simulate no AI provider key configured
    st_mod.cache_resource.side_effect = lambda f: f  # pass-through decorator: run the REAL function
    st_mod.sidebar = MagicMock()
    st_mod.sidebar.radio.return_value = sidebar_choice
    st_mod.sidebar.caption = MagicMock()

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


def run_page(page_name: str) -> None:
    # fresh stubs + fresh module cache each run so `page = st.sidebar.radio(...)`
    # picks up the right value and utils modules re-import cleanly.
    for mod in list(sys.modules):
        if mod.startswith("utils") or mod == "app":
            del sys.modules[mod]

    st_stub = make_streamlit_stub(page_name)
    plotly_mod, express_mod, go_mod = make_plotly_stub()
    sys.modules["streamlit"] = st_stub
    sys.modules["plotly"] = plotly_mod
    sys.modules["plotly.express"] = express_mod
    sys.modules["plotly.graph_objects"] = go_mod

    print(f"--- Running page: {page_name} ---")
    runpy.run_path("app.py", run_name="__main__")
    print(f"--- OK: {page_name} completed without exception ---\n")


if __name__ == "__main__":
    for page in [
        "Overview", "Data Explorer", "Exploratory Data Analysis",
        "\U0001F9E9 Student Segmentation",
        "\U0001F9E0 AI Insight Copilot",
        "\U0001F3AF What-If Simulator",
        "\U0001F916 Placement Prediction", "\U0001F916 Salary Prediction",
        "\U0001F916 Model Evaluation",
        "\U0001F4D8 Project Summary",
    ]:
        run_page(page)
    print("ALL PAGES EXECUTED WITHOUT ERROR")
