"""
ml_models.py
------------
STAGE 3 - Placement Prediction (supervised classification).

Pure sklearn/pandas module (no Streamlit) so it can be unit-tested and
reused by the AI Insight Copilot later without pulling in the UI layer.

Everything here is fit at runtime on the ACTUAL dataset. No metric is
hard-coded. Preprocessing is fit only on the training split (never on the
full dataset) to avoid preprocessing leakage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
    mean_absolute_error, root_mean_squared_error, r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from utils.data_processing import (
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, RANDOM_STATE,
    get_classification_data, get_regression_data,
)

TEST_SIZE = 0.20
POSITIVE_LABEL = "Placed"  # Placed -> 1, Not Placed -> 0

MODEL_REGISTRY = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": lambda: DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE),
}

MODEL_EXPLANATIONS = {
    "Logistic Regression": (
        "Estimates the probability of a binary outcome (Placed / Not Placed) as a "
        "weighted combination of the input features, passed through a curve that "
        "keeps the result between 0 and 1."
    ),
    "Decision Tree": (
        "Learns a sequence of yes/no rules on the input features (e.g. 'is Degree % "
        "above X?') that split students into groups with more similar outcomes."
    ),
}

METRIC_EXPLANATIONS = {
    "Accuracy": "The share of all predictions (placed and not placed) that were correct.",
    "Precision": "Of the students the model predicted as Placed, the share that were actually Placed.",
    "Recall": "Of the students actually Placed, the share the model correctly identified.",
    "F1 Score": "The harmonic mean of Precision and Recall - a single balance between the two.",
    "ROC-AUC": "How well the model separates the two classes across all possible thresholds (1.0 = perfect, 0.5 = random).",
}


# ---------------------------------------------------------------------------
# Preprocessing pipeline (shared shape for both models; scaling is harmless
# for the Decision Tree since tree splits are invariant to monotonic scaling)
# ---------------------------------------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(transformers=[
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def build_model_pipeline(model_name: str) -> Pipeline:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {list(MODEL_REGISTRY)}.")
    return Pipeline(steps=[
        ("preprocess", build_preprocessor()),
        ("classifier", MODEL_REGISTRY[model_name]()),
    ])


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class ModelEvaluation:
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    confusion: np.ndarray
    roc_fpr: np.ndarray | None
    roc_tpr: np.ndarray | None
    feature_names: list[str]
    coefficients: dict[str, float] | None = None      # Logistic Regression only
    feature_importances: dict[str, float] | None = None  # Decision Tree only


@dataclass
class ClassificationResults:
    random_state: int
    test_size: float
    n_train: int
    n_test: int
    feature_columns: list[str]
    excluded_columns: list[str]
    pipelines: dict[str, Pipeline] = field(default_factory=dict)
    evaluations: dict[str, ModelEvaluation] = field(default_factory=dict)
    X_test: pd.DataFrame | None = None
    y_test: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Training + evaluation
# ---------------------------------------------------------------------------

def _encode_target(y: pd.Series) -> np.ndarray:
    return (y.str.lower() == POSITIVE_LABEL.lower()).astype(int).to_numpy()


def train_and_evaluate(clean_df: pd.DataFrame) -> ClassificationResults:
    X, y_raw = get_classification_data(clean_df)  # already excludes sl_no, gender, salary, status
    y = _encode_target(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )

    results = ClassificationResults(
        random_state=RANDOM_STATE,
        test_size=TEST_SIZE,
        n_train=len(X_train),
        n_test=len(X_test),
        feature_columns=list(X.columns),
        excluded_columns=["sl_no", "gender", "salary", "status"],
        X_test=X_test,
        y_test=y_test,
    )

    for model_name in MODEL_REGISTRY:
        pipeline = build_model_pipeline(model_name)
        pipeline.fit(X_train, y_train)  # preprocessor fit ONLY on training data
        results.pipelines[model_name] = pipeline

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        roc_auc = None
        fpr = tpr = None
        if len(set(y_test)) == 2:  # ROC-AUC only valid with both classes present
            roc_auc = float(roc_auc_score(y_test, y_proba))
            fpr, tpr, _ = roc_curve(y_test, y_proba)

        feature_names = list(pipeline.named_steps["preprocess"].get_feature_names_out())
        classifier = pipeline.named_steps["classifier"]

        coefficients = None
        importances = None
        if isinstance(classifier, LogisticRegression):
            coefficients = dict(zip(feature_names, classifier.coef_[0].tolist()))
        elif isinstance(classifier, DecisionTreeClassifier):
            importances = dict(zip(feature_names, classifier.feature_importances_.tolist()))

        results.evaluations[model_name] = ModelEvaluation(
            model_name=model_name,
            accuracy=float(accuracy_score(y_test, y_pred)),
            precision=float(precision_score(y_test, y_pred, zero_division=0)),
            recall=float(recall_score(y_test, y_pred, zero_division=0)),
            f1=float(f1_score(y_test, y_pred, zero_division=0)),
            roc_auc=roc_auc,
            confusion=confusion_matrix(y_test, y_pred),
            roc_fpr=fpr,
            roc_tpr=tpr,
            feature_names=feature_names,
            coefficients=coefficients,
            feature_importances=importances,
        )

    return results


def comparison_table(results: ClassificationResults) -> pd.DataFrame:
    rows = []
    for name, ev in results.evaluations.items():
        rows.append({
            "Model": name,
            "Accuracy": round(ev.accuracy, 3),
            "Precision": round(ev.precision, 3),
            "Recall": round(ev.recall, 3),
            "F1 Score": round(ev.f1, 3),
            "ROC-AUC": round(ev.roc_auc, 3) if ev.roc_auc is not None else None,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Single-row prediction (for the Placement Prediction page)
# ---------------------------------------------------------------------------

def predict_single(results: ClassificationResults, model_name: str, input_values: dict) -> dict:
    """input_values must contain exactly `results.feature_columns` keys."""
    missing = [c for c in results.feature_columns if c not in input_values]
    if missing:
        raise ValueError(f"Missing input values for: {missing}")

    row = pd.DataFrame([{c: input_values[c] for c in results.feature_columns}])
    pipeline = results.pipelines[model_name]
    pred = int(pipeline.predict(row)[0])
    proba = float(pipeline.predict_proba(row)[0, 1])

    return {
        "predicted_status": "Placed" if pred == 1 else "Not Placed",
        "predicted_probability_placed": proba,
        "model_name": model_name,
    }


# ===========================================================================
# STAGE 4 - SALARY REGRESSION
# ===========================================================================
#
# Trained only on rows with an observed salary (placed students). Uses the
# SAME feature exclusion policy as classification: sl_no, gender, status and
# (obviously) salary itself never enter the predictor set.

REGRESSION_TEST_SIZE = 0.20

REGRESSION_MODEL_REGISTRY = {
    "Linear Regression": lambda: LinearRegression(),
    "Random Forest Regressor": lambda: RandomForestRegressor(
        n_estimators=300, max_depth=6, random_state=RANDOM_STATE
    ),
}

REGRESSION_MODEL_EXPLANATIONS = {
    "Linear Regression": (
        "Fits a straight-line relationship between the input features and salary - "
        "each feature contributes an additive amount, weighted by its coefficient."
    ),
    "Random Forest Regressor": (
        "Averages the predictions of many decision trees, each trained on a random "
        "subset of the data and features, to produce a smoother salary estimate."
    ),
}

REGRESSION_METRIC_EXPLANATIONS = {
    "MAE": "Mean Absolute Error - the average size of the prediction error, in rupees.",
    "RMSE": "Root Mean Squared Error - like MAE, but penalizes large errors more heavily.",
    "R2": "R-squared - the share of the variation in salary that the model explains (1.0 = perfect, 0.0 = no better than the average).",
}

SMALL_SAMPLE_WARNING = (
    "Salary modelling uses only records with observed salary values. Because this "
    "subset is relatively small, evaluation results should be interpreted as an "
    "educational modelling exercise rather than a production salary estimator."
)


@dataclass
class RegressionEvaluation:
    model_name: str
    mae: float
    rmse: float
    r2: float
    y_test: np.ndarray
    y_pred: np.ndarray
    residuals: np.ndarray
    feature_names: list[str]
    coefficients: dict[str, float] | None = None       # Linear Regression only
    feature_importances: dict[str, float] | None = None  # Random Forest only


@dataclass
class RegressionResults:
    random_state: int
    test_size: float
    n_salary_records: int
    n_train: int
    n_test: int
    feature_columns: list[str]
    excluded_columns: list[str]
    pipelines: dict[str, Pipeline] = field(default_factory=dict)
    evaluations: dict[str, RegressionEvaluation] = field(default_factory=dict)
    X_test: pd.DataFrame | None = None


def train_and_evaluate_salary(clean_df: pd.DataFrame) -> RegressionResults:
    X, y = get_regression_data(clean_df)  # already placed-only; excludes sl_no, gender, status, salary

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=REGRESSION_TEST_SIZE, random_state=RANDOM_STATE,
    )

    results = RegressionResults(
        random_state=RANDOM_STATE,
        test_size=REGRESSION_TEST_SIZE,
        n_salary_records=len(X),
        n_train=len(X_train),
        n_test=len(X_test),
        feature_columns=list(X.columns),
        excluded_columns=["sl_no", "gender", "status", "salary"],
        X_test=X_test,
    )

    for model_name in REGRESSION_MODEL_REGISTRY:
        pipeline = Pipeline(steps=[
            ("preprocess", build_preprocessor()),
            ("regressor", REGRESSION_MODEL_REGISTRY[model_name]()),
        ])
        pipeline.fit(X_train, y_train)  # preprocessor fit ONLY on training data
        results.pipelines[model_name] = pipeline

        y_pred = pipeline.predict(X_test)
        residuals = y_test.to_numpy() - y_pred

        feature_names = list(pipeline.named_steps["preprocess"].get_feature_names_out())
        regressor = pipeline.named_steps["regressor"]

        coefficients = None
        importances = None
        if isinstance(regressor, LinearRegression):
            coefficients = dict(zip(feature_names, regressor.coef_.tolist()))
        elif isinstance(regressor, RandomForestRegressor):
            importances = dict(zip(feature_names, regressor.feature_importances_.tolist()))

        results.evaluations[model_name] = RegressionEvaluation(
            model_name=model_name,
            mae=float(mean_absolute_error(y_test, y_pred)),
            rmse=float(root_mean_squared_error(y_test, y_pred)),
            r2=float(r2_score(y_test, y_pred)),
            y_test=y_test.to_numpy(),
            y_pred=y_pred,
            residuals=residuals,
            feature_names=feature_names,
            coefficients=coefficients,
            feature_importances=importances,
        )

    return results


def regression_comparison_table(results: RegressionResults) -> pd.DataFrame:
    rows = []
    for name, ev in results.evaluations.items():
        rows.append({
            "Model": name,
            "MAE (INR)": round(ev.mae, 0),
            "RMSE (INR)": round(ev.rmse, 0),
            "R2": round(ev.r2, 3),
        })
    return pd.DataFrame(rows)


def predict_salary(results: RegressionResults, model_name: str, input_values: dict) -> dict:
    """input_values must contain exactly `results.feature_columns` keys."""
    missing = [c for c in results.feature_columns if c not in input_values]
    if missing:
        raise ValueError(f"Missing input values for: {missing}")

    row = pd.DataFrame([{c: input_values[c] for c in results.feature_columns}])
    pipeline = results.pipelines[model_name]
    predicted_salary = float(pipeline.predict(row)[0])

    return {
        "predicted_salary": predicted_salary,
        "model_name": model_name,
    }


# ===========================================================================
# STAGE 5 - STUDENT SEGMENTATION (K-MEANS)
# ===========================================================================
#
# Unsupervised - no target variable. Excludes sl_no, gender, status, salary:
# segmentation is based only on student profile characteristics available
# BEFORE the placement outcome. Placement/salary may be examined AFTER
# clustering as a purely descriptive, clearly-labeled comparison - never as
# an input to the clustering itself.

CLUSTER_NUMERIC_FEATURES = NUMERIC_FEATURES
CLUSTER_CATEGORICAL_FEATURES = CATEGORICAL_FEATURES
CLUSTER_EXCLUDED = ["sl_no", "gender", "status", "salary"]
CLUSTER_K_RANGE = list(range(2, 7))  # 2 through 6, inclusive


def prepare_segmentation_features(clean_df: pd.DataFrame) -> pd.DataFrame:
    cols = CLUSTER_NUMERIC_FEATURES + CLUSTER_CATEGORICAL_FEATURES
    return clean_df[cols].copy()


def build_segmentation_pipeline() -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(transformers=[
        ("num", numeric_pipeline, CLUSTER_NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CLUSTER_CATEGORICAL_FEATURES),
    ])


@dataclass
class SegmentationDiagnostics:
    feature_names: list[str]
    X_transformed: np.ndarray
    elbow_scores: dict          # k -> inertia
    silhouette_scores: dict     # k -> silhouette score (None if undefined for that k)
    n_students: int
    k_range: list[int]


def calculate_elbow_scores(X_transformed: np.ndarray, k_range: list[int] = CLUSTER_K_RANGE) -> dict:
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km.fit(X_transformed)
        scores[k] = float(km.inertia_)
    return scores


def calculate_silhouette_scores(X_transformed: np.ndarray, k_range: list[int] = CLUSTER_K_RANGE) -> dict:
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_transformed)
        if len(set(labels)) > 1:
            scores[k] = float(silhouette_score(X_transformed, labels))
        else:
            scores[k] = None
    return scores


def compute_segmentation_diagnostics(clean_df: pd.DataFrame,
                                      k_range: list[int] = CLUSTER_K_RANGE) -> SegmentationDiagnostics:
    """Fits the preprocessing pipeline once, then evaluates every K in
    k_range for the Elbow / Silhouette diagnostics. Preprocessing is fit on
    ALL students since clustering is unsupervised (no train/test split is
    meaningful here - there is no target to leak)."""
    X_raw = prepare_segmentation_features(clean_df)
    preprocessor = build_segmentation_pipeline()
    X_transformed = preprocessor.fit_transform(X_raw)
    feature_names = list(preprocessor.get_feature_names_out())

    return SegmentationDiagnostics(
        feature_names=feature_names,
        X_transformed=X_transformed,
        elbow_scores=calculate_elbow_scores(X_transformed, k_range),
        silhouette_scores=calculate_silhouette_scores(X_transformed, k_range),
        n_students=len(X_raw),
        k_range=list(k_range),
    )


def recommend_k(diagnostics: SegmentationDiagnostics) -> int:
    """A DIAGNOSTIC suggestion only (highest silhouette score in range) -
    not an absolute truth. The UI must show this as a suggestion and let
    the user pick any K themselves."""
    valid = {k: s for k, s in diagnostics.silhouette_scores.items() if s is not None}
    if not valid:
        return diagnostics.k_range[0]
    return max(valid, key=valid.get)


@dataclass
class ClusterProfile:
    cluster_id: int
    count: int
    pct_of_total: float
    ssc_avg: float
    hsc_avg: float
    degree_avg: float
    etest_avg: float
    mba_avg: float
    academic_avg: float          # mean of the 5 percentages above
    workex_yes_pct: float
    top_hsc_stream: str
    top_degree_type: str
    top_specialisation: str
    label: str


@dataclass
class SegmentationResult:
    k: int
    labels: np.ndarray
    kmeans: KMeans
    profiles: list[ClusterProfile]
    overall_academic_avg: float
    overall_workex_pct: float
    pca_df: pd.DataFrame
    diagnostics: SegmentationDiagnostics


def _label_cluster(academic_diff: float, workex_diff: float) -> str:
    """Neutral, data-driven label based on how this cluster's academic
    average and work-experience share compare to the overall dataset.
    Thresholds are modest and symmetric; if neither condition is clearly
    met, the cluster is described as 'Balanced Profile' rather than
    inventing a narrative."""
    if academic_diff >= 3.0:
        return "Higher Academic Profile"
    if academic_diff <= -3.0:
        return "Emerging Academic Profile"
    if workex_diff >= 15.0:
        return "Work-Experience-Oriented Profile"
    return "Balanced Profile"


def get_cluster_profiles(clean_df: pd.DataFrame, labels: np.ndarray) -> tuple[list[ClusterProfile], float, float]:
    """Computes profile statistics from the ORIGINAL (unscaled) dataframe,
    never from the scaled/encoded matrix."""
    df = clean_df.copy()
    df["_cluster"] = labels
    total = len(df)

    df["_academic_avg_row"] = df[CLUSTER_NUMERIC_FEATURES].mean(axis=1)
    overall_academic_avg = float(df["_academic_avg_row"].mean())
    overall_workex_pct = float((df["workex"].str.lower() == "yes").mean() * 100)

    profiles: list[ClusterProfile] = []
    for cluster_id in sorted(df["_cluster"].unique()):
        sub = df[df["_cluster"] == cluster_id]
        academic_avg = float(sub["_academic_avg_row"].mean())
        workex_pct = float((sub["workex"].str.lower() == "yes").mean() * 100)

        profile = ClusterProfile(
            cluster_id=int(cluster_id),
            count=len(sub),
            pct_of_total=round(100 * len(sub) / total, 1),
            ssc_avg=round(float(sub["ssc_p"].mean()), 1),
            hsc_avg=round(float(sub["hsc_p"].mean()), 1),
            degree_avg=round(float(sub["degree_p"].mean()), 1),
            etest_avg=round(float(sub["etest_p"].mean()), 1),
            mba_avg=round(float(sub["mba_p"].mean()), 1),
            academic_avg=round(academic_avg, 1),
            workex_yes_pct=round(workex_pct, 1),
            top_hsc_stream=str(sub["hsc_s"].mode().iloc[0]) if not sub["hsc_s"].mode().empty else "N/A",
            top_degree_type=str(sub["degree_t"].mode().iloc[0]) if not sub["degree_t"].mode().empty else "N/A",
            top_specialisation=str(sub["specialisation"].mode().iloc[0]) if not sub["specialisation"].mode().empty else "N/A",
            label=_label_cluster(academic_avg - overall_academic_avg, workex_pct - overall_workex_pct),
        )
        profiles.append(profile)

    return profiles, overall_academic_avg, overall_workex_pct


def get_pca_projection(clean_df: pd.DataFrame, X_transformed: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """PCA is used ONLY to visualize the clustering result in 2D. Clustering
    itself already happened on the full preprocessed feature space."""
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_transformed)
    pca_df = pd.DataFrame({
        "pc1": coords[:, 0],
        "pc2": coords[:, 1],
        "cluster": labels.astype(str),
        "ssc_p": clean_df["ssc_p"].values,
        "hsc_p": clean_df["hsc_p"].values,
        "degree_p": clean_df["degree_p"].values,
        "workex": clean_df["workex"].values,
        "specialisation": clean_df["specialisation"].values,
    })
    return pca_df


def run_segmentation(clean_df: pd.DataFrame, k: int,
                      diagnostics: SegmentationDiagnostics | None = None) -> SegmentationResult:
    if diagnostics is None:
        diagnostics = compute_segmentation_diagnostics(clean_df)

    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(diagnostics.X_transformed)

    profiles, overall_academic_avg, overall_workex_pct = get_cluster_profiles(clean_df, labels)
    pca_df = get_pca_projection(clean_df, diagnostics.X_transformed, labels)

    return SegmentationResult(
        k=k,
        labels=labels,
        kmeans=kmeans,
        profiles=profiles,
        overall_academic_avg=round(overall_academic_avg, 1),
        overall_workex_pct=round(overall_workex_pct, 1),
        pca_df=pca_df,
        diagnostics=diagnostics,
    )


def post_clustering_outcomes(clean_df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """OPTIONAL, POST-clustering descriptive comparison only. These outcomes
    were NOT used to create the clusters - salary/status never entered the
    feature matrix (see CLUSTER_EXCLUDED)."""
    df = clean_df.copy()
    df["_cluster"] = labels
    rows = []
    for cluster_id in sorted(df["_cluster"].unique()):
        sub = df[df["_cluster"] == cluster_id]
        placed = sub[sub["status"].str.lower() == "placed"]
        placement_rate = round(100 * len(placed) / len(sub), 1) if len(sub) else 0.0
        placed_with_salary = placed[placed["salary"].notna()]
        avg_salary = float(placed_with_salary["salary"].mean()) if len(placed_with_salary) else None
        rows.append({
            "cluster": int(cluster_id),
            "students": len(sub),
            "placement_rate_pct": placement_rate,
            "avg_salary_placed": round(avg_salary, 0) if avg_salary is not None else None,
            "salary_sample_size": len(placed_with_salary),
        })
    return pd.DataFrame(rows)


@dataclass
class SegmentationInsight:
    fact: str
    insight: str
    possible_action: str


def generate_segmentation_insights(profiles: list[ClusterProfile]) -> list[SegmentationInsight]:
    """Deterministic, calculated purely from the profile statistics already
    computed above - no LLM, no invented traits."""
    if len(profiles) < 2:
        return []

    insights: list[SegmentationInsight] = []

    by_academic = sorted(profiles, key=lambda p: p.academic_avg, reverse=True)
    top, bottom = by_academic[0], by_academic[-1]
    if top.cluster_id != bottom.cluster_id:
        insights.append(SegmentationInsight(
            fact=(f"Cluster {top.cluster_id} has the highest observed average academic score "
                  f"({top.academic_avg:.1f}%, n={top.count}); Cluster {bottom.cluster_id} has the "
                  f"lowest ({bottom.academic_avg:.1f}%, n={bottom.count})."),
            insight="Academic percentage averages differ across the discovered segments.",
            possible_action=("Academic support resources could be prioritized for segments with "
                              "lower observed academic averages, subject to further review."),
        ))

    by_workex = sorted(profiles, key=lambda p: p.workex_yes_pct, reverse=True)
    top_we, bottom_we = by_workex[0], by_workex[-1]
    if top_we.cluster_id != bottom_we.cluster_id and (top_we.workex_yes_pct - bottom_we.workex_yes_pct) >= 10:
        insights.append(SegmentationInsight(
            fact=(f"Cluster {top_we.cluster_id} has the highest observed proportion of students with "
                  f"work experience ({top_we.workex_yes_pct:.1f}%); Cluster {bottom_we.cluster_id} has "
                  f"the lowest ({bottom_we.workex_yes_pct:.1f}%)."),
            insight="Work experience is a distinguishing characteristic between these segments.",
            possible_action=("Career-support programs could examine whether students in the "
                              "lower-work-experience segment have access to relevant internship or "
                              "practical-exposure opportunities."),
        ))

    return insights


SEGMENTATION_SMALL_SAMPLE_WARNING = (
    "This segmentation is exploratory. The dataset contains 215 students, so cluster profiles "
    "may change with a larger or different sample. K-Means is also sensitive to feature "
    "selection, scaling, the number of clusters chosen, and the overall dataset composition."
)


# ===========================================================================
# STAGE 7 - WHAT-IF SIMULATOR
# ===========================================================================
#
# Pure scenario helpers built entirely on top of the EXISTING Stage 3
# ClassificationResults / predict_single - no new model is trained, no
# preprocessing is duplicated or re-fit. This module only decides WHICH
# input rows to score and packages the results; the trained pipeline
# (fit once, in train_and_evaluate) does all the actual scaling/encoding
# and probability calculation via its own predict_proba().

WHATIF_FEATURE_LABELS = {
    "ssc_p": "SSC %", "hsc_p": "HSC %", "degree_p": "Degree %",
    "etest_p": "E-test %", "mba_p": "MBA %",
    "ssc_b": "SSC Board", "hsc_b": "HSC Board", "hsc_s": "HSC Stream",
    "degree_t": "Degree Type", "workex": "Work Experience",
    "specialisation": "MBA Specialisation",
}

WHATIF_DATASET_SIZE_NOTE = "Dataset size: 215 students. This simulator is an educational scenario-analysis tool based on a relatively small dataset."

WHATIF_LIMITATION_NOTE = (
    "This simulator shows how the trained model responds to hypothetical input profiles. It does "
    "not predict an individual's guaranteed outcome and does not establish causal relationships "
    "between a changed attribute and placement. Results depend on the small dataset size, the "
    "observational nature of the data, the limited set of available features, and the specific "
    "model selected."
)


def baseline_profile(clean_df: pd.DataFrame) -> dict:
    """A default student profile calculated from the dataset: median for
    numeric features, most-frequent category for categorical features. Not
    presented as a real individual student."""
    profile = {}
    for col in NUMERIC_FEATURES:
        profile[col] = float(clean_df[col].median())
    for col in CATEGORICAL_FEATURES:
        mode = clean_df[col].mode()
        profile[col] = str(mode.iloc[0]) if not mode.empty else None
    return profile


def detect_profile_changes(baseline: dict, whatif: dict) -> list[tuple[str, object, object]]:
    """Returns [(feature, old_value, new_value), ...] for every feature that
    differs between the two profiles, in a stable feature order."""
    changes = []
    for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        old, new = baseline.get(col), whatif.get(col)
        if isinstance(old, float) and isinstance(new, float):
            if abs(old - new) > 1e-9:
                changes.append((col, old, new))
        elif old != new:
            changes.append((col, old, new))
    return changes


@dataclass
class ScenarioComparison:
    model_name: str
    prob_baseline: float
    prob_whatif: float
    diff_pp: float               # percentage points, prob_whatif - prob_baseline
    class_baseline: str
    class_whatif: str
    changes: list


def compare_scenarios(results: ClassificationResults, model_name: str,
                       baseline: dict, whatif: dict) -> ScenarioComparison:
    """Scores BOTH profiles through the SAME already-fitted pipeline
    (results.pipelines[model_name]) via the existing predict_single() -
    no manual scaling/encoding, no new model fit."""
    pred_a = predict_single(results, model_name, baseline)
    pred_b = predict_single(results, model_name, whatif)
    prob_a = pred_a["predicted_probability_placed"]
    prob_b = pred_b["predicted_probability_placed"]
    return ScenarioComparison(
        model_name=model_name,
        prob_baseline=prob_a,
        prob_whatif=prob_b,
        diff_pp=round((prob_b - prob_a) * 100, 1),
        class_baseline=pred_a["predicted_status"],
        class_whatif=pred_b["predicted_status"],
        changes=detect_profile_changes(baseline, whatif),
    )


def simulate_single_numeric_feature(results: ClassificationResults, model_name: str,
                                     base_profile: dict, feature: str,
                                     n_points: int = 11) -> pd.DataFrame:
    """Holds every input fixed at base_profile except `feature`, which is
    swept across a range derived from the ACTUAL dataset (0-100 for a
    percentage feature), and scores each point with the real trained model.
    Used for the 'Model response curve' line chart."""
    lo, hi = 0.0, 100.0
    values = [round(lo + i * (hi - lo) / (n_points - 1), 1) for i in range(n_points)]
    rows = []
    for v in values:
        profile = dict(base_profile)
        profile[feature] = v
        pred = predict_single(results, model_name, profile)
        rows.append({"value": v, "probability": pred["predicted_probability_placed"]})
    return pd.DataFrame(rows)


def simulate_categorical_feature(results: ClassificationResults, model_name: str,
                                  base_profile: dict, feature: str,
                                  categories: list[str]) -> pd.DataFrame:
    """Holds every input fixed at base_profile except `feature`, scored once
    per category actually observed in the dataset."""
    rows = []
    for cat in categories:
        profile = dict(base_profile)
        profile[feature] = cat
        pred = predict_single(results, model_name, profile)
        rows.append({"category": cat, "probability": pred["predicted_probability_placed"]})
    return pd.DataFrame(rows)


# ===========================================================================
# GENERIC (DATASET-AGNOSTIC) ML LAYER
# ===========================================================================
#
# Everything above this line is BENCHMARK-SPECIFIC and UNCHANGED - it still
# assumes the campus-placement column names and is exactly what Stages 3-7
# use. Everything below makes NO assumption about column names: it receives
# a dataframe, a chosen target column, and a chosen feature-column list from
# the caller (app.py), and works with whatever those turn out to be. This is
# what powers the app's GENERIC mode for a dataset that isn't the benchmark.

GENERIC_CLASSIFICATION_MODELS = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": lambda: DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE),
}
GENERIC_REGRESSION_MODELS = {
    "Linear Regression": lambda: LinearRegression(),
    "Random Forest Regressor": lambda: RandomForestRegressor(n_estimators=300, max_depth=6, random_state=RANDOM_STATE),
}


@dataclass
class GenericTaskValidation:
    """Returned BEFORE attempting to train anything, so the UI can show a
    clear reason instead of a stack trace when a dataset/target genuinely
    isn't usable for the requested task."""
    ok: bool
    reason: str = ""


def validate_generic_target(df: pd.DataFrame, target_col: str, task: str,
                             min_rows: int = 20) -> GenericTaskValidation:
    if target_col not in df.columns:
        return GenericTaskValidation(False, f"Column '{target_col}' does not exist in this dataset.")
    series = df[target_col].dropna()
    if len(series) < min_rows:
        return GenericTaskValidation(False, f"Only {len(series)} non-missing values in '{target_col}' - "
                                             f"need at least {min_rows} for a meaningful train/test split.")
    n_unique = series.nunique()
    if n_unique <= 1:
        return GenericTaskValidation(False, f"'{target_col}' has only {n_unique} unique value(s) - "
                                             f"there is nothing to predict.")
    if task == "classification":
        if pd.api.types.is_numeric_dtype(series) and n_unique > 15:
            return GenericTaskValidation(False, f"'{target_col}' looks like a continuous numeric column "
                                                 f"({n_unique} unique values) - consider Regression instead.")
        if n_unique > 15:
            return GenericTaskValidation(False, f"'{target_col}' has {n_unique} distinct categories - "
                                                 f"too many for a reliable classification target here.")
    elif task == "regression":
        if not pd.api.types.is_numeric_dtype(series):
            return GenericTaskValidation(False, f"'{target_col}' is not numeric - Regression needs a "
                                                 f"numeric continuous target. Consider Classification instead.")
    else:
        return GenericTaskValidation(False, f"Unknown task '{task}'.")
    return GenericTaskValidation(True)


def build_generic_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """Same shape as the benchmark preprocessor, but built from WHATEVER
    numeric/categorical column lists are passed in - no hard-coded names."""
    transformers = []
    if numeric_cols:
        transformers.append(("num", Pipeline(steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", Pipeline(steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_cols))
    if not transformers:
        raise ValueError("No usable numeric or categorical feature columns were provided.")
    return ColumnTransformer(transformers=transformers)


@dataclass
class GenericModelResult:
    task: str                 # "classification" or "regression"
    model_name: str
    target: str
    feature_columns: list
    n_train: int
    n_test: int
    metrics: dict
    positive_label: object = None   # classification only
    pipeline: object = None


def train_generic_classifier(df: pd.DataFrame, target_col: str, feature_cols: list[str],
                              model_name: str = "Logistic Regression") -> GenericModelResult:
    """Binary classification only (multiclass targets are rejected by
    validate_generic_target before this is called, with a clear reason -
    kept intentionally simple and reliable rather than guessing at
    one-vs-rest strategies for an arbitrary uploaded dataset)."""
    working = df[[target_col] + feature_cols].dropna(subset=[target_col])
    y_raw = working[target_col]
    classes = sorted(y_raw.unique(), key=str)
    if len(classes) != 2:
        raise ValueError(f"train_generic_classifier requires a binary target; '{target_col}' has {len(classes)} classes.")
    positive_label = classes[-1]  # deterministic (alphabetical/numeric-last), documented to the user in the UI
    y = (y_raw == positive_label).astype(int)
    X = working[feature_cols]

    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify,
    )

    model_fn = GENERIC_CLASSIFICATION_MODELS.get(model_name, GENERIC_CLASSIFICATION_MODELS["Logistic Regression"])
    pipeline = Pipeline(steps=[
        ("preprocess", build_generic_preprocessor(numeric_cols, categorical_cols)),
        ("classifier", model_fn()),
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)) if y_test.nunique() > 1 else None,
    }
    return GenericModelResult(
        task="classification", model_name=model_name, target=target_col, feature_columns=feature_cols,
        n_train=len(X_train), n_test=len(X_test), metrics=metrics, positive_label=positive_label,
        pipeline=pipeline,
    )


def train_generic_regressor(df: pd.DataFrame, target_col: str, feature_cols: list[str],
                             model_name: str = "Linear Regression") -> GenericModelResult:
    working = df[[target_col] + feature_cols].dropna(subset=[target_col])
    y = working[target_col].astype(float)
    X = working[feature_cols]

    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    model_fn = GENERIC_REGRESSION_MODELS.get(model_name, GENERIC_REGRESSION_MODELS["Linear Regression"])
    pipeline = Pipeline(steps=[
        ("preprocess", build_generic_preprocessor(numeric_cols, categorical_cols)),
        ("regressor", model_fn()),
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(root_mean_squared_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
    }
    return GenericModelResult(
        task="regression", model_name=model_name, target=target_col, feature_columns=feature_cols,
        n_train=len(X_train), n_test=len(X_test), metrics=metrics, pipeline=pipeline,
    )


@dataclass
class GenericClusterProfile:
    cluster_id: int
    count: int
    pct_of_total: float
    numeric_means: dict           # {col: mean} for the numeric feature columns used
    categorical_modes: dict       # {col: most-common value} for the categorical feature columns used


@dataclass
class GenericSegmentationResult:
    k: int
    labels: object
    elbow_scores: dict
    silhouette_scores: dict
    profiles: list
    pca_df: pd.DataFrame
    numeric_cols: list
    categorical_cols: list


def run_generic_segmentation(df: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str],
                              k: int, k_range: list[int] = CLUSTER_K_RANGE) -> GenericSegmentationResult:
    """Generic K-Means over WHATEVER numeric/categorical columns the caller
    selects (after excluding identifiers/target in app.py) - reuses the same
    calculate_elbow_scores/calculate_silhouette_scores used by the benchmark
    segmentation, which are already column-agnostic."""
    feature_df = df[numeric_cols + categorical_cols].copy()
    preprocessor = build_generic_preprocessor(numeric_cols, categorical_cols)
    X_transformed = preprocessor.fit_transform(feature_df)

    elbow = calculate_elbow_scores(X_transformed, k_range)
    silhouette = calculate_silhouette_scores(X_transformed, k_range)

    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(X_transformed)

    profiles = []
    total = len(feature_df)
    labeled_df = feature_df.copy()
    labeled_df["_cluster"] = labels
    for cluster_id in sorted(set(labels)):
        sub = labeled_df[labeled_df["_cluster"] == cluster_id]
        numeric_means = {c: round(float(sub[c].mean()), 2) for c in numeric_cols if pd.api.types.is_numeric_dtype(sub[c])}
        categorical_modes = {}
        for c in categorical_cols:
            mode = sub[c].mode()
            categorical_modes[c] = str(mode.iloc[0]) if not mode.empty else "N/A"
        profiles.append(GenericClusterProfile(
            cluster_id=int(cluster_id), count=len(sub), pct_of_total=round(100 * len(sub) / total, 1),
            numeric_means=numeric_means, categorical_modes=categorical_modes,
        ))

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_transformed)
    pca_df = pd.DataFrame({"pc1": coords[:, 0], "pc2": coords[:, 1], "cluster": labels.astype(str)})
    for c in (numeric_cols + categorical_cols)[:3]:  # a few original columns for hover context, no identifiers
        pca_df[c] = df[c].values

    return GenericSegmentationResult(
        k=k, labels=labels, elbow_scores=elbow, silhouette_scores=silhouette, profiles=profiles,
        pca_df=pca_df, numeric_cols=numeric_cols, categorical_cols=categorical_cols,
    )


# ---------------------------------------------------------------------------
# Manual test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from utils.data_processing import load_data, clean_data

    raw = load_data()
    clean, _ = clean_data(raw)
    results = train_and_evaluate(clean)

    print(f"Train rows: {results.n_train}, Test rows: {results.n_test}, "
          f"random_state={results.random_state}, test_size={results.test_size}")
    print("\n=== COMPARISON TABLE (CLASSIFICATION) ===")
    print(comparison_table(results).to_string(index=False))

    for name, ev in results.evaluations.items():
        print(f"\n=== {name} ===")
        print("Confusion matrix:\n", ev.confusion)
        if ev.coefficients:
            top = sorted(ev.coefficients.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
            print("Top |coefficient| features:", top)
        if ev.feature_importances:
            top = sorted(ev.feature_importances.items(), key=lambda kv: kv[1], reverse=True)[:5]
            print("Top importance features:", top)

    # sample prediction using the first test row's actual feature values
    sample_row = results.X_test.iloc[0].to_dict()
    pred = predict_single(results, "Logistic Regression", sample_row)
    print("\nSample prediction (Logistic Regression):", pred)

    print("\n\n=== STAGE 4: SALARY REGRESSION ===")
    reg_results = train_and_evaluate_salary(clean)
    print(f"Salary records: {reg_results.n_salary_records}, "
          f"Train: {reg_results.n_train}, Test: {reg_results.n_test}, "
          f"random_state={reg_results.random_state}")
    print("\n=== COMPARISON TABLE (REGRESSION) ===")
    print(regression_comparison_table(reg_results).to_string(index=False))

    for name, ev in reg_results.evaluations.items():
        print(f"\n=== {name} ===")
        if ev.coefficients:
            top = sorted(ev.coefficients.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
            print("Top |coefficient| features:", top)
        if ev.feature_importances:
            top = sorted(ev.feature_importances.items(), key=lambda kv: kv[1], reverse=True)[:5]
            print("Top importance features:", top)

    sample_row = reg_results.X_test.iloc[0].to_dict()
    pred = predict_salary(reg_results, "Linear Regression", sample_row)
    print("\nSample salary prediction (Linear Regression):", pred)

    print("\n\n=== STAGE 5: STUDENT SEGMENTATION ===")
    diagnostics = compute_segmentation_diagnostics(clean)
    print(f"Students used: {diagnostics.n_students}")
    print("Elbow (inertia) by K:", {k: round(v, 1) for k, v in diagnostics.elbow_scores.items()})
    print("Silhouette by K:", {k: (round(v, 3) if v is not None else None)
                                for k, v in diagnostics.silhouette_scores.items()})
    suggested_k = recommend_k(diagnostics)
    print(f"Diagnostic-suggested K (highest silhouette): {suggested_k}")

    seg = run_segmentation(clean, k=suggested_k, diagnostics=diagnostics)
    print(f"\nOverall academic avg: {seg.overall_academic_avg}%, overall workex%: {seg.overall_workex_pct}%")
    for p in seg.profiles:
        print(f"\nCluster {p.cluster_id} [{p.label}] - n={p.count} ({p.pct_of_total}%)")
        print(f"  SSC={p.ssc_avg} HSC={p.hsc_avg} Degree={p.degree_avg} "
              f"E-test={p.etest_avg} MBA={p.mba_avg} (academic avg={p.academic_avg})")
        print(f"  Work experience: {p.workex_yes_pct}% | Top stream: {p.top_hsc_stream} | "
              f"Top degree type: {p.top_degree_type} | Top specialisation: {p.top_specialisation}")

    print("\n=== POST-CLUSTERING OUTCOMES (descriptive only, not used to cluster) ===")
    print(post_clustering_outcomes(clean, seg.labels).to_string(index=False))

    print("\n=== SEGMENTATION INSIGHTS ===")
    for ins in generate_segmentation_insights(seg.profiles):
        print(ins)

    print("\n\n=== STAGE 7: WHAT-IF SIMULATOR ===")
    base = baseline_profile(clean)
    print("Baseline profile:", base)

    whatif = dict(base)
    whatif["ssc_p"] = min(100.0, base["ssc_p"] + 10)
    whatif["degree_p"] = min(100.0, base["degree_p"] + 10)
    whatif["workex"] = "Yes"

    for model_name in results.pipelines:
        cmp = compare_scenarios(results, model_name, base, whatif)
        print(f"\n{model_name}: baseline={cmp.prob_baseline:.3f} whatif={cmp.prob_whatif:.3f} "
              f"diff={cmp.diff_pp:+.1f}pp classes=({cmp.class_baseline} -> {cmp.class_whatif})")
        print("  Changes:", cmp.changes)

    curve = simulate_single_numeric_feature(results, "Logistic Regression", base, "degree_p", n_points=6)
    print("\nSingle-feature (degree_p) response curve, Logistic Regression:")
    print(curve.to_string(index=False))

    cat_df = simulate_categorical_feature(results, "Logistic Regression", base, "workex", ["Yes", "No"])
    print("\nCategorical (workex) comparison, Logistic Regression:")
    print(cat_df.to_string(index=False))
