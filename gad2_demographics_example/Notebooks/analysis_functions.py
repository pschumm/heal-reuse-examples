"""
Analysis functions for the GAD-2 data reuse example.

Covers: data loading, HEAL CDE mapping utilities, per-study cleaning/recoding,
comparison plots, cluster analysis, model training, and result visualizations.
"""

import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> pd.DataFrame:
    """Load a CSV or Excel file, inferred from extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path, low_memory=False)
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file format: {ext}")


# ---------------------------------------------------------------------------
# HEAL CDE mapping utilities
# ---------------------------------------------------------------------------

def load_mappings(path: str) -> dict:
    """Load HEAL variable mappings from a JSON file."""
    with open(path) as f:
        return json.load(f)


def get_study_variable(standard_id: str, study_id: str, mappings: dict):
    """Return the first study-specific column name for a standard variable ID."""
    for var in mappings["variables"]:
        if var["id"].lower() == standard_id.lower():
            names = var["metadata"].get("study_variable_mappings", {}).get(study_id, [])
            return names[0] if names else None
    return None


def get_standard_labels(standard_id: str, mappings: dict) -> dict:
    """Return {int_value: short_label} from a standard variable's permissible_values."""
    for var in mappings["variables"]:
        if var["id"].lower() == standard_id.lower():
            pv = var["metadata"].get("permissible_values", {})
            return {int(k): v.split(",")[0].strip() for k, v in pv.items()}
    return {}


# ---------------------------------------------------------------------------
# Recoding helpers
# ---------------------------------------------------------------------------

def recode_empstat(x):
    """Recode employment status values — used only for HDP00619."""
    if x == 1:
        return 1
    elif x in [3, 4, 5, 6]:
        return 2
    elif x in [2, 7]:
        return 3
    elif x == 8:
        return 4
    return x


def recode_sex_hdp1233(x):
    """Recode sex values for HDP01233 to HDP00619 encoding."""
    if x in [1, 2]:
        return x
    elif x in [3, 4]:
        return 0
    return np.nan


def recode_sex_hdp619(x):
    """Recode sex values for HDP00619 to CDE labels."""
    if x == 1:
        return 1   # Male
    elif x == 0:
        return 2   # Female
    elif x == 2:
        return 3   # Unknown
    return np.nan


def recode_ethnic(x):
    """Recode ethnicity values for HDP01233."""
    if x == 1:
        return 1
    elif x in [2, 3, 4]:
        return 0
    return np.nan


def recode_ethnic_hdp619(x):
    """Recode ethnicity values for HDP00619 to standard encoding."""
    if x == 1:
        return 1   # Hispanic or Latino
    elif x == 0:
        return 2   # Not Hispanic or Latino
    return np.nan


def recode_maristat_hdp619(x):
    """Recode marital status values for HDP00619 to standard encoding."""
    mapping = {4: 1, 3: 2, 1: 3, 5: 5, 2: 6}
    return mapping.get(x, np.nan)


# ---------------------------------------------------------------------------
# Per-study cleaning
# ---------------------------------------------------------------------------

def clean_hdp00619(df_619: pd.DataFrame, mappings: dict) -> pd.DataFrame:
    """Clean and standardize the HDP00619 dataset."""
    study_id = "HDP00619"
    standard_ids = ["Age", "Sex", "ETHNIC", "MARISTAT", "EMPSTAT",
                    "GAD2FeelNervScale", "GAD2NotStopWryScale"]

    df_619.columns = df_619.columns.str.strip()

    print(f"  Variable mappings for {study_id}:")
    study_mappings = {}
    for std_id in standard_ids:
        col = get_study_variable(std_id, study_id, mappings)
        study_mappings[std_id] = col
        print(f"    {std_id:<26} -> '{col}'")

    cols_619 = [study_mappings[s] for s in standard_ids]
    df_619 = df_619[cols_619 + ["scrid"]]
    df_619 = (
        df_619
        .groupby("scrid", as_index=False)
        .agg(lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else np.nan)
    )
    df_619 = df_619.drop(columns=["scrid"]).dropna()
    df_619[cols_619] = df_619[cols_619].astype(int)

    df_619["gad_total"] = (
        df_619[study_mappings["GAD2FeelNervScale"]]
        + df_619[study_mappings["GAD2NotStopWryScale"]]
    )
    df_619["higher_anxiety"] = (df_619["gad_total"] >= 6).astype(int)
    df_619[study_mappings["EMPSTAT"]]   = df_619[study_mappings["EMPSTAT"]].apply(recode_empstat)
    df_619[study_mappings["Sex"]]       = df_619[study_mappings["Sex"]].apply(recode_sex_hdp619)
    df_619[study_mappings["MARISTAT"]]  = df_619[study_mappings["MARISTAT"]].apply(recode_maristat_hdp619)
    df_619[study_mappings["ETHNIC"]]    = df_619[study_mappings["ETHNIC"]].apply(recode_ethnic_hdp619)

    df_619.rename(columns={study_mappings[v]: v for v in standard_ids}, inplace=True)
    return df_619


def clean_hdp01233(
    df_1233_demo: pd.DataFrame,
    df_1233_gad: pd.DataFrame,
    mappings: dict,
) -> pd.DataFrame:
    """Clean and standardize the HDP01233 dataset."""
    study_id = "HDP01233"
    standard_ids = ["Age", "Sex", "ETHNIC", "MARISTAT", "EMPSTAT",
                    "GAD2FeelNervScale", "GAD2NotStopWryScale"]

    print(f"  Variable mappings for {study_id}:")
    study_mappings = {}
    for std_id in standard_ids:
        col = get_study_variable(std_id, study_id, mappings)
        study_mappings[std_id] = col
        print(f"    {std_id:<26} -> '{col}'")

    df_1233 = pd.merge(df_1233_demo, df_1233_gad, on="subject_id", how="inner")
    cols_1233 = [study_mappings[s] for s in standard_ids]
    df_1233 = df_1233[cols_1233 + ["visit"]]
    df_1233 = df_1233[~df_1233["visit"].astype(str).str.contains("2")]
    df_1233 = df_1233.dropna()
    df_1233[cols_1233] = df_1233[cols_1233].astype(int)

    df_1233["gad_total"] = (
        df_1233[study_mappings["GAD2FeelNervScale"]]
        + df_1233[study_mappings["GAD2NotStopWryScale"]]
    )
    df_1233["higher_anxiety"] = (df_1233["gad_total"] >= 6).astype(int)
    df_1233 = df_1233.drop(columns=["visit"])

    df_1233.rename(columns={study_mappings[v]: v for v in standard_ids}, inplace=True)
    return df_1233


# ---------------------------------------------------------------------------
# Comparison plots
# ---------------------------------------------------------------------------

def plot_scatter_comparison(
    df_619: pd.DataFrame,
    df_1233: pd.DataFrame,
    variable: str,
    variable_label: str | None = None,
) -> None:
    """Scatter plot of a variable vs GAD total score for both datasets."""
    label = variable_label or variable
    plt.figure(figsize=(8, 6))
    plt.scatter(df_619[variable], df_619["gad_total"], label="HDP00619", alpha=0.6)
    plt.scatter(df_1233[variable], df_1233["gad_total"], label="HDP01233", alpha=0.6)
    plt.xlabel(label)
    plt.ylabel("GAD Total Score")
    plt.title(f"{label} vs GAD Total Score (Two Datasets)")
    legend = plt.legend(loc="upper right", fontsize=11, framealpha=0.9)
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_linewidth(1.5)
    plt.tight_layout()
    plt.show()


def plot_feature_vs_gad(
    df: pd.DataFrame,
    feature_name: str,
    feature_label: str,
    ax=None,
    label_map: dict | None = None,
) -> None:
    """Plot a feature against GAD total score.

    Categorical (label_map provided): box plot per category.
    Continuous (no label_map): scatter plot with linear regression line.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 6))

    if label_map is not None:
        categories = sorted(label_map.keys())
        groups = [df[df[feature_name] == v]["gad_total"].dropna().values for v in categories]
        bp = ax.boxplot(groups, patch_artist=True, medianprops=dict(color="red", linewidth=2))
        for patch in bp["boxes"]:
            patch.set_facecolor("lightsteelblue")
        ax.set_xticks(range(1, len(categories) + 1))
        ax.set_xticklabels([label_map[k] for k in categories], rotation=30, ha="right")
    else:
        ax.scatter(df[feature_name], df["gad_total"], alpha=0.5, edgecolors="k", linewidths=0.5)
        X_feat = df[[feature_name]].values
        y_gad = df["gad_total"].values
        mask = ~(np.isnan(X_feat.flatten()) | np.isnan(y_gad) |
                 np.isinf(X_feat.flatten()) | np.isinf(y_gad))
        X_clean, y_clean = X_feat[mask].reshape(-1, 1), y_gad[mask]
        if len(X_clean) > 0:
            lr = LinearRegression().fit(X_clean, y_clean)
            X_line = np.linspace(X_clean.min(), X_clean.max(), 100).reshape(-1, 1)
            ax.plot(X_line, lr.predict(X_line), "r-", linewidth=2,
                    label=f"Linear Fit (coef={lr.coef_[0]:.3f})")
            ax.legend(fontsize=8)

    ax.set_xlabel(feature_label)
    ax.set_ylabel("GAD Total Score")
    ax.set_title(f"{feature_label} vs GAD Total Score")

    if standalone:
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Cluster analysis
# ---------------------------------------------------------------------------

def run_kmeans(
    df: pd.DataFrame,
    features: list,
    n_clusters: int = 3,
    random_state: int = 42,
):
    """Scale features, run K-means, and project to 2D via PCA.

    Returns (cluster_labels, pca_coords, variance_explained).
    """
    X_scaled = StandardScaler().fit_transform(df[features])
    labels = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit_predict(X_scaled)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    return labels, coords, pca.explained_variance_ratio_


def plot_pca_clusters(
    coords,
    color_values: list,
    color_label: str,
    title: str,
    ax=None,
) -> None:
    """PCA scatter plot colored by a categorical grouping variable."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 5))
    unique = sorted(set(color_values))
    colors = plt.cm.Set1.colors
    for i, val in enumerate(unique):
        idx = [j for j, v in enumerate(color_values) if v == val]
        ax.scatter(coords[idx, 0], coords[idx, 1], label=str(val),
                   alpha=0.6, color=colors[i % len(colors)])
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend(title=color_label, fontsize=8)
    if standalone:
        plt.tight_layout()
        plt.show()


def get_cluster_profiles(
    df: pd.DataFrame,
    features: list,
    cluster_labels,
) -> pd.DataFrame:
    """Return mean feature values per cluster."""
    tmp = df[features].copy()
    tmp["Cluster"] = cluster_labels
    return tmp.groupby("Cluster")[features].mean().round(2)


# ---------------------------------------------------------------------------
# Model preparation
# ---------------------------------------------------------------------------

def prepare_model_data(df: pd.DataFrame):
    """Return (X, y_continuous, y_binary) ready for modeling."""
    X = df[["Age", "Sex", "MARISTAT", "EMPSTAT", "ETHNIC"]].apply(pd.to_numeric)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X, df["gad_total"], df["higher_anxiety"]


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_linear_regression(X_train, y_train, X_test, y_test) -> dict:
    """Train and evaluate a linear regression model."""
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "model": model,
        "r2":    r2_score(y_test, y_pred),
        "rmse":  np.sqrt(mean_squared_error(y_test, y_pred)),
        "predictions": y_pred,
    }


def train_logistic_regression(X_train, y_train, X_test, y_test) -> dict:
    """Train and evaluate a logistic regression model with balanced class weights."""
    model = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "model":                 model,
        "accuracy":              accuracy_score(y_test, y_pred),
        "confusion_matrix":      confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
        "predictions":           y_pred,
    }


def train_decision_tree(X_train, y_train, X_test, y_test, X) -> dict:
    """Train and evaluate a decision tree with balanced class weights."""
    model = DecisionTreeClassifier(random_state=42, class_weight="balanced").fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "model":         model,
        "accuracy":      accuracy_score(y_test, y_pred),
        "predictions":   y_pred,
        "feature_names": X.columns,
    }


def train_random_forest(X_train, y_train, X_test, y_test) -> dict:
    """Train and evaluate a random forest with balanced class weights."""
    model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "model":       model,
        "accuracy":    accuracy_score(y_test, y_pred),
        "predictions": y_pred,
    }


# ---------------------------------------------------------------------------
# Result visualizations
# ---------------------------------------------------------------------------

def plot_decision_tree(tree_model, feature_names) -> None:
    """Visualize a fitted decision tree."""
    plt.figure(figsize=(20, 10))
    plot_tree(tree_model, feature_names=feature_names,
              class_names=["Low Anxiety", "High Anxiety"], filled=True)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(rf_model, X: pd.DataFrame) -> pd.Series:
    """Bar chart of random forest feature importances. Returns the Series."""
    importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    importance.plot(kind="bar")
    plt.title("Random Forest Feature Importance")
    plt.ylabel("Importance")
    plt.xlabel("Features")
    plt.tight_layout()
    plt.show()
    return importance


def plot_linear_coefficients(lin_model, X: pd.DataFrame) -> pd.Series:
    """Bar chart of linear regression coefficients. Returns the Series."""
    coefficients = pd.Series(lin_model.coef_, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    coefficients.plot(kind="bar")
    plt.title("Linear Regression Coefficients")
    plt.ylabel("Coefficient Value")
    plt.xlabel("Features")
    plt.axhline(y=0, color="r", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
    return coefficients
