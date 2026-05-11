#!/usr/bin/env python
# coding: utf-8

# # Analysis Functions for GAD-2 Dataset
# This notebook contains all data processing, cleaning, and analysis functions.

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, confusion_matrix, classification_report


# ## Data Loading Functions

def load_dataset(path):
    """Load a single dataset from a CSV or Excel file, inferred from the file extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path, low_memory=False)
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


# ## Mapping Utility Functions

def load_mappings(path):
    """Load HEAL variable mappings from a JSON file."""
    with open(path) as f:
        return json.load(f)


def get_study_variable(standard_id, study_id, mappings):
    """Return the first study-specific column name for a given standard variable ID and study."""
    for var in mappings["variables"]:
        if var["id"].lower() == standard_id.lower():
            names = var["metadata"].get("study_variable_mappings", {}).get(study_id, [])
            return names[0] if names else None
    return None


def get_standard_labels(standard_id, mappings):
    """Return {int_value: short_label} from a standard variable's permissible_values."""
    for var in mappings["variables"]:
        if var["id"].lower() == standard_id.lower():
            pv = var["metadata"].get("permissible_values", {})
            return {int(k): v.split(",")[0].strip() for k, v in pv.items()}
    return {}


# ## Recoding Functions

def recode_empstat(x):
    """Recode employment status values - used only for HDP00619."""
    if x == 1:
        return 1
    elif x in [3, 4, 5, 6]:
        return 2
    elif x in [2, 7]:
        return 3
    elif x == 8:
        return 4
    else:
        return x


def recode_sex_hdp1233(x):
    """Recode sex values for HDP01233 to HDP00619 values."""
    if x in [1, 2]:
        return x
    elif x in [3, 4]:
        return 0
    else:
        return np.nan

def recode_sex_hdp619(x):
    """Recode sex values for HDP00619 to CDE labels."""
    if x == 1:
        return x # Male
    elif x == 0:
        return 2 # Female
    elif x == 2:
        return 3 # Unknown
    else:
        return np.nan

def recode_ethnic(x):
    """Recode ethnicity values for HDP01233."""
    if x == 1:
        return 1
    elif x in [2, 3, 4]:
        return 0
    else:
        return np.nan

def recode_ethnic_hdp619(x):
    """Recode ethnicity values for HDP00619 to Standard."""
    if x == 1:
        return 1 # Hispanic or Latino
    elif x == 0:
        return 2 # Not Hispanic or Latino
    else:
        return np.nan

def recode_maristat_hdp619(x):
    """Recode marital status values for HDP00619 to Standard Variables."""
    if x==4:
        return 1 # Divorced
    elif x==3:
        return 2 # Married
    elif x==1:
        return 3 # Never married
    elif x==5:
        return 5 # Widowed
    elif x==2:
        return 6 # Domestic Partner
    else:
        return np.nan


# ## Data Cleaning Functions

def clean_hdp00619(df_619, mappings=None):
    """Clean and process HDP00619 dataset."""
    study_id = "HDP00619"

    df_619.columns = df_619.columns.str.strip()

    standard_ids = ["Age", "Sex", "ETHNIC", "MARISTAT", "EMPSTAT", "GAD2FeelNervScale", "GAD2NotStopWryScale"]
    study_mappings = {}
    if mappings:
        print(f"  Variable mappings for {study_id}:")
        for std_id in ["Age", "Sex", "ETHNIC", "MARISTAT", "EMPSTAT", "GAD2FeelNervScale", "GAD2NotStopWryScale"]:
            col = get_study_variable(std_id, study_id, mappings)
            study_mappings[std_id] = col
            print(f"    {std_id:<26} -> '{col}'")

    cols_619 = [study_mappings[s] for s in standard_ids]

    df_619 = df_619[ cols_619 + ["scrid"]]

    df_619 = (
        df_619
        .groupby("scrid", as_index=False)
        .agg(lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else np.nan)
    )
    df_619 = df_619.drop(columns=["scrid"])
    df_619 = df_619.dropna()

    df_619[cols_619] = df_619[cols_619].astype(int)

    df_619["gad_total"] = df_619[study_mappings['GAD2FeelNervScale']] + df_619[study_mappings['GAD2NotStopWryScale']]
    df_619["higher_anxiety"] = (df_619["gad_total"] >= 6).astype(int)
    df_619[study_mappings['EMPSTAT']] = df_619[study_mappings['EMPSTAT']].apply(recode_empstat)
    df_619[study_mappings['Sex']] = df_619[study_mappings['Sex']].apply(recode_sex_hdp619)
    df_619[study_mappings['MARISTAT']] = df_619[study_mappings['MARISTAT']].apply(recode_maristat_hdp619)
    df_619[study_mappings['ETHNIC']] = df_619[study_mappings['ETHNIC']].apply(recode_ethnic_hdp619)

    df_619.rename(columns={study_mappings[v]:v for v in standard_ids}, inplace=True)
    return df_619


def clean_hdp01233(df_1233_demo, df_1233_gad, mappings=None):
    """Clean and process HDP01233 dataset."""
    study_id = "HDP01233"
    standard_ids = ["Age", "Sex", "ETHNIC", "MARISTAT", "EMPSTAT", "GAD2FeelNervScale", "GAD2NotStopWryScale"]
    study_mappings = {}
    if mappings:
        print(f"  Variable mappings for {study_id}:")
        for std_id in standard_ids:
            col = get_study_variable(std_id, study_id, mappings)
            study_mappings[std_id] = col
            print(f"    {std_id:<26} -> '{col}'")

    df_1233 = pd.merge(df_1233_demo, df_1233_gad, on="subject_id", how="inner")

    cols_1233 = [study_mappings[s] for s in standard_ids]
    cols_keep = cols_1233 + ["visit"]
    df_1233 = df_1233[cols_keep]
    df_1233 = df_1233[~df_1233["visit"].astype(str).str.contains("2")]
    df_1233 = df_1233.dropna()

    # cols_to_int = ["age", "sex", "ethnic", "marstat", "empstat", "gad701", "gad702"]
    df_1233[cols_1233] = df_1233[cols_1233].astype(int)

    df_1233["gad_total"] = df_1233[study_mappings['GAD2FeelNervScale']] + df_1233[study_mappings['GAD2NotStopWryScale']]
    df_1233["higher_anxiety"] = (df_1233["gad_total"] >= 6).astype(int)
    df_1233 = df_1233.drop(columns=["visit"])

    df_1233.rename(columns={study_mappings[v]:v for v in standard_ids}, inplace=True)
    return df_1233


# ## Plotting Functions

def plot_scatter_comparison(df_619, df_1233, variable, variable_label=None):
    """Create scatter plot comparing a variable vs GAD total for both datasets."""
    label = variable_label if variable_label else variable
    plt.figure(figsize=(8, 6))

    plt.scatter(df_619[variable], df_619["gad_total"], label="HDP00619", alpha=0.6)
    plt.scatter(df_1233[variable], df_1233["gad_total"], label="HDP01233", alpha=0.6)

    plt.xlabel(label)
    plt.ylabel("GAD Total Score")
    plt.title(f"{label} vs GAD Total Score (Two Datasets)")
    legend = plt.legend(loc='upper right', fontsize=11, framealpha=0.9)
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(1.5)
    plt.tight_layout()
    plt.show()


def plot_feature_vs_gad(df, feature_name, feature_label, ax=None, label_map=None):
    """Plot a feature vs total GAD score.

    Categorical (label_map provided): box plot showing GAD distribution per category.
    Continuous (no label_map): scatter plot with linear regression line.
    Pass ax to draw into an existing subplot; omit to create a standalone figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        standalone = True
    else:
        standalone = False

    if label_map is not None:
        # Categorical: box plot per category
        categories = sorted(label_map.keys())
        groups = [df[df[feature_name] == v]["gad_total"].dropna().values for v in categories]
        bp = ax.boxplot(
            groups,
            patch_artist=True,
            medianprops=dict(color="red", linewidth=2),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor("lightsteelblue")
        ax.set_xticks(range(1, len(categories) + 1))
        ax.set_xticklabels([label_map[k] for k in categories], rotation=30, ha="right")
    else:
        # Continuous: scatter + linear regression
        ax.scatter(df[feature_name], df["gad_total"], alpha=0.5, edgecolors="k", linewidths=0.5)

        X_feat = df[[feature_name]].values
        y_gad = df["gad_total"].values
        mask = ~(np.isnan(X_feat.flatten()) | np.isnan(y_gad) |
                 np.isinf(X_feat.flatten()) | np.isinf(y_gad))
        X_feat_clean = X_feat[mask].reshape(-1, 1)
        y_gad_clean = y_gad[mask]

        if len(X_feat_clean) > 0:
            lr = LinearRegression()
            lr.fit(X_feat_clean, y_gad_clean)
            X_line = np.linspace(X_feat_clean.min(), X_feat_clean.max(), 100).reshape(-1, 1)
            y_line = lr.predict(X_line)
            ax.plot(X_line, y_line, "r-", linewidth=2,
                    label=f"Linear Fit (coef={lr.coef_[0]:.3f})")
            ax.legend(fontsize=8)

    ax.set_xlabel(feature_label)
    ax.set_ylabel("GAD Total Score")
    ax.set_title(f"{feature_label} vs GAD Total Score")

    if standalone:
        plt.tight_layout()
        plt.show()


# ## Cluster Analysis Functions

def run_kmeans(df, features, n_clusters=3, random_state=42):
    """Scale features, run K-means, and project to 2D via PCA.
    Returns cluster labels, PCA coordinates, and variance explained per component.
    """
    X_scaled = StandardScaler().fit_transform(df[features])
    labels = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit_predict(X_scaled)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    return labels, coords, pca.explained_variance_ratio_


def plot_pca_clusters(coords, color_values, color_label, title, ax=None):
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


def get_cluster_profiles(df, features, cluster_labels):
    """Return mean feature values per cluster as a DataFrame."""
    tmp = df[features].copy()
    tmp['Cluster'] = cluster_labels
    return tmp.groupby('Cluster')[features].mean().round(2)


# ## Model Preparation Functions

def prepare_model_data(df):
    """Prepare features and targets for modeling."""
    X = df[['Age', 'Sex', 'MARISTAT', 'EMPSTAT', 'ETHNIC']]
    y_linear = df['gad_total']
    y_logistic = df['higher_anxiety']

    X = X.apply(pd.to_numeric)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    return X, y_linear, y_logistic


# ## Model Training Functions

def train_linear_regression(X_train, y_train_lin, X_test, y_test_lin):
    """Train and evaluate linear regression model."""
    lin_model = LinearRegression()
    lin_model.fit(X_train, y_train_lin)
    y_pred_lin = lin_model.predict(X_test)
    return {
        'model': lin_model,
        'r2': r2_score(y_test_lin, y_pred_lin),
        'rmse': np.sqrt(mean_squared_error(y_test_lin, y_pred_lin)),
        'predictions': y_pred_lin,
    }


def train_logistic_regression(X_train, y_train_log, X_test, y_test_log):
    """Train and evaluate logistic regression model with balanced class weights."""
    log_model = LogisticRegression(max_iter=1000, class_weight='balanced')
    log_model.fit(X_train, y_train_log)
    y_pred_log = log_model.predict(X_test)
    return {
        'model': log_model,
        'accuracy': accuracy_score(y_test_log, y_pred_log),
        'confusion_matrix': confusion_matrix(y_test_log, y_pred_log),
        'classification_report': classification_report(y_test_log, y_pred_log),
        'predictions': y_pred_log,
    }


def train_decision_tree(X_train, y_train_log, X_test, y_test_log, X):
    """Train and evaluate decision tree classifier with balanced class weights."""
    tree_model = DecisionTreeClassifier(random_state=42, class_weight='balanced')
    tree_model.fit(X_train, y_train_log)
    y_pred_tree = tree_model.predict(X_test)
    return {
        'model': tree_model,
        'accuracy': accuracy_score(y_test_log, y_pred_tree),
        'predictions': y_pred_tree,
        'feature_names': X.columns,
    }


def train_random_forest(X_train, y_train_log, X_test, y_test_log):
    """Train and evaluate random forest classifier with balanced class weights."""
    rf_model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train_log)
    y_pred_rf = rf_model.predict(X_test)
    return {
        'model': rf_model,
        'accuracy': accuracy_score(y_test_log, y_pred_rf),
        'predictions': y_pred_rf,
    }


# ## Visualization Functions for Model Results

def plot_decision_tree(tree_model, feature_names):
    """Visualize decision tree."""
    plt.figure(figsize=(20, 10))
    plot_tree(
        tree_model,
        feature_names=feature_names,
        class_names=["Low Anxiety", "High Anxiety"],
        filled=True
    )
    plt.tight_layout()
    plt.show()


def plot_feature_importance(rf_model, X):
    """Plot random forest feature importance."""
    importance = pd.Series(
        rf_model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    importance.plot(kind="bar")
    plt.title("Random Forest Feature Importance")
    plt.ylabel("Importance")
    plt.xlabel("Features")
    plt.tight_layout()
    plt.show()

    return importance


def plot_linear_coefficients(lin_model, X):
    """Plot linear regression coefficients."""
    coefficients = pd.Series(
        lin_model.coef_,
        index=X.columns
    ).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    coefficients.plot(kind="bar")
    plt.title("Linear Regression Coefficients")
    plt.ylabel("Coefficient Value")
    plt.xlabel("Features")
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    return coefficients

