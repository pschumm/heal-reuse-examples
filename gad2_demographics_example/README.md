# GAD-2 Anxiety Analysis: A Data Reuse Example for NIH HEAL Datasets

**Author:** Urja Mehta  
**Organization:** Renaissance Computing Institute (RENCI), UNC–Chapel Hill  
**Program:** HEAL Data Reuse Internship  
**Date:** April 2026

---

## Overview

This project demonstrates **data reuse** by combining two independent HEAL studies — HDP00619 and HDP01233 — both of which measured anxiety using the GAD-2 instrument alongside demographic variables. Neither study alone has enough participants for robust subgroup analysis, but together they form a dataset of 281 records used to explore how age, sex, ethnicity, marital status, and employment relate to anxiety severity.

### What is GAD-2?

The GAD-2 is a two-question screening tool for generalized anxiety disorder. Each question is scored 0–3, giving a total range of **0–6**. A score ≥ 6 is used here as the threshold for "higher anxiety."

---

## Research Questions

1. Is there a relationship between demographic variables and total anxiety scores?
2. Can we predict anxiety severity (GAD total score) using demographic features?
3. Can we classify individuals into high vs. low anxiety groups based on demographic characteristics?
4. Which demographic and socioeconomic factors are most strongly associated with anxiety levels, and which are most important for predicting anxiety classification?

---

## Datasets

| Dataset | Format | Records (after cleaning) |
|---|---|---|
| HDP00619 | CSV | 82 |
| HDP01233 | Excel (demographics + GAD-2) | 199 |
| **Combined** | — | **281** |

Variables used (HEAL standard names): `Age`, `Sex`, `ETHNIC`, `MARISTAT`, `EMPSTAT`, `GAD2FeelNervScale`, `GAD2NotStopWryScale`

---

## How It Works: Notebook Walkthrough

The analysis is split across two notebooks — `analysis_functions.ipynb` defines all helper functions, and `presentation.ipynb` runs the full analysis end-to-end.

### 1. Imports and Setup
Loads libraries and runs `analysis_functions.ipynb` via `%run` to make all helper functions available.

### 2. Load, Standardize, and Clean Data
Raw data is loaded first and displayed so you can see the original column names and structure. The key challenge — different studies use different column names for the same variables (e.g., `gender` vs `sex`, `gad701` vs `gad2feelnervscale`) — is solved using **HEAL Common Data Element (CDE) mappings** stored in `heal-mappings.json`. This file maps each standard variable name to the study-specific column name, so cleaning is programmatic rather than manual. After cleaning, both datasets use the same standard column names and value encodings.

### 3. Merge the Datasets
Both cleaned datasets are previewed, then concatenated into a single 281-record dataset. The shared structure (enabled by CDEs) makes this a straightforward `pd.concat`.

### 4. Research Question 1 — Exploratory Analysis
A 2×3 panel shows each demographic variable against GAD total score:
- **Age**: scatter plot with linear regression line
- **Sex, Ethnicity, Marital Status, Employment**: box plots showing the distribution of GAD scores per category, with labels drawn from the HEAL standard variable definitions

### 4b. Cluster Analysis
Since the two studies recruited distinct populations (pain patients vs. OUD patients), K-means clustering and PCA visualization are used to ask: do the two populations occupy separate regions of the feature space, or do anxiety-driven patterns cut across study boundaries? This is a direct test of the value of combining the datasets.

### 5. Prepare Data for Modeling
Features and targets are prepared. Class imbalance (few high-anxiety cases) is addressed with:
- **Stratified train/test split** — preserves the high-anxiety proportion in both sets
- **Balanced class weights** in all classifiers — penalizes errors on the minority class more heavily

### 6. Research Question 2 — Linear Regression
Linear regression predicts GAD total score. The coefficient plot (addressing RQ4) identifies which features have the strongest linear association with anxiety.

### 7. Research Question 3 — Classification Models
Logistic regression, decision tree, and random forest classify individuals as high vs. low anxiety. The confusion matrix and F1 score for the high-anxiety class are the key evaluation metrics — accuracy alone is misleading with imbalanced classes.

### 8. Research Question 4 — Random Forest Feature Importance
Random forest feature importance ranks which demographic variables are most predictive of anxiety classification, complementing the regression coefficients from the previous section.

### 9. Answers to Research Questions
A summary cell prints answers to all five research questions using the computed model results.

---

## Repository Structure

```
gad2_demographics_example/
├── data/
│   ├── heal-mappings.json          # HEAL CDE variable mappings
│   ├── HDP00619.csv
│   ├── HDP01233_Demographics.xlsx
│   └── HDP01233_Gad2.xlsx
├── Notebooks/
│   ├── analysis_functions.ipynb    # All helper functions (cleaning, plotting, models)
│   └── presentation.ipynb          # Main analysis notebook
└── README.md
```

---

## Setup

**Requirements:**
```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
scikit-learn>=1.2.0
openpyxl>=3.1.0
jupyter>=1.0.0
```

**Install and run:**
```bash
git clone https://github.com/heal-data-stewards/heal-reuse-examples.git
cd heal-reuse-examples/gad2_demographics_example
pip install -r requirements.txt
jupyter notebook Notebooks/presentation.ipynb
```

**To modify functions:** edit `analysis_functions.ipynb` — `presentation.ipynb` imports it automatically via `%run`.

---

## Limitations

- **Cross-sectional data**: Cannot establish causation
- **Small combined sample**: 281 records limits statistical power, particularly for subgroup analyses
- **Class imbalance**: High-anxiety cases are rare in this sample, making classification challenging
- **Limited features**: Income, education, and clinical history are not available but likely relevant
- **Dataset heterogeneity**: HDP00619 and HDP01233 recruited participants under different study contexts

---

## Acknowledgments

Completed as part of the HEAL Data Reuse internship at RENCI, UNC–Chapel Hill.  
Thanks to: Hina Shah, Liezl Mae Fos, Julie Horvath, and the NIH HEAL Initiative.

---

## License

MIT License — see [LICENSE](../../LICENSE) for details.
