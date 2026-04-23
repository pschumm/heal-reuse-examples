# GAD-2 Anxiety Analysis: Demographic Predictors of Anxiety

**Author:** Urja Mehta  
**Organization:** Renaissance Computing Institute (RENCI), UNC–Chapel Hill  
**Program:** HEAL Data Reuse Internship  
**Date:** April 2026

---

## Project Overview

This project analyzes anxiety levels using the GAD-2 (Generalized Anxiety Disorder-2) screening tool across two large healthcare datasets: **HDP00619** and **HDP01233**. The goal is to identify demographic and socioeconomic predictors of anxiety and develop machine learning models to predict anxiety severity and classify individuals into risk categories.

As part of the HEAL (Helping to End Addiction Long-term) Data Reuse initiative, this work demonstrates how healthcare data can be integrated, cleaned, and analyzed to generate actionable insights for mental health research.

---

## Background

### What is GAD-2?

The GAD-2 is a brief, validated screening tool for generalized anxiety disorder consisting of two questions:

1. Feeling nervous, anxious, or on edge (gad701)  
2. Not being able to stop or control worrying (gad702)

Scores range from **0–6**, with scores **≥3** indicating possible anxiety disorder requiring further evaluation.

---

### Datasets Used

- **HDP00619**: CSV format, 619 participant records with demographics and GAD-2 scores  
- **HDP01233**: Excel format (Demographics + GAD-2)
- **Combined Dataset**: 1,852 total participants after data cleaning and integration  

---

## Research Questions

1. **What demographic and socioeconomic factors are most strongly associated with anxiety levels (GAD-2 scores)?**

2. **Can we predict anxiety severity (GAD total score) using demographic features such as age, sex, ethnicity, marital status, and employment?**

3. **What is the relationship between each individual demographic feature and total anxiety scores?**

4. **Can we classify individuals into high vs. low anxiety groups based on their demographic characteristics?**

5. **Which features are most important for predicting anxiety classification?**

---

## Technical Approach

### Data Processing Pipeline

```
Raw Data → Data Cleaning → Feature Engineering → Model Training → Validation
```

### Data Processing Steps

- **Data Integration**: Merged heterogeneous datasets  
- **Data Cleaning**: Removed duplicates, handled missing values, standardized variables  
- **Feature Engineering**:
  - Created `gad_total`
  - Created `higher_anxiety` (gad_total ≥ 6)
  - One-hot encoded categorical variables  

---

### Features Analyzed

| Feature | Description | Type |
|--------|------------|------|
| age | Age in years | Continuous |
| sex | Sex/Gender | Categorical |
| ethnic | Ethnicity | Binary |
| marstat | Marital status | Categorical |
| empstat | Employment status | Categorical |

**Note:** `height` was excluded due to no theoretical relevance.

---

## Machine Learning Models

- **Linear Regression** → Predict anxiety score  
- **Logistic Regression** → Classify high vs. low anxiety  
- **Decision Tree** → Interpretable rules  
- **Random Forest** → Best-performing ensemble model  

---

## Repository Structure

```
.
├── gad2_demographics_example/
│   ├── analysis_functions.ipynb
│   ├── presentation.ipynb
│   └── data/
├── README.md
├── LICENSE
├── .gitignore
```

## Setup and Installation

### Requirements
```bash
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
scikit-learn>=1.2.0
openpyxl>=3.1.0  # For Excel file support
jupyter>=1.0.0
```


## Setup

```bash
git clone https://github.com/heal-data-stewards/heal-reuse-examples.git
cd GAD2-Anxiety-Analysis
pip install -r requirements.txt
jupyter notebook

## Usage

### For Presentations
1. Open `presentation.ipynb`
2. Ensure data files are in the same directory
3. Run all cells sequentially
4. The notebook will:
   - Display research questions
   - Load and clean data
   - Generate exploratory plots
   - Train all models
   - Answer research questions with results

### For Development
1. Open `analysis_functions.ipynb` to modify functions
2. The presentation notebook automatically imports these functions using `%run`
3. Make changes to functions and re-run presentation to see updates

## Key Findings

### Model Performance
- **Linear Regression**: R² = 0.1145
- **Logistic Regression**: Accuracy = 0.9649
- **Decision Tree**: Accuracy = 0.9474
- **Random Forest**: Accuracy = 0.9649

### Most Important Predictors
Based on Random Forest feature importance:
1. Age
2. Ethnicity
3. mployment Status


## Data Privacy and Ethics

- All data used in this analysis is from de-identified, publicly available research datasets
- No personally identifiable information (PII) is included
- Analysis follows HIPAA compliance guidelines for healthcare data
- This work is part of the NIH HEAL Initiative promoting responsible data reuse

## Limitations

1. **Cross-sectional data**: Cannot establish causation
2. **Self-reported measures**: Subject to response bias
3. **Limited features**: Other factors (income, education, medical history) not available
4. **Dataset heterogeneity**: Different collection methods between HDP00619 and HDP01233

## Future Directions

- Incorporate additional socioeconomic variables (income, education level)
- Longitudinal analysis to track anxiety changes over time
- Deep learning approaches for more complex pattern recognition
- External validation on independent datasets
- Development of web-based risk calculator for clinical use

## Technical Contributions

This project demonstrates:
- **Data Integration**: Merging heterogeneous healthcare datasets
- **Data Quality**: Robust cleaning and validation pipelines
- **Reproducibility**: Modular, well-documented code
- **Visualization**: Clear, publication-quality plots
- **Machine Learning**: Multiple modeling approaches with proper validation
- **Research Communication**: Question-driven analysis framework

## Acknowledgments

This work was completed as part of the HEAL Data Reuse internship at the Renaissance Computing Institute (RENCI), UNC–Chapel Hill. Special thanks to:
- RENCI research computing infrastructure
- NIH HEAL Initiative for data access
- Hina Shah, Liezl Mae Fos. Julie Horvath

## Citation

If you use this code or analysis approach, please cite:

```
Urja Mehta. (2026). GAD-2 Anxiety Analysis: Demographic Predictors of Anxiety. 
RENCI HEAL Data Reuse Project. University of North Carolina at Chapel Hill.
```

## Contact

**Urja Mehta**  
ata Analyst Research Intern 
Renaissance Computing Institute (RENCI)  
University of North Carolina at Chapel Hill  
Email: urjam@renci.org  
GitHub: Urjamehta2

## License

This project is licensed under the MIT License - see LICENSE file for details.


