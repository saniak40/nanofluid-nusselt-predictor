# Predictive Modeling of Nanofluid Thermodynamics using Ensemble Machine Learning

## Overview
This repository contains a robust, academic-grade machine learning pipeline designed to predict the Nusselt number for various nanofluids. By mapping core thermophysical properties (Reynolds number, thermal conductivities, nanoparticle size, and concentration), this project leverages advanced ensemble regressors to model convective heat transfer with high accuracy ($R^2 > 90\%$).

The workflow bridges core chemical engineering principles with data analytics, utilizing SHAP (SHapley Additive exPlanations) to prove the models adhere to fundamental fluid dynamic laws.

## Key Features
*   **Data Processing:** Automated outlier filtration, logarithmic feature scaling, and rigorous 75/25 train-test splitting to prevent data leakage across 1,400+ experimental samples.
*   **Ensemble Modeling:** Implementation and evaluation of nine distinct regressors, including XGBoost, LightGBM, CatBoost, and Random Forest, heavily regularized to prevent overfitting.
*   **Interpretability:** Integration of SHAP beeswarm/bar plots and Partial Dependence Plots (PDPs) in a 2x2 grid to isolate the thermodynamic contribution of individual fluid properties.
*   **Automated Diagnostics:** Seamless export of detailed descriptive statistics and row-by-row prediction errors to Excel for peer-review auditing.

## Tech Stack
*   **Language:** Python 3.x
*   **Data Manipulation:** Pandas, NumPy
*   **Machine Learning:** Scikit-Learn, XGBoost, LightGBM, CatBoost
*   **Visualization:** Matplotlib, Seaborn, SHAP

## Visual Insights

### 1. Model Performance (Actual vs. Predicted)
*(Add your Actual vs Predicted image here by uploading it to the repo and linking the path: e.g., `![Performance](assets/3_Actual_vs_Predicted.png)`)*

### 2. Feature Importance (SHAP Analysis)
*(Add your SHAP Beeswarm image here: e.g., `![SHAP](assets/5_SHAP_Beeswarm.png)`)*
The SHAP analysis mathematically validates classical empirical heat transfer correlations, demonstrating that the Reynolds Number is the primary driving force behind the Nusselt number.

### 3. Thermodynamic Response (Partial Dependence Plots)
*(Add your PDP 2x2 Grid image here: e.g., `![PDPs](assets/6_PDP_Plots_2x2.png)`)*

## How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/nanofluid-thermodynamics-ml.git](https://github.com/yourusername/nanofluid-thermodynamics-ml.git)