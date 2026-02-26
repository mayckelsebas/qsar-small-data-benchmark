# QSAR Small-Data Benchmark

## Overview

This repository contains the fully reproducible computational framework supporting the study:

**"Descriptor Selection vs Algorithmic Complexity in Small-Data QSAR: A Benchmark Study (n = 19)"**

The objective of this work is to evaluate whether modern machine learning (ML) algorithms outperform a parsimonious classical QSAR model under strict small-data conditions.

The dataset consists of 19 L-mannitol derivatives evaluated against Plasmepsin II (Log Ki as dependent variable).

---

## Scientific Rationale

In small datasets (n ≈ 10–30), model stability is often more influenced by dimensionality control than by algorithmic complexity.  

This study benchmarks:

- A classical Multiple Linear Regression (MLR) model (3 selected descriptors)
- Several ML algorithms
- Full descriptor set vs reduced descriptor set

All models are evaluated using identical Leave-One-Out Cross-Validation (LOOCV) protocols to ensure methodological fairness.

---

## Dataset

File: `dataset_standard.csv`

- 19 compounds
- 29 structural descriptors
- No quantum descriptors
- Dependent variable: `Log Ki`
- Standard international CSV formatting (comma separator, decimal point)

---

## Models Evaluated

### Classical Model
- Linear Regression
- 3 selected descriptors: Mi, nCs, nHDon

### Machine Learning Models
- Ridge
- Lasso
- SVR (RBF)
- Random Forest
- KNN
- XGBoost

Two dimensionality scenarios were tested:
1. Reduced descriptor set (3 descriptors)
2. Full descriptor set (29 descriptors)

---

## Validation Protocol

- Leave-One-Out Cross-Validation (LOOCV)
- R²_LOOCV
- MAE
- RMSE
- Q²_F1
- Q²_F2
- Golbraikh–Tropsha parameters (k and R²₀)
- Y-randomization test
- Paired t-test
- Wilcoxon signed-rank test
- Bootstrap 95% confidence intervals

Random seed fixed to ensure reproducibility.

---

## Key Findings

- The classical 3-descriptor model achieved:

  R²_LOOCV = 0.7324  
  MAE = 0.3706  
  RMSE = 0.4787  

- The best ML full-descriptor model (XGBoost) achieved:

  R²_LOOCV = 0.6677  
  MAE = 0.3912  
  RMSE = 0.5334  

- Differences were not statistically significant.
- Descriptor dimensionality exerted greater influence on stability than algorithmic complexity.

---

## Reproducibility

To reproduce the results:

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
