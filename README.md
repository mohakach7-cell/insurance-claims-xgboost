# Motor-insurance claims: high-value claim detection with XGBoost

Master's thesis project (Finance and Data Science, FSJES Mohammedia, Hassan II University), built with real claims data from an insurer. The scores were then loaded into a Power BI dashboard.

## Problem
An insurer cannot review every motor claim by hand. This project builds a model that gives each claim a risk score, so analysts can look first at the claims most worth a closer review.

## Data
- About 260,750 motor claims, 23 columns (company, counterparty company, recourse type, liability rate, vehicle usage, claim and statement dates, amounts).
- The data is confidential, so it is **not included** here. To run the notebook, provide your own `BASE.csv` with the same structure.

## Target (important)
Confirmed fraud labels were not available. The target is the claim-size class `SUP 10000` (claims above 10,000), which is 14.65% of the data (38,207 claims). It is used as a **proxy** for claims that deserve a closer look, so this is not a confirmed-fraud detector.

## Approach
1. Dropped empty columns, identifiers, and amount-derived columns (`Montant Réel`, `Montant Recours`, `Quintile`) to avoid target leakage.
2. Imputed missing values (median / mode) and converted dates and usage codes to numeric features.
3. Engineered 5 features (start of week, recent claim, high season, shared liability, same company on both sides).
4. Stratified 80/20 split, then SMOTE on the training set only, and scaling fitted on the training set only.
5. Compared Logistic Regression, KNN, linear SVM, Random Forest, XGBoost and an Isolation Forest.
6. Exported a fraud score and a risk level (Low / Moderate / High) per test claim for Power BI.

## Results (held-out test set, 52,150 claims)
| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| **XGBoost** | 0.676 | 0.551 | 0.607 | **0.829** |
| Random Forest | 0.920 | 0.508 | 0.655 | 0.823 |
| Linear SVM | 0.583 | 0.558 | 0.570 | 0.802 |
| Logistic Regression | 0.799 | 0.522 | 0.631 | 0.802 |
| KNN | 0.678 | 0.537 | 0.599 | 0.779 |
| Isolation Forest (unsupervised) | 0.163 | 0.167 | n/a | n/a |

XGBoost had the best ROC-AUC and was used for the scores.

## Limitations and next steps
- **Proxy target.** The model predicts claim size, not confirmed fraud.
- **One dominant feature.** Recourse type carries about 82% of XGBoost's feature importance. Flat-rate recourses are small claims, so it may partly encode claim size. Next step: retrain without it and compare.
- **Cross-validation scores are optimistic.** SMOTE was applied before cross-validation, so the CV AUCs (up to 0.82) are computed on resampled data. The test-set figures above come from untouched data and are the ones to trust.
- **Possible duplicates.** Several top-scored claims share an identical score, so check for duplicate rows across train and test.
- **Arbitrary risk thresholds.** The 0.30 and 0.70 cut-offs were not tuned to a business cost.

## How to run
```bash
pip install -r requirements.txt
# put your own BASE.csv next to the notebook, then:
jupyter notebook notebook.ipynb
```
