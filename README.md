# LendingClub Loan Default Prediction

Predicting loan default from borrower and loan characteristics using L2-regularized logistic regression and LightGBM.

**Data:** LendingClub accepted loans 2007-2018 Q4 ([Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)). First 200,000 rows used; 176,083 loans with known outcomes (fully paid or charged off), ~20% default rate.

**Methods:** Stratified 80/20 train/test split; median imputation and standardization fit on training data only; class-weighted logistic regression (L2); LightGBM with early stopping on a validation split from the training data.

**Results (held-out test set):**
| Model | AUC-ROC |
|---|---|
| Logistic Regression (L2) | 0.7259 |
| LightGBM | 0.7340 |

**Key findings:** FICO score, DTI, loan term, and loan amount are among the strongest predictors. Gradient boosting gave only a small lift over logistic regression, suggesting most of the signal is roughly linear.

**Limitations:** Non-random sample (first 200K rows), possible look-ahead features, single train/test split, state and purpose may proxy for demographics.

**To run:** Download the CSV from Kaggle, place it in the same folder as the notebook, and run `lending_club_project_revamped.ipynb`.
