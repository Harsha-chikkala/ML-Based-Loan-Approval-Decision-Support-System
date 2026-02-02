# ML-Based Loan Approval Decision Support System

## Project Overview

This project implements an end-to-end Machine Learning–based Decision Support System for loan approval.  
Instead of producing only binary predictions, the system outputs **approval probabilities**, converts them into **actionable business decisions**, and provides transparent explanations and recommendations.

The focus of the project is decision quality, interpretability, and real-world applicability, not just model accuracy.

---

## Objectives

- Build a supervised ML system for loan approval
- Handle real-world data challenges (missing values, imbalance)
- Use **probability-based decisions** instead of hard classifications
- Provide **explainable and deterministic reasoning**
- Simulate how ML systems are used in financial decision workflows

---

## Dataset Description

- Dataset: Loan Approval Dataset (tabular)
- Records: ~1000 applicants
- Features include:
  - Demographic attributes (Gender, Married, Dependents, Education)
  - Employment information
  - Financial attributes (Income, Loan Amount, Loan Term)
  - Credit history
- Target variable:
  - `Loan_Status` → Approved (Y) / Not Approved (N)

---

## Mini Sub-Project: Label Completion for Test Data

### Problem
The original dataset was split into:
- `train.csv` → labeled
- `test.csv` → unlabeled

A fully labeled dataset was required for downstream decision modeling.

### Solution
A **supervised Random Forest classifier** was trained using `train.csv` and then used to predict `Loan_Status` for `test.csv`.

Steps:
1. Train Random Forest on labeled training data
2. Predict loan approval outcomes for test data
3. Concatenate ground-truth labels and model-inferred labels
4. Save a unified dataset (`loan_full_labeled.csv`)

### Why This Matters
- Avoids blind assumptions
- Increases dataset size
- Maintains transparency between true and inferred labels
- Reflects real-world data engineering practices

---

## 🧱 Data Preparation & Preprocessing

### Feature Selection
Dropped:
- `Loan_ID` → Identifier only
- `Property_Area` → Location bias, not actionable

Retained features are:
- Known at application time
- Interpretable
- Actionable

### Missing Value Handling
- Categorical features → Mode
- Numerical features → Median

### Encoding
- Binary categorical variables encoded numerically
- Ordinal meaning preserved where applicable

### Scaling
- Numerical features standardized
- Required for stable Logistic Regression coefficients

---

## Target Imbalance Handling

- Target distribution: ~72% approved, ~28% rejected
- No artificial balancing applied
- Stratified splitting used
- Evaluation focuses on **precision and recall**, not accuracy alone

This preserves **probability calibration**, which is critical for decision systems.

---

## Model Selection

### Primary Model: Logistic Regression

**Why Logistic Regression?**
- Produces calibrated probabilities
- Interpretable coefficients
- Stable on tabular data
- Widely used in financial risk modeling

The goal is **trustworthy decisions**, not black-box performance.

---

## Model Output

The model outputs:
- `P(Loan Approved | Applicant Data)`

This probability becomes the foundation for decision-making.

---

## Decision Logic (Core System Design)

Approval probabilities are converted into **three decision buckets**:

| Probability Range | Decision |
|------------------|----------|
| ≥ 0.70 | APPROVE |
| 0.40 – 0.69 | REVIEW |
| < 0.40 | REJECT |

This mirrors real-world workflows where uncertain cases require human review.

---

## Explainability (Without SHAP)

Explainability is achieved using **intrinsic Logistic Regression properties**:

- Each feature has a coefficient
- Feature contribution = coefficient × feature value
- Contributions are ranked by impact

This ensures:
- Deterministic explanations
- No post-hoc approximation instability
- Full transparency

---

## Recommendation Engine

Based on high-impact feature contributions:
- Negative contributors → corrective suggestions
- Positive contributors → reinforcement

Examples:
- High loan amount → recommend reducing loan size
- Weak credit history → request additional documentation
- Strong income → maintain income proof

Recommendations are:
- Rule-based
- Impact-aware
- Actionable

---

## Evaluation Strategy

- Stratified train–validation split
- Metrics:
  - Precision
  - Recall
  - F1-score
- Emphasis on minimizing risky false approvals

## Author - Sri Harsha Vardhan Chikkala
