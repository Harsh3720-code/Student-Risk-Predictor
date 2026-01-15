# 🎓 Student Risk Prediction using Machine Learning

This project builds a complete **end-to-end machine learning system** to predict whether a student is **at risk of failing** based on demographic, social, and academic features.

The focus is not only on accuracy, but on building a **decision-oriented, explainable, and reproducible ML pipeline** suitable for a real-world early-warning system in education.

---

## 📌 Problem Statement

Educational institutions often want to identify students who are at risk of failing **as early as possible** so that timely academic support can be provided.

This project formulates the task as a **binary classification problem**:

- **0** → Not at risk  
- **1** → At risk of failing  

The goal is to **maximize the detection of at-risk students**, even if this means accepting some false alarms.

---

## 📊 Dataset

- Source: UCI Student Performance Dataset  
- Number of students: 395  
- Original features: Demographic, social, family, and academic attributes  
- Target variable created as:

```python
at_risk = (G3 < 10)

at_risk = (G3 < 10)

where G3 is the final grade and to avoid data leakages G1,G2,G3 were removed from the input features.



🧠 Machine Learning Pipeline

The project follows a full applied ML pipeline:

1. Exploratory Data Analysis (EDA)

2. Data preprocessing:

One-hot encoding for categorical features

Standard scaling for numerical features

ColumnTransformer + Pipeline

3. Train-test split

4. Model training

5. Model evaluation

6. ROC analysis and threshold tuning


🤖 Models Used

1. Logistic Regression (baseline)

2. Random Forest (non-linear, stronger model)


📈 Evaluation Strategy

Metrics used:

1. Precision

2. Recall

3. F1-score

4. Confusion matrix

5. ROC-AUC

Because the dataset is moderately imbalanced and the goal is to catch at-risk students, recall for the positive class is prioritised.


🎯 Threshold Tuning

Instead of using the default threshold (0.5), multiple thresholds were evaluated and tested.

Key result:

1. Threshold = 0.4 provides a strong balance:

2. Recall (at-risk) ≈ 0.50

3. Accuracy ≈ 0.71

4. Much better than default threshold!

5. Lower thresholds increase recall further but cause too many false positives.


