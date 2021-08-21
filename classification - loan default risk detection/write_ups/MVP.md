# Home Credit Default Risk

---

- In this project, I will explore the application data and build several models to predict whether the applicant is able to pay back their loan if accepted for Home Credit Bank.

- Impacts of this project for the business:
    - Help the bank to determine essentially good and bad credit clients.
    - By balancing both precision and recall (prioritizing F1), the model will help both the bank and clients by not rejecting people with *fair* and *okay* credit, also not easily accepting people with *essentially bad* credit.

- I built some baseline models and some advanced models to compare and choose the best one to tune.

- As the first glance, `XGBoost` performs the best base on their ROC AUC scores.

![](https://github.com/luongtruong77/loan_risk_classification/blob/main/notebooks/roc_curve_comparison_plot.png?raw=true)

- I also include the table of **f1 score** since I will try to optimize f1 score based on the project's purpose.

    Baseline KNN | Baseline Log_Reg | Random Forest | XGBoost
    --- | --- | --- | ---
    0.1427 | 0.0001 | 0.0167 | 0.2915

- Comments: 
    - Even though `KNN` does not have the good ROC score, it has an OKAY f1 score.
    - `Baseline Log_Reg` has a terribly low f1 score.
    - `XGBoost` performs very well on both ROC AUC score and F1 score.

- Further works:
    - Build `SGDClassifier, AdaBoost` and (possibly) `Multi-layer Perceptron (MLPClassifer)`
    - Tune `XGBoost` hyper parameters.