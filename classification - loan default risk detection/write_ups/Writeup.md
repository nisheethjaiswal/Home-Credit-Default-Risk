# Home Credit Default Risk Classification
#### Are applicants capable of repaying their loans?

---

Steven L Truong

---

## Abstract
---
In this project, I will build classification model to predict if potential clients are capable of paying their loans if accepted to help [Home Credit](http://www.homecredit.net/) ensure to not reject people with potentially good credit, also not to mistakenly accept clients who essentially have bad credit scores. I will build several models (Logistic Regression, KNN, Random Forest, XGBoost, etc) and choose the final model to optimize and improve. The final product should help the bank to optimize their process of screening and monitoring their loan applications.

## Design
---
- I will build the model to answer to question of the bank: Is this applicant eligible for loan?
- The model will predict **Yes/No** based on historical credit data and appropriate algorithm.
- I will do **exploratory analysis** on my dataset to seek insights and do **features engineer/selection** to ensure the model has the optimized input.
- I will few baseline models (Logistic Regression, KNN, SGD) and move on in to more advanced models (AdaBoost, XGBoost, LightGBM), compare them and choose the best model to optimize.
- XGBoost is the chosen model based on dedicated metrics (ROC AUC score and F2 score) to be optimized and deployment.

## Data
---
- The dataset is provided by [Home Credit](http://www.homecredit.net/) and can be downloaded via [Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data). 
- The original dataset has over *300,000* data points and *122* features to work with. After doing EDA and features engineering, I have arrived with the final dataset consists of over *300,000* data points and *250* features to build the models.
- The **EDA notebook** can be found [HERE](https://github.com/luongtruong77/loan_risk_classification/blob/main/notebooks/EDA_features_engineering.ipynb)

## Algorithms
---
##### Feature Engineering
- Construct polynomial features between several income sources.
- Construct domain features (such as **percentage of credit vs income, percentage of annuity income vs total income,** etc.)
- Imputting missing values, convert categorial features to binary dummy variables.
- Apply oversampling to deal with imbalanced dataset.
- The **features engineering notebook** can be found [HERE](http://localhost:8888/notebooks/Metis_Bootcamp/Home_Credit_loan_risk/notebooks/EDA_features_engineering.ipynb)

##### Build the models
- I've tried quite a few algorithms:
    - Logistic Regression
    - K-Nearest Neighbors (KNN)
    - Random Forest
    - Naive Bayes
    - Stochastic Gradient Descent (SGD)
    - AdaBoost
    - XGBoost (notebook on how I train and tune for this algorithm can be found [HERE](https://github.com/luongtruong77/loan_risk_classification/blob/main/notebooks/XGBoost_train_tune.ipynb))
    - LightGBM (notebook on how I train and GridSearch for this algorithm can be found [HERE](https://github.com/luongtruong77/loan_risk_classification/blob/main/notebooks/lightgbm_params_search.ipynb))
    - Multilayer perceptron (MLP)
- The **models comparison notebook** can be found [HERE](https://github.com/luongtruong77/loan_risk_classification/blob/main/notebooks/Models_comparison.ipynb)
- The chosen model is **XGBoost** based on 2 metrics: `ROC AUC score` and `F-beta (beta=2) score`. The reason I chose F2 is because of the purpose of business is to flag people with essential bad credit but not too strict on the `precision` either since we want to give people chances.
![](https://github.com/luongtruong77/loan_risk_classification/blob/main/figures/roc_curve_comparison_plot.png?raw=true)
![](https://github.com/luongtruong77/loan_risk_classification/blob/main/figures/f2_comparison.png?raw=true)

##### Evalution
- Based on the baseline XGBoost model which has F2 score of **0.3921**, I will run GridSearchCV for small subset of randomly selected data, and apply the best parameters to the full dataset.
- The optimized model has slightly better performance.
- The **XGBoost parameters tunning notebook** can be found [HERE](https://github.com/luongtruong77/loan_risk_classification/blob/main/notebooks/XGBoost_train_tune.ipynb)
- It has F2 of 0.463 for train set and 0.432 for test set.
![](https://github.com/luongtruong77/loan_risk_classification/blob/main/figures/optimized_metrics.png?raw=true)

### Tools
---
- Python
- Pandas
- Numpy
- Matplotlib
- Seaborn
- Scikit-learn
- xgboost
- lightGBM
- Google Sheets

### Communication
---
- All the notebooks can be found [HERE](https://github.com/luongtruong77/loan_risk_classification/tree/main/notebooks)
- The link to my presentation can be found [HERE]()

