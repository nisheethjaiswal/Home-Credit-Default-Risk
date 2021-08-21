# Home Credit Default Risk
#### Are applicants capable of repaying their loans?
---

Steven L Truong

---

#### What is this project about?
- Home Credit is an international non-bank financial institution providing installment financial loans to unbanked people who have insufficient or non-existent credit histories.
- They want to laverage the power of data, statistical and machine learning methods to make predictions whether potential clients are capable of repaying their loans if accepted.
- By doing so, they make sure not to reject people who are capable of repaying their loans, hence helping people to achieve their goals.


#### Task:
The task is to predict future payment behavior of clients from application, demographic and historical credit behavior data.


#### Data description:
- The dataset is acquired from [Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data) which compiles multiple small datasets. 
- The main dataset is the `application.csv`, along with other usefule sets such as `bureau.csv` and `credit_card_balance.csv`.
- The data has more than *3,000,000* data points with more than *100* features for us to explore and choose from.

#### Algorithms:
- I am planning to use:
    - Logistic Regression
    - K-Nearest Neighbor (KNN)
    - Random Forest
    - Gradient Boosted Tree

- For the purpose of the business, we would like to ensure that we grant the loans for individuals who are capable of repaying their loans, so the `precision` score will be our priority. Of course we don't want to accidently reject potential customers, but if we grant loans for people who are not be able to pay back, we will lose our money.

#### Tools:
Tools I intend to use in this project:
- Python
- Pandas
- Numpy
- Scikit-learn


#### MVP:
- Baseline model to classify whether the client is able to pay back their loan.
- Feature engineering to improve the models and try several algorithms.