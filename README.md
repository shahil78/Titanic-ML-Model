Titanic Survival Prediction

This repository contains an analysis of the Titanic dataset, where we attempt to predict the survival of passengers using various machine learning models. The models trained include Random Forest, Logistic Regression, Support Vector Classifier (SVC), and XGBoost.

Table of Contents
Introduction
Dataset
Data Preprocessing
Exploratory Data Analysis
Modeling
Random Forest
Logistic Regression
SVC
XGBoost
Hyperparameter Tuning
Model Evaluation
Conclusion

Introduction

The Titanic disaster is one of the most infamous shipwrecks in history, and the goal of this project is to predict which passengers survived using machine learning models. This analysis uses various models to explore the relationships between features and the probability of survival.

Dataset

The dataset used for this project is from Kaggle's competition: Titanic - Machine Learning from Disaster. It consists of the following key features:

Survived: Whether the passenger survived (1) or not (0).
Pclass: Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd).
Sex: Gender of the passenger.
Age: Age of the passenger.
Fare: The fare paid by the passenger.
Parch: Number of parents/children aboard the Titanic.
Embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

Data Preprocessing

Missing Values:

Filled missing values in the Age column with the median age.
Dropped the Cabin column due to a large number of missing values.
Filled missing values in the Embarked column with the mode.
Filled missing values in Fare for the test dataset with the median fare.

Feature Engineering:

Dropped unnecessary columns like PassengerId, Name, and Ticket.
Applied Label Encoding to categorical features such as Sex and Embarked.
Exploratory Data Analysis
Exploratory analysis was performed using visualizations, including:
- Bar plots to analyze survival rates based on Pclass and Sex.
- Histograms to visualize the distribution of continuous variables like Age and Fare.
- Correlation Matrix to study the relationship between features.

Modeling

We used four different models to predict survival:

1. Random Forest
A robust ensemble model that aggregates multiple decision trees. It handles overfitting well and is suitable for datasets with mixed feature types.

2. Logistic Regression
A linear model often used for binary classification tasks like this. It provides interpretable coefficients for feature importance.

3. SVC (Support Vector Classifier)
This model aims to find a hyperplane that best separates the classes. We used SVC with probability estimation enabled to calculate the ROC AUC score.

4. XGBoost
An advanced boosting algorithm that is efficient and often yields high accuracy. XGBoost was fine-tuned with hyperparameter tuning to enhance performance.

Hyperparameter Tuning

For XGBoost, we performed hyperparameter tuning using GridSearchCV, optimizing parameters like:

n_estimators
learning_rate
max_depth
subsample
colsample_bytree

The best model configuration was selected based on the ROC AUC score.

Model Evaluation

Each model was evaluated using:

Accuracy
ROC AUC Score
Classification Report (Precision, Recall, F1-Score)
Confusion Matrix

Among the models, XGBoost provided the best performance in terms of both accuracy and ROC AUC score.

Conclusion:

The XGBoost model, after hyperparameter tuning, was the most accurate in predicting survival on the Titanic dataset. While all models performed reasonably well, XGBoost stood out in both precision and overall model performance.
