# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import streamlit as st


def load_data(file_path):
    """
    Load the dataset
    """
    data = pd.read_csv('C:\Users\win10\Desktop\copper')
    return data


def data_cleaning(data):
    """
    Data Cleaning
    """
    # Remove irrelevant columns
    data = data.drop(['id', 'item_date', 'customer', 'delivery date', 'product_ref'], axis=1)

    # Handle missing values
    data = data.fillna(data.mean())

    # Remove outliers using IQR
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    data = data[~((data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR))).any(axis=1)]

    # Identify skewness in the dataset
    skewness = data.skew()

    # Perform appropriate data transformations to handle skewness
    data['selling_price'] = np.log1p(data['selling_price'])
    data['quantity tons'] = np.sqrt(data['quantity tons'])
    data['thickness'] = np.log1p(data['thickness'])
    data['width'] = np.sqrt(data['width'])

    # Encode categorical variables using one-hot encoding
    data = pd.get_dummies(data, columns=['country', 'status', 'item type', 'application', 'material_ref'])

    return data


def split_data(data):
    """
    Split the dataset into training and testing sets
    """
    X = data.drop(['selling_price', 'status_LOST', 'status_WON'], axis=1)
    y_reg = data['selling_price']
    y_clf = data['status_WON']
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(X, y_reg, y_clf, test_size=0.2, random_state=42)

    return X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test


def feature_scaling(X_train, X_test):
    """
    Feature Scaling
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test


def regression_models(X_train, X_test, y_reg_train, y_reg_test):
    """
    Regression Model Building and Evaluation
    """
    # Linear Regression
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_reg_train)
    y_reg_pred = lin_reg.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
    r2 = r2_score(y_reg_test, y_reg_pred)

    # Decision Tree Regression
    dt_reg = DecisionTreeRegressor()
    dt_reg.fit(X_train, y_reg_train)
    y_reg_pred = dt_reg.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
    r2 = r2_score(y_reg_test, y
