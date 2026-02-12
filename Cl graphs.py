import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures


# Reading a CSV file into a DataFrame
data = pd.read_csv('C:\\Users\\eltjo\\OneDrive - Delft University of Technology\\Persoonlijke\\Tycho-and-Eltjo-Repository\\AirfoilSelfNoise.csv') # skiprows=2 skips the first two rows

#Cleaning the data by removing entries with missing values and/or not numbers
data_clean = data.dropna()

#Removing outliers using Z-score method
z_scores = np.abs(stats.zscore(data_clean['SSPL']))
data_clean = data_clean[(z_scores < 3)]

def train_test_validation_split(X, y, test_size, cv_size):
    # collective size of test and cv sets
    test_cv_size = test_size+cv_size

    # split data into train and test - cross validation subsets
    X_train, X_testcv, y_train, y_testcv = train_test_split(
        X, y, test_size=test_cv_size, random_state=0, shuffle=True)

    # split test - cross validation sets into test and cross validation subsets
    X_test, X_cv, y_test, y_cv = train_test_split(
        X_testcv, y_testcv, test_size=cv_size/test_cv_size, random_state=0, shuffle=True)

    # return split data
    return [X_train, y_train, X_test, y_test, X_cv, y_cv]

# Make X and y
X = data_clean.drop('SSPL', axis=1).to_numpy()
y = data_clean['SSPL'].to_numpy()

X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.3, random_state=0)

degree_list = np.arange(1,21)
model_list = np.zeros(len(degree_list), dtype=object)

mse_list = []

for degree in degree_list:
    #regression model
    poly = PolynomialFeatures(degree=degree)
    poly_features = poly.fit_transform(X_train)

    poly_reg = LinearRegression()
    poly_reg_model = poly_reg.fit(poly_features, y_train)

    #validation
    poly_features_cv = poly.fit_transform(X_cv)
    y_pred = poly_reg_model.predict(poly_features_cv)

    mse = mean_squared_error(y_cv, y_pred)
    mse_list.append(mse)

model = model_list[np.argmin(mse_list)]

f = 3500
alpha = 6.7
c = 0.67
U_inifinity = 40
delta = 0.00266337

X_pred = np.array([f, alpha, c, U_inifinity, delta]).reshape(1, -1)
features_pred = poly_list[np.argmin(mse_list)].fit_transform(X_pred)

SSPL = model.predict(features_pred)[0]

print(SSPL)










