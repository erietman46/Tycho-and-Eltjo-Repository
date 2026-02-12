from matplotlib import pyplot as plt
import scipy as sp
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

#HOW TO READ DATA


data = pd.read_csv('AirfoilSelfNoise.csv')   # Reads the csv files, splits the data into a dataframe, separates by commas
data_clean=data.dropna() # Drops any rows with missing values
z_scores=np.abs(sp.stats.zscore(data_clean['SSPL'])) # Calculates the z-scores for the 'SSPL' column to identify outliers
newdata=data_clean[(z_scores<3)]    # Keeps only the rows where the z-score is less than 3, so datapoints with less than three standard deviations
#effectively removing outliers



##LINEAR REGRESSION


#HOW TO SPLIT DATA INTO TRAINING, TESTING, AND CROSS-VALIDATION SETS

def train_test_validation_split(X, y, test_size, cv_size): # Function to split the data into training, testing, and cross-validation sets, done with random shuffling 
    #and a specified random state for reproducibility
    
    # collective size of test and cv sets, so test size and validation size together
    test_cv_size = test_size+cv_size 

    # split data into train and test - cross validation subsets
    X_train, X_testcv, y_train, y_testcv = train_test_split(
        X, y, test_size=test_cv_size, random_state=0, shuffle=True)

    # split test - cross validation sets into test and cross validation subsets
    X_test, X_cv, y_test, y_cv = train_test_split(
        X_testcv, y_testcv, test_size=cv_size/test_cv_size, random_state=0, shuffle=True)

    # return split data
    return [X_train, y_train, X_test, y_test, X_cv, y_cv]


X_train, y_train, X_test, y_test, X_cv, y_cv = train_test_validation_split(newdata['alpha'], newdata['SSPL'], test_size=0.2, cv_size=0.1)

  #If you print X_train, you see two columns, the first one is the index and the second one is the 'alpha' column, which is the dataset we are using for regression

X_train_np = X_train.to_numpy() # Converts the training data from a pandas Series to a NumPy array, better for regression analysis
y_train_np = y_train.to_numpy() # Converts the target variable from a pandas Series to Numpy array, better for regression analysis
X_test_np = X_test.to_numpy() # Converts the test data from a pandas Series to a NumPy array, better for regression analysis
y_test_np = y_test.to_numpy() # Converts the target variable from a pandas Series to Numpy array, better for regression analysis
X_cv_np = X_cv.to_numpy() # Converts the cross-validation data from a pandas Series to a NumPy array, better for regression analysis
y_cv_np = y_cv.to_numpy() # Converts the target variable from a pandas Series to


#USING LINEAR REGRESSION WITH SKLEARN

#Step 1: Create a linear regression model
model = LinearRegression() # the model is now set up and ready to be trained on the data, it will learn the relationship between 
#the 'alpha' feature and the 'SSPL' target variable,


#Step 2: Fit the model to the training data
model.fit(X_train_np.reshape(-1, 1), y_train_np) # fit means: the model will learn the relationship between the 'alpha' feature and the 'SSPL' target variable 
#from the training data

#We reshape X_train_np to be a 2D array with one column, because the LinearRegression model expects the input data to be in this format. 
# The reshape(-1, 1) function reshapes the array to have one column and as many rows as needed 
# resphape(rows, columns) -1 means that the number of rows will be automatically determined based on the original size of the array, 
# and 1 means that there will be one column.

# So [1, 2, 3] becomes [[1], [2], [3]] after reshaping, which is necessary for the linear regression model to process the data correctly.


#Step 3: Evaluate the model on the test set
y_pred = model.predict(X_test_np.reshape(-1, 1)) 
# the test data is used becasue we want to evaluate how well the model generalizes to unseen data, 
# we use the test set to make predictions and then compare those predictions to the actual values in the test set.

# y_pred is a NumPy array containing the predicted values of 'SSPL' for each corresponding value of 'alpha' in the test set

# We will then compare the predicted values with the actual values using the mean squared error
# The lower this value, the better the model fits the data
mse = mean_squared_error(y_test_np, y_pred)
print(f'Mean Squared Error: {mse}')
# Compares every predicted value in y_pred with the actual value in y_test, calculates the squared difference for each pair, 
# and then averages those squared differences to get the mean squared error.

# 4. Make prediction using the model
X_pred = 10 # Example value of 'alpha' for which we want to predict 'SSPL'
y_pred_single = model.predict(np.array([[X_pred]]))   
print(f'Predicted value for X = {X_pred}: {y_pred_single[0]}')





# POLYNOMIAL REGRESSION

poly=PolynomialFeatures(degree=3) 

poly_features = poly.fit_transform(X_train_np.reshape(-1, 1)) # Transforms the original training data into polynomial features of degree 3, 
#which allows the model to capture non-linear relationships between 'alpha' and 'SSPL'.
#Creates the matrix form of for a like [1, x, x^2, x^3] for each value of 'alpha' in the training set, where x is the original feature value.

poly_reg_model = LinearRegression() # Creates a new linear regression model that will be used to fit the polynomial features

poly_reg_model.fit(poly_features, y_train_np) # Fits the polynomial regression model to the transformed training data

poly_features_test = poly.transform(X_test_np.reshape(-1, 1))
y_pred_poly = poly_reg_model.predict(poly_features_test)
mse_poly = mean_squared_error(y_test_np, y_pred_poly)
print(f'Polynomial (degree=3) Mean Squared Error: {mse_poly}')





# MULTIVARIABLE REGRESSION
# we are now testing the relationship between multiple features and the target variable, instead of just one feature as in linear regression.

X_multiple = data_clean.drop('SSPL', axis=1).to_numpy() # We drop the 'SSPL' column from the dataset to create the 
#feature matrix X, which contains all the features except the target variable.
# Returns a matrix with the features as columns and the samples as rows, so we have 10000 rows or something.

X_train_multiple, X_test_multiple, y_train_multiple, y_test_multiple = train_test_split(X_multiple, data_clean['SSPL'].to_numpy(), test_size=0.2, random_state=0, shuffle=True)

poly=PolynomialFeatures(degree=3) # We create polynomial features of degree 3 for the multiple regression model

poly_features_train = poly.fit_transform(X_train_multiple) # We transform the original training data into polynomial features of degree 3,
poly_features_test = poly.transform(X_test_multiple)

poly_reg_model = LinearRegression() # We create a new linear regression model that will be used to fit the polynomial features for multiple regression

poly_reg_model.fit(poly_features_train, y_train_multiple) # We fit the polynomial regression model to the transformed training data for multiple regression

y_pred_multi = poly_reg_model.predict(poly_features_test)
mse_multi = mean_squared_error(y_test_multiple, y_pred_multi)
print(f'Multivariable Polynomial (degree=3) Mean Squared Error: {mse_multi}')








#Underfitting and Overfitting, determining the best degree for polynomial regression, and regularization techniques will be covered in the next notebook.

degrees = np.arange(1,21) # We will test polynomial degrees from 1 to 20 to find the best degree for our polynomial regression model
# returns an array of integers from 1 to 20, which represent the degrees of the polynomial features we want to test.

mse_list = []

X_train_m, y_train_m, X_test_m, y_test_m, X_cv_m, y_cv_m = train_test_validation_split(
    pd.Series(list(range(len(X_multiple)))), data_clean['SSPL'], test_size=0.2, cv_size=0.1
)

train_idx = X_train_m.to_numpy()
test_idx = X_test_m.to_numpy()
cv_idx = X_cv_m.to_numpy()

X_train_multi = X_multiple[train_idx]
X_test_multi2 = X_multiple[test_idx]
X_cv_multi = X_multiple[cv_idx]
y_train_multi = y_train_m.to_numpy()
y_cv_multi = y_cv_m.to_numpy()

for degree in degrees:
    #Regression model
    poly = PolynomialFeatures(degree=degree) # We create polynomial features of the current degree in the loop
    poly_features_train = poly.fit_transform(X_train_multi) # We transform the original training data into polynomial features of the current degree
    poly_features_test = poly.transform(X_test_multi2) # We also transform the test data into polynomial features of the current degree to evaluate the model
    poly_reg_model = LinearRegression() # We create a new linear regression model for the current degree of polynomial features
    poly_reg_model.fit(poly_features_train, y_train_multi) # We fit the polynomial regression model to the transformed training data for the current degree
    
    #validation 
    poly_features_cv = poly.transform(X_cv_multi) # We transform the cross-validation data into polynomial features of the current degree to evaluate the model
    y_cv_pred = poly_reg_model.predict(poly_features_cv) # We make predictions on the cross-validation set using the fitted model for the current degree

    mse = mean_squared_error(y_cv_multi, y_cv_pred) # We calculate the mean squared error for the predictions on the cross-validation set to evaluate the model's performance for the current degree

    mse_list.append(mse) # We append the mean squared error for the current degree to the list of MSE values

best_degree = int(degrees[int(np.argmin(mse_list))])
print(f'Best degree by CV MSE: {best_degree}')
print(f'Best CV MSE: {min(mse_list)}')

plt.figure()
plt.plot(degrees, mse_list, marker='o')
plt.xlabel('Polynomial Degree')
plt.ylabel('CV Mean Squared Error')
plt.title('Degree vs CV MSE')
plt.show()
