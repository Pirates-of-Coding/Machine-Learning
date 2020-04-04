#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#importing dataset (X-> Independent and Y-> Dependent Varibale)
dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:,:-1]
Y = dataset.iloc[:,-1]
print("Dataset Imported")


#Encoding categorical data
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [3])],remainder='passthrough') 
        # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
        # Leave the rest of the columns untouched
newEncoded_X = np.array(ct.fit_transform(X), dtype=np.float)

#to avoid dummy variable trap(it is automatically done by libraries)
newEncoded_X = newEncoded_X[:,1:]

#splitting  model into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(newEncoded_X,Y,test_size = 0.2, random_state=0)


#fitting Multiple Regression Model into training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)
print("Model Trained")

#5 independent variable and 1 dependent variable, so graph will be complex 
#and in higher dimensions

#predicting the value of test result
Y_predict = regressor.predict(X_train)

#building the optimal model using Backward Elimination
#only consider those independent variable that have appreciable impact on dependent
#variable(that is statistically significant) and eliminate others
#Backward Elimination steps :-
#Select a significance level (SL = 0.05)
#fit the full model with all possible predictors
#Consider the predictor with highest P values if P>SL go to next step else done
#Remove the predictor
#fit model without this variable

import statsmodels.api as sm
newEncoded_X = np.append(np.ones(shape=(50,1), dtype="int"),newEncoded_X,axis=1)
# X_optimal will contain every independent variable initially but we'll remove them
#by backward elimination later

X_optimal = newEncoded_X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog= X_optimal).fit()
print(regressor_OLS.summary())


X_optimal = newEncoded_X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog=Y, exog=X_optimal).fit()
print(regressor_OLS.summary())

X_optimal = newEncoded_X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog=Y, exog=X_optimal).fit()
print(regressor_OLS.summary())

X_optimal = newEncoded_X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog=Y, exog=X_optimal).fit()
print(regressor_OLS.summary())

X_optimal = newEncoded_X[:,[0,3]]
regressor_OLS = sm.OLS(endog=Y, exog=X_optimal).fit()
print(regressor_OLS.summary())

