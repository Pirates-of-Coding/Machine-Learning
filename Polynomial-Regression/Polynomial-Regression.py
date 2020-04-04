#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2].values
print("Dataset Imported")

#fitting Linear Regression to dataset
from sklearn.linear_model import LinearRegression
lin_regressor1 = LinearRegression()
lin_regressor1.fit(X,Y)

#fitting Polynomial Regression to dataset
from sklearn.preprocessing import PolynomialFeatures
poly_regressor = PolynomialFeatures(degree = 4)
X_poly = poly_regressor.fit_transform(X)
lin_regressor2 = LinearRegression()
lin_regressor2.fit(X_poly, Y)
print("Model Trained")

#visualising the linear regression model
plt.title("Linear Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.scatter(X,Y, color="red",label="Original Values")
plt.plot(X,lin_regressor1.predict(X), label="Model Prediction")
plt.legend(loc=0)  #to show indicator for lines that which line 
#represent what.
print("Showing Graph for Linear Regression(Close it to see Polynomial Regression Graph)")
plt.show()

#visualising the polynomial regression model
plt.title("Polynomial Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.scatter(X,Y, color="red",label="Original Values")
#plt.plot(X,lin_regressor2.predict(X_poly))
#generalised >>>>>>>
plt.plot(X,lin_regressor2.predict(poly_regressor.fit_transform(X)), label="Model Prediction")
plt.legend(loc=0)
print("Showing Graph for Polynomial Regression")
plt.show()


