#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2].values

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

#visualising the linear regression model
plt.title("Truth or Bluff(Linear Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.scatter(X,Y, color="red")
plt.plot(X,lin_regressor1.predict(X))

#visualising the polynomial regression model
plt.title("Truth or Bluff(Linear Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.scatter(X,Y, color="red")
#plt.plot(X,lin_regressor2.predict(X_poly))
#generalised >>>>>>>
plt.plot(X,lin_regressor2.predict(poly_regressor.fit_transform(X)))
