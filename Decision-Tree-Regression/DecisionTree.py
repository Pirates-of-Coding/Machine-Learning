#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset 
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2:3]
print("Dataset Imported")

#fitting the Decision Tree Regression to dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, Y)
print("Model trained")

# #visualising the Decision Tree Regressor
# plt.title("Decision Tree Regression")
# plt.xlabel("Position Level")
# plt.ylabel("Salary")
# plt.scatter(X,Y, color="red",label="Original Values")
# plt.plot(X,regressor.predict(X), label="Model Prediction")
# print("Graph opened")
# plt.legend(loc=0)
# plt.show()

# Visualising the Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red',label="Original Values")
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue', label="Model Prediction")
plt.title('Decision Tree Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.legend(loc=0)
print("Graph opened")
plt.show()
