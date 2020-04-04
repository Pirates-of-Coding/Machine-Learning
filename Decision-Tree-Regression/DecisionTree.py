#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset 
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1:2]
Y = dataset.iloc[:,2:3]
print("Dataset Imported")

#fitting the Decision Tree Regression to dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, Y)
print("Model trained")

#visualising the Decision Tree Regressor
plt.title("Truth or Bluff(Decision Tree Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.scatter(X,Y, color="red")
plt.plot(X,regressor.predict(X))
print("Graph opened")
plt.show()
