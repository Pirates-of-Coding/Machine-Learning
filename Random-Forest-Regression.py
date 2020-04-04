#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2]
print("Dataset Imported")

#fitting the Random Forest Regressor to dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300, random_state=0)
regressor.fit(X,Y)

# Visualising the Random Forest Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red',label="Original Values")
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue', label="Model Prediction")
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.legend(loc=0)
print("Graph opened")
plt.show()


a = np.array([6.5],dtype="int").reshape(-1,1)
b= regressor.predict(a)  
print(b)