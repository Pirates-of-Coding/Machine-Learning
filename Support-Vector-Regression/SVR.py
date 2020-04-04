#importing libararies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
#.values will give the values in an array. (shape: (n,1))
#.ravel will convert that array shape to (n, )
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2:3].values
print("Dataset Imported")

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y)

#fitting SVR to dataset
from sklearn.svm import SVR
regressor = SVR(kernel = "rbf")
regressor.fit(X,Y.ravel())
print("Model Trained")

#visualising the SVR results
plt.title("SVR")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.scatter(X,Y, color="red",label="Original Values")
print("Now showing graph")
plt.plot(X,regressor.predict(X), label="Model Prediction")
plt.legend(loc=0)
plt.show()
