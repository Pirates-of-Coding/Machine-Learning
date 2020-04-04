#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset (X-> Independent and Y-> Dependent Varibale)
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:,:-1]  #is same as [:,0]
Y = dataset.iloc[:,1]
print("Dataset Imported")

#splitting the data set into training and test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=1/3, random_state=0) 

#fitting Simple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression() 
regressor.fit(X_train, Y_train)    
print("Model Trained")     

#predicting the Test set result
Y_predict = regressor.predict(X_test)

#visualising the training set result
plt.title("Salary vs Experience (Training Set)")
plt.xlabel("Salary")
plt.ylabel("Years of Experience")
plt.scatter(X_train,Y_train, color = "red")
plt.plot(X_train,regressor.predict(X_train), color="blue")
plt.scatter(X_test,Y_test,color="green")
print("Showing Graph")
plt.show()
