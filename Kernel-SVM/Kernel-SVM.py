#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv("Social_Network_Ads.csv")
X = dataset.iloc[:,2:4].values
Y = dataset.iloc[:,4].values
print("Dataset imported")

#splitting data into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.25, random_state=0)
print("Data is splitted into training and test model")

#Feature Scaling data because it is highly deviated or 
#has significant variations
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
print("Training and Test set is scaled")


#fitting Kernal SVM Classifier to the Training Set
from sklearn.svm import SVC
classifier = SVC(kernel = "rbf",random_state=0)
classifier.fit(X_train,Y_train)
print("Model trained")

#predicting the test set result
Y_pred = classifier.predict(X_test)

#Making the confusion matrix to evaluate predictive power of our model
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
print("Confusion Matrix :- ")
print(cm)

print("Total Correct prediction = ", cm[0][0] + cm[1][1] )
print("Total Incorrect prediction = ", cm[0][1] + cm[1][0] )

# Visualising the Training set results
from matplotlib.colors import ListedColormap
#supress warning for 'c' argument looks like a single numeric...
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
X_set, Y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
print(Y_set.shape)
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Kernel SVM(Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
print("Graph for Training Set is showing(Close it to see graph for Test Set ")
plt.show()


# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, Y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Kernel SVM(Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
print("Graph for Test set is showing")
plt.show()
