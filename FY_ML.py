#1. importing the libraries
import numpy as np #mathematical tool
import matplotlib.pyplot as plt #plotting charts
import pandas as pd #import and manage datasets


 #2. importing the dataset
dataset=pd.read_csv('Database1.csv')
#print(dataset.shape[1])
nBits=dataset.shape[1]-2
X=dataset.iloc[:,1:nBits+1].values #matrix of independent variable
y=dataset.iloc[:,nBits+1].values #matrix of dependent variable


#5. Splitting the dataset into Training set and Test Set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
#X_train=X_train.tolist()

##6. Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc_X=StandardScaler()
#X_train=sc_X.fit_transform(X_train) #when you are applying your 
##(StandardScaler) object to your training set, you've to fit the 
##object to the training set and then transform it.
#
#X_test=sc_X.transform(X_test) #sc_X object is already fitted to the training set
##and hence it is not needed to fit, only tranform.


##Fitting Logisitc Regression to the Training Set
#from sklearn.linear_model import LogisticRegression
##X_train=[X_train]
#X_train=X_train.reshape(-1,1)
#classifier=LogisticRegression(random_state=0) #classifier is our LogisticRegression object
#classifier.fit(X_train,y_train)
#
#
#
##Predicitng the test set results
#X_test=X_test.reshape(-1,1)
#y_pred=classifier.predict(X_test)



#Fitting Logisitc Regression to the Training Set
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0) #classifier is our LogisticRegression object
classifier.fit(X_train,y_train)

#Predicitng the test set results
y_pred=classifier.predict(X_test)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix #it's a function, hence small letters
cm=confusion_matrix(y_test,y_pred) #calculates the predictive power of the logistic regression



accuracy=(cm[0][0]+cm[1][1])/(cm[0][0]+cm[1][1]+cm[1][0]+cm[0][1])
print(accuracy*100)






#
## Visualising the Training set results
#from matplotlib.colors import ListedColormap
#X_set, y_set = X_train, y_train
#X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
#                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
#plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
#plt.xlim(X1.min(), X1.max())
#plt.ylim(X2.min(), X2.max())
#for i, j in enumerate(np.unique(y_set)):
#    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                c = ListedColormap(('red', 'green'))(i), label = j)
#plt.title('Logistic Regression (Training set)')
#plt.xlabel('Age')
#plt.ylabel('Estimated Salary')
#plt.legend()
#plt.show()
#
#
#
## Visualising the Test set results
#from matplotlib.colors import ListedColormap
#X_set, y_set = X_test, y_test
#X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
#                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
#plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
#plt.xlim(X1.min(), X1.max())
#plt.ylim(X2.min(), X2.max())
#for i, j in enumerate(np.unique(y_set)):
#    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                c = ListedColormap(('red', 'green'))(i), label = j)
#plt.title('Logistic Regression (Test set)')
#plt.xlabel('Age')
#plt.ylabel('Estimated Salary')
#plt.legend()
#plt.show()
#
#
#
