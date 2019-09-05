import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

names = ['sepal-length','sepal-width','petal-length','petal-width','class']

dataSet = pd.read_csv(url,names=names)

x = dataSet.iloc[:,:-1].values
y = dataSet.iloc[:,4].values


from sklearn.model_selection import train_test_split

# split the data into train set and test test
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20)

from sklearn.neighbors import KNeighborsClassifier

error = []

for i in range(1,40):
    #define the clf
    clf = KNeighborsClassifier(n_neighbors=i)
    # fit data to classifier
    clf.fit(x_train,y_train)
    #predict the y value with x
    y_pred = clf.predict(x_test)
    #calculate the mean value of false negetive values
    error.append(np.mean(y_pred!=y_test))

plt.plot(range(1,40),error)
plt.title('Error rate k values')
plt.xlabel('k values')
plt.ylabel('Error rate')
plt.show()
# clf = KNeighborsClassifier(n_neighbors=1)

# clf.fit(x_train,y_train)

# y_pred = clf.predict(x_test)

# from sklearn.metrics import classification_report,confusion_matrix
# #print confusion matrix -->(true positive,true negitive,....)
# print(confusion_matrix(y_test,y_pred))
# #print classification report
# print(classification_report(y_test,y_pred))


    