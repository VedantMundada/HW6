import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
path = "C:/Users/Vedant/Desktop/UIUC/SEM 1/Machine Learning/Week 6/ccdefault.csv"
data=pd.read_csv(path)
print(data.head())
data = data.drop('ID',axis=1)
print(data.head())
data=data.values

X = data[:,:22]
y = data[:,23]


#%%
testacc=[]
trainacc=[]
for i in range(1,11):
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.10, random_state=i, stratify = y)
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    dtc = DecisionTreeClassifier(max_depth=3)
    dtc.fit(X_train, y_train)
    y_pred = dtc.predict(X_test)
    y_train_pred = dtc.predict(X_train)
    testacc.append(dtc.score(X_test,y_test))
    trainacc.append(dtc.score(X_train,y_train))
    
plt.plot(testacc,label='Out of sample accuracy')
plt.plot(trainacc,label='In sample accuracy')
plt.legend(loc='upper right')
plt.title("Accuracy scores (Decision Tree")
plt.xlabel('Random state')
plt.ylabel('Accuracy')
plt.show()

testacc = np.array(testacc)
trainacc = np.array(trainacc)
print("The training accuracy is = ",trainacc)
print("The test accuracy is = ",testacc)
print("Mean of the in sample accuracy scores = ", trainacc.mean())
print("Mean of the out of sample accuracy scores = ", testacc.mean())
print("Standard deviation of the in sample accuracy scores = ", trainacc.std())
print("Standard deviation of the out of sample accuracy scores = ", testacc.std())

print("The training accuracy is = ",trainacc.mean(),"+/-",trainacc.std())
print("The test accuracy is = ",testacc.mean(),"+/-",testacc.std())

#%%

from sklearn.model_selection import StratifiedKFold
testacc2=[]
trainacc2=[]

X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y,test_size=0.10, random_state=10, stratify = y)
scaler = preprocessing.StandardScaler().fit(X_train2)
X_train2 = scaler.transform(X_train)
X_test2 = scaler.transform(X_test)
kfold = StratifiedKFold(n_splits = 10).split(X_train2,y_train2)
dtc = DecisionTreeClassifier(max_depth=3)

for k, (train,test) in enumerate(kfold):
    dtc.fit(X_train[train], y_train[train])
    trainacc2.append(dtc.score(X_train[test], y_train[test]))
    testacc2.append(dtc.score(X_test,y_test))
    print('Fold :',k+1 ,' Class dist :', np.bincount(y_train[train]),' Training set Accuracy:',dtc.score(X_train[train], y_train[train]) , 'Test set accuracy :',dtc.score(X_train[test], y_train[test]))
    
#%%
from sklearn.model_selection import cross_val_score
dtc2 = DecisionTreeClassifier(max_depth=3)
kfold = StratifiedKFold(n_splits = 10).split(X_train2,y_train2)
scores = cross_val_score(estimator = dtc , X = X_train, y=y_train, cv=kfold , n_jobs = 1)
dtc2.fit(X_train,y_train)
y_pred3=dtc2.predict(X_test)
print("CV accuracy Scores ",scores)
print("CV acuracy is ", np.mean(scores),"+/-", np.std(scores))
print("Test accuracy ",dtc2.score(X_test,y_test))



#%%
from sklearn.utils import resample
print("Number of class 0 samples before ",X[y==0].shape[0])
X_resamp,y_resamp = resample(X[y==0],y[y==0],replace=False,n_samples=X[y==1].shape[0] , random_state=123)
print("Number of class 0 samples after ",X_resamp.shape[0])
X_fin=np.vstack((X[y==1],X_resamp))
y_fin=np.hstack((y[y==1],y_resamp))

testacc3=[]
trainacc3=[]
for i in range(1,11):
    X_train3, X_test3, y_train3, y_test3 = train_test_split(X_fin, y_fin,test_size=0.10, random_state=i, stratify = y_fin)
    scaler3 = preprocessing.StandardScaler().fit(X_train3)
    X_train3 = scaler3.transform(X_train3)
    X_test3 = scaler3.transform(X_test3)
    dtc3 = DecisionTreeClassifier(max_depth=3)
    dtc3.fit(X_train3, y_train3)
    y_pred3 = dtc3.predict(X_test3)
    y_train_pred3 = dtc3.predict(X_train3)
    testacc3.append(dtc3.score(X_test3,y_test3))
    trainacc3.append(dtc3.score(X_train3,y_train3))
    
plt.plot(testacc3,label='Out of sample accuracy')
plt.plot(trainacc3,label='In sample accuracy')
plt.legend(loc='upper right')
plt.title("Accuracy scores (upsampling)")
plt.xlabel('Random state')
plt.ylabel('Accuracy')
plt.show()

testacc3 = np.array(testacc3)
trainacc3 = np.array(trainacc3)

print("Mean of the in sample accuracy scores = ", trainacc3.mean())
print("Mean of the out of sample accuracy scores = ", testacc3.mean())
print("Standard deviation of the in sample accuracy scores = ", trainacc3.std())
print("Standard deviation of the out of sample accuracy scores = ", testacc3.std())

print("The training accuracy is = ",trainacc3.mean(),"+/-",trainacc3.std())
print("The test accuracy is = ",testacc3.mean(),"+/-",testacc3.std())

#%%
print("My name is Vedant Mundada")
print("My NetID is:vkm3")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")

















