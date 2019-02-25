#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as P
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn import tree
train_data=P.read_csv('train_set.csv')
ytrain=train_data.pop('click').values
#del row['click'], row['id'], row['hour'],row['device_id'], row['device_ip'] ,row['device_model'] , row['device_conn_type'],row['device_type']
y_train=ytrain[0:10000]
y_train
xtrain=train_data.drop(columns=['id','device_id','device_ip','device_model','device_type','hour','device_conn_type'],axis=1).values
x_train=xtrain[0:10000]
x_train[0]
type(x_train[0][0])
labelenode_x= LabelEncoder()
for c in range(16):
    x_train[:,c]=labelenode_x.fit_transform(x_train[:,c])
x_train[0]
onehotencoder  = OneHotEncoder(categorical_features= 'all')
x_train = onehotencoder.fit_transform(x_train).toarray()
x_train.shape
labelencoder_y = LabelEncoder()
y_train = labelencoder_y.fit_transform(y_train)
y_train
from sklearn.model_selection import train_test_split
x__train,x__test,y__train,y__test=train_test_split(x_train, y_train, test_size=0.25, random_state=0)
x__train.shape
x__test.shape
y__train.shape
y__test.shape

#Decision tree
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
parameters = {'max_depth': [5],'min_samples_split':[10,30,50]}
decision_tree = DecisionTreeClassifier(criterion='gini')
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(decision_tree, parameters,n_jobs=-1, cv=3)
grid_search.fit(x__train, y__train)
print(grid_search.best_params_)
predicted = grid_search.predict(x__test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y__test,predicted)*100
print("Accuracy = {}".format(accuracy)) 


# In[4]:


from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
parameters = {'max_depth': [10],'min_samples_split':[10,30,50]}
decision_tree = DecisionTreeClassifier(criterion='gini')
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(decision_tree, parameters,n_jobs=-1, cv=3)
grid_search.fit(x__train, y__train)
print(grid_search.best_params_)
predicted = grid_search.predict(x__test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y__test,predicted)*100
print("Accuracy = {}".format(accuracy)) 


# In[8]:


from sklearn.linear_model import LogisticRegression
# param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],'penalty':['l1','l2']}
C = np.logspace(0, 4, 10)
penalty = ['l1','l2']
hyperparameters = dict(C=C, penalty=penalty)
clf = GridSearchCV(LogisticRegression(), hyperparameters)
gridsearch=GridSearchCV(cv=None,estimator=LogisticRegression(C=5.0, intercept_scaling=1,   
               dual=False, fit_intercept=True,tol=0.0001),param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]})
clf.fit(x__train,y__train)
pred = clf.predict(x__test)
accuracy_score(y__test,pred)
print(gridsearch.best_params_)
prediction=gridsearch.predict(x__test)
prediction.shape
clf.fit(x__train,y__train)
prediction.shape
y__test
accuracy=accuracy_score(y__test,prediction)*100
accuracy


# In[11]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(x__train,y__train)
Y_predict = clf.predict(x__test)
accuracy = accuracy_score(y__test,Y_predict)*100
print("Accuracy = {}".format(accuracy)) 


# In[12]:


from sklearn.naive_bayes import MultinomialNB
mul=MultinomialNB()
mul.fit(x__train,y__train)
predictmb=mul.predict(x__test)
accuracymb=accuracy_score(y__test,predictmb)
accuracymb


# In[ ]:




