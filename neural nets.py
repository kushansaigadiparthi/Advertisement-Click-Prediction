
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('train.csv')


# In[22]:


dataset.head()


# In[23]:


labeldata = dataset.pop('click').values
# del row['click'], row['id'], row['hour'],row['device_id'], row['device_ip'] ,row['device_model'] , row['device_conn_type'],row['device_type']
featuredata = dataset.drop(columns=['id','hour','device_id','device_ip','device_model','device_conn_type','device_type'],axis=1).values
featuredata[0]
featuredata.shape
featuredata[0]
labeldata.shape
from sklearn.preprocessing import LabelEncoder , OneHotEncoder


# In[24]:


featuredata[0]


# In[25]:


labelencoder_x = LabelEncoder()
for i in range(16):
    featuredata[:,i] = labelencoder_x.fit_transform(featuredata[:,i])


# In[26]:


featuredata[0]


# In[27]:


onehotencoder  = OneHotEncoder(categorical_features= 'all')
featuredata = onehotencoder.fit_transform(featuredata).toarray()

featuredata[0]
labelencoder_y = LabelEncoder()
labeldata = labelencoder_y.fit_transform(labeldata)


# In[28]:


featuredata[0] #after one-hot encoding 


# In[29]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(featuredata, labeldata, test_size = 0.2, random_state = 0)


# In[30]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[31]:


X_train.shape


# In[32]:


import keras 


# In[33]:


from keras.models import Sequential


# In[34]:


from keras.layers import Dense,Activation


# In[35]:


classifier = Sequential()

classifier.add(Dense(units=425 , kernel_initializer= "uniform",input_dim = 856))
classifier.add(Activation('relu'))

classifier.add(Dense(units=425 , kernel_initializer= "uniform"))
classifier.add(Activation('relu'))

classifier.add(Dense(units=425 , kernel_initializer= "uniform"))
classifier.add(Activation('relu'))

classifier.add(Dense(units=425 , kernel_initializer= "uniform"))
classifier.add(Activation('relu'))

classifier.add(Dense(units=1 , kernel_initializer= "uniform",))
classifier.add(Activation('sigmoid'))

classifier.compile(optimizer= 'adam' , loss= 'binary_crossentropy' , metrics= ['accuracy'])


# In[36]:


classifier.fit(X_train,y_train , batch_size=30 , epochs= 200)


# In[71]:


ypredicted = classifier.predict(X_test)


# In[72]:


for i in range(ypredicted.shape[0]):
    if ypredicted[i]>0.5:
        ypredicted[i]=1
    else:
        ypredicted[i] = 0
    


# In[73]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, ypredicted)


# In[74]:


from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, ypredicted)


In[69]:


count = 0
for i in range(ypredicted.shape[0]):
    if ypredicted[i] == 0:
        count+=1

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units=425 , kernel_initializer= "uniform",input_dim = 856))
    classifier.add(Activation('relu'))
    classifier.add(Dense(units=425 , kernel_initializer= "uniform"))
    classifier.add(Activation('relu'))
    classifier.add(Dense(units=425 , kernel_initializer= "uniform"))
    classifier.add(Activation('relu'))
    classifier.add(Dense(units=425 , kernel_initializer= "uniform"))
    classifier.add(Activation('relu'))
    classifier.add(Dense(units=1 , kernel_initializer= "uniform",))
    classifier.add(Activation('sigmoid'))
    classifier.compile(optimizer= 'adam' , loss= 'binary_crossentropy' , metrics= ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [30, 45,60],
              'epochs': [150, 250],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,n_jobs=-1)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

