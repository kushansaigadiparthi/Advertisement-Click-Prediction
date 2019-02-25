#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('train.csv')
labeldata = dataset.pop('click').values
# del row['click'], row['id'], row['hour'],row['device_id'], row['device_ip'] ,row['device_model'] , row['device_conn_type'],row['device_type']
featuredata = dataset.drop(columns=['id','hour','device_id','device_ip','device_model','device_conn_type','device_type'],axis=1).values

featuredata[0]
featuredata.shape
featuredata[0]
labeldata.shape
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder_x = LabelEncoder()
for i in range(16):
    featuredata[:,i] = labelencoder_x.fit_transform(featuredata[:,i])
    
featuredata[0]
onehotencoder  = OneHotEncoder(categorical_features= 'all')
featuredata = onehotencoder.fit_transform(featuredata).toarray()
featuredata[0]
labelencoder_y = LabelEncoder()
labeldata = labelencoder_y.fit_transform(labeldata)


# In[24]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


# In[31]:


featuredata.shape


# In[37]:


model = LogisticRegression()
rfe = RFE(model, 600)
fit = rfe.fit(featuredata, labeldata)
print("Num Features: {}" .format(fit.n_features_))
print("Selected Features: {}".format(fit.support_))
print("Feature Ranking:{}".format(fit.ranking_))


# In[38]:


from sklearn.decomposition import PCA


# In[40]:


pca = PCA(n_components=600)
fit = pca.fit(featuredata)
# summarize components
print("Explained Variance:{}".format(fit.explained_variance_ratio_))
print(fit.components_)


# In[46]:


pca.score


# In[ ]:




