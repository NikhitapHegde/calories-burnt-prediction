#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.ensemble import RandomForestRegressor

import warnings 
warnings.filterwarnings('ignore')


# In[3]:


df=pd.read_csv('combined_user_data.csv')
df


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


sb.scatterplot(x=df['Height'], y=df['Weight']) 
plt.show()


# In[8]:


df.columns.tolist()


# In[9]:


features = ['Age', 'Height', 'Weight', 'Duration'] 
  
plt.subplots(figsize=(15, 10)) 
for i, col in enumerate(features): 
    plt.subplot(2, 2, i + 1) 
    x = df.sample(1000) 
    sb.scatterplot(x=x[col], y=x['Calories']) 
plt.tight_layout() 
plt.show() 


# In[10]:


features = df.select_dtypes(include='float').columns 
  
plt.subplots(figsize=(15, 10)) 
for i, col in enumerate(features): 
    plt.subplot(2, 3, i + 1) 
    sb.distplot(df[col]) 
plt.tight_layout() 
plt.show()


# In[11]:


df.replace({'male':0, 'female':1}, inplace=True)
df.head()


# In[12]:


plt.figure(figsize=(8, 8)) 
sb.heatmap(df.corr() > 0.9, 
           annot=True, 
           cbar=False) 
plt.show() 


# In[13]:


to_remove = ['Weight', 'Duration'] 
df.drop(to_remove, axis=1, inplace=True) 


# In[14]:


features = df.drop(['User_ID', 'Calories'], axis=1) 
target = df['Calories'].values 
  
X_train, X_val,Y_train, Y_val = train_test_split(features, target, 
                                      test_size=0.1, 
                                      random_state=22) 
X_train.shape, X_val.shape 


# In[15]:


scaler = StandardScaler() 
X_train = scaler.fit_transform(X_train) 
X_val = scaler.transform(X_val) 


# In[16]:


from sklearn.metrics import mean_absolute_error as mae 
models = [LinearRegression(), Lasso(), RandomForestRegressor(), Ridge()] 
  
for i in range(4): 
    models[i].fit(X_train, Y_train) 
  
    print(f'{models[i]} : ') 
  
    train_preds = models[i].predict(X_train) 
    print('Training Error : ', mae(Y_train, train_preds)) 
  
    val_preds = models[i].predict(X_val) 
    print('Validation Error : ', mae(Y_val, val_preds)) 
    print() 





