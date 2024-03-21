#!/usr/bin/env python
# coding: utf-8

# In[45]:


#sourav kumar
import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[46]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV


# In[47]:


df = pd.read_csv('OLA_trips_dataset.csv')
df.head()


# In[48]:


df.shape


# In[49]:


df.describe()


# In[50]:


df.isnull().sum()


# In[51]:


distance_travelled = pd.DataFrame(df['distance_travelled'].value_counts())
distance_travelled


# In[52]:


sns.countplot(df['distance_travelled'])


# In[53]:


df.drop(['gender' , 'category'] , axis = 1)


# In[54]:


df.drop(['gender' , 'reason'] , axis = 1)


# In[55]:


attrition_dummies = pd.get_dummies(df['distance_travelled'])
attrition_dummies.head()


# In[56]:


attrition_dummies = pd.get_dummies(df['month'])
attrition_dummies.head()


# In[57]:


attrition_dummies = pd.get_dummies(df['category'])
attrition_dummies.head()


# In[58]:


df = pd.concat([df, distance_travelled] , axis = 1)
df.head()


# In[59]:


df.head(5)


# In[60]:


df.dtypes


# In[61]:


df.info()


# In[62]:


(df.isnull().sum()/len(df))


# In[63]:


df = df.dropna()


# In[64]:


(df.isnull().sum()/len(df))


# In[65]:


df = df.drop('category', axis = 1)


# In[66]:


df = df.drop('distance_travelled', axis = 1)


# In[67]:


df = df.drop('gender', axis = 1)


# In[68]:


plt.figure(figsize = (25,20))
sns.set(color_codes = True)
plt.subplot(3,2,2)
sns.histplot(df['time_taken'], kde = False)

plt.subplot(3,2,3)
sns.histplot(df['toll'], kde = False)

plt.subplot(3,2,4)
sns.histplot(df['driver_base_cost'], kde = False)

plt.subplot(3,2,5)
sns.histplot(df['total_tax'], kde = False)


# In[ ]:





# In[ ]:




