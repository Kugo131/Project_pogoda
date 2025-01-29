#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import seaborn as sns
import statsmodels.formula.api as smf
import math
import statsmodels.api as sm
import linearmodels.iv.model as lm

from linearmodels import PanelOLS
from scipy.optimize import minimize


# In[2]:


df = pd.read_csv('/mnt/HC_Volume_18315164/home-jupyter/jupyter-b-kugotov/5 number/knn2023.csv')
df


# In[3]:


import sklearn


# In[4]:


df.isna().sum()


# In[5]:


from sklearn.preprocessing import StandardScaler


# In[6]:


knn1 = df.drop(["Company","Year"], axis=1).copy()
scaler = StandardScaler()
knn1 = pd.DataFrame(scaler.fit_transform(knn1), columns = knn1.columns)


# In[7]:


Company = df[["Company","Year"]]
Company


# In[8]:


from sklearn.impute import KNNImputer


# In[9]:


knn_imputer = KNNImputer(n_neighbors = 5, weights = 'uniform')


# In[10]:


knn1 = pd.DataFrame(knn_imputer.fit_transform(knn1), columns = knn1.columns)


# In[11]:


knn1.isna().sum(), knn1.shape


# In[12]:


knn1 = pd.DataFrame(scaler.inverse_transform(knn1), columns = knn1.columns)
knn1


# In[13]:


knn1.to_excel('./knn2023.xlsx', sheet_name='knn2023', index=False)


# In[14]:


knn1["a1"]=knn1["a1"].apply(lambda x: math.log(x))


# In[15]:


knn1["ec1"]=knn1["ec1"].apply(lambda x: math.log(x))


# In[16]:


knn1["ec2"]=knn1["ec1"].apply(lambda x: math.log(x))


# In[17]:


knn1["ec3"]=knn1["ec3"].apply(lambda x: math.log(x))


# In[18]:


knn1["e1"]=knn1["e1"].apply(lambda x: math.log(x))


# In[19]:


knn1["e2"]=knn1["e2"].apply(lambda x: math.log(x))


# In[20]:


knn1["s1"]=knn1["s1"].apply(lambda x: math.log(x))


# In[21]:


knn1["s2"]=knn1["s2"].apply(lambda x: math.log(x))


# In[22]:


knn1["s3"]=knn1["s3"].apply(lambda x: math.log(x))


# In[23]:


knn1["s4"]=knn1["s4"].apply(lambda x: math.log(x))


# In[24]:


knn1["s5"]=knn1["s5"].apply(lambda x: math.log(x))


# In[25]:


knn1["m1"]=knn1["m1"].apply(lambda x: math.log(x))


# In[26]:


knn1["m2"]=knn1["m2"].apply(lambda x: math.log(x))


# In[27]:


knn1["p1"]=knn1["p1"].apply(lambda x: math.log(x))


# In[28]:


knn1.to_excel('./knn1log2023.xlsx', sheet_name='knn1log2023', index=False)


# In[29]:


descriptive_stat = knn1.describe()


# In[30]:


print("Описательная статистика:")
print(descriptive_stat)


# In[31]:


descriptive_stat.to_excel('./stat2023.xlsx', sheet_name='stat2023', index=False)


# In[32]:


knn1 = pd.concat([Company, knn1], axis=1)
knn1


# In[33]:


knn1.to_excel('./knn1itog2023.xlsx', sheet_name='knn1itog2023', index=False)


# In[ ]:





# In[34]:


knn3 = knn1.set_index(['Company','Year'])
knn3


# In[35]:


returns = knn3.pct_change().dropna()
correlation_matrix = returns.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Корреляция")
plt.show()


# In[36]:


knn4 = knn1


# In[37]:


knn4.drop(["Year"],axis=1, inplace= True)


# In[38]:


knn4.set_index(['Company'])


# In[39]:


model_2 =smf.ols(formula = 'p1 ~ a2 + a3 + a4 + a5 + e1 + e2', data=knn4).fit()
print(model_2.summary())


# In[40]:


model_21 =smf.ols(formula = 'p1 ~ a2 + a3 + a4 + a5 + e1 + e2 + ec1', data=knn4).fit()
print(model_21.summary())


# In[42]:


model_22 =smf.ols(formula = 'p1 ~ a2 + a3 + a4 + a5 + e1 + e2 + ec2', data=knn4).fit()
print(model_22.summary())


# In[43]:


model_23 =smf.ols(formula = 'p1 ~ a2 + a3 + a4 + a5 + e1 + e2 + ec3', data=knn4).fit()
print(model_23.summary())


# In[ ]:




