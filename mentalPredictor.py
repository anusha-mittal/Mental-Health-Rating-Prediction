#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import joblib


# In[13]:


filename = 'enc.sav'
loaded_enc = joblib.load(filename)


# In[14]:


print(int(loaded_enc[0].transform(["Maybe"])))


# In[15]:


filename = 'mymodel.sav'
model = joblib.load(filename)


# In[16]:


def preprocess(arr):
    #label encoding
    arr[2]=int(loaded_enc[0].transform([arr[2]]))
    arr[5]=int(loaded_enc[1].transform([arr[5]]))
    arr[10]=int(loaded_enc[2].transform([arr[10]]))
    arr=np.array(arr)
    return arr
    
    
    


# In[17]:


arr=[19,1,"Maybe",4,4,"No",4,5,3,2,"No"]


# In[18]:


arr1=[20,2,"No",4,4,"Yes",4,5,3,2,"Yes"]


# In[ ]:





# In[19]:


def predict(arr):
    X=preprocess(arr)
    X=X.reshape(1, -1)
    #print(X)
    y=model.predict(X)
    return int(y)
    
    


# In[20]:


y=predict(arr)


# In[21]:


print(y)


# In[22]:


y=predict(arr1)
print(y)


# In[ ]:




