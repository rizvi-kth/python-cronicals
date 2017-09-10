
# coding: utf-8

# In[28]:


import numpy as np
X=np.array([[1],
            [0],
            [1],
            [0]]).T
X.shape


# In[29]:


wh = np.array([[.42,.88,.55],
               [.10,.73,.68],
               [.60,.18,.47],
               [.92,.11,.52]])
wh.shape


# In[30]:


bh = np.array([.46,.72,.08])
bh


# In[31]:


Z=np.dot(X,wh) + bh
Z


# In[ ]:




