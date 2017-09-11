
# coding: utf-8

# In[33]:


#Sigmoid Function
def sigmoid (x):
    return 1/(1 + np.exp(-x))


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


# In[43]:


Act = sigmoid(Z)
Act
#Act.shape


# In[41]:


wh_1_2 = np.array([[ 0.3,  0.25,  0.23]]).T
wh_1_2
#wh_1_2.shape


# In[46]:


bh_1_2 = 0.69
Output_layer_input = np.dot( Act, wh_1_2) + bh_1_2
Output = sigmoid(Output_layer_input)
Output

