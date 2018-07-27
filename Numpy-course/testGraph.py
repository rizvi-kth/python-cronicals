
#%%
import numpy as np
from matplotlib import pyplot as plt

#plt.plot([2,5,7,9],[3,5,6,8])
#plt.show()

#https://www.youtube.com/watch?v=pQv6zMlYJ0A


data = np.genfromtxt('data.txt', delimiter=',',dtype=( float, float, float, float, '|U8'))
print(data)

#%%
print(data[:,0])

