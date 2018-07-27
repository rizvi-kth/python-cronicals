
#%%
import numpy as np
from matplotlib import pyplot as plt

#plt.plot([2,5,7,9],[3,5,6,8])
#plt.show()

#https://www.youtube.com/watch?v=pQv6zMlYJ0A


data = np.genfromtxt('data.txt', delimiter=',',dtype=( float, float, float, float, '|U8'))
print(type(data))
print(data.shape)
print(data)

#%%
arr = []
arr2= []
for l in data:
    arr = np.append(arr, l[1])
    arr2=np.append(arr2,l[2])

print(arr)
print(arr2)

#%%
#print(data[:10])
plt.plot( arr, 'ro')
plt.plot( arr2, 'b.')
plt.show()
