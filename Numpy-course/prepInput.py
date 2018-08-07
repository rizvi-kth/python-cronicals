#%%
import numpy as np
from matplotlib import pyplot as plt

#plt.plot([2,5,7,9],[3,5,6,8])
#plt.show()

#https://www.youtube.com/watch?v=pQv6zMlYJ0A

data = np.genfromtxt('data/iris.txt', delimiter=',',dtype=( float, float, float, float, '|U8'))
print(type(data))
print(data.shape)
print(data)

#%%
# Imput preparation

sampleArr = []
isFirst = True

for l in data:
   
    sampleArr = np.array([l[0],l[1],l[2],l[3]])
    #print(sampleArr)
    if (isFirst):
        dataArr = [sampleArr]
        isFirst = False
    else:
        dataArr = np.append(dataArr, [sampleArr], axis = 0)


    #arr2=np.append(arr2,l[2])
print('Prepared input:')
print(dataArr)
#print(arr2)

#%%
outputs = []

# Output structure count ('Iris-set' 'Iris-ver' 'Iris-vir')
for l in data:
    sampleArr = np.array(l[4])
    outputs = np.append(outputs,sampleArr)

print('Unique outputs:')
print(np.unique(outputs))
print('Output nodes needed:')
print(np.unique(outputs).shape)
print('Sample count:')
print(outputs.shape)
print('Output matrix would be:')
print('%i X %i' %(outputs.shape[0], np.unique(outputs).shape[0]))


#%%
rows = outputs.shape[0]
cols = np.unique(outputs).shape[0]
outArr = np.zeros((rows,cols))
for indx,item in enumerate(outArr):
    #print(outputs[indx])
    if(outputs[indx] == 'Iris-set'):
        item[0] = 1
    if(outputs[indx] == 'Iris-ver'):
        item[1] = 1
    if(outputs[indx] == 'Iris-vir'):
        item[2] = 1
    
print(outArr)

#for indx,item in enumerate(b):
#    print('%s = %i' %(item,indx))
    


#%%
#print(data[:10])
#plt.plot( arr, 'ro')
#plt.plot( arr2, 'b.')
#plt.show()
