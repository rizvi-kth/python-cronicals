#Read file content as stdin in a python script
#python readfile.py < /etc/file.try:
# Pass is no statemt
# Continue skips the loop
wanted = 7
found_index = -1
l = [4, 8, 7, 2, 9, 7]
for i, e in enumerate(l):
    if e == wanted:
        found_index = i
        break
print("Found index {}".format(found_index))

# Use index
l = [4, 8, 7, 2, 9, 7]
l.index(2)

#Take a random number
import random
goal = random.randint(100, 999)

#Excercise 7 (first part)
[i for i in range(int(1e6)) if str(i ** 2) == str(i ** 2)[::-1] ]
["{}{}".format(i,j) for i in range(10) for j in range(10)]

# Take some random number from a list
import random
L = [1,2,3,4,5,6,7,8,9]
random.sample(L, 3)    

#!!!!!!!!
list(filter(lambda s: s%2 == 0 , L))

# Doing the combination
from itertools import combinations
len(list( combinations(range(16), 3) ))

sum([1,2])
len([1,2])
mean = sum([1,2])/len([1,2]) 


#***  Mapping and filter  ***
#Map modifies the list elements - mutates the list from one to another form.
#Lambda function shoul be modifying the individual element of the list.
list(map(lambda s: s ** 2,[1,3.0,4,6,6]))
list(map(int,[1,3.0,4,6,6]))

#Filter matches each element with a criteria/constraint and evaluates True/False.
#Lambda functioin should be returning True/False and based on the result new list can be filtered with a criteria.
list(filter(lambda s: s%2 == 0,[1,3.0,4,6,6]))

#Declate a dictinary
d = dict([[1,None],[2,None]])
d1={1:None, 3:None}
# Dictioary building with out a key 
d = {"key", 12}
d = {"key":"value", 12:1000}
# Hash value of a key
hash("key")
#Only way to add in dictionary
d['key'] = 9999

#Experiment with d
d = {"key":"value", 12:1000}
hash("key") #4366877726699808123
hash(12) #12
d[4366877726699808123] = "sd"
d #{'key': 'value', 12: 1000, 4366877726699808123: 'sd'}
d[4366877726699808123] = 12
d #{'key': 'value', 12: 1000, 4366877726699808123: 12}
hash(4366877726699808123) #2061034717486114172



#Format to hex
"{:x}".format(12345) # 3034
int("3034",16)
#Align
"{:>30}{:^30}".format(12345, 5050)

