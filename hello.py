#print("Hello world")
# _ prints/holds the last value computed. 
type(32)
from datetime import date
from datetime import timedelta
conf_date = date(2017,1,1)
to_day = date.today()
diff = conf_date - to_day
diff.days

date.today() + timedelta(1)

# Integer result division

#ctrl + R - search command history
#masak.org/carl/tmp/slides-day1.pdf
#masak.org/carl/tmp/exercises.pdf

# 4 Power 3
4 ** 3
# Modulo/ remainder
1970 % 100


import math
math.sqrt(18)
#math.sqrt(-18)
dir(math)
help(math.sqrt)
help(str.upper)
print( "abr".upper())

# Use this import to get print() from v3 to v2.7 (from __future__ import print_function, division, ...)
# Anything with __XX__ means a function of an operator or basic functionality
#from __future__ import print_function
print ("test") 

#id() function givs the memory addres of a referance type
id([1,2,3])
# X and y will ahve the same referance memory location 
x = y = [1,2,3]
id(x)
id(y)
#Compare referance type
x is y

# Split a string
"test test2".split()
#False
bool([])
#True
bool([1])
L = []
if L:
    print ("L is not empty")

#Multiply a string
"Na" * 15
"Yes" + str(15) + "Na"
"string" + str(7)
print("#" * 15)
"-- {:4.2f} -- ".format(12345.3333)

#Find your age
birth_day = "19790927"
my_year = int(birth_day[0:4])
from datetime import *
this_year = datetime.today().year
this_year - my_year

#None will not print anything in the console
None

l = [1,5,2,]
l.sort() # Mutable - so workes
l.append(7)
l.insert(9,2)
# List slicing
l[:3]
l[2:3]
l[2:]
"hello world"[::-1]
#copy list
l2 = l[:]

for i in l:
    print (i)

list(range(1,5))

#Pairwise zip two lists 
list(zip(l,l2)) # returns a List of tauples pair wise elements of l and l2

#Multiply 2 lists of elements and put it another list
l3 = [a*b for b in l for a in l2]
#Little advanced - call a function
#l3 = [printIfSame(a,b) for b in l for a in l2]
#If two list has common element
l3 = [a == b and print(a) for b in l for a in l2]

#Take an input
x = input("Enter:")
print("You have given {}".format(int(x)))
