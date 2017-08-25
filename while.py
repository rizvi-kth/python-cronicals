import math

def is_palind(v):
    w = str(v)
    org = math.sqrt(v)
    #while not found: 
        #w = input()
    if w == w [::-1]:
        print("This is a palindrom:{} of {}".format(w,org))
    # else:
    #     print("This is NOT a palindrom")

n = 1
while n < 1e6:
    v = n ** 2
    #print(v)
    is_palind(v)
    n += 1


# for i in range(int(1e6)):
#     print(i)

# The program can be written as a command by this
#[i for i in range(int(1e6)) if str(i ** 2) == str(i ** 2)[::-1] ]
#["{}{}".format(i,j) for i in range(10) for j in range(10)]


