import numpy as np
import sys

if __name__ == '__main__':
    # print("Hello, World!")
    # r = np.inner([1, 2], [1, 2])
    # print(r)

    # w = sys.stdin.readlines()
    # print('word:', w)

    # m1 = sys.stdin.read()  # input("M1: ")
    # m2 = sys.stdin.read()  # input("M2: ")
    nums = []
    for line in sys.stdin:
        numbers = line.split()
        # print(numbers)
        nums.append(list(map(int, numbers)))

    # print(nums[0])
    # print(nums[1])

    # m1 = list(map(int, m1.strip().split()))
    # m2 = list(map(int, m2.strip().split()))
    # #
    # print(m1)
    # print(m2)
    #
    r = np.inner(nums[1], nums[1])
    print(r)

    r = np.outer(nums[1], nums[1])
    print(r)
