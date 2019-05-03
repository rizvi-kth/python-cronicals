from itertools import permutations


if __name__ == '__main__':
    # fptr = open(os.environ['OUTPUT_PATH'], 'w')

    word = input()
    k = int(input())

    for i in list(permutations(word, k)):
        print(i)

    # arr = []
    #
    # for _ in range(arr_count):
    #     arr_item = int(input().strip())
    #     arr.append(arr_item)
    #
    # k = int(input().strip())
    #
    # res = findNumber(arr, k)

    # print(res)
