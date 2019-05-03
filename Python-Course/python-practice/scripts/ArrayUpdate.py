



def arrayManipulation(n, queries):
    print(n)
    print(queries)

    arr = [0] * n
    print(arr)

    for items3 in queries:
        print(items3)
        strIdx = items3[0] - 1
        endIdx = items3[1]
        for y in range(strIdx, endIdx):
            arr[y] += items3[2]

        print(arr)

    return max(arr)

    # # ints = str.split()
    # items3 = list(map(int, str.split()))
    # print(items3)
    # strIdx = items3[0]
    # endIdx = items3[1]+1
    # for x in range(strIdx, endIdx):
    #     arr[x] += items3[2]
    #
    # print(arr)


if __name__ == '__main__':
    f = open("input.txt", "r")
    # if f.mode == 'r':
    #     contents = f.read()
    #     print(contents)
    #
    queries = []
    header = True
    # or, readlines reads the individual line into a list
    if f.mode == 'r':
        fl = f.readlines()
        for x in fl:
            if header:
                nm = x.split()
                n = int(nm[0])
                m = int(nm[1])
            else:
                # for _ in range(m):
                queries.append(list(map(int, x.rstrip().split())))
            header = False

        result = arrayManipulation(n, queries)
        print(result)








    f.close()

