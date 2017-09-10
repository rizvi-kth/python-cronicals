with open('words', encoding="utf-8") as f:
    L = {}
    for line in f:
        line = line.rstrip("\n\r")
        #line = line.rstrip("\n\r")

        if "'" in line:
            continue

        if len(line) == 5:
            nbrs = my_neb(line)
            L[line] = []   
            
print(L)

for k1 in words:
    for k2 in words:
        if neighbors(k1, k2):
            words[k1].append(k2)


print(len(L))