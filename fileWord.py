with open('words', encoding="utf-8") as f:
    L = []
    for line in f:
        line = line.rstrip("\n\r")
        #line = line.rstrip("\n\r")

        if "'" in line:
            continue

        if len(line) == 5:
            L.append(line)    
        # if line[::-1] == line:
        #     print(line)

print(L)
print(len(L))
