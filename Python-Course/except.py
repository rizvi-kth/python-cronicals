try:
    with open(r"Python-Course\words") as f:
        for line in f:
            last_line = line.rsplit()

    print(last_line)
except FileNotFoundError as e:
    raise FileNotFoundError("Wanted to read file words; but file not found! >> " + e.__str__())
