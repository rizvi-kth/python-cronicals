
months = [
    "January",
    "Fabruary",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "Octber",
    "November",
    "December",
    ]
month_days = [31,28,30,31,30,31,30,31,30,31,30,31,]

#import sys
#command = int(sys.argv[1])

for i,j in zip(months,month_days):
    print("{} {}".format(i,j))

