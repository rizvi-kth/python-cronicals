import sys

def roman(n):    
    if n < 4:
        return "I" * n
    if n == 4:
        return "IV" 
    if n > 4 and n < 9:
        return "V" + "I" * (n - 5)
    if n == 9:
        return "IX"
    if n > 9:
        return "X" + "I" * (n - 10)
        
    return

def assertEqual(first, second, msg=""):
    if first != second:
        msg += "\n{0!r} != {1!r}".format(first, second)
        raise AssertionError(msg)

def test_roman():
    assertEqual(roman(1), "I")
    assertEqual(roman(2), "II")
    assertEqual(roman(3), "III")
    assertEqual(roman(4), "IV")
    assertEqual(roman(5), "V")
    assertEqual(roman(6), "VI")
    assertEqual(roman(7), "VII")
    assertEqual(roman(8), "VIII")
    assertEqual(roman(9), "IX")
    assertEqual(roman(10), "X")


    print("All tests passed successfully.")

script_name = sys.argv[0]
arguments = sys.argv[1:]

USAGE = "Usage: {0} <number>".format(script_name)

if len(arguments) == 0 or len(arguments) > 1:
    print(USAGE)
elif arguments[0] == "test":
    test_roman()
else:
    number = int(arguments[0])
    print(roman(number))
