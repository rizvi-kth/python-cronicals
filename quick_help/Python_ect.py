
# Measure execution time for a given code
import time
start = time.time()
lcs_df.show() # <--- Target statement to measure
end = time.time()
hours, rem = divmod(end - start, 3600)
minutes, seconds = divmod(rem, 60)
print("Execution time (sec) : {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
