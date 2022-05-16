# using time module
import time
  
# ts stores the time in seconds
ts = int(time.time())

f = open("observations/myfile"+str(ts)+".txt", "w")
f.write("0,0,1,-1 0.33")
f.close()