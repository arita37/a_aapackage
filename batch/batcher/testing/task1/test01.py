#!/usr/bin/env python
'''
#START_HEADER_SCRIPT  ###################################################################################
# header: parameters for manage.py
NSPLIT=2
ARRAY=[ [1, 2, 3, 4], [2, 3, 4, 5] ]
#DATA_VAR_TO_SPLIT= [  "f1",  "f2", "f3"  ]        # Pandas dataframe or numpy array
#DATA_NSPLIT=   [  1, 3, 4 ]
#END_HEADER_SCRIPT ######################################################################################
'''


import time, sys, os, arrow
def now_str() : return arrow.now().format('YYYY-MM-DD HH:mm:ss')


with open("task01.txt", "a") as f :
 f.write("\n\n ------------------------------------------")
 time.sleep(5)
 f.write( now_str() + " aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n")
 time.sleep(5)
 f.write( now_str()  + " qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq\n")
 time.sleep(1)
 f.write( now_str() + " bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n")
 time.sleep(1)
 f.write( now_str()  + " ccccccccccccccccccccccccccccccc\n")

 #for name in os.listdir("task/task_"+str(sys.argv[1])+"/input/"):
 #   f.write(name)

 #time.sleep(5)


with open("dt_now.txt", 'w') as f:
    f.write(now_str())


