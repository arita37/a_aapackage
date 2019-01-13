#!/usr/bin/env python
'''
#START_HEADER_SCRIPT  #########################################################
# header: parameters for manage.py
NSPLIT=2
ARRAY=[ [1, 2, 3, 4], [2, 3, 4, 5] ]
#DATA_VAR_TO_SPLIT= [  "f1",  "f2", "f3"  ]   # Pandas dataframe or numpy array
#DATA_NSPLIT=   [  1, 3, 4 ]
#END_HEADER_SCRIPT ############################################################
'''
from __future__ import print_function
import time
import arrow


def now_str():
    return arrow.now().format('YYYY-MM-DD HH:mm:ss')

time.sleep(5)

print("------------------ %s ------------------\n" % now_str())
for i in range(0, 90):
    print("%s" % str(i))

with open("dt_now.txt", 'w') as f:
    f.write(now_str())
