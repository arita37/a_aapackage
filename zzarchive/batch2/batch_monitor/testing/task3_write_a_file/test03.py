#!/usr/bin/env python
"""
#START_HEADER_SCRIPT  #########################################################
# header: parameters for manage.py
NSPLIT=2
ARRAY=[ [1, 2, 3, 4], [2, 3, 4, 5] ]
#DATA_VAR_TO_SPLIT= [  "f1",  "f2", "f3"  ]   # Pandas dataframe or numpy array
#DATA_NSPLIT=   [  1, 3, 4 ]
#END_HEADER_SCRIPT ############################################################
"""
from __future__ import print_function

import math
import os
import time

import arrow


def now_str():
    return arrow.now().format("YYYY-MM-DD HH:mm:ss")


time.sleep(5)

filename = "%s%s%s" % (os.path.abspath("."), os.sep, "file_test03.txt")

f = open(filename, "w")
f.write("------------------ %s ------------------\n" % now_str())
try:
    for i in range(0, 180):
        f.write("%s\n" % math.radians(i))
finally:
    try:
        f.close()
    except:
        pass


with open("dt_now.txt", "w") as f:
    f.write(now_str())
