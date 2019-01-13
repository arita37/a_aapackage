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
import arrow
import sys


def now_str():
    return arrow.now().format('YYYY-MM-DD HH:mm:ss')

print(now_str(), file=sys.stderr)

with open("dt_now.txt", 'w') as f:
    f.write(now_str())

sys.stderr.write('spam2\n')

print('spam3', file=sys.stderr)
