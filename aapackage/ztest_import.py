# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from future import standard_library
standard_library.install_aliases()
from builtins import next;      from builtins import map
from builtins import zip;       from builtins import str
from builtins import range;     from past.builtins import basestring
from past.utils import old_div; from builtins import object
############################################################################################
print("Start")

import os, sys

from collections import OrderedDict
from tabulate import tabulate
from datetime import datetime; from datetime import timedelta
from calendar import isleap






############################################################################################
import numpy as np,  pandas as pd, copy, scipy as sci, math as mth
from attrdict import AttrDict as dict2


import time,  shutil,  IPython, gc, copy
import matplotlib.pyplot as plt
import urllib3

from numba import jit, float32

import sklearn as sk
import tensorflow as tf
import arrow
import requests,  re, regex
from bs4 import BeautifulSoup

import calendar
import copy
import matplotlib.pyplot as plt
import numba
from dateutil import parser
from numba import jit, int32, float32, float64, int64
from tabulate import tabulate




############################################################################################
from aapackage import globalvar
from aapackage import allmodule
from aapackage import datanalysis
# import fast
from aapackage import function_custom


from aapackage import portfolio
# import rstatpy


from aapackage import util
from aapackage import util_ml
from aapackage import util_aws
from aapackage import util_release
# import util_search
from aapackage import util_spark
from aapackage import util_sql
from aapackage import util_web


from aapackage import util




############################################################################################
try :
  import matplotlib.pyplot as plt
  from matplotlib.collections import LineCollection

except Exception as e:
  print(e)







############################################################################################
print("Success")



print( [ x  for x in globals().keys() if x[0] != "_" ] )



############################################################################################
'''
import tensorflow as tf, numpy as np, pandas as pd, sys, os, argparse, arrow; from tabulate import tabulate
import util_min, util_ml as util_ml




import util
'''




############### util    ####################################################################
'''
Regression test : generate meta data of function

package_name, 
name,     aws_function
name_type1, 
name_type2, 
name_type3, 
name_comment
name_doc


arg_name, 
arg_type1, 
arg_default,
arg_comment, 


from . import util

'''



























