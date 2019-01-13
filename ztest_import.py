# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function



############################################################################################
print("Start")

import os, sys
from attrdict import AttrDict as dict2

import numpy as np,  pandas as pd, copy, scipy as sci, math as mth
import requests,  re
from bs4 import BeautifulSoup
from collections import OrderedDict

from tabulate import tabulate
from datetime import datetime; from datetime import timedelta; from calendar import isleap



import tensorflow as tf
import arrow


############################################################################################
import allmodule
import datanalysis
import fast
import function_custom
import globalvar

import multiprocessfunc
import portfolio
import portfolio_withdate
import rstatpy


import util
import util_ml 
import util_aws
import util_release
import util_search
import util_spark
import util_sql
import util_web






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



























