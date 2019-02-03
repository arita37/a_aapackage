# -*- coding: utf-8 -*-
'''Daemon monitoring batch '''
import os, sys
import numpy as np

import argparse
import gc
import logging
import socket
from time import sleep, time
import ujson
import arrow

###############################################################################
import util

from util_batch import *
from util_log import *



############### Variable definition ###########################################
logging.basicConfig(level=logging.INFO)
APP_ID   = __file__ + ',' + str(os.getpid()) + ',' + str(socket.gethostname())



#########Logging ##############################################################
def load_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_config",
                      help="Select the path for a .csv containing batch optimization parameters.\n One row per optimization.")

    parser.add_argument("--file_log", default="logfile_batchdaemon.txt",
                      help="Wether optimization will run locally or on a dedicated AWS instance.")

    options = parser.parse_args()
    return options




if __name__ != '__main__' :
  args     = load_arguments()   
  LOG_FILE = args.file_log
  
  while 1 :
   py_proc = server_pyprocess()   
   server_monitor_load(file_log)
   wait(60)   
   printlog("logs")   











