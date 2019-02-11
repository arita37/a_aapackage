# -*- coding: utf-8 -*-
"""
Run task

python main.py  --hyperparam      --subprocess_script sub


"""
import os, sys, socket
import numpy as np, pandas as pd
import random
import toml
import time
import logging
import psutil
import arrow





APP_ID = __file__ + ',' + str(os.getpid()) + ',' + str(socket.gethostname())


def logs( s='', s1='', s2='', s3='', s4='', s5='', s6='', s7='', s8='', s9='' , APP_ID=""):
    # APP_ID = log_get_APPID() 
    try:
        prefix = APP_ID + ',' + arrow.utcnow().to('Japan').format("YYYYMMDD_HHmmss,")
        s = ','.join([prefix, str(s), str(s1), str(s2), str(s3), str(s4), str(s5),
                      str(s6), str(s7), str(s8), str(s9) ])

        # logging.info(s)
        print(s)
    except Exception as e:
        # logging.info(e)
        print(e)


def log_get_APPID() :
  APP_ID = __file__ + ',' + str(os.getpid()) + ',' + str(socket.gethostname())
  return APP_ID


def os_getparent(dir0):
    return os.path.abspath(os.path.join(dir0, os.pardir))


def ps_wait_process_complete(subprocess_list):
    for pid in subprocess_list:
        while True:
            try:
                pr = psutil.Process(pid)
                try:
                    pr_status = pr.status()
                except TypeError:  # psutil < 2.0
                    pr_status = pr.status
                except psutil.NoSuchProcess:  # pragma: no cover
                    break
                # Check if process status indicates we should exit
                if pr_status in [psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD]:
                    break
            except:
                break
            time.sleep(0.5)




def batch_result_folder(prefix) :
  BATCH_RESULT =  prefix + os.path.dirname(os.path.realpath(__file__)).split("/")[-1] +"/"
  if not os.path.isdir(BATCH_RESULT):
        os.mkdir(BATCH_RESULT)

  return BATCH_RESULT


def load_data_session(file_data , method ="spyder") :
  if method == "spyder"  :
    try :
      from spyderlib.utils.iofuncs import load_dictionary
      globals().update(load_dictionary(filedata)[0])
    except :
      pass

  if method == "shelve"  :
    import shelve
    "dict of var in myshelf.db"
    try :
      with shelve.open(file_data, flag='r') as db:
        for key in db:
         print(key,) 
         globals()[key] = db[key]
    except :
       pass    
   



def save_results(  BATCH_RESULT, ddict ):
    if not os.path.isdir(BATCH_RESULT):
        os.mkdir(BATCH_RESULT)

    import shelve
    "dict of var in myshelf.db"
    with shelve.open(file_data, flag='w') as db:
        db["res"] = ddict

    with open(BATCH_RESULT+"/result%i.txt" % ii, 'a') as ff :
        result = str(ddict)
        ff.write(result + "\n")
